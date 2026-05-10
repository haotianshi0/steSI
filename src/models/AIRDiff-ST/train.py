import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_AIRGATE_ROOT = os.path.join(_PROJECT_ROOT, "src", "models", "AIRGate-ST")
sys.path.insert(0, _AIRGATE_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "shared"))

DEFAULT_DATA_PATH = os.path.join(
    _PROJECT_ROOT, "data", "151673", "adata_ai_compressed.h5ad"
)

from shared.data_utils import (  # noqa: E402
    load_data,
    filter_sites,
    train_val_split,
    compute_ratio,
    eval_metrics,
    load_external_masks,
    validate_prediction_output,
)
from inner_mask_utils import make_inner_validation_split  # noqa: E402


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * -(np.log(10000.0) / max(half - 1, 1))
        )
        x = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


class ConditionalDenoiser(nn.Module):
    """Coordinate DDPM denoiser conditioned on train-observed summaries."""

    def __init__(self, n_spots: int, n_sites: int, emb_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.spot_emb = nn.Embedding(n_spots, emb_dim)
        self.site_emb = nn.Embedding(n_sites, emb_dim)
        self.time_emb = TimeEmbedding(emb_dim)
        in_dim = emb_dim * 3 + 10
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        spot_ids: torch.Tensor,
        site_ids: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_features: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.cat(
            [
                self.spot_emb(spot_ids),
                self.site_emb(site_ids),
                self.time_emb(t),
                x_t.unsqueeze(1),
                cond_features,
            ],
            dim=1,
        )
        return self.net(h).squeeze(1)


def make_beta_schedule(steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.linspace(1e-4, 2e-2, steps, device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def build_condition_features(
    ratio: np.ndarray,
    train_mask: np.ndarray,
    depth: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    train_values = np.where(train_mask, ratio, np.nan)
    train_depth = np.where(train_mask, depth, np.nan)

    site_count = train_mask.sum(axis=0).astype(np.float32)
    spot_count = train_mask.sum(axis=1).astype(np.float32)
    global_mean = float(np.nanmean(train_values))
    site_sum = np.nansum(train_values, axis=0)
    spot_sum = np.nansum(train_values, axis=1)
    site_mean = np.divide(site_sum, site_count, out=np.full_like(site_sum, global_mean, dtype=np.float32), where=site_count > 0)
    spot_mean = np.divide(spot_sum, spot_count, out=np.full_like(spot_sum, global_mean, dtype=np.float32), where=spot_count > 0)
    site_mean = np.where(np.isfinite(site_mean), site_mean, global_mean)
    spot_mean = np.where(np.isfinite(spot_mean), spot_mean, global_mean)

    global_depth = float(np.nanmean(train_depth))
    site_depth_sum = np.nansum(train_depth, axis=0)
    spot_depth_sum = np.nansum(train_depth, axis=1)
    site_depth = np.divide(site_depth_sum, site_count, out=np.full_like(site_depth_sum, global_depth, dtype=np.float32), where=site_count > 0)
    spot_depth = np.divide(spot_depth_sum, spot_count, out=np.full_like(spot_depth_sum, global_depth, dtype=np.float32), where=spot_count > 0)
    site_depth = np.where(np.isfinite(site_depth), site_depth, global_depth)
    spot_depth = np.where(np.isfinite(spot_depth), spot_depth, global_depth)

    positive_values = np.where(train_mask, train_values > 0, np.nan)
    site_positive_sum = np.nansum(positive_values, axis=0)
    spot_positive_sum = np.nansum(positive_values, axis=1)
    site_positive = np.divide(site_positive_sum, site_count, out=np.zeros_like(site_positive_sum, dtype=np.float32), where=site_count > 0)
    spot_positive = np.divide(spot_positive_sum, spot_count, out=np.zeros_like(spot_positive_sum, dtype=np.float32), where=spot_count > 0)
    site_positive = np.where(np.isfinite(site_positive), site_positive, 0.0)
    spot_positive = np.where(np.isfinite(spot_positive), spot_positive, 0.0)

    n_spots, n_sites = ratio.shape
    feat = np.stack(
        [
            site_mean[cols],
            spot_mean[rows],
            np.full(rows.shape, global_mean, dtype=np.float32),
            np.log1p(site_count[cols]) / np.log1p(max(n_spots, 1)),
            np.log1p(spot_count[rows]) / np.log1p(max(n_sites, 1)),
            np.log1p(site_depth[cols]) / np.log1p(max(global_depth, 1.0)),
            np.log1p(spot_depth[rows]) / np.log1p(max(global_depth, 1.0)),
            site_positive[cols],
            spot_positive[rows],
        ],
        axis=1,
    )
    return feat.astype(np.float32)


def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(x0)
    ab = alpha_bar[t]
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise
    return x_t, noise


def train_epoch(
    model: ConditionalDenoiser,
    optimizer: torch.optim.Optimizer,
    coords: torch.Tensor,
    x0: torch.Tensor,
    cond: torch.Tensor,
    alpha_bar: torch.Tensor,
    batch_size: int,
) -> float:
    model.train()
    n = coords.shape[0]
    perm = torch.randperm(n, device=x0.device)

    total = 0.0
    nb = 0
    for start in range(0, perm.numel(), batch_size):
        idx = perm[start : start + batch_size]
        t = torch.randint(0, alpha_bar.numel(), (idx.numel(),), device=x0.device)
        x_t, noise = q_sample(x0[idx], t, alpha_bar)
        pred_noise = model(coords[idx, 0], coords[idx, 1], x_t, t, cond[idx])
        signal_w = torch.where(x0[idx] > 0, torch.tensor(2.0, device=x0.device), torch.tensor(1.0, device=x0.device))
        loss = (signal_w * (pred_noise - noise) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.item())
        nb += 1
    return total / max(nb, 1)


def sample_inner_target_indices(
    train_x: np.ndarray,
    frac: float,
    seed: int,
    high_threshold: float,
    high_weight: float,
) -> np.ndarray:
    """Sample observed train entries to hide from the conditional context."""
    n = int(train_x.shape[0])
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    n_target = max(1, int(n * min(max(frac, 1e-6), 1.0)))
    n_target = min(n_target, n - 1)
    weights = np.ones(n, dtype=np.float64)
    weights[train_x >= high_threshold] *= float(high_weight)
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=n_target, replace=False, p=weights)


@torch.no_grad()
def sample_predictions(
    model: ConditionalDenoiser,
    coords_np: np.ndarray,
    cond_np: np.ndarray,
    shape: tuple[int, int],
    beta: torch.Tensor,
    alpha: torch.Tensor,
    alpha_bar: torch.Tensor,
    device: torch.device,
    batch_size: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = coords_np.shape[0]
    samples = []
    coords_all = torch.tensor(coords_np, dtype=torch.long, device=device)
    cond_all = torch.tensor(cond_np, dtype=torch.float32, device=device)
    for _ in range(n_samples):
        out = np.empty(n, dtype=np.float32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            coords = coords_all[start:end]
            cond = cond_all[start:end]
            x = torch.randn(end - start, device=device)
            for step in range(beta.numel() - 1, -1, -1):
                t = torch.full((end - start,), step, dtype=torch.long, device=device)
                eps = model(coords[:, 0], coords[:, 1], x, t, cond)
                coef = beta[step] / torch.sqrt((1.0 - alpha_bar[step]).clamp_min(1e-8))
                mean = (x - coef * eps) / torch.sqrt(alpha[step])
                if step > 0:
                    x = mean + torch.sqrt(beta[step]) * torch.randn_like(x)
                else:
                    x = mean
            out[start:end] = x.clamp(0.0, 1.0).cpu().numpy()
        samples.append(out)
    sample_arr = np.stack(samples, axis=0)
    mean_flat = sample_arr.mean(axis=0)
    std_flat = sample_arr.std(axis=0)
    pred = np.full(shape, np.nan, dtype=np.float32)
    unc = np.full(shape, np.nan, dtype=np.float32)
    pred[coords_np[:, 0], coords_np[:, 1]] = mean_flat
    unc[coords_np[:, 0], coords_np[:, 1]] = std_flat
    return pred, unc


def main() -> None:
    parser = argparse.ArgumentParser(description="Observed-mask conditional diffusion for ratio imputation.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--emb_dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--sample_batch_size", type=int, default=262144)
    parser.add_argument("--n_samples", type=int, default=3)
    parser.add_argument("--inner_eval_samples", type=int, default=3,
                        help="Number of diffusion samples averaged for inner epoch selection.")
    parser.add_argument("--predict_unobserved", action="store_true",
                        help="Also sample originally unobserved entries. Default samples holdout only.")
    parser.add_argument("--inner_mask_frac", type=float, default=0.6,
                        help="Fraction of fit-train entries hidden from condition and used as denoising targets each epoch.")
    parser.add_argument("--inner_high_threshold", type=float, default=0.05)
    parser.add_argument("--inner_high_weight", type=float, default=2.0)
    parser.add_argument("--inner_val_frac", type=float, default=0.10)
    parser.add_argument("--min_depth", type=int, default=10)
    parser.add_argument("--min_obs", type=int, default=10)
    parser.add_argument("--holdout", type=float, default=0.10)
    parser.add_argument("--train_mask_path", default=None)
    parser.add_argument("--val_mask_path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", default="results/AIRDiff-ST")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    A_raw, G_raw, _coords, _ = load_data(args.data_path)
    A, G, _keep = filter_sites(A_raw, G_raw, min_obs=args.min_obs)
    observed_for_mask = (A + G) >= args.min_depth if args.min_depth > 0 else (A + G) > 0
    external_masks = load_external_masks(
        args.train_mask_path,
        args.val_mask_path,
        shape=A.shape,
        observed_mask=observed_for_mask,
    )
    if external_masks is None:
        outer_train_mask, val_mask = train_val_split(
            A, G, holdout_frac=args.holdout, seed=args.seed, min_depth=args.min_depth
        )
    else:
        outer_train_mask, val_mask = external_masks

    fit_train_mask, inner_val_mask = make_inner_validation_split(
        outer_train_mask, frac=args.inner_val_frac, seed=args.seed + 10_000, min_train_per_site=1
    )
    train_mask = fit_train_mask if inner_val_mask.any() else outer_train_mask
    epoch_select_mask = inner_val_mask if inner_val_mask.any() else train_mask

    depth = A + G
    ratio = np.zeros_like(A, dtype=np.float32)
    valid = depth > 0
    ratio[valid] = (G[valid] / depth[valid]).astype(np.float32)
    gt_ratio = compute_ratio(A, G, mask=observed_for_mask)

    train_coords_np = np.argwhere(train_mask).astype(np.int64)
    train_x_np = ratio[train_mask].astype(np.float32)
    beta, alpha, alpha_bar = make_beta_schedule(args.diffusion_steps, device)
    model = ConditionalDenoiser(A.shape[0], A.shape[1], args.emb_dim, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    best_state = None
    best_inner_rmse = float("inf")
    print(
        f"[Config] model=conditional_diffusion device={device} epochs={args.epochs} "
        f"steps={args.diffusion_steps} train={int(train_mask.sum()):,} "
        f"inner_val={int(inner_val_mask.sum()):,} final_holdout={int(val_mask.sum()):,} "
        f"inner_mask_frac={args.inner_mask_frac}"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        target_idx_np = sample_inner_target_indices(
            train_x_np,
            frac=args.inner_mask_frac,
            seed=args.seed * 1000 + epoch,
            high_threshold=args.inner_high_threshold,
            high_weight=args.inner_high_weight,
        )
        target_coords_np = train_coords_np[target_idx_np]
        visible_mask = train_mask.copy()
        visible_mask[target_coords_np[:, 0], target_coords_np[:, 1]] = False
        target_cond_np = build_condition_features(
            ratio,
            visible_mask,
            depth,
            target_coords_np[:, 0],
            target_coords_np[:, 1],
        )
        target_coords_t = torch.tensor(target_coords_np, dtype=torch.long, device=device)
        target_x_t = torch.tensor(train_x_np[target_idx_np], dtype=torch.float32, device=device)
        target_cond_t = torch.tensor(target_cond_np, dtype=torch.float32, device=device)

        loss = train_epoch(
            model,
            optimizer,
            target_coords_t,
            target_x_t,
            target_cond_t,
            alpha_bar,
            args.batch_size,
        )
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            eval_coords_np = np.argwhere(epoch_select_mask).astype(np.int64)
            eval_cond_np = build_condition_features(
                ratio, train_mask, depth, eval_coords_np[:, 0], eval_coords_np[:, 1]
            )
            pred_eval, _ = sample_predictions(
                model,
                eval_coords_np,
                eval_cond_np,
                A.shape,
                beta,
                alpha,
                alpha_bar,
                device,
                args.sample_batch_size,
                args.inner_eval_samples,
            )
            m = eval_metrics(pred_eval, A, G, epoch_select_mask)
            print(
                f"[Epoch {epoch:04d}/{args.epochs}] loss={loss:.5f} "
                f"inner_rmse={m['rmse']:.5f} inner_r={m['r']:.4f} time={time.time()-t0:.1f}s"
            )
            if m["rmse"] < best_inner_rmse:
                best_inner_rmse = m["rmse"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"[Epoch {epoch:04d}/{args.epochs}] loss={loss:.5f} time={time.time()-t0:.1f}s")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if args.predict_unobserved:
        pred_coords_np = np.argwhere(~outer_train_mask).astype(np.int64)
    else:
        pred_coords_np = np.argwhere(val_mask).astype(np.int64)
    pred_cond_np = build_condition_features(ratio, train_mask, depth, pred_coords_np[:, 0], pred_coords_np[:, 1])
    pred, unc = sample_predictions(
        model,
        pred_coords_np,
        pred_cond_np,
        A.shape,
        beta,
        alpha,
        alpha_bar,
        device,
        args.sample_batch_size,
        args.n_samples,
    )

    out_dir = os.path.join(args.output_dir, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    train_ratio = gt_ratio.copy()
    train_ratio[~outer_train_mask] = np.nan
    fit_train_ratio = gt_ratio.copy()
    fit_train_ratio[~train_mask] = np.nan
    pred[outer_train_mask] = gt_ratio[outer_train_mask]
    pred = validate_prediction_output("AIRDiff-ST", pred, gt_ratio, outer_train_mask, val_mask)

    np.save(os.path.join(out_dir, "pred_ratio.npy"), pred)
    np.save(os.path.join(out_dir, "pred_uncertainty.npy"), unc)
    np.save(os.path.join(out_dir, "gt_ratio.npy"), gt_ratio)
    np.save(os.path.join(out_dir, "train_ratio.npy"), train_ratio)
    np.save(os.path.join(out_dir, "fit_train_ratio.npy"), fit_train_ratio)
    np.save(os.path.join(out_dir, "inner_val_mask.npy"), inner_val_mask)
    np.save(os.path.join(out_dir, "fit_train_mask.npy"), train_mask)
    np.save(os.path.join(out_dir, "val_mask.npy"), val_mask)

    final_metrics = eval_metrics(pred, A, G, val_mask)
    with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"model=AIRDiff-ST\nseed={args.seed}\nepochs={args.epochs}\n"
            f"diffusion_steps={args.diffusion_steps}\ninner_eval_samples={args.inner_eval_samples}\n"
            f"inner_mask_frac={args.inner_mask_frac}\n"
            f"inner_mask_semantics=target entries are removed from condition features during training\n"
            f"best_inner_rmse={best_inner_rmse:.6f}\n"
            f"rmse={final_metrics['rmse']:.6f}\nmae={final_metrics['mae']:.6f}\n"
            f"r={final_metrics['r']:.6f}\ncos={final_metrics['cos']:.6f}\n"
        )
    print(f"[Done] AIRDiff-ST best_inner_rmse={best_inner_rmse:.5f}")
    print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    main()
