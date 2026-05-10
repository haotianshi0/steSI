"""
AIRGate-ST training entry point.

AIRGate-ST is a two-stage zero-inflation model for spatial A-to-I editing
ratio imputation. It separates editability from editing strength:

  gate  = sigmoid(logit_gate)
  ratio = sigmoid(logit_ratio)
  pred  = gate * ratio

The final model uses spot/site embeddings, site-specific bias terms, spatial
KNN aggregation, and a learned spatial attention gate. Training values are
preserved exactly in the final prediction matrix; only held-out observed
entries are used for metric evaluation.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_AIRGATE_ROOT = _HERE
sys.path.insert(0, _AIRGATE_ROOT)
_PROJECT_ROOT = os.path.abspath(os.path.join(_AIRGATE_ROOT, "..", "..", ".."))
DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "151673", "adata_ai_compressed.h5ad")

from shared.data_utils import (
    load_data, filter_sites, build_knn_graph,
    train_val_split, compute_ratio, eval_metrics, plot_heatmaps,
    plot_diagnostics,
    load_external_masks, validate_prediction_output,
)

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src", "shared"))
from inner_mask_utils import make_inner_validation_split  # noqa: E402


# Model

class TwoStageModel(nn.Module):
    """
    AIRGate-ST two-stage zero-inflation model.

    Embeddings
    ----------
    e_spot : (n_spots, emb_dim) spot latent representation
    e_site : (n_sites, emb_dim) site latent representation
    b_spot_gate / b_spot_ratio : per-spot bias
    b_site_gate / b_site_ratio : per-site bias

    The spatial branch augments spot embeddings with inverse-distance weighted
    KNN neighbour information. The attention branch lets the gate learn which
    neighbouring spots are most informative for editability.
    """

    def __init__(self, n_spots: int, n_sites: int,
                 emb_dim: int = 32,
                 use_spatial: bool = False,
                 use_attn_gate: bool = False,
                 knn_k: int = 6,
                 separate_emb: bool = True,
                 site_ratio_logit_init=None):
        super().__init__()
        self.n_spots       = n_spots
        self.n_sites       = n_sites
        self.emb_dim       = emb_dim
        self.use_spatial   = use_spatial
        self.use_attn_gate = use_attn_gate
        self.knn_k         = knn_k
        self.separate_emb  = separate_emb

        if separate_emb:
            # Independent embeddings prevent gate-to-ratio interference.
            self.e_spot_gate  = nn.Embedding(n_spots, emb_dim)
            self.e_site_gate  = nn.Embedding(n_sites, emb_dim)
            self.e_spot_ratio = nn.Embedding(n_spots, emb_dim)
            self.e_site_ratio = nn.Embedding(n_sites, emb_dim)
        else:
            self.e_spot = nn.Embedding(n_spots, emb_dim)
            self.e_site = nn.Embedding(n_sites, emb_dim)

        self.b_spot_gate  = nn.Embedding(n_spots, 1)
        self.b_site_gate  = nn.Embedding(n_sites, 1)
        self.b_spot_ratio = nn.Embedding(n_spots, 1)
        self.b_site_ratio = nn.Embedding(n_sites, 1)

        if use_spatial:
            self.log_idw_power = nn.Parameter(torch.zeros(1))  # softplus -> >0

        if use_attn_gate:
            self.attn_gate_mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1),
            )

        self._init_weights(site_ratio_logit_init)

    def _init_weights(self, site_ratio_logit_init=None):
        if self.separate_emb:
            for emb in (self.e_spot_gate, self.e_site_gate,
                        self.e_spot_ratio, self.e_site_ratio):
                nn.init.normal_(emb.weight, std=0.02)
        else:
            nn.init.normal_(self.e_spot.weight, std=0.02)
            nn.init.normal_(self.e_site.weight, std=0.02)
        nn.init.zeros_(self.b_spot_gate.weight)
        nn.init.constant_(self.b_site_gate.weight, -2.0)  # prior: mostly unedited
        nn.init.zeros_(self.b_spot_ratio.weight)

        # Warm-start ratio head from per-site mean editing strength.
        if site_ratio_logit_init is not None:
            init_t = torch.tensor(site_ratio_logit_init, dtype=torch.float32)
            self.b_site_ratio.weight.data.copy_(init_t.view(-1, 1))
        else:
            nn.init.zeros_(self.b_site_ratio.weight)

    def _spatial_aggregate(self, e_spot: torch.Tensor,
                            knn_idx: torch.Tensor,
                            knn_dist: torch.Tensor) -> torch.Tensor:
        """
        Aggregate KNN neighbour embeddings using IDW weights.
        e_spot: (n_spots, emb_dim)
        Returns augmented e_spot (n_spots, emb_dim).
        """
        power = F.softplus(self.log_idw_power) + 1.0  # >1
        w = 1.0 / (knn_dist ** power + 1e-6)          # (n_spots, k)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)

        # Gather neighbour embeddings
        flat_idx = knn_idx.reshape(-1)                 # (n*k,)
        e_nbr    = e_spot[flat_idx]                    # (n*k, emb)
        e_nbr    = e_nbr.reshape(self.n_spots,
                                  self.knn_k, self.emb_dim)  # (n, k, emb)
        agg = (w.unsqueeze(-1) * e_nbr).sum(dim=1)    # (n, emb)
        return e_spot + 0.5 * agg                      # residual

    def forward(self, spot_ids: torch.Tensor, site_ids: torch.Tensor,
                knn_idx=None, knn_dist=None):
        """Returns (gate_prob, ratio_prob) each (B,)."""
        if self.separate_emb:
            e_s_gate_all = self.e_spot_gate.weight
            e_s_rat_all  = self.e_spot_ratio.weight
            if self.use_spatial and knn_idx is not None:
                e_s_gate_all = self._spatial_aggregate(e_s_gate_all, knn_idx, knn_dist)
                e_s_rat_all  = self._spatial_aggregate(e_s_rat_all,  knn_idx, knn_dist)
            e_s_gate = e_s_gate_all[spot_ids]
            e_s_rat  = e_s_rat_all[spot_ids]
            e_t_gate = self.e_site_gate(site_ids)
            e_t_rat  = self.e_site_ratio(site_ids)
            dot_gate = (e_s_gate * e_t_gate).sum(-1)
            dot_rat  = (e_s_rat  * e_t_rat).sum(-1)
            e_s_base_gate = self.e_spot_gate(spot_ids)
        else:
            e_s_all = self.e_spot.weight
            if self.use_spatial and knn_idx is not None:
                e_s_all = self._spatial_aggregate(e_s_all, knn_idx, knn_dist)
            e_s = e_s_all[spot_ids]
            e_t = self.e_site(site_ids)
            dot_gate = dot_rat = (e_s * e_t).sum(-1)
            e_s_gate = e_s
            e_s_base_gate = self.e_spot(spot_ids)

        if self.use_attn_gate and knn_idx is not None:
            gate_input = torch.cat([e_s_base_gate, e_s_gate], dim=-1)
            gate_logit = self.attn_gate_mlp(gate_input).squeeze(-1) + \
                         self.b_site_gate(site_ids).squeeze(-1)
        else:
            gate_logit = dot_gate + \
                         self.b_spot_gate(spot_ids).squeeze(-1) + \
                         self.b_site_gate(site_ids).squeeze(-1)

        ratio_logit = dot_rat + \
                      self.b_spot_ratio(spot_ids).squeeze(-1) + \
                      self.b_site_ratio(site_ids).squeeze(-1)

        return torch.sigmoid(gate_logit), torch.sigmoid(ratio_logit)

    @torch.no_grad()
    def predict_all(self, device: torch.device,
                    knn_idx=None, knn_dist=None,
                    chunk: int = 2048) -> np.ndarray:
        """Returns (n_spots, n_sites) predicted ratio matrix."""
        out = np.empty((self.n_spots, self.n_sites), dtype=np.float32)

        if self.separate_emb:
            e_s_gate = self.e_spot_gate.weight
            e_s_rat  = self.e_spot_ratio.weight
            if self.use_spatial and knn_idx is not None:
                e_s_gate = self._spatial_aggregate(e_s_gate, knn_idx, knn_dist)
                e_s_rat  = self._spatial_aggregate(e_s_rat,  knn_idx, knn_dist)
            e_s_base_gate = self.e_spot_gate.weight
        else:
            e_s = self.e_spot.weight
            if self.use_spatial and knn_idx is not None:
                e_s = self._spatial_aggregate(e_s, knn_idx, knn_dist)
            e_s_gate = e_s_rat = e_s
            e_s_base_gate = self.e_spot.weight

        bsg = self.b_spot_gate.weight.squeeze(-1)
        bsr = self.b_spot_ratio.weight.squeeze(-1)

        if self.use_attn_gate and knn_idx is not None:
            gate_in = torch.cat([e_s_base_gate, e_s_gate], dim=-1)
            gate_spot_logit = self.attn_gate_mlp(gate_in).squeeze(-1)   # (n_spots,)
        else:
            gate_spot_logit = None

        for s in range(0, self.n_sites, chunk):
            e = min(s + chunk, self.n_sites)
            if self.separate_emb:
                e_t_gate = self.e_site_gate.weight[s:e]
                e_t_rat  = self.e_site_ratio.weight[s:e]
                dot_gate = e_s_gate @ e_t_gate.T
                dot_rat  = e_s_rat  @ e_t_rat.T
            else:
                e_t = self.e_site.weight[s:e]
                dot_gate = dot_rat = e_s_gate @ e_t.T

            btg = self.b_site_gate.weight[s:e, 0]
            btr = self.b_site_ratio.weight[s:e, 0]

            if gate_spot_logit is not None:
                gate = torch.sigmoid(gate_spot_logit.unsqueeze(1) + btg.unsqueeze(0))
            else:
                gate = torch.sigmoid(dot_gate + bsg.unsqueeze(1) + btg.unsqueeze(0))

            ratio = torch.sigmoid(dot_rat + bsr.unsqueeze(1) + btr.unsqueeze(0))

            out[:, s:e] = (gate * ratio).cpu().float().numpy()

        return out


# Loss

def combined_loss(gate: torch.Tensor, ratio: torch.Tensor,
                  true_ratio: torch.Tensor, depth: torch.Tensor,
                  alpha: float = 1.0,
                  extreme_lambda: float = 0.0,
                  extreme_pivot: float = 0.1,
                  p_high_natural: float = -1.0,
                  batch_high_frac: float = -1.0,
                  zero_anchor_beta: float = 0.0) -> torch.Tensor:
    """
    L = wBCE(gate, is_edited) + alpha * weighted_MSE(ratio, true_ratio).

    Fix MSE weight = depth_w * (1 + extreme_lambda * max(r - pivot, 0)).
    High-ratio samples (systematically under-predicted) get a much larger
    share of the ratio gradient.

    Method-B gate de-bias: when the batch is oversampled for high-r entries
    (p_high_natural >= 0 and batch_high_frac > 0), each sample's gate BCE is
    weighted so the *effective* label distribution equals the natural one.
      w_high = p_high_natural / batch_high_frac
      w_low  = (1 - p_high_natural) / (1 - batch_high_frac)
    This lets the ratio head keep its strong gradient signal (bhf=0.5)
    without teaching the gate to over-fire on unseen cells.
    """
    is_edited = (true_ratio > 0).float()

    per_sample_bce = F.binary_cross_entropy(gate, is_edited, reduction="none")
    if (p_high_natural >= 0.0 and
            0.0 < batch_high_frac < 1.0):
        is_high = (true_ratio > extreme_pivot).float()
        w_high  = p_high_natural / batch_high_frac
        w_low   = (1.0 - p_high_natural) / (1.0 - batch_high_frac)
        w_gate  = is_high * w_high + (1.0 - is_high) * w_low
        gate_loss = (w_gate * per_sample_bce).sum() / (w_gate.sum() + 1e-6)
    else:
        gate_loss = per_sample_bce.mean()

    edit_mask = is_edited.bool()
    if edit_mask.sum() > 0:
        d       = depth[edit_mask]
        r_true  = true_ratio[edit_mask]
        r_pred  = ratio[edit_mask]

        w_depth   = d / (d.mean() + 1e-6)
        w_extreme = 1.0 + extreme_lambda * torch.clamp(r_true - extreme_pivot, min=0.0)
        w         = w_depth * w_extreme

        ratio_loss = (w * (r_pred - r_true) ** 2).sum() / (w.sum() + 1e-6)
    else:
        ratio_loss = torch.tensor(0.0, device=gate.device)

    # Zero-anchor: penalize gate*ratio>0 at observed r=0.
    # Direct anti-hallucination signal at supervised zero positions.
    if zero_anchor_beta > 0.0:
        zero_mask_local = ~edit_mask
        if zero_mask_local.sum() > 0:
            pred_at_zero = gate[zero_mask_local] * ratio[zero_mask_local]
            zero_anchor_loss = (pred_at_zero ** 2).mean()
        else:
            zero_anchor_loss = torch.tensor(0.0, device=gate.device)
        return gate_loss + alpha * ratio_loss + zero_anchor_beta * zero_anchor_loss

    return gate_loss + alpha * ratio_loss


# Training loop

def run_epoch(model: TwoStageModel,
              optimizer: torch.optim.Optimizer,
              train_coords: torch.Tensor,
              ratio_t: torch.Tensor, depth_t: torch.Tensor,
              knn_idx, knn_dist,
              alpha: float, batch_size: int,
              device: torch.device,
              extreme_lambda: float = 0.0,
              extreme_pivot: float = 0.1,
              balanced_high_frac: float = 0.0,
              high_idx=None, low_idx=None,
              p_high_natural: float = -1.0,
              zero_anchor_beta: float = 0.0) -> float:
    """
    Fix when balanced_high_frac > 0, each batch pulls that fraction of
    entries from the high-ratio pool (with replacement) and the rest from
    the low-ratio pool (sequential). An 'epoch' ends once the low pool has
    been traversed once.
    """
    model.train()
    total = 0.0
    nb    = 0

    use_balanced = (balanced_high_frac > 0.0 and
                    high_idx is not None and low_idx is not None and
                    high_idx.numel() > 0 and low_idx.numel() > 0)

    def _step(idx):
        nonlocal total, nb
        s_ids = train_coords[idx, 0]
        g_ids = train_coords[idx, 1]
        gate, ratio = model(s_ids, g_ids, knn_idx, knn_dist)
        loss = combined_loss(gate, ratio, ratio_t[idx], depth_t[idx],
                             alpha=alpha,
                             extreme_lambda=extreme_lambda,
                             extreme_pivot=extreme_pivot,
                             p_high_natural=p_high_natural,
                             batch_high_frac=balanced_high_frac if use_balanced else -1.0,
                             zero_anchor_beta=zero_anchor_beta)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
        nb    += 1

    if use_balanced:
        n_high = int(round(batch_size * balanced_high_frac))
        n_low  = max(batch_size - n_high, 1)
        perm_low = low_idx[torch.randperm(low_idx.numel(), device=device)]
        n_batches = (perm_low.numel() + n_low - 1) // n_low
        for b in range(n_batches):
            lo_b    = perm_low[b * n_low : b * n_low + n_low]
            hi_pick = torch.randint(0, high_idx.numel(), (n_high,), device=device)
            hi_b    = high_idx[hi_pick]
            _step(torch.cat([lo_b, hi_b]))
    else:
        n    = train_coords.shape[0]
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            _step(perm[start : start + batch_size])

    return total / max(nb, 1)


# Main

def main():
    parser = argparse.ArgumentParser(
        description="AIRGate-ST two-stage spatial zero-inflation model")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--emb_dim",    type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int,   default=65536)
    parser.add_argument("--knn_k",      type=int,   default=6)
    parser.add_argument("--alpha",      type=float, default=1.0,
                        help="Weight for ratio MSE loss vs gate BCE")
    parser.add_argument("--min_depth",  type=int,   default=10)
    parser.add_argument("--min_obs",    type=int,   default=10)
    parser.add_argument("--holdout",    type=float, default=0.10)
    parser.add_argument("--train_mask_path", default=None, help="Optional external train_mask.npy; must pair with --val_mask_path")
    parser.add_argument("--val_mask_path", default=None, help="Optional external val_mask.npy; must pair with --train_mask_path")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--data_path",  default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", default="results/AIRGate-ST")

    # Core AIRGate-ST options. Defaults are the final artifact configuration.
    parser.add_argument("--no_separate_emb", dest="separate_emb",
                        action="store_false",
                        help="Share embeddings between gate and ratio")
    parser.add_argument("--extreme_lambda", type=float, default=3.0,
                        help="Fix weight = 1 + lambda * max(r - pivot, 0); 0 disables")
    parser.add_argument("--extreme_pivot", type=float, default=0.1,
                        help="Pivot r for both extreme weighting and high/low pool split")
    parser.add_argument("--no_init_site_bias", dest="init_site_bias",
                        action="store_false",
                        help="Keep b_site_ratio at 0 instead of logit(site_mean)")
    parser.add_argument("--balanced_high_frac", type=float, default=0.5,
                        help="Fix fraction of each batch drawn from r>pivot pool; 0 disables")
    parser.add_argument("--zero_anchor_beta", type=float, default=0.0,
                        help="Penalize gate*ratio>0 at observed r=0; "
                             "0 disables (default), >0 enables anti-hallucination signal.")
    parser.add_argument("--gate_sampling_debias", action="store_true",
                        help="Reweight gate BCE back to the natural high-ratio frequency when "
                             "balanced_high_frac oversamples high-ratio entries.")
    parser.add_argument("--inner_val_frac", type=float, default=0.10,
                        help="Fraction of outer train_mask held out internally for epoch selection")
    parser.set_defaults(separate_emb=True, init_site_bias=True)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Config] model=AIRGate-ST  device={device}  "
          f"epochs={args.epochs}  emb_dim={args.emb_dim}  seed={args.seed}")
    print(f"[Fixes] separate_emb={args.separate_emb}  "
          f"extreme_lambda={args.extreme_lambda}  "
          f"init_site_bias={args.init_site_bias}  "
          f"balanced_high_frac={args.balanced_high_frac}  "
          f"pivot={args.extreme_pivot}")

    # Data
    A_raw, G_raw, coords, _ = load_data(args.data_path)
    A, G, _keep = filter_sites(A_raw, G_raw, min_obs=args.min_obs)
    observed_for_mask = (A + G) >= args.min_depth if args.min_depth > 0 else (A + G) > 0
    external_masks = load_external_masks(
        args.train_mask_path,
        args.val_mask_path,
        shape=A.shape,
        observed_mask=observed_for_mask,
    )
    if external_masks is None:
        train_mask, val_mask = train_val_split(
            A, G, holdout_frac=args.holdout,
            seed=args.seed, min_depth=args.min_depth,
        )
    else:
        train_mask, val_mask = external_masks
    outer_train_mask = train_mask.copy()
    fit_train_mask, inner_val_mask = make_inner_validation_split(
        outer_train_mask,
        frac=args.inner_val_frac,
        seed=args.seed + 10_000,
        min_train_per_site=1,
    )
    if inner_val_mask.any():
        train_mask = fit_train_mask
    epoch_select_mask = inner_val_mask if inner_val_mask.any() else train_mask
    print(
        f"[InnerVal] fit_train={int(train_mask.sum()):,}  "
        f"inner_val={int(inner_val_mask.sum()):,}  "
        f"final_holdout={int(val_mask.sum()):,}"
    )

    depth = A + G
    valid = depth > 0
    ratio = np.zeros_like(A)
    ratio[valid] = G[valid] / depth[valid]

    t_idx  = np.argwhere(train_mask)
    t_spot = torch.tensor(t_idx[:, 0], dtype=torch.long,  device=device)
    t_site = torch.tensor(t_idx[:, 1], dtype=torch.long,  device=device)
    t_r    = torch.tensor(ratio[train_mask], dtype=torch.float32, device=device)
    t_d    = torch.tensor(depth[train_mask], dtype=torch.float32, device=device)
    train_coords = torch.stack([t_spot, t_site], dim=1)

    n_spots, n_sites = A.shape

    # Per-site mean editing ratio over all observed train cells, including r=0.
    # This gives the ratio head a conservative site-level starting point.
    site_ratio_logit_init = None
    if args.init_site_bias:
        sum_r  = (ratio * train_mask).sum(axis=0)
        n_obs  = train_mask.sum(axis=0)
        mean_r = np.where(n_obs > 0, sum_r / np.maximum(n_obs, 1),
                          0.01).astype(np.float32)
        mean_r = np.clip(mean_r, 0.005, 0.995)
        site_ratio_logit_init = np.log(mean_r / (1.0 - mean_r)).astype(np.float32)
        print(f"[InitSiteBias] mean_r (all observed): min={mean_r.min():.3f}  "
              f"median={np.median(mean_r):.3f}  max={mean_r.max():.3f}  "
              f"sites_observed={int((n_obs>0).sum())}/{n_sites}")

    # High/low pools are indexed over train_coords rows.
    r_train_np = ratio[train_mask].astype(np.float32)
    high_pos   = torch.tensor(np.where(r_train_np >  args.extreme_pivot)[0],
                              dtype=torch.long, device=device)
    low_pos    = torch.tensor(np.where(r_train_np <= args.extreme_pivot)[0],
                              dtype=torch.long, device=device)
    print(f"[Pool] high-r entries={high_pos.numel():,}  "
          f"low-r entries={low_pos.numel():,}  "
          f"high_frac={high_pos.numel()/max(len(r_train_np),1):.4f}")
    p_high_natural = high_pos.numel() / max(len(r_train_np), 1) if args.gate_sampling_debias else -1.0
    print(f"[GateDebias] enabled={args.gate_sampling_debias}  p_high_natural={p_high_natural:.6f}")

    # Final AIRGate-ST always uses spatial KNN features and the attention gate.
    use_spatial = True
    knn_idx_np, knn_dist_np = build_knn_graph(coords, k=args.knn_k)
    knn_idx  = torch.tensor(knn_idx_np,  dtype=torch.long,  device=device)
    knn_dist = torch.tensor(knn_dist_np, dtype=torch.float32, device=device)

    # Model
    model = TwoStageModel(
        n_spots               = n_spots,
        n_sites               = n_sites,
        emb_dim               = args.emb_dim,
        use_spatial           = use_spatial,
        use_attn_gate         = True,
        knn_k                 = args.knn_k,
        separate_emb          = args.separate_emb,
        site_ratio_logit_init = site_ratio_logit_init,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] n_params={n_params:,}  n_spots={n_spots}  n_sites={n_sites}")

    # Train
    best_inner_rmse = float("inf")
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = run_epoch(
            model, optimizer, train_coords, t_r, t_d,
            knn_idx, knn_dist, args.alpha, args.batch_size, device,
            extreme_lambda     = args.extreme_lambda,
            extreme_pivot      = args.extreme_pivot,
            balanced_high_frac = args.balanced_high_frac,
            high_idx           = high_pos,
            low_idx            = low_pos,
            p_high_natural     = p_high_natural,
            zero_anchor_beta   = args.zero_anchor_beta,
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            pred = model.predict_all(device, knn_idx, knn_dist)
            m    = eval_metrics(pred, A, G, epoch_select_mask)
            print(f"[Epoch {epoch:04d}/{args.epochs}] "
                  f"loss={loss:.5f}  inner_rmse={m['rmse']:.5f}  "
                  f"inner_r={m['r']:.4f}  time={time.time()-t0:.1f}s")
            if m["rmse"] < best_inner_rmse:
                best_inner_rmse = m["rmse"]
                best_state = {k: v.cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            print(f"[Epoch {epoch:04d}/{args.epochs}] "
                  f"loss={loss:.5f}  time={time.time()-t0:.1f}s")

    # Output
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    pred = model.predict_all(device, knn_idx, knn_dist)   # full, no NaN mask

    out_dir = os.path.join(args.output_dir, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    gt_ratio    = compute_ratio(A, G, mask=observed_for_mask)
    train_ratio = gt_ratio.copy()
    train_ratio[~outer_train_mask] = np.nan
    fit_train_ratio = gt_ratio.copy()
    fit_train_ratio[~train_mask] = np.nan
    pred[outer_train_mask] = gt_ratio[outer_train_mask]   # keep all outer training values exact
    pred = validate_prediction_output("AIRGate-ST", pred, gt_ratio, outer_train_mask, val_mask)
    np.save(os.path.join(out_dir, "pred_ratio.npy"),  pred)
    np.save(os.path.join(out_dir, "gt_ratio.npy"),    gt_ratio)
    np.save(os.path.join(out_dir, "train_ratio.npy"), train_ratio)
    np.save(os.path.join(out_dir, "fit_train_ratio.npy"), fit_train_ratio)
    np.save(os.path.join(out_dir, "inner_val_mask.npy"), inner_val_mask)
    np.save(os.path.join(out_dir, "fit_train_mask.npy"), train_mask)
    np.save(os.path.join(out_dir, "val_mask.npy"),    val_mask)
    print(f"[Saved] core arrays -> {out_dir}")

    try:
        plot_heatmaps(
            gt_ratio    = gt_ratio,
            train_ratio = train_ratio,
            pred_ratio  = pred,
            val_mask    = val_mask,
            save_dir    = out_dir,
            prefix      = "AIRGate-ST",
            top_k       = 50,
        )
    except Exception as exc:
        print(f"[PlotWarning] legacy heatmaps skipped: {exc}")

    try:
        plot_diagnostics(
            gt_ratio                 = gt_ratio,
            pred_ratio               = pred,
            val_mask                 = val_mask,
            save_dir                 = out_dir,
            prefix                   = "AIRGate-ST",
            train_ratio_for_baseline = train_ratio,
        )
    except Exception as exc:
        print(f"[PlotWarning] diagnostics skipped: {exc}")

    final_metrics = eval_metrics(pred, A, G, val_mask)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"model=AIRGate-ST\nseed={args.seed}\n"
                f"epochs={args.epochs}\nbest_inner_rmse={best_inner_rmse:.6f}\n"
                f"rmse={final_metrics['rmse']:.6f}\n"
                f"mae={final_metrics['mae']:.6f}\n"
                f"r={final_metrics['r']:.6f}\n"
                f"cos={final_metrics['cos']:.6f}\n")

    print(f"\n[Done] AIRGate-ST  best_inner_rmse={best_inner_rmse:.5f}")
    print(f"       output -> {out_dir}")


if __name__ == "__main__":
    main()






