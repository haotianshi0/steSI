import numpy as np
import torch


def make_inner_validation_split(
    train_mask: np.ndarray,
    frac: float,
    seed: int,
    min_train_per_site: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an outer training mask into fit and inner-validation masks.

    The split is per-site so columns with sparse training support keep at least
    ``min_train_per_site`` fit entries whenever possible. The final holdout
    mask must not be used for epoch selection.
    """
    train_mask = train_mask.astype(bool, copy=False)
    fit_mask = train_mask.copy()
    inner_val_mask = np.zeros_like(train_mask, dtype=bool)
    if frac <= 0.0:
        return fit_mask, inner_val_mask

    rng = np.random.default_rng(seed)
    for site in range(train_mask.shape[1]):
        rows = np.where(train_mask[:, site])[0]
        if rows.size <= min_train_per_site:
            continue
        n_inner = min(int(rows.size * frac), rows.size - min_train_per_site)
        if n_inner <= 0 and frac > 0.0 and rows.size > min_train_per_site:
            n_inner = 1
        if n_inner <= 0:
            continue
        picked = rng.choice(rows, size=n_inner, replace=False)
        inner_val_mask[picked, site] = True
        fit_mask[picked, site] = False
    return fit_mask, inner_val_mask


def sample_inner_mask(
    train_coords: torch.Tensor,
    true_ratio: torch.Tensor,
    frac: float,
    seed: int | None = None,
    high_ratio_threshold: float = 0.05,
    high_ratio_weight: float = 2.0,
) -> torch.Tensor:
    """Sample a dynamic inner-mask inside training-observed entries.

    Parameters
    ----------
    train_coords:
        Tensor of shape (N, 2), kept for caller alignment. The function returns
        a boolean mask over the same N entries.
    true_ratio:
        Ratio values for the N train entries.
    frac:
        Fraction of train entries to hide from the reconstruction objective.
    seed:
        Optional deterministic seed for ablations.
    high_ratio_threshold:
        Entries above this ratio receive extra sampling probability.
    high_ratio_weight:
        Multiplicative probability boost for high-ratio entries.

    Notes
    -----
    This is a light-weight internal re-mask helper. The outer validation
    mask remains fixed; this inner mask is sampled during training only.
    """
    if train_coords.shape[0] != true_ratio.shape[0]:
        raise ValueError("train_coords and true_ratio must have the same first dimension")
    n = int(true_ratio.shape[0])
    n_mask = int(n * frac)
    out = torch.zeros(n, dtype=torch.bool, device=true_ratio.device)
    if n_mask <= 0:
        return out

    weights = torch.ones(n, dtype=torch.float32, device=true_ratio.device)
    weights = torch.where(true_ratio >= high_ratio_threshold, weights * high_ratio_weight, weights)
    weights = weights / weights.sum().clamp_min(1e-12)

    if seed is not None:
        generator = torch.Generator(device=true_ratio.device)
        generator.manual_seed(seed)
        idx = torch.multinomial(weights, n_mask, replacement=False, generator=generator)
    else:
        idx = torch.multinomial(weights, n_mask, replacement=False)
    out[idx] = True
    return out


def split_train_inner(
    train_coords: torch.Tensor,
    ratio_t: torch.Tensor,
    depth_t: torch.Tensor,
    inner_mask: torch.Tensor,
):
    """Return visible and inner-target tensors for an inner-mask training step."""
    visible = ~inner_mask
    return {
        "visible_coords": train_coords[visible],
        "visible_ratio": ratio_t[visible],
        "visible_depth": depth_t[visible],
        "inner_coords": train_coords[inner_mask],
        "inner_ratio": ratio_t[inner_mask],
        "inner_depth": depth_t[inner_mask],
    }


def weighted_inner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    depth: torch.Tensor,
    zero_weight: float = 1.0,
    positive_weight: float = 2.0,
) -> torch.Tensor:
    """Depth-weighted MSE for dynamically hidden train entries."""
    if pred.numel() == 0:
        return pred.sum() * 0.0
    depth_w = depth / depth.mean().clamp_min(1e-6)
    signal_w = torch.where(target > 0, positive_weight, zero_weight)
    weight = depth_w * signal_w
    return (weight * (pred - target) ** 2).sum() / weight.sum().clamp_min(1e-6)
