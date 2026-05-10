"""
Shared data utilities for all imputation architectures.
Handles: loading, site filtering, KNN graph, train/val split, heatmap plotting.
"""
import os
import numpy as np
import anndata as ad
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SHARED_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SHARED_DIR, "..", "..", "..", ".."))
DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "151673", "adata_ai_compressed.h5ad")


# Loading

def load_data(path=DEFAULT_DATA_PATH):
    """
    Load A (adenosine) and G (inosine/guanosine) count matrices + spatial coords.
    Returns: A (n_spots x n_sites), G, coords (n_spots x 2), adata
    """
    adata = ad.read_h5ad(path)

    A = adata.layers["A"]
    G = adata.layers["G"]
    if sparse.issparse(A):
        A = A.toarray()
    if sparse.issparse(G):
        G = G.toarray()
    A = A.astype(np.float32)
    G = G.astype(np.float32)

    if "x_pixel" in adata.obs.columns and "y_pixel" in adata.obs.columns:
        coords = np.column_stack(
            [adata.obs["x_pixel"].values, adata.obs["y_pixel"].values]
        ).astype(np.float32)
    elif "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    else:
        coords = np.column_stack(
            [adata.obs["x"].values, adata.obs["y"].values]
        ).astype(np.float32)

    print(f"[Data] shape={A.shape}  coords={coords.shape}")
    return A, G, coords, adata


def filter_sites(A, G, min_obs=10):
    """
    Keep only sites where ratio > 0 (i.e. G > 0) in more than min_obs spots.
    Per the project protocol: 'non-zero bin count > 10' means G > 0 in >10 bins.
    Returns filtered A, G, and boolean keep mask (n_sites,).
    """
    g_nonzero_per_site = (G > 0).sum(axis=0)   # count bins where editing observed
    keep = g_nonzero_per_site > min_obs
    print(f"[Filter] sites: {A.shape[1]} -> {keep.sum()} (G>0 in >{min_obs} bins)")
    return A[:, keep], G[:, keep], keep


def apply_min_depth_mask(A, G, min_depth=10):
    """
    Return boolean mask of entries with sufficient coverage.
    True = entry is valid / observed with enough depth.
    """
    depth = A + G
    if min_depth <= 0:
        return depth > 0
    return depth >= min_depth


# Spatial graph

def build_knn_graph(coords, k=6):
    """
    Build KNN spatial graph.
    Returns:
      knn_indices: (n_spots, k) int64
      knn_dists:   (n_spots, k) float32, normalized by median
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
    dists, idxs = nbrs.kneighbors(coords)
    dists = dists[:, 1:].astype(np.float32)   # remove self
    idxs  = idxs[:, 1:].astype(np.int64)
    med   = float(np.median(dists)) + 1e-6
    dists = dists / med
    print(f"[KNN] k={k}, median_dist={med:.2f} (normalised)")
    return idxs, dists


# Train / validation split

def train_val_split(A, G, holdout_frac=0.10, seed=42, min_depth=10):
    """
    Hold out holdout_frac of observed entries for validation.
    Returns:
      train_mask, val_mask: bool arrays (n_spots, n_sites)
    Only entries passing min_depth are eligible.
    """
    depth = A + G
    if min_depth <= 0:
        observed = depth > 0
    else:
        observed = depth >= min_depth

    obs_idx = np.argwhere(observed)            # (N_obs, 2)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(obs_idx))
    n_val = int(len(obs_idx) * holdout_frac)

    val_idx   = obs_idx[perm[:n_val]]
    train_idx = obs_idx[perm[n_val:]]

    val_mask   = np.zeros(A.shape, dtype=bool)
    train_mask = np.zeros(A.shape, dtype=bool)
    val_mask  [val_idx[:, 0],   val_idx[:, 1]]   = True
    train_mask[train_idx[:, 0], train_idx[:, 1]] = True

    print(f"[Split] train={train_mask.sum():,}  val={val_mask.sum():,}  "
          f"(holdout={holdout_frac:.0%})")
    return train_mask, val_mask




def load_external_masks(train_mask_path=None, val_mask_path=None, shape=None, observed_mask=None):
    """Load a shared train/validation mask pair, or return None if not requested."""
    if not train_mask_path and not val_mask_path:
        return None
    if not train_mask_path or not val_mask_path:
        raise ValueError("Both --train_mask_path and --val_mask_path must be provided together.")

    train_mask = np.load(train_mask_path).astype(bool)
    val_mask = np.load(val_mask_path).astype(bool)
    if shape is not None and (train_mask.shape != shape or val_mask.shape != shape):
        raise ValueError(
            f"External mask shape mismatch: train={train_mask.shape}, val={val_mask.shape}, expected={shape}"
        )
    if np.any(train_mask & val_mask):
        raise ValueError("External train_mask and val_mask overlap.")
    if observed_mask is not None:
        outside = (train_mask | val_mask) & ~observed_mask
        if np.any(outside):
            raise ValueError(f"External masks include {int(outside.sum())} entries outside observed_mask.")
        if not np.array_equal(train_mask | val_mask, observed_mask):
            missing = int((observed_mask & ~(train_mask | val_mask)).sum())
            raise ValueError(f"External masks omit {missing} observed entries; expected train_mask | val_mask == observed_mask.")

    total = int(train_mask.sum() + val_mask.sum())
    frac = val_mask.sum() / max(total, 1)
    print(
        f"[ExternalMask] train={int(train_mask.sum()):,}  val={int(val_mask.sum()):,}  "
        f"holdout={frac:.1%}"
    )
    return train_mask, val_mask

# Ratio computation

def compute_ratio(A, G, mask=None):
    """
    Compute editing ratio G/(A+G).  NaN where unobserved (or mask=False).
    mask: optional bool array; True entries are treated as observed.
    """
    depth = A + G
    ratio = np.full(A.shape, np.nan, dtype=np.float32)
    if mask is None:
        obs = depth > 0
    else:
        obs = mask & (depth > 0)
    ratio[obs] = G[obs] / depth[obs]
    return ratio


def validate_prediction_output(name, pred_ratio, gt_ratio, train_mask, val_mask):
    """Validate final imputation output without treating unobserved predictions as missing."""
    pred_ratio = pred_ratio.astype(np.float32, copy=False)
    if pred_ratio.shape != gt_ratio.shape:
        raise ValueError(f"{name}: shape mismatch pred={pred_ratio.shape}, gt={gt_ratio.shape}")
    pred_finite = np.isfinite(pred_ratio)
    observed_mask = train_mask | val_mask
    if np.any(observed_mask & ~pred_finite):
        raise ValueError(f"{name}: observed train/holdout entries contain NaN/Inf")
    finite_pred = pred_ratio[pred_finite]
    if finite_pred.size and np.any((finite_pred < 0.0) | (finite_pred > 1.0)):
        raise ValueError(f"{name}: finite pred_ratio values outside [0, 1]")
    if np.any(val_mask & ~np.isfinite(pred_ratio)):
        raise ValueError(f"{name}: metric/holdout entries contain NaN")
    if not np.allclose(pred_ratio[train_mask], gt_ratio[train_mask], equal_nan=False):
        raise ValueError(f"{name}: train entries must preserve all training-visible gt_ratio values exactly")
    return pred_ratio


# Metrics

def eval_metrics(pred_ratio, A_true, G_true, val_mask):
    """
    Evaluate on held-out entries.
    Returns dict with rmse, mae, pearson r, cosine similarity.
    """
    depth = A_true + G_true
    obs   = val_mask & (depth > 0)
    if obs.sum() == 0:
        return {"rmse": float("nan"), "mae": float("nan"),
                "r": float("nan"), "cos": float("nan")}

    true_r = G_true[obs] / depth[obs]
    pred_r = pred_ratio[obs]

    rmse = float(np.sqrt(np.mean((pred_r - true_r) ** 2)))
    mae  = float(np.mean(np.abs(pred_r - true_r)))

    if np.std(pred_r) > 1e-9 and np.std(true_r) > 1e-9:
        r = float(np.corrcoef(pred_r, true_r)[0, 1])
    else:
        r = 0.0

    nt = float(np.linalg.norm(true_r))
    np_ = float(np.linalg.norm(pred_r))
    cos = float(np.dot(pred_r, true_r) / (nt * np_ + 1e-12)) if nt > 1e-9 and np_ > 1e-9 else 0.0

    return {"rmse": rmse, "mae": mae, "r": r, "cos": cos}


# Plotting

def _cluster_order(matrix: np.ndarray, axis: int) -> np.ndarray:
    """Hierarchical clustering order (average linkage) along one axis."""
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, leaves_list
    n = matrix.shape[axis]
    if n <= 1:
        return np.arange(n)
    obs = np.nan_to_num(matrix if axis == 0 else matrix.T, nan=0.0)
    if obs.shape[0] <= 1 or np.allclose(obs, obs[0]):
        return np.arange(n)
    d = pdist(obs, metric="euclidean")
    if np.allclose(d, 0.0):
        return np.arange(n)
    return leaves_list(linkage(d, method="average"))


def _select_top_sites(gt_ratio: np.ndarray, val_mask: np.ndarray,
                      top_k: int = 50):
    """
    Select top_k most informative sites among those appearing in val_mask.
    Score = variance(gt_ratio) + 0.1 * nonzero_freq(gt_ratio)
    Returns sorted column indices into the (already filtered) site axis.
    """
    _, col_ids = np.where(val_mask)
    unique_cols = np.unique(col_ids)
    if unique_cols.size == 0:
        unique_cols = np.arange(gt_ratio.shape[1])

    sub = gt_ratio[:, unique_cols]
    var_score  = np.nanvar(sub, axis=0)
    nz_score   = np.nanmean(np.nan_to_num(sub, nan=0.0) > 0, axis=0)
    score      = var_score + 0.1 * nz_score
    top_n      = min(top_k, unique_cols.size)
    top_cols   = unique_cols[np.argsort(score)[-top_n:]]
    return top_cols


def _draw_panels(arrays, titles, col_order, save_path, suptitle="",
                 is_predicted=None):
    """
    Render side-by-side panels with shared colorbar.
    Colormap: RdBu_r (matches ground truth style), white=NaN.
    vmin=0, vmax=1 fixed (ratio is always in [0,1]).

    is_predicted: list of bool, same length as arrays.
        True: predicted panel: show all values, no small-value threshold.
        False: GT/train panel: mask values <=1e-8 as NaN (suppress noise).
    """
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")   # NaN -> gray (white = ratio=0.5, must be distinct)

    if is_predicted is None:
        is_predicted = [False] * len(arrays)

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 7), constrained_layout=True)
    if n == 1:
        axes = [axes]

    im = None
    for ax, arr, title, is_pred in zip(axes, arrays, titles, is_predicted):
        ordered = arr[:, col_order]
        heat = ordered.copy()
        if not is_pred:
            # GT / masked-train: suppress noise near zero
            heat[np.nan_to_num(heat, nan=0.0) <= 1e-8] = np.nan
        im = ax.imshow(heat, aspect="auto", cmap=cmap,
                       vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.8)
        cbar.set_label("A-to-I editing ratio")

    if suptitle:
        fig.suptitle(suptitle, fontsize=9, y=1.01)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved -> {save_path}")


def plot_heatmaps(gt_ratio: np.ndarray,
                  train_ratio: np.ndarray,
                  pred_ratio: np.ndarray,
                  val_mask: np.ndarray,
                  save_dir: str,
                  prefix: str = "",
                  top_k: int = 50) -> None:
    """
    Generate two comparison heatmaps:

    1. full_heatmap.png
       Panels: Ground Truth | Masked Train | Predicted (full imputed matrix)
       - GT and Masked Train show NaN at unobserved positions.
       - Predicted shows model output at ALL positions (including imputed).

    2. holdout_heatmap.png
       Panels: Ground Truth | Predicted  (only at val_mask positions)
       - Direct apples-to-apples comparison on held-out entries only.

    Column ordering (both figures):
       Hierarchical clustering on top_k most informative sites, using GT as
       reference.  Rows keep natural (spatial) order.
    """
    os.makedirs(save_dir, exist_ok=True)
    top_cols = _select_top_sites(gt_ratio, val_mask, top_k=top_k)

    # Column order: cluster on GT submatrix (top_k sites, all spots)
    gt_sub   = gt_ratio[:, top_cols]
    col_order = _cluster_order(gt_sub, axis=1)   # order within top_cols
    # Map back: col_order indexes into top_cols
    final_col_order = top_cols[col_order]         # absolute site indices

    col_idx = np.arange(len(final_col_order))   # already ordered
    tag = f"{prefix}_" if prefix else ""

    # Figure 1: full matrix
    _draw_panels(
        arrays=[
            gt_ratio    [:, final_col_order],
            train_ratio [:, final_col_order],
            pred_ratio  [:, final_col_order],
        ],
        titles=["Ground Truth", "Masked Train", "Predicted (Full Imputation)"],
        is_predicted=[False, False, True],
        col_order=col_idx,
        save_path=os.path.join(save_dir, f"{tag}full_heatmap.png"),
        suptitle=f"Top-{top_k} sites by variance | rows=natural spatial order",
    )

    # Figure 2: holdout only
    gt_hold   = gt_ratio.copy();   gt_hold  [~val_mask] = np.nan
    pred_hold = pred_ratio.copy(); pred_hold[~val_mask] = np.nan

    _draw_panels(
        arrays=[
            gt_hold  [:, final_col_order],
            pred_hold[:, final_col_order],
        ],
        titles=["Ground Truth (holdout)", "Predicted (holdout)"],
        is_predicted=[False, True],
        col_order=col_idx,
        save_path=os.path.join(save_dir, f"{tag}holdout_heatmap.png"),
        suptitle=f"Holdout 10% only | Top-{top_k} sites | rows=natural spatial order",
    )


# Diagnostics

def plot_diagnostics(gt_ratio: np.ndarray,
                     pred_ratio: np.ndarray,
                     val_mask: np.ndarray,
                     save_dir: str,
                     prefix: str = "",
                     train_ratio_for_baseline: np.ndarray = None) -> dict:
    """
    Diagnostic figure on holdout entries only:
      (a) Scatter: GT vs Predicted, with y=x diagonal and GT-binned means.
      (b) Bar chart of RMSE / MAE stratified by GT value range.

    Also writes diagnostics.txt with per-bin counts and metrics, plus
    an overall mean-baseline comparison (predict per-site train mean).

    Returns a summary dict.
    """
    os.makedirs(save_dir, exist_ok=True)
    tag = f"{prefix}_" if prefix else ""

    sel = val_mask & np.isfinite(gt_ratio) & np.isfinite(pred_ratio)
    if sel.sum() == 0:
        print("[Diag] no valid holdout entries, skipping")
        return {}

    y = gt_ratio [sel].astype(np.float64)
    p = pred_ratio[sel].astype(np.float64)

    # (a) Scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    # Density via hexbin (robust to tens of thousands of points).
    hb = ax.hexbin(y, p, gridsize=50, cmap="viridis",
                   mincnt=1, bins="log", extent=(0, 1, 0, 1))
    ax.plot([0, 1], [0, 1], "r--", lw=1, label="y = x")

    # Overlay: mean of predictions within GT bins
    edges = np.linspace(0.0, 1.0, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.clip(np.digitize(y, edges) - 1, 0, len(centers) - 1)
    bin_mean_pred = np.array([
        p[bin_id == i].mean() if (bin_id == i).any() else np.nan
        for i in range(len(centers))
    ])
    ax.plot(centers, bin_mean_pred, "o-", color="orange",
            lw=2, ms=6, label="mean pred per GT bin")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Ground truth ratio")
    ax.set_ylabel("Predicted ratio")
    ax.set_title(f"Holdout scatter  (n = {sel.sum():,})")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(hb, ax=ax, label="log10 count")

    # (b) Binned RMSE / MAE
    bins = [(0.0, 0.1, "[0,0.1]"),
            (0.1, 0.5, "(0.1,0.5]"),
            (0.5, 0.9, "(0.5,0.9]"),
            (0.9, 1.001, "(0.9,1.0]")]

    names, rmses, maes, counts = [], [], [], []
    for lo, hi, lab in bins:
        m = (y > lo) & (y <= hi) if lo > 0 else (y >= lo) & (y <= hi)
        names.append(lab)
        counts.append(int(m.sum()))
        if m.any():
            rmses.append(float(np.sqrt(np.mean((p[m] - y[m]) ** 2))))
            maes .append(float(np.mean(np.abs(p[m] - y[m]))))
        else:
            rmses.append(np.nan); maes.append(np.nan)

    ax = axes[1]
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w / 2, rmses, w, label="RMSE", color="steelblue")
    ax.bar(x + w / 2, maes,  w, label="MAE",  color="coral")
    for i, c in enumerate(counts):
        ax.text(i, 0.005, f"n={c:,}", ha="center", va="bottom",
                fontsize=8, rotation=0)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Error")
    ax.set_title("Error stratified by GT bin")
    ax.set_ylim(0, max([v for v in rmses + maes if np.isfinite(v)] + [0.1]) * 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    save_path = os.path.join(save_dir, f"{tag}diagnostics.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] saved -> {save_path}")

    # Mean-baseline comparison
    baseline_rmse = float("nan")
    if train_ratio_for_baseline is not None:
        site_mean = np.nanmean(train_ratio_for_baseline, axis=0)
        site_mean = np.where(np.isfinite(site_mean), site_mean,
                             np.nanmean(site_mean))
        _, col = np.where(sel)
        baseline_pred = site_mean[col]
        baseline_rmse = float(np.sqrt(np.mean((baseline_pred - y) ** 2)))

    # Write diagnostics.txt
    overall_rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    lines = [
        f"# {tag}diagnostics",
        f"n_holdout={sel.sum():,}",
        f"overall_rmse={overall_rmse:.6f}",
        f"baseline_site_mean_rmse={baseline_rmse:.6f}  (NaN if train_ratio not provided)",
        f"improvement_vs_baseline={baseline_rmse - overall_rmse:+.6f}",
        "",
        "bin\tcount\trmse\tmae",
    ]
    for name, cnt, rm, ma in zip(names, counts, rmses, maes):
        lines.append(f"{name}\t{cnt}\t{rm:.6f}\t{ma:.6f}")

    txt_path = os.path.join(save_dir, f"{tag}diagnostics.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Diag] wrote    -> {txt_path}")

    return {
        "n": int(sel.sum()),
        "overall_rmse": overall_rmse,
        "baseline_rmse": baseline_rmse,
        "bins": list(zip(names, counts, rmses, maes)),
    }



