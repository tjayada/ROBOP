"""Shared utilities for ROBOP analysis scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


# np.trapz was renamed np.trapezoid in NumPy 2.0 (the old name still warns).
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def add_auc(errors_mm, ceiling_mm: float) -> float:
    """DREAM/CtRNet ADD-AUC on a 0.01 mm grid with inclusive thresholds."""
    delta = 0.01  # mm  (DREAM delta_threshold = 0.00001 m)
    thresholds = np.arange(0.0, float(ceiling_mm), delta)
    add_sorted = np.sort(np.asarray(errors_mm, dtype=np.float64).ravel())
    if add_sorted.size == 0:
        return float("nan")
    frac_below = np.searchsorted(add_sorted, thresholds, side="right") / add_sorted.size
    return float(_TRAPZ(frac_below, dx=delta) / float(ceiling_mm))


def load_results(path: str | Path) -> tuple[dict, list[dict]]:
    """Load a results JSON, return (summary, queries)."""
    with open(path) as f:
        data = json.load(f)
    return data["summary"], data["queries"]


def detect_dataset(summary: dict) -> str:
    """Return the dataset name from the summary dict."""
    return summary.get("dataset", "unknown")


def is_nemo(queries: list[dict]) -> bool:
    """True if the result file comes from the NeMO estimator (has conf_stats in diagnostics)."""
    for q in queries:
        d = q.get("diagnostics")
        if d is not None:
            return "conf_stats" in d
    return False


# Signal extraction

def extract_signals(queries: list[dict]) -> dict[str, np.ndarray]:
    """Extract per-frame metric and diagnostic arrays from NeMO results.

    ``conf_50`` is the mean above threshold, while ``conf_count_50`` is the
    number above threshold. Missing or post-PnP-only signals are NaN.
    """
    n = len(queries)
    add_m_mm       = np.full(n, np.nan)
    pnp_failed     = np.zeros(n, dtype=bool)
    conf_50        = np.full(n, np.nan)
    conf_count_50  = np.full(n, np.nan)
    reproj         = np.full(n, np.nan)
    inlier         = np.full(n, np.nan)
    anc_r          = np.full(n, np.nan)
    iou_conf       = np.full(n, np.nan)
    iou_modal      = np.full(n, np.nan)
    iou_amodal     = np.full(n, np.nan)
    frame_time_ms  = np.full(n, np.nan)

    for i, q in enumerate(queries):
        add_m_mm[i]   = q.get("add_m_mm", np.nan)
        pnp_failed[i] = bool(q.get("pnp_failed", False))

        d = q.get("diagnostics")
        if d is None:
            continue

        frame_time_ms[i] = d.get("frame_time_ms", np.nan)

        cs = d.get("conf_stats", {}).get("thr_0.50", {})
        conf_50[i]       = cs.get("mean_conf", np.nan)
        conf_count_50[i] = cs.get("count", np.nan)

        pnp = d.get("pnp") or {}
        reproj[i] = pnp.get("mean_reproj_error_px", np.nan)
        inlier[i] = pnp.get("inlier_ratio", np.nan)

        aln = d.get("alignment") or {}
        anc_r[i] = aln.get("anchor_rotation_error_deg", np.nan)

        iou = d.get("iou") or {}
        iou_conf[i]   = (iou.get("conf_vs_render") or {}).get("thr_0.50", np.nan)
        iou_modal[i]  = iou.get("modal_vs_render", np.nan)
        iou_amodal[i] = iou.get("amodal_vs_render", np.nan)

    return {
        "add_m_mm":      add_m_mm,
        "pnp_failed":    pnp_failed,
        "conf_50":       conf_50,
        "conf_count_50": conf_count_50,
        "reproj":        reproj,
        "inlier":        inlier,
        "anc_r":         anc_r,
        "iou_conf":      iou_conf,
        "iou_modal":     iou_modal,
        "iou_amodal":    iou_amodal,
        "frame_time_ms": frame_time_ms,
    }


# Error decomposition

def _rotation_errors(R_pred: np.ndarray, R_gt: np.ndarray) -> tuple[float, float, float, float]:
    """
    Decompose rotation error into geodesic (scalar) and signed Euler angles (xyz, degrees).

    R_err = R_pred.T @ R_gt maps predicted frame to GT frame.
    Euler angles are extracted in xyz order (OpenCV convention: Rx=pitch, Ry=yaw, Rz=roll).
    Returns (geodesic_deg, rx_deg, ry_deg, rz_deg).
    """
    from scipy.spatial.transform import Rotation
    R_err = R_pred.T @ R_gt
    cos = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    geodesic = float(np.degrees(np.arccos(cos)))
    rx, ry, rz = Rotation.from_matrix(R_err).as_euler("xyz", degrees=True)
    return geodesic, float(rx), float(ry), float(rz)


def error_decomposition(queries: list[dict], dataset: str) -> dict[str, np.ndarray]:
    """
    Decompose pose error into per-axis translation and rotation components.

    For panda_orb and craves: uses est_pose - gt_pose (4×4 matrices).
    For baxter: uses ee_cam_pred - ee_3d_gt (EE position only; rotation N/A).

    Translation returned in mm (signed for X/Y/Z, unsigned for lateral/depth).
    Rotation returned in degrees.

    Returns arrays (excluding PnP failures):
        x_mm, y_mm, z_mm    : signed per-axis translation error
        lateral_mm           : sqrt(X²+Y²)
        depth_mm             : |Z|
        rot_geodesic_deg     : geodesic rotation error (NaN for Baxter)
    """
    xs, ys, zs = [], [], []
    laterals, depths = [], []
    geodesics, rxs, rys, rzs = [], [], [], []

    for q in queries:
        if q.get("pnp_failed"):
            continue

        if dataset == "baxter":
            pred  = np.array(q["ee_cam_pred"], dtype=np.float64)
            gt    = np.array(q["ee_3d_gt"],    dtype=np.float64)
            t_err = (pred - gt) * 1000.0
            geodesics.append(np.nan)
            rxs.append(np.nan)
            rys.append(np.nan)
            rzs.append(np.nan)
        else:
            est  = q.get("est_pose")
            gt_p = q.get("gt_pose")
            if est is None or gt_p is None:
                continue
            est_arr = np.array(est,  dtype=np.float64)
            gt_arr  = np.array(gt_p, dtype=np.float64)
            t_err   = (est_arr[:3, 3] - gt_arr[:3, 3]) * 1000.0
            geo, rx, ry, rz = _rotation_errors(est_arr[:3, :3], gt_arr[:3, :3])
            geodesics.append(geo)
            rxs.append(rx)
            rys.append(ry)
            rzs.append(rz)

        xs.append(float(t_err[0]))
        ys.append(float(t_err[1]))
        zs.append(float(t_err[2]))
        laterals.append(float(np.sqrt(t_err[0]**2 + t_err[1]**2)))
        depths.append(float(abs(t_err[2])))

    return {
        "x_mm":             np.array(xs),
        "y_mm":             np.array(ys),
        "z_mm":             np.array(zs),
        "lateral_mm":       np.array(laterals),
        "depth_mm":         np.array(depths),
        "rot_geodesic_deg": np.array(geodesics),
        "rx_deg":           np.array(rxs),
        "ry_deg":           np.array(rys),
        "rz_deg":           np.array(rzs),
    }


# Confidence gating

def naive_score(signals: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight min-max score from confidence, reprojection, inliers and anchor error."""
    components = {
        "conf_50": False,   # False = don't invert
        "reproj":  True,
        "inlier":  False,
        "anc_r":   True,
    }
    normed = []
    for key, invert in components.items():
        arr = signals[key].copy()
        valid = ~np.isnan(arr)
        if valid.sum() < 2:
            continue
        mn, mx = arr[valid].min(), arr[valid].max()
        rng = mx - mn
        if rng < 1e-12:
            arr[valid] = 0.5
        else:
            arr[valid] = (arr[valid] - mn) / rng
        if invert:
            arr[valid] = 1.0 - arr[valid]
        normed.append(arr)

    if not normed:
        return np.full(len(signals["conf_50"]), np.nan)

    stack = np.stack(normed, axis=0)
    with np.errstate(all="ignore"):
        score = np.nanmean(stack, axis=0)
    all_nan = np.all(np.isnan(stack), axis=0)
    score[all_nan] = np.nan
    return score


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUROC where label=1 means 'good' (low error) and score is confidence.
    Both arrays must be the same length with no NaN.
    """
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def pareto_curve(
    scores: np.ndarray,
    add_m_mm: np.ndarray,
    failure_threshold_mm: float = 100.0,
    n_steps: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Pareto frontier: AUC@100mm vs coverage.

    At each score threshold t, frames with score >= t are "retained" (coverage),
    and ADD AUC is computed only on those frames.

    Returns (coverage, auc) arrays of length n_steps.
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    coverages, aucs = [], []
    total = len(scores)

    for t in thresholds:
        mask = scores >= t
        if mask.sum() < 5:
            continue
        coverage = mask.sum() / total
        arr = add_m_mm[mask]
        auc = add_auc(arr, failure_threshold_mm)
        coverages.append(coverage)
        aucs.append(auc)

    return np.array(coverages), np.array(aucs)


def operating_point(
    scores: np.ndarray,
    add_m_mm: np.ndarray,
    coverage_target: float,
    failure_threshold_mm: float = 100.0,
    total_n: Optional[int] = None,
) -> dict:
    """
    Find the threshold that retains approximately `coverage_target` fraction of frames.

    scores / add_m_mm may already be pre-filtered to valid-score frames.
    total_n should be the full dataset size so coverage is expressed as a
    fraction of ALL frames, not just frames with valid scores.
    Returns threshold, actual coverage, AUC at that coverage, and baseline AUC (all frames).
    """
    if total_n is None:
        total_n = len(scores)
    t = float(np.quantile(scores, 1.0 - coverage_target))
    mask = scores >= t
    actual_coverage = float(mask.sum() / total_n)
    arr_filtered = add_m_mm[mask]
    arr_all = add_m_mm
    auc_filtered = add_auc(arr_filtered, failure_threshold_mm)
    auc_all      = add_auc(arr_all,      failure_threshold_mm)
    return {
        "coverage_target":  coverage_target,
        "actual_coverage":  round(actual_coverage, 3),
        "threshold":        round(t, 4),
        "auc_filtered":     round(auc_filtered, 4),
        "auc_all":          round(auc_all, 4),
        "auc_gain":         round(auc_filtered - auc_all, 4),
    }


# Printing

def print_header(title: str) -> None:
    width = max(60, len(title) + 4)
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_table(headers: list[str], rows: list[list], col_width: int = 18) -> None:
    """Print a fixed-width text table."""
    fmt = "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("-" * (col_width * len(headers)))
    for row in rows:
        formatted = []
        for v in row:
            if isinstance(v, float):
                formatted.append(f"{v:.4f}")
            else:
                formatted.append(str(v))
        print(fmt.format(*formatted))


# Spearman correlation table

def spearman_table(
    signals: dict[str, np.ndarray],
    target: np.ndarray,
    signal_names: Optional[list[str]] = None,
) -> None:
    """Print Spearman ρ between each signal and target (e.g. add_m_mm)."""
    from scipy.stats import spearmanr

    if signal_names is None:
        signal_names = ["conf_50", "conf_count_50", "reproj", "inlier", "anc_r",
                        "iou_conf", "iou_modal", "iou_amodal"]

    rows = []
    for name in signal_names:
        arr = signals.get(name)
        if arr is None:
            continue
        valid = ~np.isnan(arr) & ~np.isnan(target)
        if valid.sum() < 10:
            rows.append([name, "n/a", "n/a", str(int(valid.sum()))])
            continue
        rho, pval = spearmanr(arr[valid], target[valid])
        rows.append([name, f"{rho:+.3f}", f"{pval:.2e}", str(int(valid.sum()))])

    print_table(["Signal", "Spearman ρ", "p-value", "N"], rows)


# Fixed order keeps series colors consistent across figures.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CATEGORICAL

DIVERGING = (BLUE, RED)

# Ordered good/bad/fail groups.
# so identity survives grayscale print and CVD (secondary encoding, not color alone).
SEVERITY    = {"good": BLUE, "bad": YELLOW, "fail": RED}
SEVERITY_LS = {"good": "-",  "bad": "--",   "fail": ":"}

# Text + chrome tokens (light surface): guides recede, data stays prominent.
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7"}

# Robots keep one color across figures, assigned by this canonical order.
_DATASET_ORDER = ["panda_orb", "baxter", "craves", "hydra_lbr", "hydra_xarm"]


def dataset_color(name: str) -> str:
    key = name.split()[0]
    if key not in _DATASET_ORDER:
        _DATASET_ORDER.append(key)
    return CATEGORICAL[_DATASET_ORDER.index(key) % len(CATEGORICAL)]


# Estimators keep ONE color across every figure (canonical model->color map).
# Keyed by lowercase model token AND display name so either lookup works.
MODEL_COLOR = {
    "nemo": BLUE,      "NeMO": BLUE,
    "gigapose": ORANGE, "GigaPose": ORANGE,
    "megapose": AQUA,  "MegaPose": AQUA,
    "foundpose": YELLOW, "FoundPose": YELLOW,
}


def model_color(name: str) -> str:
    """Canonical color for an estimator; falls back to primary ink if unknown."""
    return MODEL_COLOR.get(name, MODEL_COLOR.get(name.lower(), INK["primary"]))


def setup_matplotlib() -> None:
    """Shared figure style: system sans, hairline solid grid, recessive axes,
    the validated categorical palette as the default cycle."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "axes.edgecolor": INK["axis"], "axes.linewidth": 0.8,
        "axes.labelcolor": INK["secondary"], "text.color": INK["primary"],
        "xtick.color": INK["muted"], "ytick.color": INK["muted"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.8,
        "grid.linestyle": "-", "grid.alpha": 1.0,
        "legend.frameon": False,
    })
