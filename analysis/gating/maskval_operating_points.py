"""MaskVal AP-target operating points and the Table III signal-quality summary.

Follows Quentin & Goehring (MaskVal, arXiv:2409.03556, Sec. V): threshold the certainty
c = iou_cnos so the retained set reaches a target average precision AP at a pose-error
tolerance e_t. Emits one file per AP target to figures/gating/: gating_tables_ap99.md (main,
carries Table III) and gating_tables_ap95.md (appendix, looser target, transfers less well).
Each file has, per deployable regime (coarse / refined):

  Table III  - signal quality of the IoU certainty: Spearman(1-IoU, ADD), AUC-AR, and the
               AUROC of catastrophic-failure (> 100 mm) detection, per dataset (main file only).
  Operating  - one table per regime, robots as rows (MaskVal Table IV shape): the self-
               calibrated per-robot ceiling and the Hydra-frozen transfer as side-by-side
               column blocks, so the transfer cost reads left-to-right. t* is calibrated on
               pooled Hydra (lbr/xarm/meca) to hold AP >= target at tolerance e_t, frozen, and
               applied to panda-orb and CRAVES (MaskVal Sec. V-E, here cross-robot). Coarse and
               refined share the 100/50 mm tolerances so the two modalities compare directly.

Reads the per-frame {iou, add} caches written by risk_coverage_from_masks.py. You deploy a
coarse OR a refined estimator, so the two are pooled and reported separately. Baxter is
excluded: its benchmark exposes only end-effector GT, which a whole-arm silhouette gate
cannot be scored against.

Deviations from the paper, all deliberate: pose error is ADD not MDD (eq. 12), the
DREAM/CtRNet home metric; AP/AR are set-level (eq. 3-5) not per-image-averaged (eq. 13-15),
since our benchmark has one instance per frame; the reference mask is CNOS/SAM (mode-b),
not the estimator's own stage-1 mask; the visibility weighting (eq. 11) is off.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_utils import compute_auroc  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2.0

AP_HEADLINE = 0.99          # target for the AUC-AR summary column (MaskVal's industrial default)
FAIL_MM = 100.0             # catastrophic-failure line for AUROC (all regimes)

# tolerance grid for the AUC-AR summary; refined estimators are accurate, so tighter.
GRID = {"coarse": np.arange(5.0, 155.0, 5.0), "refined": np.arange(2.0, 52.0, 2.0)}

# operating-point tolerances (mm); coarse and refined share 100/50 mm for comparison, refined
# adds a tight 30 mm. One table file per AP target (0.99 main, 0.95 appendix).
TOLS = {"coarse": [100.0, 50.0], "refined": [100.0, 50.0, 30.0]}
AP_TARGETS = [(0.99, "gating_tables_ap99.md"), (0.95, "gating_tables_ap95.md")]

GATING_DIR = Path(__file__).resolve().parents[2] / "figures" / "gating"  # repo-relative, any cwd

CALIB = ["hydra_lbr", "hydra_xarm", "hydra_meca"]
VALIDATION = ["panda_orb", "craves"]
DATASET_LABEL = {"hydra": "Hydra (calibration)", "panda_orb": "panda-orb", "craves": "CRAVES"}


# IO

def _is_extra(variant):
    v = variant.lower()
    return "bootstrap" in v or "modular" in v


def load_pool(cache_dir, datasets, regime):
    """Concatenate (iou, add) across the deployable variants of the given datasets, keeping
    only the requested regime ('coarse' = no refiner, 'refined' = refiner)."""
    ious, adds = [], []
    for f in sorted(cache_dir.glob("*.json")):
        variant, dataset = f.stem.split("__", 1)
        if dataset not in datasets or _is_extra(variant):
            continue
        is_refined = "refiner" in variant
        if regime == "coarse" and is_refined:
            continue
        if regime == "refined" and not is_refined:
            continue
        c = json.load(open(f))
        ious.append(np.asarray(c["iou"], float))
        adds.append(np.asarray(c["add"], float))
    if not ious:
        return None, None
    return np.concatenate(ious), np.concatenate(adds)


# MaskVal metrics (eq. 3-5), set-level

def metrics_at(cert, add, e_t, t):
    """AP/AR and coverage for keeping frames with certainty >= t at tolerance e_t."""
    good = add <= e_t
    kept = cert >= t
    tp = int(np.sum(kept & good))
    fp = int(np.sum(kept & ~good))
    n = len(add)
    return {
        "t": float(t),
        "AP":  tp / (tp + fp) if (tp + fp) else float("nan"),   # precision of retained
        "AR":  tp / n,                                          # recall over all frames
        "coverage": float(kept.mean()),
    }


def calibrate(cert, add, e_t, target):
    """Lowest threshold whose thresholded retained set has AP >= target (max recall subject to
    AP). Evaluated at tie-group boundaries so the deployed threshold's realized AP actually
    meets the target - a top-k prefix cut can land inside a tie (e.g. IoU 0), which thresholding
    cannot separate. Returns metrics at that threshold, or None if the target is unreachable."""
    order = np.argsort(-cert, kind="stable")     # highest certainty first
    sc = cert[order]
    good = (add <= e_t)[order]
    ap = np.cumsum(good) / np.arange(1, len(good) + 1)     # AP of frames kept at threshold sc[k]
    boundary = np.ones(len(sc), dtype=bool)               # last index of each tied group
    boundary[:-1] = sc[1:] != sc[:-1]
    ok = boundary & (ap >= target)
    if not ok.any():
        return None
    k = int(np.max(np.where(ok)[0]))             # largest kept set (max recall) meeting AP
    return metrics_at(cert, add, e_t, float(sc[k]))


def ar_curve(cert, add, e_grid, target):
    """AR (at AP >= target) over the e_t grid, for the AUC-AR summary."""
    ar = []
    for e in e_grid:
        m = calibrate(cert, add, e, target)
        ar.append(m["AR"] if m else 0.0)
    return np.array(ar)


def auc_ar(e_grid, ar):
    """Area under the AR-vs-e_t curve, normalized to the tolerance span (MaskVal Table III)."""
    return float(_trapz(ar, e_grid) / (e_grid[-1] - e_grid[0]))


# markdown tables

def _f(x, spec="{:.3f}"):
    return "-" if (isinstance(x, float) and np.isnan(x)) else spec.format(x)


def md_table_iii(panels, grid):
    """Spearman(1-IoU, ADD), AUC-AR, AUROC(fail@100mm) per dataset."""
    from scipy.stats import spearmanr
    out = ["| dataset | Spearman(1-IoU, ADD) | AUC-AR | AUROC(fail@100 mm) |",
           "|---|---|---|---|"]
    for label, cert, add in panels:
        ar = ar_curve(cert, add, grid, AP_HEADLINE)
        rho = spearmanr(1.0 - cert, add).correlation
        ok = (add <= FAIL_MM).astype(int)
        auroc = compute_auroc(cert, ok) if 0 < ok.sum() < len(ok) else float("nan")
        out.append(f"| {label} | {_f(rho)} | {_f(100 * auc_ar(grid, ar), '{:.1f}')} | {_f(auroc)} |")
    return out


def md_operating(cache_dir, regime, ap, tols):
    """Merged operating-point table (MaskVal Table IV shape) at one AP target: robots as rows,
    self-calibrated ceiling and Hydra-transfer as side-by-side column blocks. Transfer freezes
    the Hydra row's threshold and applies it to panda-orb / CRAVES; Hydra's own transfer cells
    are '-' (it is the calibration source, so its transfer is its self). The AP target is fixed
    per table (one file per target), so it is not a column."""
    short = {"hydra": "Hydra", "panda_orb": "panda-orb", "craves": "CRAVES"}
    order = ["hydra"] + VALIDATION
    pools = {"hydra": load_pool(cache_dir, CALIB, regime)}
    for d in VALIDATION:
        pools[d] = load_pool(cache_dir, [d], regime)
    out = ["| robot | tolerance | self reject if | self AP | self cov "
           "| transfer AP | transfer cov |",
           "|---|---|---|---|---|---|---|"]
    for e_t in tols:
        hy = calibrate(*pools["hydra"], e_t, ap)          # frozen threshold for transfer
        for d in order:
            cert, add = pools[d]
            m = calibrate(cert, add, e_t, ap)             # self-calibrated on this robot
            if m is None:
                s_rej, s_ap, s_cov = "unreachable", "-", "-"
            else:
                s_rej = f"IoU>={m['t']:.2f}"
                s_ap, s_cov = _f(m["AP"]), f"{m['coverage']:.2%}"
            if d == "hydra" or hy is None:
                t_ap, t_cov = "-", "-"                     # Hydra is the calibration source
            else:
                tv = metrics_at(cert, add, e_t, hy["t"])
                t_ap, t_cov = _f(tv["AP"]), f"{tv['coverage']:.2%}"
            out.append(f"| {short[d]} | {e_t:.0f} mm | {s_rej} | {s_ap} | {s_cov} "
                       f"| {t_ap} | {t_cov} |")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=GATING_DIR / "iou_cache",
                    help="iou_cache/ written by risk_coverage_from_masks.py")
    ap.add_argument("--out-dir", type=Path, default=GATING_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # one file per AP target: 0.99 (main, carries Table III) and 0.95 (appendix).
    for target, fname in AP_TARGETS:
        main_file = target == AP_TARGETS[0][0]
        md = [f"# MaskVal IoU gate - operating points (AP target {target:.2f})", ""]
        if not main_file:
            md += [f"Appendix companion to the {AP_TARGETS[0][0]:.2f} tables; signal quality "
                   "(Table III) is reported there. A looser AP target transfers less reliably "
                   "across unseen robots than the conservative one.", ""]
        for regime in ("coarse", "refined"):
            ch, ah = load_pool(args.cache_dir, CALIB, regime)
            if ch is None:
                print(f"skip {regime}: no Hydra caches under {args.cache_dir}")
                continue
            md += [f"## {regime}", ""]
            if main_file:
                panels = [(DATASET_LABEL["hydra"], ch, ah)]
                for d in VALIDATION:
                    c, a = load_pool(args.cache_dir, [d], regime)
                    if c is not None:
                        panels.append((DATASET_LABEL[d], c, a))
                md += [f"Hydra calibration set: n={len(ah)}, within 100 mm = "
                       f"{int((ah <= FAIL_MM).sum())} ({(ah <= FAIL_MM).mean():.2%}).", "",
                       "### Signal quality (Table III)", ""]
                md += md_table_iii(panels, GRID[regime])
                md += [""]
            md += ["### Operating points: self-calibrated ceiling vs Hydra-transfer", ""]
            md += md_operating(args.cache_dir, regime, target, TOLS[regime])
            md += [""]
        md += ["The self columns are the per-robot ceiling (each arm's own threshold); the "
               "transfer columns freeze Hydra's threshold and apply it. The frozen threshold runs "
               "slightly lenient on panda-orb (AP dips just under target) and conservative on "
               "CRAVES (coverage below its self-calibrated ceiling); transfer holds best in the "
               "refined regime. The coverage a fixed threshold yields is always robot-specific.",
               ""]
        out = args.out_dir / fname
        out.write_text("\n".join(md))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
