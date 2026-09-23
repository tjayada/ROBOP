"""Risk-coverage figures for the MaskVal CNOS gate, computed directly from mask dirs.

IoU = IoU(estimator projected-mesh silhouette, CNOS/SAM detection mask), joined per
frame, paired with the per-frame ADD from the results JSON. Frames are ordered by IoU,
the lowest progressively dropped, and the retained-set ADD-AUC plotted vs coverage
against the ADD-ordered oracle (the ceiling) and the random floor. The fraction of the
oracle's AUC gain the IoU gate recovers is drawn on each panel.

Three directory roots, discovered not hardcoded:
  --est-masks  <est_masks_root>/{variant}/{dataset}/est_masks.json.gz
  --cnos       robot-detector/detections/sam/{alias}.json.gz
  --results    <results_root>/{variant}/{run}/results.json[.zip|.gz]  (ADD source)

Selective-prediction / risk-coverage protocol, Jaeger et al. ICLR'23. IoU comes from the
true projected-mesh masks (not a convex-hull silhouette), and every model variant present
under the est_masks root is picked up automatically.
"""
import argparse
import gzip
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_utils import add_auc, setup_matplotlib, INK  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2.0

GATING_DIR = Path(__file__).resolve().parents[2] / "figures" / "gating"  # repo-relative, any cwd

# dataset (est_masks subdir) -> CNOS detection file stem
CNOS_ALIAS = {"hydra_lbr": "lbr_med7", "hydra_meca": "meca", "hydra_xarm": "xarm",
              "baxter": "baxter", "craves": "craves", "panda_orb": "panda_orb"}
DATASET_LABEL = {"hydra_lbr": "Hydra LBR", "hydra_meca": "Hydra Meca",
                 "hydra_xarm": "Hydra xArm", "baxter": "Baxter", "craves": "CRAVES",
                 "panda_orb": "panda-orb"}
DATASET_ORDER = ["hydra_lbr", "hydra_xarm", "hydra_meca", "craves", "baxter", "panda_orb"]

# Baxter is scored over 0-400 mm (CtRNet endpoint protocol); everything else over 0-100 mm.
CEILING_MM = {"baxter": 400.0}
DEFAULT_CEILING_MM = 100.0

# family -> colour (validated categorical slots 1/2/3/4; see analysis_utils.CATEGORICAL)
FAMILY_COLOR = {"NeMO": "#2a78d6", "GigaPose": "#eb6834",
                "MegaPose": "#1baf7a", "FoundPose": "#eda100"}
FAMILY_TOKEN = {"nemo": "NeMO", "gigapose": "GigaPose",
                "megapose": "MegaPose", "foundpose": "FoundPose"}

C_ORACLE = INK["muted"]   # ceiling, kept recessive so the gate curve stays prominent

SCALE = 100.0     # AUC drawn on the 0-100 convention (DREAM/CtRNet), not 0-1
MIN_N = 1         # compute down to a single retained frame (tails are noisy by design)
MAX_POINTS = 300  # coverage samples per curve; keeps the 32k-frame datasets tractable
XVIEW_MIN = 0.10  # but only display coverage >= this; the sub-0.1 tail is single-frame noise


# variant naming

def parse_variant(variant):
    """Derive display metadata from an est_masks/results variant dirname."""
    fam = FAMILY_TOKEN.get(variant.split("_")[0], variant.split("_")[0])
    bootstrap = "bootstrap" in variant
    refined = "refiner" in variant
    modular = "modular" in variant.lower()
    label = fam + (" bootstrap" if bootstrap else "") + (" +ref" if refined else "")
    if modular:
        label += " (mod)"
    ls = "-."  if bootstrap else (0, (1, 1)) if modular else "--" if refined else "-"
    return {"family": fam, "label": label, "color": FAMILY_COLOR.get(fam, INK["primary"]),
            "ls": ls, "refined": refined, "extra": bootstrap or modular,
            "sort": (list(FAMILY_COLOR).index(fam) if fam in FAMILY_COLOR else 9,
                     refined, modular, bootstrap)}


# IO

def _decode_flat(rle):
    """Custom column-major RLE (order='F', mirror of eval_utils._rle_to_mask) -> flat
    bool. IoU is order-invariant as long as both masks share the encoding, so the 2D
    reshape is skipped; the (H, W) size is returned for the shape-match guard."""
    counts = np.asarray(rle["counts"], dtype=np.int64)
    vals = np.zeros(len(counts), dtype=np.uint8)
    vals[1::2] = 1
    return np.repeat(vals, counts).astype(bool), tuple(rle["size"])


def _load_gz_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _masks_by_frame(data):
    """detections list -> {frame_id: rle} for found=true entries with a segmentation."""
    out = {}
    for d in data.get("detections", []):
        if d.get("found") and d.get("segmentation") is not None:
            out[str(d["frame_id"])] = d["segmentation"]
    return out


def load_results_queries(results_root, variant, dataset):
    """Locate and load the results file for (variant, dataset); return its query list.
    Matches the run subdir by the dataset token, and reads .json / .json.zip / .json.gz."""
    vdir = results_root / variant
    if not vdir.is_dir():
        return None
    runs = [d for d in vdir.iterdir() if d.is_dir() and dataset in d.name]
    if not runs:
        return None
    run = runs[0]
    for name in ("results.json", "results.json.zip", "results.json.gz"):
        p = run / name
        if not p.exists():
            continue
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                inner = next(n for n in z.namelist() if n.endswith("results.json")
                             and not n.startswith("__MACOSX"))
                return json.load(z.open(inner))["queries"]
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt") as f:
            return json.load(f)["queries"]
    return None


# IoU computation (cached)

def iou_add(est_path, dataset, results_root, variant, cnos_root, cache_dir, recompute):
    """Per-frame (iou_cnos, add_mm) for one (variant, dataset), plus the abstention count.

    Frames come from the results file (authoritative ADD + order). A frame gets IoU 0 when
    the estimator abstained (pnp_failed) or produced no mask, and when the detector found
    nothing - all are dropped first by the gate, and dropping them would overstate the model.
    """
    cache = cache_dir / f"{variant}__{dataset}.json"
    if cache.exists() and not recompute:
        c = json.load(open(cache))
        return np.asarray(c["iou"]), np.asarray(c["add"]), c["n_abstain"]

    queries = load_results_queries(results_root, variant, dataset)
    if queries is None:
        return None, None, 0
    est = _masks_by_frame(_load_gz_json(est_path))
    cnos = _masks_by_frame(_load_gz_json(cnos_root / f"{CNOS_ALIAS[dataset]}.json.gz"))

    iou, add, n_abstain, n_bad_size = [], [], 0, 0
    for q in queries:
        a = q.get("add_m_mm")
        if a is None:
            continue
        fid = str(q.get("frame_id"))
        add.append(float(a))
        if q.get("pnp_failed") or fid not in est:
            iou.append(0.0)
            n_abstain += 1
            continue
        if fid not in cnos:
            iou.append(0.0)
            continue
        me, se = _decode_flat(est[fid])
        mc, sc = _decode_flat(cnos[fid])
        if se != sc:
            iou.append(0.0)
            n_bad_size += 1
            continue
        inter = int(np.logical_and(me, mc).sum())
        union = int(np.logical_or(me, mc).sum())
        iou.append(inter / union if union else 0.0)

    iou, add = np.asarray(iou), np.asarray(add)
    if n_bad_size:
        print(f"    ! {variant}/{dataset}: {n_bad_size} frames skipped (mask size mismatch)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"iou": iou.round(4).tolist(), "add": add.round(3).tolist(),
               "n_abstain": n_abstain}, open(cache, "w"))
    return iou, add, n_abstain


# risk-coverage

def risk_coverage(order_desc, add, ceiling):
    """Retain the top-k by `order_desc` (higher kept longer); return (coverage, ADD-AUC)
    over a grid of k, capped at MAX_POINTS so 32k-frame datasets stay fast."""
    idx = np.argsort(-order_desc, kind="stable")
    n = len(add)
    ordered = add[idx]
    ks = np.unique(np.linspace(MIN_N, n, min(MAX_POINTS, n - MIN_N + 1)).astype(int))
    cov = ks / n
    auc = np.array([add_auc(ordered[:k], ceiling) for k in ks])
    return cov, auc


def aurc(cov, auc):
    """Area under the (coverage, AUC) curve, normalized to the covered span."""
    o = np.argsort(cov)
    c, a = cov[o], auc[o]
    return float(_trapz(a, c) / (c[-1] - c[0]))


def recovered(ic, ia, oc, oa, base):
    """IoU-gate AUC gain as a fraction of the oracle gain, both over the random floor."""
    g_orc = aurc(oc, oa) - base
    return (aurc(ic, ia) - base) / g_orc if abs(g_orc) > 1e-9 else float("nan")


# figures

def panel(ax, iou, add, meta, ceiling, n_abstain, n_total):
    base = add_auc(add, ceiling)                        # full-coverage AUC = random floor
    ic, ia = risk_coverage(iou, add, ceiling)           # IoU-ordered
    oc, oa = risk_coverage(-add, add, ceiling)          # oracle (drop worst ADD)
    frac = recovered(ic, ia, oc, oa, base)
    gain = aurc(ic, ia) - base                          # absolute AUC uplift the gate buys

    # to remove every abstention the gate must drop below this coverage: shade the band
    # of coverages that still retain some (they sit at the top of the drop order, IoU 0).
    if n_abstain:
        ax.axvspan(1.0 - n_abstain / n_total, 1.0, color=INK["muted"], alpha=0.10, lw=0)

    base_d, ia_d, oa_d = base * SCALE, ia * SCALE, oa * SCALE
    ax.fill_between(ic, base_d, ia_d, color=meta["color"], alpha=0.10, lw=0)
    ax.plot(oc, oa_d, color=C_ORACLE, lw=1.4, ls="--", label="ADD-ordered")
    ax.axhline(base_d, color=INK["muted"], lw=1.2, ls=(0, (1, 1.5)), label="random ordering")
    ax.plot(ic, ia_d, color=meta["color"], lw=2.2, label="IoU-ordered", solid_capstyle="round")

    # ratio + absolute AUC gain: a high ratio off a tiny achievable gain (weak model)
    # is correct but easy to over-read, so the parenthetical grounds it.
    ax.text(0.04, 0.06, f"recovers {frac*100:.0f}%  ({gain*SCALE:+.0f} AUC)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK["secondary"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))
    if n_abstain:
        # bottom-right: the curves rise toward the top-right, so this corner stays clear
        ax.text(0.96, 0.06, f"{n_abstain} abstain", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=INK["muted"])
    ax.set_title(meta["label"])
    ax.set_xlim(1.02, XVIEW_MIN - 0.03)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 102)
    return frac


def _save(fig, outdir, name):
    p = outdir / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    print(f"  wrote {p.name} (+ .pdf)")


def fig_indepth(dataset, series, ceiling, outdir, minimal):
    import matplotlib.pyplot as plt
    fracs = {}
    if minimal:
        # one column per family, row 0 = coarse, row 1 = its refiner
        fams = [f for f in FAMILY_COLOR if any(m["family"] == f for m, *_ in series)]
        ncol, nrow = len(fams), 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        used = set()
        for meta, iou, add, n_ab in series:
            r, c = (1 if meta["refined"] else 0), fams.index(meta["family"])
            used.add((r, c))
            fracs[meta["label"]] = panel(axes[r][c], iou, add, meta, ceiling, n_ab, len(add))
        for r in range(nrow):
            for c in range(ncol):
                if (r, c) not in used:
                    axes[r][c].set_visible(False)
        for c in range(ncol):
            rows = [r for r in range(nrow) if (r, c) in used]
            if rows:
                axes[max(rows)][c].set_xlabel("coverage")
        for r in range(nrow):
            cols = [c for c in range(ncol) if (r, c) in used]
            if cols:
                axes[r][min(cols)].set_ylabel(f"ADD-AUC @ {ceiling:.0f} mm")
        first = axes[0][0]
    else:
        n = len(series)
        ncol = min(n, 4)
        nrow = (n + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow),
                                 sharex=True, sharey=True, squeeze=False)
        axf = axes.ravel()
        for ax, (meta, iou, add, n_ab) in zip(axf, series):
            fracs[meta["label"]] = panel(ax, iou, add, meta, ceiling, n_ab, len(add))
        for ax in axf[n:]:
            ax.set_visible(False)
        for ax in axf[max(0, n - ncol):n]:
            ax.set_xlabel("coverage")
        for r in range(nrow):
            axes[r, 0].set_ylabel(f"ADD-AUC @ {ceiling:.0f} mm")
        first = axf[0]
    # Neutral proxy handles: the three curves encode an *ordering role*, not a model,
    # so the shared legend must not borrow any single panel's model color (else the
    # "IoU-ordered" swatch reads as one estimator). Panel titles carry the model.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=C_ORACLE, lw=1.4, ls="--", label="ADD-ordered"),
        Line2D([0], [0], color=INK["muted"], lw=1.2, ls=(0, (1, 1.5)),
               label="random ordering"),
        Line2D([0], [0], color=INK["primary"], lw=2.2, solid_capstyle="round",
               label="IoU-ordered"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    # No in-image suptitle: the LaTeX \caption names the figure (and avoids baking a
    # lowercased platform / CRAVES-vs-OWI naming into the raster).
    fig.tight_layout(rect=(0, 0.05, 1, 1.0))
    _save(fig, outdir, f"risk_coverage_indepth_{dataset}")
    plt.close(fig)
    print("    recovered (whole-curve captured): "
          + ", ".join(f"{k} {v*100:.0f}%" for k, v in fracs.items()))


def fig_models(dataset, series, ceiling, outdir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 5))
    for meta, iou, add, _ in series:
        ic, ia = risk_coverage(iou, add, ceiling)
        keep = ic >= XVIEW_MIN
        ax.plot(ic[keep], ia[keep] * SCALE, color=meta["color"], ls=meta["ls"], lw=2.0,
                label=meta["label"])
    ax.set_xlim(1.02, XVIEW_MIN - 0.03)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 102)
    ax.set_xlabel("coverage")
    ax.set_ylabel(f"ADD-AUC @ {ceiling:.0f} mm (retained frames)")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    # No in-image suptitle: the LaTeX \caption names the figure.
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    _save(fig, outdir, f"risk_coverage_models_{dataset}")
    plt.close(fig)


def discover(est_root):
    """(variant, dataset, est_masks path) for every dump under the est_masks root."""
    for vdir in sorted(p for p in est_root.iterdir() if p.is_dir()):
        for ddir in sorted(p for p in vdir.iterdir() if p.is_dir()):
            f = ddir / "est_masks.json.gz"
            if f.exists() and ddir.name in CNOS_ALIAS:
                yield vdir.name, ddir.name, f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est-masks", type=Path, required=True)
    ap.add_argument("--cnos", type=Path, required=True, help="dir of {stem}.json.gz detections")
    ap.add_argument("--results", type=Path, required=True, help="results root (ADD source)")
    ap.add_argument("--outdir", type=Path, default=GATING_DIR)
    ap.add_argument("--recompute", action="store_true", help="ignore the IoU cache")
    ap.add_argument("--full", action="store_true",
                    help="show every variant in an auto-grid; default drops bootstrap/modular "
                         "and lays panels out as one column per family, coarse over refiner")
    args = ap.parse_args()

    setup_matplotlib()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "iou_cache"

    by_dataset = {}
    for variant, dataset, est_path in discover(args.est_masks):
        iou, add, n_ab = iou_add(est_path, dataset, args.results, variant, args.cnos,
                                 cache_dir, args.recompute)
        if iou is None or len(iou) < 5:
            print(f"skip {variant}/{dataset}: no results / too few frames")
            continue
        by_dataset.setdefault(dataset, []).append((parse_variant(variant), iou, add, n_ab))

    minimal = not args.full
    for dataset in [d for d in DATASET_ORDER if d in by_dataset]:
        series = sorted(by_dataset[dataset], key=lambda s: s[0]["sort"])
        if minimal:
            series = [s for s in series if not s[0]["extra"]]
        ceiling = CEILING_MM.get(dataset, DEFAULT_CEILING_MM)
        print(f"\n=== {dataset} ({len(series)} estimators) ===")
        fig_indepth(dataset, series, ceiling, args.outdir, minimal)
        fig_models(dataset, series, ceiling, args.outdir)


if __name__ == "__main__":
    main()
