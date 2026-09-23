#!/usr/bin/env python3
"""Pose error against detection quality on Panda-Orb.

Per-frame CNOS/SAM detection-mask IoU (vs ground-truth mask) joined with
per-frame ADD for all eight benchmark configurations (four coarse stages, four
with the shared MegaPose refiner). Same data as iou_vs_add.py.

Note: the gating iou_cache quantity is a DIFFERENT IoU (mask rendered at the
estimated pose vs detection), the confidence-gating signal. Do not mix.

Figure: median ADD in equal 0.05 IoU bins, bins with fewer than 30 frames
omitted, coarse configurations only (the shared MegaPose refiner never sees
the detection: it re-crops the image around the coarse estimate, so the
detection enters the pipeline at the coarse stage). One grey bar series behind
the curves shows how many frames each bin holds; the detections are shared
across configurations, so a single series serves all curves.

Failures enter at the 1000 mm sentinel (results_metrics.FAIL_MM).

Writes a per-frame CSV cache into --out so plot iterations skip the mask
decode, and the figure (pdf + png) next to it.

    python analysis/iou_add_detection_curve.py --gt-masks <dir> --detections <dir> \\
        --results <results_root> --out <fig_dir>
"""
import argparse
import csv
import gzip
import json
import zipfile
from pathlib import Path

import numpy as np

FAIL_MM = 1000.0          # failure sentinel, results_metrics.py
EDGES = np.arange(0.30, 1.001, 0.05)
MIN_PER_BIN = 30

# Panda-Orb result subdirectory per configuration (same mapping as
# iou_vs_add.py, extended by the two remaining refined rows). Dict order is the
# legend order: coarse block first, then the refined rows.
MODELS = {
    "NeMO coarse":       ("nemo_coarse", "panda_orb_nemo_shards_final"),
    "MegaPose coarse":   ("megapose_coarse", "panda_orb_megapose_coarse_shards_final"),
    "GigaPose coarse":   ("gigapose_coarse", "panda_orb_gigapose_shards_final"),
    "FoundPose coarse":  ("foundpose_coarse", "panda_orb_foundpose_shards_final"),
    "NeMO refined":      ("nemo_coarse_megapose_refiner", "panda_orb_refine_nemo_shards_final"),
    "MegaPose refined":  ("megapose_coarse_megapose_refiner", "panda_orb_megapose_shards_final"),
    "GigaPose refined":  ("gigapose_coarse_megapose_refiner", "panda_orb_gigapose_megapose_shards_final"),
    "FoundPose refined": ("foundpose_coarse_megepose_refiner", "panda_orb_refine_foundpose_shards_final"),
}


def load_gz(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def load_results(d):
    if (d / "results.json").exists():
        return json.load(open(d / "results.json"))
    z = d / "results.json.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            return json.loads(zf.read([n for n in zf.namelist() if n.endswith(".json")][0]))
    return None


def rle_to_mask(seg):
    counts = np.asarray(seg["counts"], dtype=np.int64)
    return np.repeat(np.arange(len(counts)) % 2, counts).astype(bool)


def compute_per_frame(gt_dir, det_dir, res):
    gt = {d["frame_id"]: d for d in load_gz(gt_dir / "gt_panda_orb.json.gz")["detections"]}
    sam = {d["frame_id"]: d for d in load_gz(det_dir / "panda_orb.json.gz")["detections"]}
    iou = {}
    for fid, g in gt.items():
        s = sam.get(fid)
        if s is None or not s.get("found") or not g.get("found"):
            continue
        a, b = rle_to_mask(g["segmentation"]), rle_to_mask(s["segmentation"])
        union = np.count_nonzero(a | b)
        iou[fid] = np.count_nonzero(a & b) / union if union else 0.0
    print(f"Panda-Orb frames with GT and detection: {len(iou)}", flush=True)

    rows = []
    for mname, (root, sub) in MODELS.items():
        R = load_results(res / root / sub)
        n_fail = 0
        for q in R["queries"]:
            fid = q.get("frame_id")
            if fid not in iou:
                continue
            if q.get("est_pose") is None:
                add, n_fail = FAIL_MM, n_fail + 1
            else:
                add = q.get("add_m_mm")
            if add is None:
                continue
            rows.append((mname, fid, iou[fid], float(add)))
        print(f"  {mname}: {n_fail} failures at sentinel, cumulative rows {len(rows)}",
              flush=True)
    return rows


def load_per_frame(cache, gt_dir, det_dir, res):
    if cache.exists():
        with open(cache) as f:
            rows = [(m, f_, float(i), float(a)) for m, f_, i, a in csv.reader(f)]
        if {m for m, *_ in rows} == set(MODELS):
            return rows
        print("cache predates the 8-configuration set, recomputing", flush=True)
    rows = compute_per_frame(gt_dir, det_dir, res)
    with open(cache, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return rows


def binned_medians(ious, adds):
    """(centres, medians, counts) over EDGES, bins below MIN_PER_BIN skipped."""
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    idx = np.digitize(ious, EDGES) - 1
    xs, meds, ns = [], [], []
    for b in range(len(centres)):
        sel = adds[idx == b]
        if len(sel) >= MIN_PER_BIN:
            xs.append(centres[b])
            meds.append(float(np.median(sel)))
            ns.append(len(sel))
    return np.array(xs), np.array(meds), np.array(ns)


def main():
    from analysis_utils import model_color, setup_matplotlib
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-masks", type=Path, required=True, help="dir with gt_panda_orb.json.gz")
    ap.add_argument("--detections", type=Path, required=True, help="dir with panda_orb.json.gz (CNOS/SAM)")
    ap.add_argument("--results", type=Path, required=True, help="results root")
    ap.add_argument("--out", type=Path, default=Path("."), help="figure + cache directory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()
    rows = load_per_frame(args.out / "iou_add_perframe_panda_orb.csv",
                          args.gt_masks, args.detections, args.results)
    per_model = {m: (np.array([r[2] for r in rows if r[0] == m]),
                     np.array([r[3] for r in rows if r[0] == m])) for m in MODELS}

    # numbers for the text: first and last plotted bin for all eight configs
    for mname, (ious, adds) in per_model.items():
        xs, meds, ns = binned_medians(ious, adds)
        print(f"  {mname}: median {meds[0]:.0f} mm @IoU {xs[0]:.2f} (n={ns[0]}) "
              f"-> {meds[-1]:.0f} mm @IoU {xs[-1]:.2f} (n={ns[-1]})", flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax2 = ax.twinx()

    # shared detections: one count series serves all curves
    ref_ious = next(iter(per_model.values()))[0]
    counts, _ = np.histogram(ref_ious, bins=EDGES)
    centres = 0.5 * (EDGES[:-1] + EDGES[1:])
    ax2.bar(centres, counts, width=0.046, color="0.88", linewidth=0)
    ax2.set_ylabel("frames per bin", fontsize=9)
    ax2.set_ylim(0, counts.max() * 1.05)
    ax.set_zorder(ax2.get_zorder() + 1)  # lines above the bars
    ax.patch.set_visible(False)

    for mname, (ious, adds) in per_model.items():
        if not mname.endswith("coarse"):
            continue
        base = mname.split()[0]
        xs, meds, _ = binned_medians(ious, adds)
        ax.plot(xs, meds, marker="o", markersize=3, color=model_color(base),
                linewidth=1.6, label=base)

    ax.set_xlabel("Detection-mask IoU against ground truth")
    ax.set_ylabel("median ADD (mm)")
    ax.set_xlim(EDGES[0], 1.0)
    ax.set_ylim(bottom=0)
    # legend below the plot, matching depth_dominance (analyze_compare.py)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
              frameon=False, fontsize=9)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    for ext in ("pdf", "png"):
        out = args.out / f"iou_vs_add_panda_orb.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
