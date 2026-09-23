#!/usr/bin/env python3
"""Mask-IoU table of the detection sources against the GT arm silhouette.

Compares, per dataset, the mask IoU against the GT arm silhouette for:
  - CNOS/SAM detections      (<detections>/sam/<ds>.json.gz)
  - CNOS/FastSAM detections  (<detections>/fastsam/<ds>.json.gz)
  - NeMO bootstrap est masks (<est-masks>/<ds>/est_masks.json.gz,
                              est_mesh_projection at the estimated pose,
                              from analysis/gating/render_masks.py)

All sources share one schema: {"detections": [{frame_id, found, score?,
time_ms?, segmentation: {counts, size}}, ...]} with uncompressed COCO RLE.
IoU is computed over frames where BOTH the source and GT have found=true;
coverage (found-rate) and median per-frame runtime are reported alongside.
Baxter has no ground-truth masks and is excluded.

Stdlib only (no numpy / pycocotools): RLE decode via bytearray runs,
IoU via big-int bit ops.

    python analysis/compute_detection_iou.py --gt-masks <dir> --detections <dir> \\
        --est-masks <dir>
"""
import argparse
import gzip
import json
import statistics
from pathlib import Path

# name -> (detections file stem, gt file stem, est dir name)
DATASETS = [
    ("Panda-Orb", "panda_orb", "gt_panda_orb", "panda_orb"),
    ("OWI", "craves", "gt_craves", "craves"),
    ("LBR", "lbr_med7", "gt_hydra_lbr", "hydra_lbr"),
    ("xArm", "xarm", "gt_hydra_xarm", "hydra_xarm"),
    ("Meca", "meca", "gt_hydra_meca", "hydra_meca"),
]


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def rle_to_int(seg):
    """Uncompressed COCO RLE -> big int, bit i = pixel i (column-major)."""
    counts, size = seg["counts"], seg["size"]
    n = size[0] * size[1]
    buf = bytearray(n)
    pos = 0
    for i, run in enumerate(counts):
        if i % 2 == 1:  # odd runs are foreground
            buf[pos:pos + run] = b"\x01" * run
        pos += run
    return int.from_bytes(buf, "little")


def index_by_frame(payload):
    return {d["frame_id"]: d for d in payload["detections"]}


def score_source(gt_idx, src_idx):
    """Return (ious, coverage, median_runtime_ms) of src against GT."""
    ious, times = [], []
    n_total = len(gt_idx)
    n_found = 0
    for fid, g in gt_idx.items():
        s = src_idx.get(fid)
        if s is None or not s.get("found") or not g.get("found"):
            continue
        n_found += 1
        a, b = rle_to_int(g["segmentation"]), rle_to_int(s["segmentation"])
        inter = bin(a & b).count("1")
        union = bin(a | b).count("1")
        ious.append(inter / union if union else 0.0)
        if "time_ms" in s:
            times.append(s["time_ms"])
    cov = n_found / n_total if n_total else 0.0
    t_med = statistics.median(times) if times else None
    return ious, cov, t_med


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-masks", type=Path, required=True, help="dir of gt_<ds>.json.gz")
    ap.add_argument("--detections", type=Path, required=True, help="robot-detector detections root (sam/, fastsam/)")
    ap.add_argument("--est-masks", type=Path, required=True, help="dir of <ds>/est_masks.json.gz")
    args = ap.parse_args()

    rows = []
    for name, det_stem, gt_stem, est_stem in DATASETS:
        gt_idx = index_by_frame(load(args.gt_masks / f"{gt_stem}.json.gz"))
        srcs = {
            "nemo": index_by_frame(load(args.est_masks / est_stem / "est_masks.json.gz")),
            "fastsam": index_by_frame(load(args.detections / "fastsam" / f"{det_stem}.json.gz")),
            "sam": index_by_frame(load(args.detections / "sam" / f"{det_stem}.json.gz")),
        }
        row = {"name": name, "n_frames": len(gt_idx)}
        for key, idx in srcs.items():
            ious, cov, t_med = score_source(gt_idx, idx)
            row[key] = {
                "mean": statistics.fmean(ious) if ious else float("nan"),
                "median": statistics.median(ious) if ious else float("nan"),
                "cov": cov,
                "t_med": t_med,
            }
        rows.append(row)
        r = row
        print(f"{name:10s} n={r['n_frames']:6d}  "
              f"nemo {r['nemo']['mean']:.3f} (cov {r['nemo']['cov']:.2f})  "
              f"fastsam {r['fastsam']['mean']:.3f}  sam {r['sam']['mean']:.3f}")

    print("\n% --- LaTeX rows ---")
    for r in rows:
        cells = [r["name"]]
        for key in ("nemo", "fastsam", "sam"):
            s = r[key]
            cell = f"{s['mean']:.3f}"
            if s["cov"] < 0.999:
                cell += f"\\,{{\\scriptsize({s['cov'] * 100:.0f}\\%\\,cov.)}}"
            cells.append(cell)
        for key in ("fastsam", "sam"):
            t = r[key]["t_med"]
            cells.append(f"{t:.0f}" if t is not None else "--")
        print(" & ".join(cells) + " \\\\")


if __name__ == "__main__":
    main()
