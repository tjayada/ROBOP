#!/usr/bin/env python3
"""Benchmark summary figure: one panel per platform.

Per platform: a qualitative refined-FoundPose frame (prediction outline, red)
above a 0-100 track showing the best refined training-free estimator vs the
best per-robot-trained baseline (where one exists); a single dot for the three
Hydra arms, which have no published baseline. Layout mirrors the bottom row of
Fig. 1 of "Any Robot, Any Pose": blue dot = robot agnostic (training-free),
orange dot = robot specific (per-robot-trained).

AUC values are hardcoded (the baseline numbers are re-reported from the
respective papers and exist in no results directory):

    Panda : TF 81.74  (refined FoundPose)
            trained 89.29  (PK-ROKED, 1% real-data fine-tune)
    Baxter: TF 78.45  (refined MegaPose at its shipped K=5 budget; the best
                       training-free value on Baxter)   [AUC threshold 400 mm]
            trained 83.93  (CtRNet)
    OWI   : TF 90.88  (refined FoundPose)
            trained 84.75  (RoboPose, scored from public predictions)
    LBR   : TF 80.26  (refined FoundPose)
    xArm  : TF 84.24  (refined FoundPose)
    Meca  : TF 86.50  (refined FoundPose)

Frames come from the refined-FoundPose qualitative montages written by
visualize_results.py (3 panels: input | ground truth | prediction); the
rightmost third (prediction) is cropped out.

    python analysis/benchmark_summary.py --frames <montage_dir> --out <fig_dir>
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

C_TF = "C0"       # robot agnostic (best refined training-free) - blue, as the paper
C_TR = "C1"       # robot specific (best per-robot-trained)     - orange, as the paper
C_PRED = "red"    # prediction outline color in the cropped frames

# platform -> (frame relpath, tf_auc, [(trained_auc, trained_label), ...])
# (trained labels are documentation only; the caption names the methods)
PLATFORMS = [
    ("Panda",  "panda_orb/000000.jpg",  81.74, [(89.29, "PK-ROKED (1% real)")]),
    ("Baxter", "baxter/pose_0_0000.jpg", 78.45, [(83.93, "CtRNet")]),
    ("OWI",    "craves/00000000.jpg",    90.88, [(84.75, "RoboPose")]),
    ("LBR",    "hydra_lbr/meas0_000.jpeg", 80.26, []),
    ("xArm",   "hydra_xarm/meas0_000.jpg", 84.24, []),
    ("Meca",   "hydra_meca/meas0_000.jpg", 86.50, []),
]


def prediction_crop(frames_dir, rel):
    # rightmost third = prediction panel; then center-trim anything wider than
    # 4:3 down to 4:3 so the 16:9 arms (OWI, Hydra) are not squeezed flat
    im = Image.open(frames_dir / rel).convert("RGB")
    w, h = im.size
    crop = im.crop((2 * w // 3, 0, w, h))
    cw, ch = crop.size
    if cw / ch > 4 / 3:
        tw = int(ch * 4 / 3)
        left = (cw - tw) // 2
        crop = crop.crop((left, 0, left + tw, ch))
    return np.asarray(crop)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True, help="qualitative montage dir")
    ap.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(PLATFORMS)
    LEVEL_GAP = 8.0  # pairs closer than this: same height, nudged apart sideways

    fig = plt.figure(figsize=(9.0, 2.6))
    gs = fig.add_gridspec(2, n, height_ratios=[3.0, 1.0],
                          left=0.02, right=0.98, top=0.90, bottom=0.27,
                          hspace=0.08, wspace=0.08)

    for i, (name, frame, tf_auc, trained) in enumerate(PLATFORMS):
        axim = fig.add_subplot(gs[0, i])
        axim.imshow(prediction_crop(args.frames, frame), aspect="auto")
        axim.set_xticks([])
        axim.set_yticks([])
        for s in axim.spines.values():
            s.set_visible(False)
        axim.set_title(name, fontsize=11, pad=4)

        ax = fig.add_subplot(gs[1, i])
        ax.plot([0, 100], [0, 0], color="0.85", lw=6,
                solid_capstyle="round", zorder=1)
        # endpoint ticks below the bar, centred on its two ends (as in the
        # paper's Fig. 1), which reads cleaner than flanking the bar sideways
        ax.text(0, -0.42, "0", ha="center", va="top", fontsize=7, color="0.5")
        ax.text(100, -0.42, "100", ha="center", va="top", fontsize=7, color="0.5")

        dots = [(tr_auc, C_TR) for tr_auc, _label in trained]
        dots.append((tf_auc, C_TF))
        dots.sort(key=lambda d: d[0])
        close_pair = len(dots) == 2 and dots[1][0] - dots[0][0] < LEVEL_GAP
        for j, (x, color) in enumerate(dots):
            ax.plot(x, 0, "o", ms=7, zorder=3, color=color)
            # close pairs: same height, nudge left label left and right label
            # right so they don't overlap
            dx = 0 if not close_pair else (-5 if j == 0 else 5)
            ha = "center" if not close_pair else ("right" if j == 0 else "left")
            ax.annotate(f"{x:.1f}", (x, 0),
                        textcoords="offset points", xytext=(dx, 6),
                        ha=ha, va="bottom", fontsize=8, color=color)

        ax.set_xlim(-18, 118)
        # extra room below the bar for the 0/100 endpoint ticks
        ax.set_ylim(-0.8, 1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("ADD-AUC (%)", fontsize=9, labelpad=12)
        for s in ax.spines.values():
            s.set_visible(False)

    handles = [
        plt.Line2D([], [], color=C_PRED, lw=2, label="prediction"),
        plt.Line2D([], [], marker="o", ls="", color=C_TF, label="robot agnostic"),
        plt.Line2D([], [], marker="o", ls="", color=C_TR, label="robot specific"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    out = out_dir / "benchmark_summary.png"
    fig.savefig(out, dpi=300)
    # dpi=300 so the embedded robot photos are baked at 300 ppi (vector backends
    # otherwise embed imshow images at the low figure dpi); text/markers stay vector.
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
