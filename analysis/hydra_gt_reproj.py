#!/usr/bin/env python3
"""Hydra ground-truth validation figure.

Top row: one annotated frame per Hydra arm - detected AprilTag corners
(green) vs the tag reprojected through the bootstrapped T_cam_base (red),
rendered by hydra-data-calibration/visualize_reprojection.py. Cropped so all
three panels end up 853x720: LBR/xArm lose the empty left third, Meca loses a
sixth on each side.

Bottom row: cross-validated centre-reprojection error (mean +/- std) of our
calibration against the Hydra paper's, one point per series on a shared mm
track (the benchmark_summary.py idiom). Same quantity and presentation as the
paper (Table I), so the error-bar overlap is the reproduction claim.

Values are hardcoded from the hydra-data-calibration Monte Carlo
cross-validation table (N=9 train / 6 held-out, 5 splits, pooled over the 3
measurements per arm), PnP + RealSense: ours (mm) lbr 1.50+/-0.48,
meca 0.55+/-0.15, xarm 3.15+/-0.97; paper (mm) lbr 1.8+/-0.9, meca 0.6+/-0.3,
xarm 3.9+/-2.0.

    python analysis/hydra_gt_reproj.py --frames <reprojection_out> --out <fig_dir>
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

W, H = 1280, 720
CROP = W // 3          # LBR/xArm: drop left third
MECA_BOX = (340, 0, 340 + (W - 2 * (W // 6)), H)  # Meca: 853 px wide, but
# shifted right past the printed error text (ends ~x=330) so no fragment shows

C_OURS = "C0"          # our calibration - blue, as benchmark_summary
C_PAPER = "C3"         # Hydra paper - red

# panel -> (frame relpath, (ours mean, ours std) mm, (paper mean, paper std) mm)
PANELS = [
    ("LBR",  "lbr/measurement_0/frame_13.png",  (1.50, 0.48), (1.8, 0.9)),
    ("xArm", "xarm/measurement_0/frame_10.png", (3.15, 0.97), (3.9, 2.0)),
    ("Meca", "meca/measurement_1/frame_04.png", (0.55, 0.15), (0.6, 0.3)),
]

X_HI = 10.0            # shared mm track (xArm paper mean + std = 5.9; 10 gives headroom)


def cropped(frames_dir, rel, meca=False):
    im = Image.open(frames_dir / rel).convert("RGB")
    if meca:
        return np.asarray(im.crop(MECA_BOX))
    return np.asarray(im.crop((CROP, 0, W, H)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True,
                    help="visualize_reprojection.py output dir (lbr/, xarm/, meca/)")
    ap.add_argument("--out", type=Path, default=Path("."), help="output directory")
    args = ap.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(PANELS)
    fig = plt.figure(figsize=(7.0, 3.4))
    gs = fig.add_gridspec(2, n, height_ratios=[2.6, 1.0],
                          left=0.02, right=0.98, top=0.93, bottom=0.22,
                          hspace=0.16, wspace=0.04)

    for i, (name, frame, ours, paper) in enumerate(PANELS):
        axim = fig.add_subplot(gs[0, i])
        axim.imshow(cropped(args.frames, frame, meca=(name == "Meca")), aspect="equal")
        axim.set_xticks([])
        axim.set_yticks([])
        for s in axim.spines.values():
            s.set_visible(False)
        axim.set_title(name, fontsize=10, pad=3)

        ax = fig.add_subplot(gs[1, i])
        ax.plot([0, X_HI], [0, 0], color="0.85", lw=5,
                solid_capstyle="round", zorder=1)
        # ours above the track, paper below; error bar = std, overlap = match
        for (mean, std), color, y in ((ours, C_OURS, 0.45),
                                      (paper, C_PAPER, -0.45)):
            ax.errorbar(mean, y, xerr=std, fmt="o", ms=6, color=color,
                        ecolor=color, elinewidth=1.2, capsize=3, zorder=3)
            ax.annotate(f"{mean:.2f} ± {std:.2f}", (mean, y),
                        textcoords="offset points",
                        xytext=(0, 7 if y > 0 else -8),
                        ha="center", va="bottom" if y > 0 else "top",
                        fontsize=7, color=color)
        # shared mm-track scale: grey endpoint labels just past the track ends and
        # a little below it, the benchmark_summary.py idiom (its "0"/"100")
        pad = 0.03 * X_HI          # outward nudge past each end
        for xv, lab in ((-pad, "0"), (X_HI + pad, f"{X_HI:.0f}")):
            ax.text(xv, -0.26, lab, ha="center", va="top",   # -0.26 = gap below the bar
                    fontsize=7, color="0.5", zorder=2)
        # track spans ~80% of the panel so the labels sit just past each end
        # rather than pinched inward, with room for the "mm" unit
        ax.set_xlim(-0.11 * X_HI, 1.11 * X_HI)
        ax.set_ylim(-1.4, 1.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("centre reprojection (mm)", fontsize=8)
        for s in ax.spines.values():
            s.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_OURS,
                   label="Hydra calibration, reproduced (cross-val mean ± std)"),
        plt.Line2D([], [], marker="o", ls="", color=C_PAPER,
                   label="Hydra paper (Table I, mean ± std)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    out = out_dir / "hydra_gt_reproj.png"
    fig.savefig(out, dpi=300)
    # dpi=300 so the embedded robot photos are baked at 300 ppi (vector backends
    # otherwise embed imshow images at the low figure dpi); text/markers stay vector.
    fig.savefig(out.with_suffix(".pdf"), dpi=300)
    print(f"wrote {out} (+ .pdf)")


if __name__ == "__main__":
    main()
