"""
ROBOP result analysis.

Usage:
    python analyze.py results/panda_nemo.json
    python analyze.py results/baxter_gigapose.json --failure-threshold 50
    python analyze.py results/craves_nemo.json --plots plots/craves_nemo/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis_utils import (
    load_results, detect_dataset, is_nemo,
    extract_signals, error_decomposition,
    naive_score, compute_auroc, pareto_curve, operating_point,
    print_header, print_table, spearman_table, add_auc,
    setup_matplotlib, BLUE, ORANGE, RED, DIVERGING, SEVERITY, SEVERITY_LS, INK,
)


# Section 1: Summary

def section_summary(summary: dict, dataset: str, nemo: bool, signals: dict) -> None:
    print_header(f"Summary  [{dataset}]  ({'NeMO' if nemo else 'DINOv2'})")

    rows = []
    for key in ["add_auc_100mm", "add_auc_400mm", "add_at_100mm", "add_mean_mm"]:
        if key in summary:
            rows.append([key, summary[key]])

    if dataset == "baxter":
        for key in ["pck_auc_200px", "pck_at_50px", "mean_2d_px"]:
            if key in summary:
                rows.append([key, summary[key]])

    if dataset == "craves":
        for key in ["pck_at_0_2_all17", "pck_auc_all17", "pck_at_0_2_joints",
                    "mean_trans_err_cm", "mean_rot_euler_deg"]:
            if key in summary:
                rows.append([key, summary[key]])

    # Coverage / failures from the per-frame pnp_failed flags, not a stored counter.
    pnp_failed = signals["pnp_failed"]
    n_total = len(pnp_failed)
    n_fail  = int(np.sum(pnp_failed))
    if n_total:
        rows.append(["coverage", f"{(n_total - n_fail) / n_total:.4f}"])
        rows.append(["pnp_failures", f"{n_fail} / {n_total}  ({100 * n_fail / n_total:.1f}%)"])

    if "runtime_median_ms" in summary:
        rows.append(["runtime_median_ms", summary["runtime_median_ms"]])
        rows.append(["runtime_mean_ms",   summary["runtime_mean_ms"]])

    print_table(["Metric", "Value"], rows)


# Section 2: Failure analysis

def section_failures(queries: list[dict], signals: dict, failure_threshold_mm: float) -> None:
    print_header("Failure Analysis")

    n_total    = len(queries)
    n_pnp_fail = int(np.sum(signals["pnp_failed"]))
    add        = signals["add_m_mm"]
    n_above    = int(np.sum(add >= failure_threshold_mm))

    print(f"  Total frames       : {n_total}")
    print(f"  PnP failures       : {n_pnp_fail}  ({100*n_pnp_fail/n_total:.1f}%)")
    print(f"  ADD >= {failure_threshold_mm:.0f}mm        : {n_above}  ({100*n_above/n_total:.1f}%)")

    # ADD distribution excluding failures
    success_mask = ~signals["pnp_failed"]
    if success_mask.sum() > 0:
        add_succ = add[success_mask]
        print(f"\n  ADD distribution (PnP-success frames only, n={success_mask.sum()}):")
        for pct in [25, 50, 75, 90, 95]:
            print(f"    p{pct:02d}: {np.percentile(add_succ, pct):.1f} mm")
        print(f"    mean: {add_succ.mean():.1f} mm")

    # Oracle: what AUC would you get if you could perfectly filter failures?
    oracle_mask = add < failure_threshold_mm
    if oracle_mask.sum() > 0:
        arr_oracle = add[oracle_mask]
        auc_oracle = add_auc(arr_oracle, 100.0)
        auc_all    = add_auc(add, 100.0)
        coverage   = float(oracle_mask.sum() / n_total)
        print(f"\n  Oracle filter (keep ADD < {failure_threshold_mm:.0f}mm):")
        print(f"    coverage  : {coverage:.3f}  ({oracle_mask.sum()} / {n_total} frames)")
        print(f"    AUC all   : {auc_all:.4f}")
        print(f"    AUC oracle: {auc_oracle:.4f}  (+{auc_oracle - auc_all:.4f})")


# Section 3: Error decomposition

def section_error_decomposition(queries: list[dict], dataset: str) -> None:
    print_header("Error Decomposition")

    decomp = error_decomposition(queries, dataset)
    x   = decomp["x_mm"]
    y   = decomp["y_mm"]
    z   = decomp["z_mm"]
    lat = decomp["lateral_mm"]
    dep = decomp["depth_mm"]
    rot = decomp["rot_geodesic_deg"]
    rx  = decomp["rx_deg"]
    ry  = decomp["ry_deg"]
    rz  = decomp["rz_deg"]

    if len(x) == 0:
        print("  No valid frames for decomposition.")
        return

    print(f"  N (PnP-success frames): {len(x)}")
    print()

    def _row(name, arr, signed=True):
        a = arr[~np.isnan(arr)]
        if len(a) == 0:
            return [name, "n/a", "n/a", "n/a"]
        mean_s   = f"{a.mean():+.1f}" if signed else f"{a.mean():.1f}"
        median_s = f"{np.median(a):+.1f}" if signed else f"{np.median(a):.1f}"
        return [name, mean_s, median_s, f"{np.percentile(np.abs(a), 90):.1f}"]

    print("  Translation:")
    print_table(
        ["Component", "Mean", "Median", "|p90|"],
        [
            _row("X (mm, signed)",       x,   signed=True),
            _row("Y (mm, signed)",       y,   signed=True),
            _row("Z (mm, signed)",       z,   signed=True),
            _row("Lateral √(X²+Y²) mm", lat, signed=False),
            _row("Depth |Z| mm",        dep, signed=False),
        ],
        col_width=22,
    )
    bias = "systematic" if abs(z.mean()) > 0.1 * dep.mean() else "no systematic"
    print(f"  -> {bias} Z bias  (Z mean={z.mean():+.1f} mm along robot base-frame Z axis)")

    rot_valid = rot[~np.isnan(rot)]
    if len(rot_valid) > 0:
        print("\n  Rotation:")
        print_table(
            ["Component", "Mean", "Median", "|p90|"],
            [
                _row("Rx (°, signed)", rx, signed=True),
                _row("Ry (°, signed)", ry, signed=True),
                _row("Rz (°, signed)", rz, signed=True),
                _row("Geodesic (°)",   rot, signed=False),
            ],
            col_width=22,
        )
    else:
        print("\n  Rotation: N/A (Baxter EE-only evaluation)")


# Section 4: Signal correlations  (NeMO only)

def section_signal_correlations(signals: dict, failure_threshold_mm: float) -> None:
    print_header("Signal Correlations  (Spearman ρ vs ADD error)")

    add = signals["add_m_mm"]
    # Exclude PnP failures from correlation - sentinel 1000mm would dominate
    valid = ~signals["pnp_failed"]
    add_valid = add[valid]
    signals_valid = {k: v[valid] for k, v in signals.items()}

    print(f"  N = {valid.sum()} frames (PnP failures excluded from correlation)\n")
    spearman_table(signals_valid, add_valid)


# Section 5: AUROC per signal  (NeMO only)

def section_auroc(signals: dict, failure_threshold_mm: float) -> None:
    print_header(f"AUROC  (label: ADD < {failure_threshold_mm:.0f}mm)")

    add     = signals["add_m_mm"]
    labels  = (add < failure_threshold_mm).astype(int)

    # Skip frames with NaN scores
    signal_names = ["conf_50", "conf_count_50", "reproj", "inlier", "anc_r",
                    "iou_conf", "iou_modal", "iou_amodal"]

    rows = []
    for name in signal_names:
        arr = signals.get(name)
        if arr is None:
            continue
        # For "lower is better" signals, invert for AUROC (higher score = better)
        invert = name in ("reproj", "anc_r")
        scores = -arr if invert else arr
        valid  = ~np.isnan(scores)
        if valid.sum() < 10 or labels[valid].sum() == 0 or (1 - labels[valid]).sum() == 0:
            rows.append([name, "n/a", str(int(valid.sum()))])
            continue
        auroc = compute_auroc(scores[valid], labels[valid])
        rows.append([name, f"{auroc:.4f}", str(int(valid.sum()))])

    # Naive combined score
    score = naive_score(signals)
    valid = ~np.isnan(score)
    if valid.sum() >= 10 and labels[valid].sum() > 0 and (1 - labels[valid]).sum() > 0:
        auroc_combined = compute_auroc(score[valid], labels[valid])
        rows.append(["naive_combined", f"{auroc_combined:.4f}", str(int(valid.sum()))])

    print_table(["Signal", "AUROC", "N"], rows, col_width=20)


# Section 6: Pareto frontier + operating points  (NeMO only)

def section_gating(signals: dict, failure_threshold_mm: float) -> None:
    print_header("Confidence Gating")

    add     = signals["add_m_mm"]
    total_n = len(add)

    # naive_score gate (includes PnP-failure frames via conf_50 + anc_r)
    score = naive_score(signals)
    valid = ~np.isnan(score)
    if valid.sum() >= 20:
        score_v = score[valid]
        add_v   = add[valid]
        print("  naive_score gate  (conf_50 + reproj + inlier + anc_r)")
        print(f"  N frames with valid score: {valid.sum()} / {total_n}  "
              f"(includes PnP-failure frames where conf_50/anc_r are available)\n")
        print_table(
            ["Coverage target", "Actual cov.", "Threshold", "AUC (filtered)", "AUC (all)", "AUC gain"],
            [list(operating_point(score_v, add_v, c, failure_threshold_mm, total_n).values())
             for c in [0.9, 0.8, 0.7, 0.6]],
            col_width=16,
        )
    else:
        print("  naive_score: insufficient frames with valid scores.")

    # iou_modal gate (PnP-success frames only)
    iou_score   = signals["iou_modal"]
    iou_valid   = ~np.isnan(iou_score)
    n_pnp_succ  = int(iou_valid.sum())

    print()
    if n_pnp_succ >= 20:
        iou_score_v = iou_score[iou_valid]
        add_iou_v   = add[iou_valid]
        auc_pnp_baseline = add_auc(add_iou_v, 100.0)
        print("  iou_modal gate  (PnP-success frames only)")
        print(f"  N PnP-success frames: {n_pnp_succ} / {total_n}  "
              f"(coverage targets are relative to PnP-success frames)")
        print(f"  Baseline AUC on PnP-success frames (no IoU gate): {auc_pnp_baseline:.4f}\n")
        print_table(
            ["Coverage target", "Actual cov.", "Threshold", "AUC (filtered)", "AUC (base)", "AUC gain"],
            [list(operating_point(iou_score_v, add_iou_v, c, failure_threshold_mm,
                                  total_n=n_pnp_succ).values())
             for c in [0.9, 0.8, 0.7, 0.6]],
            col_width=16,
        )
    else:
        print("  iou_modal: insufficient PnP-success frames with valid IoU scores.")


# Section 6b: Level 1 gate - pre-PnP signal analysis  (NeMO only)

def section_level1_gating(signals: dict) -> None:
    """
    Can pre-PnP signals (conf_50, conf_count_50, anc_r) predict PnP failure?
    These are available for ALL frames, unlike IoU which requires a successful pose.
    """
    print_header("Level 1 Gate - Pre-PnP Signal Analysis")

    pnp_failed = signals["pnp_failed"]
    n_fail     = int(pnp_failed.sum())
    n_total    = len(pnp_failed)

    print(f"  PnP failures: {n_fail} / {n_total}  ({100 * n_fail / n_total:.1f}%)")
    print("  A pre-PnP gate on conf_count_50 can reduce failures before RANSAC runs.\n")

    # Distribution comparison: success vs failure frame medians
    level1_signals = [
        ("conf_count_50", False),   # (name, invert_for_auroc)
        ("conf_50",       False),
        ("anc_r",         True),
    ]

    dist_rows = []
    for name, _ in level1_signals:
        arr = signals.get(name)
        if arr is None:
            continue
        succ_vals = arr[~pnp_failed & ~np.isnan(arr)]
        fail_vals = arr[ pnp_failed & ~np.isnan(arr)]
        if len(fail_vals) < 3 or len(succ_vals) < 3:
            continue
        dist_rows.append([
            name,
            f"{np.median(succ_vals):.3f}  (n={len(succ_vals)})",
            f"{np.median(fail_vals):.3f}  (n={len(fail_vals)})",
            f"{np.median(succ_vals) - np.median(fail_vals):+.3f}",
        ])
    print_table(["Signal", "Median (success)", "Median (failure)", "Δ median"],
                dist_rows, col_width=26)

    # AUROC: label=1 = PnP success (good), label=0 = PnP failure (bad)
    labels = (~pnp_failed).astype(int)
    auroc_rows = []
    for name, invert in level1_signals:
        arr = signals.get(name)
        if arr is None:
            continue
        scores = -arr if invert else arr
        valid  = ~np.isnan(scores)
        n_valid = int(valid.sum())
        if n_valid < 10 or labels[valid].sum() == 0 or (1 - labels[valid]).sum() == 0:
            auroc_rows.append([name, "n/a", str(n_valid)])
            continue
        auroc = compute_auroc(scores[valid], labels[valid])
        auroc_rows.append([name, f"{auroc:.4f}", str(n_valid)])

    print()
    print("  AUROC predicting PnP success  (label=1=success; >0.7 is useful for gating):\n")
    print_table(["Signal", "AUROC", "N"], auroc_rows, col_width=20)


# Section 7: IoU failure check  (NeMO only)

def section_iou_analysis(signals: dict, failure_threshold_mm: float) -> None:
    print_header("IoU Signal Analysis")

    add        = signals["add_m_mm"]
    pnp_failed = signals["pnp_failed"]
    iou_modal  = signals["iou_modal"]

    # PnP failure distribution
    n_pnp = int(pnp_failed.sum())
    if n_pnp > 0:
        print(f"  PnP failure frames (n={n_pnp}): iou_modal distribution")
        modal_fail = iou_modal[pnp_failed]
        valid_fail = modal_fail[~np.isnan(modal_fail)]
        if len(valid_fail) > 0:
            print(f"    mean={valid_fail.mean():.3f}  median={np.median(valid_fail):.3f}"
                  f"  p10={np.percentile(valid_fail, 10):.3f}")
        else:
            print("    (all NaN - IoU not computed for failure frames)")

    # Success frame distribution
    success = ~pnp_failed & (add < failure_threshold_mm)
    modal_succ = iou_modal[success]
    valid_succ = modal_succ[~np.isnan(modal_succ)]
    if len(valid_succ) > 0:
        print(f"\n  Success frames ADD<{failure_threshold_mm:.0f}mm (n={success.sum()}):"
              f" iou_modal mean={valid_succ.mean():.3f}  median={np.median(valid_succ):.3f}")

    # AUROC of IoU signals predicting PnP failure
    print(f"\n  AUROC of IoU signals predicting ADD >= {failure_threshold_mm:.0f}mm:")
    labels = (add >= failure_threshold_mm).astype(int)
    for name in ["iou_conf", "iou_modal", "iou_amodal"]:
        arr = signals[name]
        # Lower IoU -> worse -> invert so higher score = more confident (good)
        scores = -arr
        valid  = ~np.isnan(scores)
        if valid.sum() < 10 or labels[valid].sum() == 0 or (1 - labels[valid]).sum() == 0:
            print(f"    {name:<16}: n/a")
            continue
        auroc = compute_auroc(scores[valid], labels[valid])
        print(f"    {name:<16}: {auroc:.4f}  (N={valid.sum()})")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse ROBOP evaluation results.")
    parser.add_argument("json_path", help="Path to results JSON.")
    parser.add_argument(
        "--failure-threshold", type=float, default=100.0,
        help="ADD threshold in mm for binary failure label (default: 100).",
    )
    parser.add_argument(
        "--plots", type=str, default=None,
        help="Directory to save matplotlib figures (optional).",
    )
    args = parser.parse_args()

    summary, queries = load_results(args.json_path)
    dataset  = detect_dataset(summary)
    nemo     = is_nemo(queries)
    thr      = args.failure_threshold

    print(f"\nLoaded {len(queries)} frames  |  dataset={dataset}"
          f"  |  model={'NeMO' if nemo else 'DINOv2'}"
          f"  |  failure_threshold={thr}mm")

    signals = extract_signals(queries)

    section_summary(summary, dataset, nemo, signals)
    section_failures(queries, signals, thr)
    section_error_decomposition(queries, dataset)

    if nemo:
        section_signal_correlations(signals, thr)
        section_auroc(signals, thr)
        section_level1_gating(signals)
        section_gating(signals, thr)
        section_iou_analysis(signals, thr)

    if args.plots:
        _save_plots(signals, queries, dataset, nemo, thr, Path(args.plots))


# Plots  (palette + style shared via analysis_utils; see setup_matplotlib)


def _plot_add_histogram(
    signals: dict, dataset: str, thr: float, out_dir: Path, plt
) -> None:
    add        = signals["add_m_mm"]
    pnp_failed = signals["pnp_failed"]
    n_total    = len(add)
    n_fail     = int(pnp_failed.sum())

    # PnP-success frames sorted for ECDF; denominator = ALL frames so the
    # curve ceiling reflects the true failure rate.
    add_succ = np.sort(add[~pnp_failed & ~np.isnan(add)])
    n_succ   = len(add_succ)
    if n_succ == 0:
        return

    ecdf_y      = np.arange(1, n_succ + 1) / n_total
    frac_at_thr = float(np.mean(add < thr))   # ADD@thr over all frames
    auc_val     = add_auc(add, thr)

    # x-axis: clip at p99 of success frames or 2×thr, whichever is larger
    x_max = max(float(np.percentile(add_succ, 99)) * 1.1, thr * 1.5)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Shaded AUC region
    n_below = int(np.sum(add_succ < thr))
    if n_below > 0:
        shade_x = np.concatenate([[0], add_succ[:n_below], [thr]])
        shade_y = np.concatenate([[0], ecdf_y[:n_below],   [frac_at_thr]])
        ax.fill_between(shade_x, shade_y, alpha=0.12, color=BLUE,
                        label=f"AUC@{thr:.0f}mm = {auc_val:.3f}")

    # ECDF step curve
    ax.step(add_succ, ecdf_y, where="post", color=BLUE, lw=2.0,
            label=f"ECDF  (n={n_succ} PnP-success frames)")

    # Threshold + intersection (pass/fail boundary)
    ax.axvline(thr, color=RED, lw=1.8, linestyle="--",
               label=f"{thr:.0f}mm threshold")
    ax.plot(thr, frac_at_thr, "o", color=RED, ms=8, zorder=6)
    ax.annotate(
        f"ADD@{thr:.0f}mm = {frac_at_thr:.3f}",
        xy=(thr, frac_at_thr),
        xytext=(thr + x_max * 0.03, frac_at_thr - 0.06),
        fontsize=9, color=INK["secondary"],
        arrowprops=dict(arrowstyle="-", color=INK["muted"], lw=1.0),
    )

    # Mean / median (reference guides: recessive ink, told apart by style)
    mean_v   = float(np.mean(add_succ))
    median_v = float(np.median(add_succ))
    for val, ls, lbl in [
        (mean_v,   "--", f"Mean {mean_v:.1f} mm"),
        (median_v, ":",  f"Median {median_v:.1f} mm"),
    ]:
        if val <= x_max:
            ax.axvline(val, color=INK["muted"], lw=1.4, linestyle=ls, label=lbl)

    # PnP failure ceiling annotation
    if n_fail > 0:
        ceiling = n_succ / n_total
        ax.axhline(ceiling, color=INK["muted"], lw=1.0, linestyle="--", alpha=0.6)
        ax.text(x_max * 0.99, ceiling + 0.01,
                f"ceiling: {n_fail} PnP failures ({100*n_fail/n_total:.1f}%)",
                ha="right", va="bottom", fontsize=8.5, color=INK["muted"])

    ax.set_xlabel("ADD error (mm)", fontsize=11)
    ax.set_ylabel("Fraction of all frames  (ADD < x)", fontsize=11)
    ax.set_title(f"ADD Error ECDF - {dataset}", fontsize=12, fontweight="bold")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(out_dir / "1_add_histogram.png", dpi=150)
        plt.close(fig)


def _plot_error_decomposition(
    queries: list, dataset: str, out_dir: Path, plt,
    component: str = "both",
) -> None:
    """
    component: "both" | "translation" | "rotation"

    Signed bias is an above/below-zero quantity, so it takes the diverging form
    (one lollipop per component): blue below zero, red above, IQR as the spread.
    """
    decomp  = error_decomposition(queries, dataset)
    x, y, z = decomp["x_mm"], decomp["y_mm"], decomp["z_mm"]
    rx, ry, rz = decomp["rx_deg"], decomp["ry_deg"], decomp["rz_deg"]
    rot     = decomp["rot_geodesic_deg"]
    has_rot = not np.all(np.isnan(rot))

    if len(x) == 0:
        return

    if component == "rotation" and not has_rot:
        print(f"  No rotation GT for dataset '{dataset}' (Baxter: EE-position only).")
        return

    show_t = component in ("both", "translation")
    show_r = component in ("both", "rotation") and has_rot

    neg_pole, pos_pole = DIVERGING

    def _lollipop(ax, vals_list, labels, unit, title):
        """Stem from zero to the mean (color = sign), IQR band + median tick."""
        means   = [np.nanmean(v)           for v in vals_list]
        medians = [np.nanmedian(v)         for v in vals_list]
        p25     = [np.nanpercentile(v, 25) for v in vals_list]
        p75     = [np.nanpercentile(v, 75) for v in vals_list]

        all_finite = np.concatenate([v[~np.isnan(v)] for v in vals_list])
        xlim   = max(float(np.percentile(np.abs(all_finite), 95)) * 1.7, 1.0)
        margin = xlim * 0.05

        ax.set_xlim(-xlim, xlim)
        ax.axvline(0, color=INK["axis"], lw=1.0, zorder=1)

        for i, (m, md, lo, hi) in enumerate(zip(means, medians, p25, p75)):
            col = pos_pole if m > 0 else neg_pole
            ax.barh(i, hi - lo, left=lo, height=0.45,
                    color=col, alpha=0.18, edgecolor="none", zorder=2)
            ax.plot([0, m], [i, i], color=col, lw=2.5,
                    zorder=3, solid_capstyle="round")
            ax.plot([md, md], [i - 0.22, i + 0.22], color=col, lw=1.4,
                    alpha=0.9, zorder=4)  # median tick
            ax.scatter([m], [i], s=80, color=col, zorder=5,
                       edgecolors="white", linewidths=2.0)  # 2px surface ring
            ha   = "left"  if m >= 0 else "right"
            xpos = m + (margin if m >= 0 else -margin)
            ax.text(xpos, i, f"{m:+.1f}", va="center", ha=ha,
                    fontsize=10, color=INK["primary"], fontweight="bold")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel(f"Signed mean error ({unit})", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.98, 0.02, "bar = IQR (25-75%), tick = median",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=INK["muted"], style="italic")

    # Layout: one lollipop per enabled component, side by side
    n_cols = sum([show_t, show_r])
    fig, axes = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.2), squeeze=False)
    col = 0

    if show_t:
        _lollipop(axes[0][col], [x, y, z], ["X", "Y", "Z (depth)"],
                  "mm", "Translation bias direction")
        col += 1

    if show_r:
        rot_v = rot[~np.isnan(rot)]
        _lollipop(axes[0][col], [rx, ry, rz], ["Rx", "Ry", "Rz"],
                  "deg", f"Rotation bias  (geodesic: mean {rot_v.mean():.1f}deg, "
                         f"median {np.median(rot_v):.1f}deg)")

    fig.suptitle(f"Error Decomposition - {dataset}  (n={len(x)} PnP-success frames)",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(out_dir / "2_error_decomposition.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _plot_signal_correlations(signals: dict, out_dir: Path, plt) -> None:
    from scipy.stats import spearmanr

    add = signals["add_m_mm"]
    valid_mask = ~signals["pnp_failed"]
    add_v = add[valid_mask]

    signal_names = ["conf_50", "reproj", "inlier", "anc_r",
                    "iou_conf", "iou_modal", "iou_amodal"]
    labels = {
        "conf_50":   "Confidence (thr 0.5)",
        "reproj":    "Reprojection error",
        "inlier":    "Inlier ratio",
        "anc_r":     "Anchor rot. error",
        "iou_conf":  "IoU conf vs render",
        "iou_modal": "IoU modal vs render",
        "iou_amodal":"IoU amodal vs render",
    }

    rhos, names = [], []
    for name in signal_names:
        arr = signals.get(name)
        if arr is None:
            continue
        arr_v = arr[valid_mask]
        valid = ~np.isnan(arr_v) & ~np.isnan(add_v)
        if valid.sum() < 10:
            continue
        rho, _ = spearmanr(arr_v[valid], add_v[valid])
        rhos.append(rho)
        names.append(labels.get(name, name))

    if not rhos:
        return

    order = np.argsort(np.abs(rhos))[::-1]
    rhos  = [rhos[i]  for i in order]
    names = [names[i] for i in order]
    neg_pole, pos_pole = DIVERGING
    colors = [pos_pole if r > 0 else neg_pole for r in rhos]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rhos) + 1.5))
    bars = ax.barh(names, rhos, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color=INK["axis"], lw=1.0)
    ax.set_xlabel("Spearman rho  (vs ADD error)")
    ax.set_title("Signal Correlations with ADD Error\n"
                 "(blue = higher signal -> lower error = good confidence proxy)")
    for bar, rho in zip(bars, rhos):
        x = rho + (0.01 if rho >= 0 else -0.01)
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{rho:+.3f}", va="center", color=INK["primary"],
                ha="left" if rho >= 0 else "right", fontsize=9)
    ax.set_xlim(-1.1, 1.1)
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(out_dir / "3_signal_correlations.png", dpi=150)
        plt.close(fig)


def _plot_auroc_bars(signals: dict, thr: float, out_dir: Path, plt) -> None:
    add    = signals["add_m_mm"]
    labels = (add < thr).astype(int)

    signal_names = ["conf_50", "conf_count_50", "reproj", "inlier", "anc_r",
                    "iou_conf", "iou_modal", "iou_amodal", "naive_combined"]
    display = {
        "conf_50":        "Confidence mean (thr 0.5)",
        "conf_count_50":  "Pixel count above conf=0.5",
        "reproj":         "Reprojection error",
        "inlier":         "Inlier ratio",
        "anc_r":          "Anchor rot. error",
        "iou_conf":       "IoU conf vs render",
        "iou_modal":      "IoU modal vs render",
        "iou_amodal":     "IoU amodal vs render",
        "naive_combined": "Naive combined score",
    }

    # Compute naive score and add to signals
    sig_aug = dict(signals)
    sig_aug["naive_combined"] = naive_score(signals)

    aurocs, names = [], []
    for name in signal_names:
        arr = sig_aug.get(name)
        if arr is None:
            continue
        invert = name in ("reproj", "anc_r")
        scores = -arr if invert else arr
        valid  = ~np.isnan(scores)
        if valid.sum() < 10 or labels[valid].sum() == 0 or (1 - labels[valid]).sum() == 0:
            continue
        auroc = compute_auroc(scores[valid], labels[valid])
        aurocs.append(auroc)
        names.append(display.get(name, name))

    if not aurocs:
        return

    order  = np.argsort(aurocs)
    aurocs = [aurocs[i] for i in order]
    names  = [names[i]  for i in order]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(aurocs) + 1.5))
    bars = ax.barh(names, aurocs, color=BLUE, edgecolor="none", height=0.6)
    ax.axvline(0.5, color=INK["muted"], lw=1.2, linestyle="--", label="Random (0.5)")
    ax.axvline(0.75, color=INK["muted"], lw=1.0, linestyle=":", label="Useful gate (~0.75)")
    ax.set_xlabel(f"AUROC  (label: ADD < {thr:.0f} mm)")
    ax.set_title("Signal AUROC - Predicting Good vs Bad Predictions")
    ax.set_xlim(0.4, 1.02)
    for bar, a in zip(bars, aurocs):
        ax.text(a + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{a:.3f}", va="center", ha="left", fontsize=9, color=INK["primary"])
    ax.legend()
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(out_dir / "4_auroc_bars.png", dpi=150)
        plt.close(fig)


def _plot_pareto_frontier(
    signals: dict, thr: float, out_dir: Path, plt, total_n: int
) -> None:
    add           = signals["add_m_mm"]
    score         = naive_score(signals)
    valid_naive   = ~np.isnan(score)
    iou_score     = signals["iou_modal"]
    valid_iou     = ~np.isnan(iou_score)

    if valid_naive.sum() < 20 and valid_iou.sum() < 20:
        return

    auc_unfiltered = add_auc(add, 100.0)
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(auc_unfiltered, color=INK["secondary"], lw=1.4, linestyle="--",
               label=f"Unfiltered (all frames)  AUC={auc_unfiltered:.3f}")

    # naive_score curve - covers all frames with valid score (includes PnP failures)
    if valid_naive.sum() >= 20:
        score_v = score[valid_naive]
        add_v   = add[valid_naive]
        cov, auc_curve = pareto_curve(score_v, add_v, thr)
        ax.plot(cov * 100, auc_curve, color=BLUE, lw=2.0,
                label=f"naive_score gate (all frames, N={valid_naive.sum()})")
        # Operating points ride the curve in its own color; the target labels them.
        for c_target in [0.9, 0.8, 0.7]:
            op = operating_point(score_v, add_v, c_target, thr, total_n=total_n)
            xop, yop = op["actual_coverage"] * 100, op["auc_filtered"]
            ax.scatter([xop], [yop], s=80, color=BLUE, zorder=5,
                       edgecolors="white", linewidths=2.0)  # 2px surface ring
            ax.annotate(f"{c_target:.0%}", (xop, yop), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color=INK["secondary"])

    # iou_modal curve - PnP-success frames only; coverage expressed over all frames
    if valid_iou.sum() >= 20:
        n_pnp_succ  = int(valid_iou.sum())
        iou_score_v = iou_score[valid_iou]
        add_iou_v   = add[valid_iou]
        auc_pnp_base = add_auc(add_iou_v, 100.0)
        cov_iou, auc_iou_curve = pareto_curve(iou_score_v, add_iou_v, thr)
        # Rescale coverage to fraction of ALL frames so x-axes are comparable
        cov_iou_global = cov_iou * n_pnp_succ / total_n
        ax.plot(cov_iou_global * 100, auc_iou_curve, color=ORANGE, lw=2.0,
                label=f"iou_modal gate (PnP-success only, N={n_pnp_succ})")
        ax.axhline(auc_pnp_base, color=INK["muted"], lw=1.2, linestyle=":",
                   label=f"PnP-success baseline  AUC={auc_pnp_base:.3f}")

    ax.set_xlabel("Coverage (% of all frames)")
    ax.set_ylabel(f"ADD AUC @ {thr:.0f} mm")
    ax.set_title("Pareto Frontier: Coverage vs Accuracy\n"
                 "(IoU curve = PnP-success frames only; x-axis normalised to all frames)")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xlim(0, 101)
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(out_dir / "5_pareto_frontier.png", dpi=150)
        plt.close(fig)


def _plot_iou_distributions(signals: dict, thr: float, out_dir: Path, plt) -> None:
    add        = signals["add_m_mm"]
    pnp_failed = signals["pnp_failed"]

    # Ordered good -> bad -> fail: severity color + its own linestyle, so the
    # three groups stay distinct in grayscale print and under colorblindness.
    good = (~pnp_failed) & (add < thr)
    bad  = (~pnp_failed) & (add >= thr) & (add < 999)
    groups = [
        (f"ADD < {thr:.0f}mm  (good)", good,       SEVERITY["good"], SEVERITY_LS["good"]),
        (f"ADD >= {thr:.0f}mm  (bad)", bad,        SEVERITY["bad"],  SEVERITY_LS["bad"]),
        ("PnP failure",                pnp_failed, SEVERITY["fail"], SEVERITY_LS["fail"]),
    ]

    iou_signals = {
        "iou_modal":  "IoU modal vs render",
        "iou_amodal": "IoU amodal vs render",
        "iou_conf":   "IoU conf vs render (thr 0.5)",
    }
    available = {k: v for k, v in iou_signals.items()
                 if not np.all(np.isnan(signals.get(k, np.array([np.nan]))))}
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, (sig_key, sig_label) in zip(axes, available.items()):
        arr      = signals[sig_key]
        has_data = False
        for label, mask, color, ls in groups:
            vals = arr[mask & ~np.isnan(arr)]
            if len(vals) < 3:
                continue
            vals_s = np.sort(vals)
            cdf    = np.arange(1, len(vals_s) + 1) / len(vals_s)
            ax.plot(vals_s, cdf, color=color, lw=2.0, linestyle=ls,
                    label=f"{label}  (n={len(vals)})")
            ax.axvline(np.median(vals_s), color=color, lw=1.0, alpha=0.6)
            has_data = True

        if has_data:
            ax.set_xlabel("IoU")
            ax.set_ylabel("Cumulative fraction of frames")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.05)
            ax.axhline(0.5, color=INK["muted"], lw=0.8, alpha=0.6)
            ax.set_title(sig_label, fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)

    fig.suptitle(f"IoU Signal CDFs by Frame Group - {thr:.0f}mm threshold\n"
                 "(thin verticals = group medians; higher IoU = better pose consistency)",
                 fontweight="bold")
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(out_dir / "6_iou_distributions.png", dpi=150)
        plt.close(fig)


def _plot_level1_distributions(signals: dict, out_dir: Path, plt) -> None:
    pnp_failed = signals["pnp_failed"]
    # The published PnP floor (256 valid correspondences = 16*16). A separate
    # inlier-ratio gate (0.3) also applies, so a point right of this line can
    # still fail; the line is the correspondence-count component only.
    MIN_CORR   = 256

    plot_signals = [
        ("conf_count_50", "Pixel count above conf=0.5\n(direct PnP failure predictor)"),
        ("conf_50",       "Mean confidence of pixels above 0.5"),
        ("anc_r",         "Anchor rotation error (°)"),
    ]
    available = [
        (k, lbl) for k, lbl in plot_signals
        if signals.get(k) is not None and not np.all(np.isnan(signals[k]))
    ]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 4.5))
    if len(available) == 1:
        axes = [axes]

    for ax, (sig_key, sig_label) in zip(axes, available):
        arr       = signals[sig_key]
        succ_vals = arr[~pnp_failed & ~np.isnan(arr)]
        fail_vals = arr[ pnp_failed & ~np.isnan(arr)]

        if len(succ_vals) < 3 or len(fail_vals) < 3:
            ax.set_visible(False)
            continue

        lo   = min(succ_vals.min(), fail_vals.min())
        hi   = max(succ_vals.max(), fail_vals.max())
        bins = np.linspace(lo, hi, 45)

        succ_c, fail_c = DIVERGING  # blue = success (good), red = failure (bad)
        ax.hist(succ_vals, bins=bins, density=True, alpha=0.55,
                color=succ_c, label=f"PnP success  (n={len(succ_vals)})", edgecolor="none")
        ax.hist(fail_vals, bins=bins, density=True, alpha=0.55,
                color=fail_c, label=f"PnP failure  (n={len(fail_vals)})", edgecolor="none")

        # Median markers
        ax.axvline(np.median(succ_vals), color=succ_c, lw=2.0, linestyle="--",
                   label=f"Success median  {np.median(succ_vals):.1f}")
        ax.axvline(np.median(fail_vals), color=fail_c, lw=2.0, linestyle="--",
                   label=f"Failure median  {np.median(fail_vals):.1f}")

        # For conf_count_50: mark the NeMO min_correspondences threshold
        if sig_key == "conf_count_50":
            ax.axvline(MIN_CORR, color=INK["primary"], lw=1.6, linestyle="-.",
                       label=f"min_correspondences = {MIN_CORR} (+ 0.3 inlier gate)")

        ax.set_xlabel(sig_key.replace("_", " "))
        ax.set_ylabel("Density")
        ax.set_title(sig_label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8.5)

    fig.suptitle("Level 1 Gate - Pre-PnP Signal Distributions\n"
                 "(available for ALL frames including PnP failures)",
                 fontweight="bold")
    fig.tight_layout()
    if out_dir is not None:
        fig.savefig(out_dir / "7_level1_distributions.png", dpi=150)
        plt.close(fig)


def _save_plots(
    signals: dict,
    queries: list,
    dataset: str,
    nemo: bool,
    thr: float,
    out_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[plots] matplotlib not available, skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    _plot_add_histogram(signals, dataset, thr, out_dir, plt)
    _plot_error_decomposition(queries, dataset, out_dir, plt)

    if nemo:
        _plot_signal_correlations(signals, out_dir, plt)
        _plot_auroc_bars(signals, thr, out_dir, plt)
        _plot_pareto_frontier(signals, thr, out_dir, plt, total_n=len(signals["add_m_mm"]))
        _plot_iou_distributions(signals, thr, out_dir, plt)
        _plot_level1_distributions(signals, out_dir, plt)

    saved = sorted(out_dir.glob("*.png"))
    print(f"\n[plots] {len(saved)} figures saved to {out_dir}/")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
