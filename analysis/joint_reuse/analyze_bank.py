"""Representation-bank analysis: stock NeMO vs config-keyed reuse bank.

Reads the stock and bank eval results.json - the expensive artifacts (one 32k GPU
run each) - and derives everything from their per-frame fields, so re-plotting never
needs a re-run:
  add_m_mm                         -> accuracy
  diagnostics.reuse_bank {hit,      -> cumulative encodes + mechanism
    d_surf_mm, entry_idx}
  timings                          -> realized encode-time saved

Emits: one figure (cumulative unique encodes vs frames - the bank saturating), a
representation-reuse table (stock / exact-dedup / bank), an accuracy table (ADD-AUC
+ mean ADD, stock vs bank), and the accuracy falsifiers (aggregate delta, hit-frame
mechanism vs d_surf, worst-cluster failure rate).

    python analyze_bank.py <stock_run> <bank_run> --plots figs/bank
Runs classify by the presence of reuse_bank diagnostics; frames pair by frame_id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_utils import (  # noqa: E402
    add_auc, setup_matplotlib, print_header, print_table, BLUE, INK,
)

FAIL_MM = 100.0
NOISE_MEAN_MM = 0.2      # run-to-run ADD noise floor (from the tolerance=0 bank-inert tripwire)
NOISE_AUC = 0.0007       # run-to-run AUC noise floor


def load_run(path):
    p = Path(path)
    rj = p / "results.json" if p.is_dir() else p
    with open(rj) as f:
        data = json.load(f)
    return data["summary"], data["queries"]


def _add_mm(q):
    if q.get("add_m_mm") is not None:
        return float(q["add_m_mm"])
    if q.get("add_m") is not None:
        return float(q["add_m"]) * 1000.0
    return np.nan


def _fid(q):
    return str(q.get("frame_id") or q.get("image_path"))


def _bank(q):
    return (q.get("diagnostics") or {}).get("reuse_bank") or {}


def fig_reuse(encodes, n_frames, out_dir):
    import matplotlib.pyplot as plt
    setup_matplotlib()
    final = int(encodes[-1])
    x = np.arange(1, len(encodes) + 1)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(x, encodes, "-", color=BLUE, lw=2.4)
    ax.axhline(final, color=INK["muted"], ls=(0, (4, 3)), lw=1.3)
    ax.annotate(f"{final} unique encodes", (n_frames * 0.6, final), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=9, color=INK["muted"])
    ax.set_xlim(0, n_frames)
    ax.set_ylim(0, max(final * 1.3, 5))
    ax.set_xlabel("frames processed")
    ax.set_ylabel("unique encodes  (bank size)")
    ax.grid(True, color=INK["grid"], lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out_dir / "representation_reuse.png"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {p} (+ .pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="stock and bank run dirs / results.json")
    ap.add_argument("--plots", type=Path, default=None)
    ap.add_argument("--round", type=int, default=4,
                    help="joint-angle rounding (rad) for the exact-dedup config count")
    args = ap.parse_args()

    stock = bank = None
    for summ, qs in (load_run(r) for r in args.runs):
        if any(_bank(q) for q in qs):
            bank = (summ, qs)
        else:
            stock = (summ, qs)
    if bank is None:
        sys.exit("no bank run found (queries need diagnostics.reuse_bank)")
    bsumm, bqs = bank
    n = len(bqs)

    # cumulative encodes: a miss encodes a new entry, a hit reuses one
    hit = np.array([bool(_bank(q).get("hit")) for q in bqs])
    encodes = np.cumsum(~hit)
    final = int(encodes[-1])
    joints = np.array([q["joints_rad"] for q in bqs], dtype=np.float64)
    distinct = len(np.unique(np.round(joints, args.round), axis=0))

    print_header(f"A2 representation bank - {bsumm.get('dataset')}  ({n} frames)")
    print_table(["policy", "encodes", "frac of frames"],
                [["stock (per frame)", f"{n}", "100.00%"],
                 ["exact-config dedup", f"{distinct}", f"{100 * distinct / n:.2f}%"],
                 ["reuse bank", f"{final}", f"{100 * final / n:.2f}%"]], col_width=20)
    print(f"amortisation: {n} / {final} = {n / final:.0f}x fewer encodes than stock; "
          f"exact dedup only {n / max(distinct, 1):.2f}x (workload is {100 * distinct / n:.0f}% distinct)")

    # accuracy, paired on common frames when a stock run is given
    badd = np.array([_add_mm(q) for q in bqs])
    rows = [["reuse bank", f"{add_auc(badd, FAIL_MM):.4f}", f"{np.nanmean(badd):.1f}"]]
    if stock:
        _, sqs = stock
        sidx = {_fid(q): q for q in sqs}
        common = [q for q in bqs if _fid(q) in sidx]
        sadd = np.array([_add_mm(sidx[_fid(q)]) for q in common])
        cadd = np.array([_add_mm(q) for q in common])
        rows = [["stock", f"{add_auc(sadd, FAIL_MM):.4f}", f"{np.nanmean(sadd):.1f}"],
                ["reuse bank", f"{add_auc(cadd, FAIL_MM):.4f}", f"{np.nanmean(cadd):.1f}"]]
    print("")
    print_table(["run", "ADD-AUC@100mm", "mean ADD (mm)"], rows, col_width=16)

    if stock:
        d_auc = add_auc(cadd, FAIL_MM) - add_auc(sadd, FAIL_MM)
        d_mean = float(np.nanmean(cadd) - np.nanmean(sadd))
        flag = "within run-to-run noise" if (abs(d_mean) < NOISE_MEAN_MM and abs(d_auc) < NOISE_AUC) \
            else "ABOVE noise floor - inspect"
        print(f"\naggregate delta (bank - stock): AUC {d_auc:+.4f}, mean ADD {d_mean:+.2f} mm  ->  {flag}")
        # mechanism: on HIT frames, does the extra error scale with reuse distance?
        # (the perturbation sweep found ~0 excess)
        hmask = np.array([bool(_bank(q).get("hit")) for q in common])
        dsurf = np.array([float(_bank(q).get("d_surf_mm", np.nan)) for q in common])
        dd = cadd - sadd
        m = hmask & np.isfinite(dsurf) & np.isfinite(dd)
        if m.sum() >= 5:
            r = float(np.corrcoef(dsurf[m], dd[m])[0, 1])
            print(f"hit frames ({int(m.sum())}): mean |dADD| {np.nanmean(np.abs(dd[m])):.2f} mm, "
                  f"corr(dADD, d_surf) {r:+.2f}")
        # cluster: worst per-entry failure rate (one bad entry must not degrade its neighbourhood)
        ent = np.array([int(_bank(q).get("entry_idx", -1)) for q in common])
        worst, worst_e = 0.0, -1
        for e in np.unique(ent[ent >= 0]):
            fr = float((cadd[ent == e] > FAIL_MM).mean())
            if (ent == e).sum() >= 10 and fr > worst:
                worst, worst_e = fr, int(e)
        if worst_e >= 0:
            print(f"worst bank entry #{worst_e}: {100 * worst:.1f}% failure rate on its cluster")

    if args.plots:
        args.plots.mkdir(parents=True, exist_ok=True)
        fig_reuse(encodes, n, args.plots)


if __name__ == "__main__":
    main()
