"""Estimate template-bank coverage with greedy farthest-point k-center.

Distances use the perturbation analysis task-space surface metric. The script
reports coverage radius, hit rate, bank size by tolerance and effective
dimension for saved sweep configurations or an explicit ``.npy`` workload.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_utils import (  # noqa: E402
    print_header, print_table, setup_matplotlib, dataset_color, INK,
)
from analyze_perturbation import (  # noqa: E402
    MOVING_EPS_M, RobotGeometry, discover_runs,
)


def build_clouds(geom: RobotGeometry, configs: np.ndarray):
    """Pose every config's probe cloud once. Returns (clouds (N,M,3), W (M,n_links)).

    W is the point->link mean-reduction matrix: clouds distances @ W gives the
    per-link mean displacement in one matmul, so a config-to-all d_surf is
    vectorised over the whole bank.
    """
    R0, t0 = geom.fk(configs[0])
    _, link_ids, _ = geom.probe(R0, t0)
    n_links = geom.n_links
    counts = np.bincount(link_ids, minlength=n_links)
    W = np.zeros((len(link_ids), n_links))
    W[np.arange(len(link_ids)), link_ids] = 1.0 / np.maximum(counts[link_ids], 1)

    clouds = np.empty((len(configs), len(link_ids), 3))
    for i, q in enumerate(configs):
        R, t = geom.fk(q)
        clouds[i] = geom.probe(R, t)[0]
    return clouds, W


def _dsurf(clouds: np.ndarray, W: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """d_surf (mm) from the ref cloud to every cloud: mean over MOVING links."""
    disp = np.linalg.norm(clouds - ref[None], axis=2)         # (N, M)
    link_mean = disp @ W                                      # (N, n_links)
    moving = link_mean > MOVING_EPS_M
    den = moving.sum(1)
    num = (link_mean * moving).sum(1)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0) * 1000.0


def dsurf_one_to_all(clouds: np.ndarray, W: np.ndarray, i: int) -> np.ndarray:
    """d_surf (mm) from config i to every config."""
    return _dsurf(clouds, W, clouds[i])


# Greedy k-center (Gonzalez farthest-point) - coverage curve + hit-rates

def greedy_kcenter(clouds: np.ndarray, W: np.ndarray, max_k: int, seed: int = 0):
    """Farthest-point insertion. Returns (order, radii, Dmat).

    order[:k] = the first k bank entries; radii[k-1] = covering radius with k
    centers (max over configs of distance to nearest center); Dmat (K,N) =
    distance from the k-th chosen center to all configs (for hit-rate curves).
    """
    N = len(clouds)
    start = int(np.random.default_rng(seed).integers(N))
    order = [start]
    D = [dsurf_one_to_all(clouds, W, start)]
    dmin = D[0].copy()
    radii = [float(dmin.max())]
    while len(order) < min(max_k, N):
        nxt = int(dmin.argmax())
        if dmin[nxt] <= 0:
            break
        order.append(nxt)
        d = dsurf_one_to_all(clouds, W, nxt)
        D.append(d)
        dmin = np.minimum(dmin, d)
        radii.append(float(dmin.max()))
    return order, np.array(radii), np.array(D)


# Online lazy bank (ours) - the deployable policy: no prior knowledge of the workload.

def online_bank(clouds: np.ndarray, W: np.ndarray, tol: float):
    """Walk configs in arrival order; encode when no bank entry is within tol, else reuse.

    Unlike greedy_kcenter (which needs the whole workload up front, so it is an oracle
    bound) this is what a deployment can actually run. Bank entries end up pairwise
    > tol apart, i.e. a tol-packing, so by the packing-covering inequality the bank is
    no larger than the optimal tol/2 cover - a bounded price for not knowing the
    workload. Returns (bank indices, cumulative encode count per frame).
    """
    bank = [0]
    cum = np.empty(len(clouds), dtype=int)
    cum[0] = 1
    for i in range(1, len(clouds)):
        if _dsurf(clouds[bank], W, clouds[i]).min() > tol:
            bank.append(i)
        cum[i] = len(bank)
    return np.array(bank), cum


def _first_k(radii: np.ndarray, r: float):
    hit = np.where(radii <= r)[0]
    return int(hit[0] + 1) if hit.size else None


def coverage_stats(radii: np.ndarray, Dmat: np.ndarray, tol: float, hit: float):
    """Rigorous k-center band + hit-rate bank size.

    Gonzalez gives radius_k ≤ 2·OPT_k, so the optimal bank for radius τ obeys
    k(2τ) ≤ k*(τ) ≤ k(τ), where k(r) = smallest greedy bank reaching radius ≤ r.
    Returns (k_lb=k(2τ), k_full=k(τ), k_hit for `hit` fraction within τ).
    """
    nearest = np.minimum.accumulate(Dmat, axis=0)          # (K, N): nearest of first k
    k_full = _first_k(radii, tol)                          # achievable, upper bound on k*
    k_lb = _first_k(radii, 2 * tol)                        # lower bound on k*
    hit_rate = (nearest <= tol).mean(axis=1)               # (K,)
    reach = np.where(hit_rate >= hit)[0]
    k_hit = int(reach[0] + 1) if reach.size else None
    return k_lb, k_full, k_hit


def effective_dim(radii: np.ndarray) -> float:
    """Intrinsic dimension from radius(k) ~ C k^(-1/d)  ->  d = -1/slope(log-log)."""
    k = np.arange(1, len(radii) + 1)
    m = (radii > 0) & (k >= 2)
    if m.sum() < 4:
        return float("nan")
    lo, hi = int(m.sum() * 0.1), int(m.sum() * 0.9) or m.sum()   # trim the ends
    x, y = np.log(k[m][lo:hi]), np.log(radii[m][lo:hi])
    if x.size < 3 or np.ptp(x) < 1e-6:
        return float("nan")
    slope = np.polyfit(x, y, 1)[0]
    return float(-1.0 / slope) if slope < -1e-6 else float("inf")


# Config sources

def configs_from_runs(roots, lenient):
    """{dataset: (robot_name, configs (N,n), median_frame_ms)} from δ=0 baselines."""
    grouped = discover_runs(roots, lenient=lenient)
    out = {}
    for ds, runs in grouped.items():
        base = runs.get(0.0) or runs[min(runs)]
        qs = base["queries"]
        configs = np.array([q["joints_rad"] for q in qs], dtype=np.float64)
        ms = [(q.get("diagnostics") or {}).get("frame_time_ms") for q in qs]
        ms = [m for m in ms if m is not None]
        out[ds] = (base["config"]["robot"]["name"], configs,
                   float(np.median(ms)) if ms else float("nan"))
    return out


# Report + figure

def report(ds, robot, configs, frame_ms, tols, hit, max_k, seed):
    N = len(configs)
    uniq = len(np.unique(np.round(configs, 4), axis=0))
    print_header(f"covering analysis - {ds}  (robot {robot}, {N} configs, "
                 f"{uniq}/{N} = {uniq/N:.0%} unique)")
    if N < 4:
        print("  too few configs to build a covering curve - skipping")
        return None

    geom = RobotGeometry(robot)
    clouds, W = build_clouds(geom, configs)
    order, radii, Dmat = greedy_kcenter(clouds, W, max_k, seed)
    d_eff = effective_dim(radii)

    print(f"coverage radius: {radii[0]:.0f} mm at k=1  ->  {radii[-1]:.1f} mm at "
          f"k={len(radii)} (of {N}).  effective dim d_eff ≈ {d_eff:.1f} "
          f"(-> N(τ) ~ (1/τ)^{d_eff:.1f}; low = manifold-like workload, bank pays off)")
    if frame_ms == frame_ms:  # not NaN
        print(f"median frame time {frame_ms:.0f} ms (from timings) - "
              f"one-time bank build ≈ k×{frame_ms:.0f} ms; net win once queries ≫ k")

    rows, k_at_loosest, online = [], None, {}
    loosest = max(tols)
    for tol in tols:
        k_lb, k_full, k_hit = coverage_stats(radii, Dmat, tol, hit)
        bank, cum = online_bank(clouds, W, tol)
        online[tol] = (bank, cum)
        if tol == loosest:
            k_at_loosest = k_full
        rows.append([
            f"{tol:g}",
            f"{k_full}" if k_full else f">{len(radii)}",
            f"[{k_lb or '?'}, {k_full or '?'}]",
            f"{k_hit}" if k_hit else f">{len(radii)}",
            f"{len(bank)}",
            f"{100*len(bank)/N:.0f}%",
        ])
    print_table(["τ = d_surf (mm)", "N(τ) 100% cover", "optimal band",
                 f"N for {hit:.0%} hit", "online bank", "online % of N"], rows, col_width=15)
    print("(N(τ) = greedy k-center bank keeping EVERY config within τ - needs the whole "
          "workload up front, so it is the ORACLE bound. optimal band [k(2τ), k(τ)] "
          "brackets the best possible bank. online bank = arrival-order lazy cache, the "
          "policy a deployment can actually run; the gap to N(τ) is the price of not "
          f"knowing the workload. Read τ = δ* from analyze_perturbation.py. {hit:.0%}-hit "
          "column: cheaper than 100% once you accept a small miss rate.)")
    prim = tols[0]
    bank_p = online[prim][0]
    print(f"online bank at τ={prim:g} mm: {len(bank_p)} encodes for {N} frames "
          f"-> {N / max(len(bank_p), 1):.1f}x amortisation" +
          (f", {(N - len(bank_p)) * frame_ms / 1000:.0f} s of encode time saved"
           if frame_ms == frame_ms else ""))
    if k_at_loosest and k_at_loosest > 0.5 * N:
        print(f"  WARNING: even at the loosest τ={loosest:g} mm the bank needs {k_at_loosest}/{N} "
              "configs: this workload does NOT cluster (random configs / too few samples) - "
              "bank ≈ workload, d_eff unreliable until N ≫ bank. Worst case; a trajectory "
              "workload is the contrast.")
    return {"ds": ds, "radii": radii, "Dmat": Dmat, "N": N, "d_eff": d_eff,
            "cum": online[prim][1], "online_k": len(bank_p), "tau": prim}


def fig_covering(results, tols, out_dir: Path):
    import matplotlib.pyplot as plt
    setup_matplotlib()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))
    for r in results:
        k = np.arange(1, len(r["radii"]) + 1)
        col = dataset_color(r["ds"])
        ax1.plot(k, r["radii"], "-", color=col, label=f"{r['ds']} (d_eff≈{r['d_eff']:.1f})")
        nearest = np.minimum.accumulate(r["Dmat"], axis=0)
        tol = tols[len(tols) // 2]
        hr = (nearest <= tol).mean(axis=1)
        ax2.plot(k, hr, "-", color=col, label=f"{r['ds']}")
        frames = np.arange(1, len(r["cum"]) + 1)
        ax3.plot(frames, r["cum"], "-", color=col,
                 label=f"{r['ds']} ({r['online_k']} encodes / {r['N']} frames)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("bank size k")
    ax1.set_ylabel("covering radius (mm d_surf)")
    ax1.set_title("coverage curve (slope -> effective dim)", fontsize=9)
    for tol in tols:
        ax1.axhline(tol, color=INK["muted"], ls=":", lw=0.8, alpha=0.6)
    ax1.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("bank size k")
    ax2.set_ylabel(f"hit-rate within τ={tols[len(tols)//2]:g} mm")
    ax2.set_title("workload coverage vs bank size", fontsize=9)
    ax2.axhline(0.95, color=INK["muted"], ls=":", lw=0.8)
    ax2.legend(fontsize=8)
    nmax = max(r["N"] for r in results)
    ax3.plot([1, nmax], [1, nmax], ls="--", lw=0.8, color=INK["muted"],
             label="no reuse (encode every frame)")
    ax3.set_xscale("log")
    ax3.set_xlabel("frames processed (arrival order)")
    ax3.set_ylabel("cumulative encodes")
    ax3.set_title(f"online bank at τ={results[0]['tau']:g} mm - flattening = bank saturating",
                  fontsize=9)
    ax3.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "covering_curves.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Template-bank covering analysis (offline).")
    ap.add_argument("roots", nargs="*", help="sweep roots / run dirs (δ=0 baseline configs)")
    ap.add_argument("--configs", type=Path, help="external workload: .npy (N, n_joints)")
    ap.add_argument("--robot", help="robot name (required with --configs)")
    ap.add_argument("--tolerances", default="5,10,20,40",
                    help="d_surf mm tolerances to report; δ* comes from the perturbation analysis")
    ap.add_argument("--hit", type=float, default=0.95, help="target hit-rate for the cheap-bank column")
    ap.add_argument("--max-k", type=int, default=500, help="cap on greedy bank size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", type=Path, default=None)
    ap.add_argument("--lenient", action="store_true", help="keep-first on duplicate runs")
    args = ap.parse_args()

    tols = [float(t) for t in args.tolerances.split(",")]
    if args.configs is not None:
        if not args.robot:
            sys.exit("--configs requires --robot")
        configs = np.load(args.configs)
        sources = {args.configs.stem: (args.robot, np.asarray(configs, float), float("nan"))}
    elif args.roots:
        sources = configs_from_runs(args.roots, args.lenient)
    else:
        sys.exit("give sweep roots, or --configs <npy> --robot <name>")

    results = []
    for ds, (robot, configs, frame_ms) in sorted(sources.items()):
        r = report(ds, robot, configs, frame_ms, tols, args.hit, args.max_k, args.seed)
        if r is not None:
            results.append(r)

    if args.plots and results:
        args.plots.mkdir(parents=True, exist_ok=True)
        fig_covering(results, tols, args.plots)
        print(f"\nfigure -> {args.plots}/covering_curves.png")


if __name__ == "__main__":
    main()
