"""Joint-noise robustness analysis (panda-orb).

Feed the frozen pipeline a perturbed config q + dq while the image shows the arm
at the true q, and measure how the ADD error grows. The added error is compared
to a rigid-fit geometric floor: the error a perfect rigid re-fit of the displaced
arm would still incur. The measured curve tracking that floor means the growth is
the geometric consequence of the wrong joints, not model breakdown; PnP itself
does not fail across the sweep.

delta* is the reuse tolerance handed to covering_analysis.py: the perturbation
at which the median added error first reaches BUDGET_FRAC of the baseline median
ADD (a chosen error budget, not a discovered knee - the model degrades smoothly).

Two entry points, one file:
  compute:  python analyze_perturbation.py <sweep_roots...> --plots figs/perturbation --save-intermediate
  re-plot:  python analyze_perturbation.py --from-npz figs/perturbation/perturbation_panda_orb.npz \\
                --plots figs/perturbation
The re-plot path needs only numpy + matplotlib (no robot geometry deps).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_utils import (  # noqa: E402
    print_header, print_table, setup_matplotlib, BLUE, ORANGE, INK,
)

FAILURE_MM = 100.0        # ADD above which a frame counts as failed
BUDGET_FRAC = 0.10        # delta* = added error reaches this fraction of baseline ADD
MOVING_EPS_M = 1e-9       # a link counts as "moving" above this displacement


# Geometry (only needed by the compute path)

def _import_registry():
    try:
        from robot_renderer import registry
    except ImportError:
        rr_src = Path(__file__).resolve().parents[2] / "external" / "robot-renderer" / "src"
        sys.path.insert(0, str(rr_src))
        from robot_renderer import registry
    return registry


class RobotGeometry:
    """FK adapter + per-link local bbox-corner probe points (meshes loaded once)."""

    def __init__(self, robot_name: str):
        import trimesh
        registry = _import_registry()
        entry = registry.get_robot_entry(robot_name)
        self.name = robot_name
        self.kin = entry["adapter_cls"](entry["urdf"])
        self.corners_local: list[np.ndarray] = []
        for files in entry["mesh_files"]:
            lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
            for f in files:
                m = trimesh.load(f, force="mesh", process=False)
                lo, hi = np.minimum(lo, m.bounds[0]), np.maximum(hi, m.bounds[1])
            self.corners_local.append(np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
        self.n_links = len(self.corners_local)

    def fk(self, q):
        return self.kin.get_joint_R_t(np.asarray(q, dtype=np.float64))

    def probe(self, R, t):
        pts, ids, w = [], [], []
        for li, corners in enumerate(self.corners_local):
            pts.append(corners @ R[li].T + t[li])
            ids.append(np.full(len(corners), li))
            w.append(np.full(len(corners), 1.0 / len(corners)))
        return np.concatenate(pts), np.concatenate(ids), np.concatenate(w)


def kabsch(P, Q, w):
    """Weighted rigid fit (R, t) minimizing sum w ||R p + t - q||^2."""
    w = w / w.sum()
    p_bar, q_bar = w @ P, w @ Q
    H = (P - p_bar).T @ ((Q - q_bar) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, q_bar - R @ p_bar


def _frame_key(q):
    return str(q.get("frame_id") or q["image_path"])


# Compute path: eval JSONs -> per-frame arrays

def discover_runs(roots, lenient=False):
    """Load one result per magnitude from the sweep roots (missing joint_noise = level 0)."""
    import yaml
    hits: dict[float, list] = defaultdict(list)
    for root in roots:
        for rj in sorted(Path(root).rglob("results.json")):
            cfg_path = rj.with_name(rj.name.replace(".json", ".config.yaml"))
            if not cfg_path.exists():
                continue
            cfg = yaml.safe_load(cfg_path.read_text())
            level = float((cfg.get("joint_noise") or {}).get("magnitude_deg", 0.0))
            with open(rj) as f:
                data = json.load(f)
            hits[level].append({"config": cfg, "queries": data["queries"]})
    grouped = {}
    for level in sorted(hits):
        if len(hits[level]) > 1 and not lenient:
            raise SystemExit(f"[discover] multiple runs at {level}° - pass --lenient to keep the first")
        grouped[level] = hits[level][0]
    return grouped


def compute_dataset(runs, dataset="panda_orb"):
    """Per-frame x per-level arrays: add, pnp_failed, d_surf, and the geometric floor
    (dadd_pred - the baseline-composed rigid null model). Frames paired across levels."""
    if 0.0 not in runs:
        raise ValueError("no magnitude-0 baseline run - paired analysis impossible.")
    levels = sorted(runs)
    robot = runs[0.0]["config"]["robot"]["name"]
    geom = RobotGeometry(robot)
    by_level = [{_frame_key(q): q for q in runs[lv]["queries"]} for lv in levels]
    keys = [k for k in by_level[0] if all(k in idx for idx in by_level)]
    F, L = len(keys), len(levels)

    D = {"dataset": dataset, "robot": robot, "levels": np.array(levels), "keys": keys,
         "add": np.full((F, L), np.nan), "pnp_failed": np.zeros((F, L), bool),
         "d_surf": np.full((F, L), np.nan), "dadd_pred": np.full((F, L), np.nan)}

    for fi, key in enumerate(keys):
        q0 = by_level[0][key]
        q_true = np.asarray(q0["joints_rad"], dtype=np.float64)
        R0, t0 = geom.fk(q_true)
        P0, link_ids, w = geom.probe(R0, t0)
        gt = q0.get("gt_pose")
        gt = None if gt is None else np.asarray(gt, dtype=np.float64)
        est0 = q0.get("est_pose")
        est0 = None if est0 is None else np.asarray(est0, dtype=np.float64)
        kp = q0.get("keypoints_base_m")
        pts = np.asarray(kp, dtype=np.float64) if kp is not None else t0  # native kps, else link origins

        for li, _lv in enumerate(levels):
            rec = by_level[li][key]
            D["add"][fi, li] = rec.get("add_m_mm", np.nan)
            D["pnp_failed"][fi, li] = bool(rec.get("pnp_failed", False))
            jn = (rec.get("diagnostics") or {}).get("joint_noise")
            dq = np.zeros_like(q_true) if jn is None else np.asarray(jn["dq_rad"], dtype=np.float64)
            if not np.any(dq):
                Pp = P0
            else:
                Rp, tp = geom.fk(q_true + dq)
                Pp = geom.probe(Rp, tp)[0]
            disp = np.linalg.norm(Pp - P0, axis=1)
            link_mean = np.array([disp[link_ids == li_].mean() for li_ in range(geom.n_links)])
            moving = link_mean > MOVING_EPS_M
            D["d_surf"][fi, li] = (link_mean[moving].mean() if moving.any() else 0.0) * 1000.0
            # rigid-fit geometric floor: PnP absorbs the rigid part, so T_hat(dq) ~= T_hat(0).G^-1;
            # the residual error of that best rigid re-fit is the floor.
            if gt is not None and est0 is not None:
                Rg, tg = kabsch(P0, Pp, w)
                pred = ((pts - tg) @ Rg) @ est0[:3, :3].T + est0[:3, 3]
                base = pts @ est0[:3, :3].T + est0[:3, 3]
                truth = pts @ gt[:3, :3].T + gt[:3, 3]
                D["dadd_pred"][fi, li] = 1000.0 * (
                    np.mean(np.linalg.norm(pred - truth, axis=1))
                    - np.mean(np.linalg.norm(base - truth, axis=1)))

    lv0 = levels.index(0.0)
    D["fail"] = D["pnp_failed"] | (D["add"] > FAILURE_MM)
    D["ok0"] = ~D["fail"][:, lv0]
    return D


# Analysis (shared by both paths)

def per_level(D):
    """Per perturbation level (>0): median/IQR added error, floor, mean d_surf, failure rates.
    Paired ADD stats use frames that succeed at both baseline and that level."""
    levels = D["levels"]
    lv0 = int(np.where(levels == 0.0)[0][0])
    ok0 = D["ok0"]
    rows = {"deg": [], "d_surf": [], "med": [], "q25": [], "q75": [],
            "floor": [], "pnp": [], "fail": []}
    for li, lv in enumerate(levels):
        if lv <= 0:
            continue
        good = ok0 & ~D["fail"][:, li]
        dadd = (D["add"][:, li] - D["add"][:, lv0])[good]
        dadd = dadd[np.isfinite(dadd)]
        fv = D["dadd_pred"][:, li][good]
        fv = fv[np.isfinite(fv)]
        rows["deg"].append(float(lv))
        rows["d_surf"].append(float(np.nanmedian(D["d_surf"][ok0, li])))
        rows["med"].append(float(np.median(dadd)) if dadd.size else np.nan)
        rows["q25"].append(float(np.percentile(dadd, 25)) if dadd.size else np.nan)
        rows["q75"].append(float(np.percentile(dadd, 75)) if dadd.size else np.nan)
        rows["floor"].append(float(np.median(fv)) if fv.size else np.nan)
        rows["pnp"].append(float(D["pnp_failed"][ok0, li].mean()))
        rows["fail"].append(float(D["fail"][ok0, li].mean()))
    return {k: np.array(v) for k, v in rows.items()}


def delta_star(D, pl):
    """Reuse tolerance: interpolate the perturbation (deg and d_surf) at which the median
    added error first reaches BUDGET_FRAC x baseline median ADD. NaN if never reached."""
    lv0 = int(np.where(D["levels"] == 0.0)[0][0])
    base_med = float(np.nanmedian(D["add"][D["ok0"], lv0]))
    budget = BUDGET_FRAC * base_med
    med = pl["med"]
    for i in range(len(med) - 1):
        if med[i] <= budget < med[i + 1]:
            f = (budget - med[i]) / (med[i + 1] - med[i])
            return {"deg": float(pl["deg"][i] + f * (pl["deg"][i + 1] - pl["deg"][i])),
                    "d_surf": float(pl["d_surf"][i] + f * (pl["d_surf"][i + 1] - pl["d_surf"][i])),
                    "budget_mm": budget, "base_mm": base_med}
    return {"deg": np.nan, "d_surf": np.nan, "budget_mm": budget, "base_mm": base_med}


def report(D, pl, ds):
    lv0 = int(np.where(D["levels"] == 0.0)[0][0])
    base = float(np.nanmedian(D["add"][D["ok0"], lv0]))
    print_header(f"A1 joint-noise robustness - {D['dataset']}  ({D['ok0'].sum()} paired frames)")
    print(f"baseline median ADD {base:.1f} mm  |  PnP failure across sweep: "
          f"max {100 * pl['pnp'].max():.1f}%  |  failure (ADD>{FAILURE_MM:.0f}mm): max {100 * pl['fail'].max():.1f}%")
    rows = [[f"{d:g}", f"{s:.1f}", f"{m:+.1f}", f"[{a:+.1f}, {b:+.1f}]", f"{fl:+.1f}",
             f"{100 * p:.1f}%", f"{100 * fr:.1f}%"]
            for d, s, m, a, b, fl, p, fr in zip(
                pl["deg"], pl["d_surf"], pl["med"], pl["q25"], pl["q75"], pl["floor"], pl["pnp"], pl["fail"])]
    print_table(["perturb (deg)", "d_surf (mm)", "median dADD", "IQR", "floor", "PnP-fail", "fail"],
                rows, col_width=14)
    print(f"\ndelta* (added error = {int(BUDGET_FRAC * 100)}% of baseline): "
          f"{ds['deg']:.2f}deg  ~=  {ds['d_surf']:.1f} mm d_surf  ->  A2 reuse radius")


def fig(D, pl, ds, out_dir):
    import matplotlib.pyplot as plt
    setup_matplotlib()
    x = np.arange(len(pl["deg"]))
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.axhline(0, color=INK["grid"], lw=0.9)
    ax.fill_between(x, pl["q25"], pl["q75"], color=BLUE, alpha=0.14, lw=0, label="measured, IQR")
    ax.plot(x, pl["floor"], "s--", color=ORANGE, lw=1.8, ms=5, label="geometry-only floor")
    ax.plot(x, pl["med"], "o-", color=BLUE, lw=2.3, ms=6, label="measured")
    if np.isfinite(ds["deg"]):
        # interpolate delta* onto the categorical axis
        deg = pl["deg"]
        xpos = np.interp(ds["deg"], deg, x)
        ax.axvline(xpos, color=INK["muted"], ls=(0, (4, 3)), lw=1.5,
                   label=f"measured error exceeds {int(BUDGET_FRAC * 100)}% of baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in pl["deg"]])
    ax.set_xlabel("joint perturbation  (deg)")
    ax.set_ylabel("added pose error  (mm)")
    ax.legend(loc="upper left", fontsize=8.8, framealpha=0.9)
    ax.grid(True, axis="y", color=INK["grid"], lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = out_dir / f"perturbation_{D['dataset']}.png"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {p} (+ .pdf)")


# IO for the re-plot path

_ARR_KEYS = ("levels", "add", "pnp_failed", "d_surf", "dadd_pred", "fail", "ok0")


def save_npz(D, path):
    np.savez_compressed(path, dataset=D["dataset"], robot=D["robot"],
                        **{k: D[k] for k in _ARR_KEYS})


def load_npz(path, dataset=None):
    z = np.load(path, allow_pickle=True)
    D = {k: z[k] for k in _ARR_KEYS if k in z}
    D["dataset"] = str(z["dataset"]) if "dataset" in z else (dataset or "panda_orb")
    D["robot"] = str(z["robot"]) if "robot" in z else "panda"
    if "ok0" not in D:  # older npz: derive it
        lv0 = int(np.where(D["levels"] == 0.0)[0][0])
        D.setdefault("fail", D["pnp_failed"] | (D["add"] > FAILURE_MM))
        D["ok0"] = ~D["fail"][:, lv0]
    return D


def main():
    ap = argparse.ArgumentParser(description="A1 joint-noise robustness (panda-orb).")
    ap.add_argument("roots", nargs="*", help="sweep roots / run dirs (compute path)")
    ap.add_argument("--from-npz", type=Path, help="re-plot from a saved intermediate (no geometry deps)")
    ap.add_argument("--dataset", default="panda_orb")
    ap.add_argument("--plots", type=Path, default=None, help="figure output dir")
    ap.add_argument("--save-intermediate", action="store_true", help="dump per-frame arrays as npz")
    ap.add_argument("--lenient", action="store_true")
    args = ap.parse_args()

    if args.from_npz:
        D = load_npz(args.from_npz, args.dataset)
    elif args.roots:
        D = compute_dataset(discover_runs(args.roots, args.lenient), args.dataset)
    else:
        sys.exit("give sweep roots (compute) or --from-npz <file> (re-plot)")

    pl = per_level(D)
    ds = delta_star(D, pl)
    report(D, pl, ds)
    if args.plots:
        args.plots.mkdir(parents=True, exist_ok=True)
        fig(D, pl, ds, args.plots)
        if args.save_intermediate and not args.from_npz:
            save_npz(D, args.plots / f"perturbation_{D['dataset']}.npz")
            print(f"intermediate -> {args.plots}/perturbation_{D['dataset']}.npz")


if __name__ == "__main__":
    main()
