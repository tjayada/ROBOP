#!/usr/bin/env python3
"""Cross-model / cross-platform error analysis over a whole results tree.

Directory-level counterpart of analyze.py (which is the single-file deep dive).
The companion paper's ("Any Robot, Any Pose") Sec. V-B error analysis is stated over ALL model x
platform combinations at once, so it cannot live in the per-file script. This
script points at a results root (like gating/risk_coverage_from_masks.py does)
and reproduces that layer:

  A. Depth dominance      - share of squared positional error on the camera z
                            axis, per method x platform and in aggregate.
  B. Coarse -> refined    - refinement rescales the error without redirecting
                            it (depth error mm + depth-share shift, paired by
                            frame_id).
  C. Refinement buckets   - a bad initialisation is rescued, a wrong one is
                            not ([25,100) / [100,400) / >=400 mm buckets).
  D. ADD vs translation   - ADD is largely a translation metric.
  E. Per-keypoint chain   - error grows along the kinematic chain.
  F. Oracle axis correct. - depth bounds what the family can reach (best rigid
                            z-shift / rotation / lateral shift, ADD recomputed).
  G. Native-signal AUROC  - the confidences each estimator already emits
                            (MegaPose best_pose_score, GigaPose
                            best_template_score, FoundPose/NeMO PnP stats).

Discovery handles both layouts in the results tree: plain ``results.json``
and zipped ``results.json.zip`` (panda_orb), skipping per-shard parts.

Usage:
    python analysis/analyze_compare.py [results_root]
    python analysis/analyze_compare.py results/ --plots plots/compare/
    python analysis/analyze_compare.py results/ --datasets panda_orb,craves --skip-fk
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analysis_utils import (  # noqa: E402
    add_auc, compute_auroc, extract_signals,
    print_header, print_table, setup_matplotlib,
    BLUE, ORANGE, AQUA, YELLOW, INK,
)

# Checked in order: "nemo" is a substring of nothing here, but "megapose"
# appears in "megapose_refiner_<model>_..." paths, so megapose comes last.
MODEL_TOKENS = ["nemo_bootstrap", "nemo", "foundpose", "gigapose", "megapose"]

DISPLAY = {"panda_orb": "panda-orb", "baxter": "Baxter", "craves": "CRAVES",
           "hydra_lbr": "hydra-lbr", "hydra_xarm": "hydra-xarm",
           "hydra_meca": "hydra-meca"}
DATASET_ORDER = ["panda_orb", "baxter", "craves",
                 "hydra_lbr", "hydra_xarm", "hydra_meca"]


# ---------------------------------------------------------------------------
# Discovery / loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    model: str          # nemo_bootstrap | nemo | megapose | gigapose | foundpose
    dataset: str        # normalised: panda_orb | baxter | craves | hydra_{lbr,xarm,meca}
    variant: str        # coarse | native | modular   (native/modular = refined)
    summary: dict
    queries: list[dict]
    path: Path
    raw_dataset: str = ""  # summary["dataset"] verbatim; FK robot names derive from it
    _frames: dict = field(default=None, repr=False)

    @property
    def refined(self) -> bool:
        return self.variant != "coarse"

    @property
    def label(self) -> str:
        v = {"coarse": "coarse", "native": "ref-native", "modular": "ref-modular"}
        return f"{self.model} ({v[self.variant]})"


_HYDRA_ROBOT_NORM = {"lbr_med7": "lbr", "xarm7": "xarm", "meca500": "meca"}


def norm_dataset(name: str) -> str:
    parts = name.split("_", 1)
    if parts[0] == "hydra" and len(parts) == 2:
        return "hydra_" + _HYDRA_ROBOT_NORM.get(parts[1], parts[1])
    return name


def _load_json(path: Path) -> dict:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist()
                          if n.endswith(".json") and not n.startswith("__MACOSX"))
            return json.loads(z.read(member))
    return json.loads(path.read_text())


def discover(root: Path, models: set[str] | None = None,
             datasets: set[str] | None = None) -> list[Run]:
    files = sorted(list(root.rglob("results.json"))
                   + list(root.rglob("results.json.zip")))
    runs = []
    for f in files:
        rel = f.relative_to(root)
        if any(part.startswith("shard_") for part in rel.parts):
            continue  # keep only merged results, drop per-shard parts
        name = str(rel)
        model = next((m for m in MODEL_TOKENS if m in name), None)
        if model is None:
            print(f"  [skip] no model token in path: {rel}")
            continue
        # Cheap path-level pre-filter: the dataset string appears in every
        # layout's path, so non-matching files (esp. the large panda zips)
        # are never loaded. Normalised hydra names share the raw prefix.
        if datasets and not any(d.split("_")[0] in name or d in name
                                for d in datasets):
            continue
        if models and model not in models:
            continue
        config = rel.parts[0] if len(rel.parts) > 2 else ""
        if "refiner" not in config:
            variant = "coarse"
        elif "MODULAR" in config or f.parent.name.startswith("megapose_refiner_"):
            variant = "modular"
        else:
            variant = "native"

        print(f"  loading {rel} ...", flush=True)
        data = _load_json(f)
        raw_dataset = data.get("summary", {}).get("dataset", "unknown")
        dataset = norm_dataset(raw_dataset)
        if datasets and dataset not in datasets:
            continue
        runs.append(Run(model, dataset, variant, data["summary"],
                        data["queries"], f, raw_dataset))
    return runs


def select_runs(runs: list[Run]) -> list[Run]:
    """Dedup duplicate loads, then keep one refined per (model, dataset).

    A results tree can carry the same run twice (e.g. results.json next to
    results.json.zip), so first drop repeats of (model, dataset, variant).
    The paper's refined endpoint is the shared MegaPose refiner; where an
    estimator has its own (native) refiner we prefer that, else fall back to
    the shared one. Result: coarse plus a single refined run per (model,
    dataset), variant label kept so native vs shared stays visible.
    """
    seen, uniq = set(), []
    for r in runs:
        key = (r.model, r.dataset, r.variant)
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    best: dict[tuple[str, str], Run] = {}
    for r in uniq:
        if not r.refined:
            continue
        key = (r.model, r.dataset)
        cur = best.get(key)
        if cur is None or (cur.variant == "modular" and r.variant == "native"):
            best[key] = r
    keep = {id(r) for r in best.values()}
    return [r for r in uniq if not r.refined or id(r) in keep]


# ---------------------------------------------------------------------------
# Per-frame arrays
# ---------------------------------------------------------------------------

def frame_arrays(run: Run) -> dict:
    """Per-frame ADD, failure flags, ids, and camera-frame translation error.

    est_pose/gt_pose are cam-from-base transforms, so the translation delta is
    a CAMERA-frame vector: z is the viewing (depth) axis. Baxter has no poses;
    its EE position error (ee_cam_pred - ee_3d_gt) is camera-frame as well.
    Rows of failed frames are NaN.
    """
    q = run._frames
    if q is not None:
        return q

    n = len(run.queries)
    add   = np.full(n, np.nan)
    fid   = []
    t_err = np.full((n, 3), np.nan)
    rot   = np.full(n, np.nan)

    from scipy.spatial.transform import Rotation
    for i, query in enumerate(run.queries):
        fid.append(str(query.get("frame_id", i)))
        add[i] = query.get("add_m_mm", np.nan)
        if query.get("pnp_failed"):
            continue
        if run.dataset == "baxter":
            pred = query.get("ee_cam_pred")
            gt   = query.get("ee_3d_gt")
            if pred is None or gt is None:
                continue
            t_err[i] = (np.asarray(pred, float) - np.asarray(gt, float)) * 1000.0
        else:
            est, gtp = query.get("est_pose"), query.get("gt_pose")
            if est is None or gtp is None:
                continue
            est, gtp = np.asarray(est, float), np.asarray(gtp, float)
            t_err[i] = (est[:3, 3] - gtp[:3, 3]) * 1000.0
            R_err = est[:3, :3].T @ gtp[:3, :3]
            cos = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
            rot[i] = float(np.degrees(np.arccos(cos)))

    run._frames = {
        "add_mm": add, "frame_id": fid, "t_err_mm": t_err,
        "rot_deg": rot, "depth_mm": np.abs(t_err[:, 2]),
        "ok": ~np.isnan(t_err[:, 0]),
    }
    return run._frames


def pair_runs(runs: list[Run]) -> list[tuple[Run, Run]]:
    """(coarse, refined) pairs matched by (model, dataset)."""
    coarse  = {(r.model, r.dataset): r for r in runs if not r.refined}
    pairs = []
    for r in runs:
        if r.refined and (r.model, r.dataset) in coarse:
            pairs.append((coarse[(r.model, r.dataset)], r))
    return pairs


def _join(c: Run, r: Run):
    """Aligned (coarse, refined) frame arrays on the shared frame_ids."""
    fc, fr = frame_arrays(c), frame_arrays(r)
    ridx = {f: i for i, f in enumerate(fr["frame_id"])}
    keep = [(i, ridx[f]) for i, f in enumerate(fc["frame_id"]) if f in ridx]
    ic = np.array([a for a, _ in keep])
    ir = np.array([b for _, b in keep])
    return fc, fr, ic, ir


# ---------------------------------------------------------------------------
# Section A: depth dominance
# ---------------------------------------------------------------------------

def section_depth_dominance(runs: list[Run], skip_fk: bool) -> None:
    print_header("A. Depth Dominance  (share of squared error on the camera z axis,"
                 " over evaluated 3D points)")

    rows, shares_all = [], []
    for r in sorted(runs, key=lambda r: (DATASET_ORDER.index(r.dataset)
                                         if r.dataset in DATASET_ORDER else 99,
                                         r.model, r.variant)):
        kd = keypoint_depth(r, skip_fk)
        denom = float(np.nansum(kd["tot2"]))
        if denom <= 0:
            continue
        share = float(np.nansum(kd["z2"]) / denom)
        n = int(np.sum(~np.isnan(kd["tot2"])))
        shares_all.append(share)
        rows.append([r.label, DISPLAY.get(r.dataset, r.dataset),
                     f"{share:.2f}", "yes" if share > 1 / 3 else "no", n])

    print_table(["Run", "Dataset", "median z-share", "> 1/3", "N"], rows, col_width=18)
    shares_all = np.array(shares_all)
    print(f"\n  over {len(shares_all)} method x platform combinations: "
          f"median {np.median(shares_all):.2f}, "
          f"{int(np.sum(shares_all > 1/3))} above the isotropic line (1/3)")
    print(f"  range {shares_all.min():.2f} - {shares_all.max():.2f}   "
          f"(paper: 0.49-0.98, median 0.73, 61/66 above)")


# ---------------------------------------------------------------------------
# Section B + C: refinement redirect / buckets
# ---------------------------------------------------------------------------

def section_refinement_pairs(runs: list[Run], skip_fk: bool) -> None:
    print_header("B/C. Coarse -> Refined  (paired per frame_id)")

    pairs = pair_runs(runs)
    if not pairs:
        print("  No coarse/refined pairs found.")
        return

    dshares = []
    for c, r in pairs:
        fc, fr, ic, ir = _join(c, r)
        kc, kr = keypoint_depth(c, skip_fk), keypoint_depth(r, skip_fk)
        ok = (fc["ok"][ic] & fr["ok"][ir]
              & ~np.isnan(kc["tot2"][ic]) & ~np.isnan(kr["tot2"][ir]))
        if ok.sum() < 5:
            continue
        dc = kc["depth_mm"][ic][ok]
        dr = kr["depth_mm"][ir][ok]
        sc = float(np.sum(kc["z2"][ic][ok]) / np.sum(kc["tot2"][ic][ok]))
        sr = float(np.sum(kr["z2"][ir][ok]) / np.sum(kr["tot2"][ir][ok]))
        ac = fc["add_mm"][ic][ok]
        ar = fr["add_mm"][ir][ok]
        dshare = sr - sc
        dshares.append(dshare)
        print(f"\n  {c.label} -> {r.label}   [{DISPLAY.get(r.dataset, r.dataset)}]"
              f"   n={ok.sum()}")
        print(f"    depth |z| mm   mean {np.mean(dc):.1f} -> {np.mean(dr):.1f}")
        print(f"    depth share    pooled {sc:.2f} -> {sr:.2f}"
              f"   (d={dshare:+.2f})")

        rows = []
        for lo, hi in [(25, 100), (100, 400), (400, np.inf)]:
            m = (ac >= lo) & (ac < hi)
            if m.sum() == 0:
                continue
            rec = float(np.mean(ar[m] < 100.0))
            rows.append([f"[{lo},{hi if np.isfinite(hi) else 'inf'})",
                         int(m.sum()),
                         f"{np.median(ac[m]):.0f} -> {np.median(ar[m]):.0f}",
                         f"{100 * rec:.0f}%"])
        print_table(["    coarse ADD bucket (mm)", "n", "median ADD c->r",
                     "% refined < 100mm"], rows, col_width=24)

    if dshares:
        d = np.asarray(dshares)
        print(f"\n  share shift over {len(d)} coarse-refined pairs: "
              f"max |d| {np.abs(d).max():.2f}, median d {np.median(d):+.2f}   "
              f"(paper: at most +/-0.20, no systematic direction)")


# ---------------------------------------------------------------------------
# Section D: ADD vs translation
# ---------------------------------------------------------------------------

def section_translation_metric(runs: list[Run]) -> None:
    print_header("D. ADD as a Translation Metric")

    rows = []
    for r in sorted(runs, key=lambda r: (r.dataset, r.model, r.variant)):
        f = frame_arrays(r)
        ok = f["ok"] & ~np.isnan(f["add_mm"])
        if ok.sum() < 5:
            continue
        t_norm = np.linalg.norm(f["t_err_mm"][ok], axis=1)
        ratio = float(t_norm.mean() / f["add_mm"][ok].mean())
        rot = f["rot_deg"][ok]
        rot_s = "n/a" if r.dataset == "baxter" else f"{np.nanmean(rot):.1f}"
        rows.append([r.label, DISPLAY.get(r.dataset, r.dataset),
                     f"{t_norm.mean():.1f}", f"{f['add_mm'][ok].mean():.1f}",
                     f"{100 * ratio:.0f}%", rot_s])

    print_table(["Run", "Dataset", "mean |t_err| mm", "mean ADD mm",
                 "|t| / ADD", "mean rot deg"], rows, col_width=18)
    print("  (paper: base translation alone reproduces 85-110% of mean ADD on"
          " panda-orb; >100% = rotation partially cancels translation at the"
          " link keypoints)")


# ---------------------------------------------------------------------------
# Keypoints (sections E + F)
# ---------------------------------------------------------------------------

def _fk_base_keypoints(dataset: str, joints_rad) -> np.ndarray | None:
    """Base-frame ADD keypoints via FK (hydra/craves); needs robot_renderer."""
    from add_keypoints import _fk_keypoints  # lazy robot_renderer import
    out = _fk_keypoints(dataset, joints_rad)
    if out is None:
        return None
    return np.asarray(out[2], dtype=np.float64)


def _query_keypoints(run: Run, q: dict, skip_fk: bool):
    """(pred, gt) ADD keypoints in camera coords (mm) for one query, or None.

    panda_orb: stored keypoints_base_m / keypoints_cam_gt. hydra/craves: FK
    from joints_rad (add_keypoints._fk_keypoints). None if the frame failed PnP
    or FK is unavailable (skip_fk or robot_renderer missing).
    """
    if q.get("pnp_failed") or q.get("est_pose") is None:
        return None
    est = np.asarray(q["est_pose"], float)
    if "keypoints_base_m" in q and "keypoints_cam_gt" in q:
        base = np.asarray(q["keypoints_base_m"], float)
        gt   = np.asarray(q["keypoints_cam_gt"], float)
    else:
        gt_p = q.get("gt_pose")
        if skip_fk or gt_p is None:
            return None
        gt_p = np.asarray(gt_p, float)
        try:
            base = _fk_base_keypoints(run.raw_dataset, q["joints_rad"])
        except Exception:
            return None
        if base is None:
            return None
        gt = (gt_p[:3, :3] @ base.T).T + gt_p[:3, 3]
    pred = (est[:3, :3] @ base.T).T + est[:3, 3]
    return pred * 1000.0, gt * 1000.0  # -> mm


def keypoint_arrays(run: Run, skip_fk: bool = False):
    """(pred, gt) stacks of per-frame ADD keypoints in camera mm, or None.

    Baxter is EE-only (None). Only PnP-success frames with usable keypoints are
    kept; returns None if none are (e.g. hydra/craves under --skip-fk).
    """
    if run.dataset == "baxter":
        return None
    preds, gts = [], []
    for q in run.queries:
        kp = _query_keypoints(run, q, skip_fk)
        if kp is not None:
            preds.append(kp[0])
            gts.append(kp[1])
    if not preds:
        return None
    return np.stack(preds), np.stack(gts)


def keypoint_depth(run: Run, skip_fk: bool = False) -> dict:
    """Per-frame squared depth (z2) and total (tot2) error over ADD keypoints.

    Paper Sec. V-B decomposes error "over the evaluated 3D points": the depth
    share is the POOLED ratio sum dz^2 / sum |err|^2 over all keypoints and
    frames, so callers sum z2/tot2 rather than averaging per-frame ratios
    (pooling reproduces the paper's 0.79 GigaPose-Panda anchor; a per-frame
    median does not). Also returns per-frame mean |dz|. Baxter has no link
    keypoints and falls back to its single end-effector error (frame_arrays).
    Arrays align with run.queries; NaN where a frame has no usable keypoints.
    """
    if run.dataset == "baxter":
        f = frame_arrays(run)
        t = f["t_err_mm"]
        return {"z2": t[:, 2] ** 2, "tot2": np.sum(t ** 2, axis=1),
                "depth_mm": f["depth_mm"], "frame_id": f["frame_id"]}

    n = len(run.queries)
    z2   = np.full(n, np.nan)
    tot2 = np.full(n, np.nan)
    dmm  = np.full(n, np.nan)
    for i, q in enumerate(run.queries):
        kp = _query_keypoints(run, q, skip_fk)
        if kp is None:
            continue
        pred, gt = kp
        dz = pred[:, 2] - gt[:, 2]
        z2[i]   = float(np.sum(dz ** 2))
        tot2[i] = float(np.sum((pred - gt) ** 2))
        dmm[i]  = float(np.mean(np.abs(dz)))
    return {"z2": z2, "tot2": tot2, "depth_mm": dmm,
            "frame_id": [str(q.get("frame_id", i))
                         for i, q in enumerate(run.queries)]}


# ---------------------------------------------------------------------------
# Section E: per-keypoint chain profile
# ---------------------------------------------------------------------------

def section_keypoint_chain(runs: list[Run], skip_fk: bool) -> None:
    print_header("E. Per-Keypoint Error Along the Kinematic Chain")

    rows = []
    for r in sorted(runs, key=lambda r: (r.dataset, r.model, r.variant)):
        kp = keypoint_arrays(r, skip_fk)
        if kp is None:
            continue
        pred, gt = kp
        per_kp = np.linalg.norm(pred - gt, axis=2).mean(axis=0)  # mm
        rows.append([r.label, DISPLAY.get(r.dataset, r.dataset), len(pred),
                     f"{per_kp[0]:.0f}", f"{per_kp[-1]:.0f}",
                     " ".join(f"{v:.0f}" for v in per_kp)])

    if rows:
        print_table(["Run", "Dataset", "N", "kp[0] mean mm", "kp[-1] mean mm",
                     "per-kp means (mm)"], rows, col_width=18)
        print("  (paper: ~100 -> 174 mm for coarse GigaPose on panda-orb; Baxter"
              " skipped, EE-only metric)")
    else:
        print("  No keypoint data available (use --skip-fk only if robot_renderer"
              " is unavailable; Baxter is EE-only by design).")


# ---------------------------------------------------------------------------
# Section F: oracle axis correction
# ---------------------------------------------------------------------------

def _best_rotation(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Batched Kabsch: rotate pred about ITS centroid onto gt (translation kept)."""
    c = pred.mean(axis=1, keepdims=True)
    P = pred - c
    G = gt - c
    H = P.transpose(0, 2, 1) @ G
    U, _, Vt = np.linalg.svd(H)
    D = np.repeat(np.eye(3)[None], len(pred), axis=0)
    D[:, 2, 2] = np.sign(np.linalg.det(Vt.transpose(0, 2, 1)
                                       @ U.transpose(0, 2, 1)))
    R = Vt.transpose(0, 2, 1) @ D @ U.transpose(0, 2, 1)
    return (R @ P.transpose(0, 2, 1)).transpose(0, 2, 1) + c


def section_oracle(runs: list[Run], skip_fk: bool) -> None:
    print_header("F. Oracle Axis Correction  (best rigid shift/rotation, ADD recomputed)")

    rows, gains = [], {"z": [], "rot": [], "lat": []}
    for r in sorted(runs, key=lambda r: (r.dataset, r.model, r.variant)):
        kp = keypoint_arrays(r, skip_fk)
        if kp is None:
            continue
        pred, gt = kp
        ceiling = 400.0 if r.dataset == "baxter" else 100.0

        def auc_of(p):
            return add_auc(np.linalg.norm(p - gt, axis=2).mean(axis=1), ceiling)

        auc_base = auc_of(pred)

        z_fix = pred.copy()
        z_fix[:, :, 2] += (gt[:, :, 2] - pred[:, :, 2]).mean(axis=1, keepdims=True)
        auc_z = auc_of(z_fix)

        lat_fix = pred.copy()
        lat_fix[:, :, :2] += (gt[:, :, :2] - pred[:, :, :2]).mean(axis=1, keepdims=True)
        auc_lat = auc_of(lat_fix)

        auc_rot = auc_of(_best_rotation(pred, gt))

        for k, v in [("z", auc_z), ("rot", auc_rot), ("lat", auc_lat)]:
            gains[k].append(v - auc_base)
        rows.append([r.label, DISPLAY.get(r.dataset, r.dataset),
                     f"{100 * auc_base:.1f}", f"+{100 * (auc_z - auc_base):.1f}",
                     f"+{100 * (auc_rot - auc_base):.1f}",
                     f"{100 * (auc_lat - auc_base):+.1f}"])

    if rows:
        print_table(["Run", "Dataset", "base AUC", "z-shift", "rotation",
                     "lateral"], rows, col_width=18)
        print(f"\n  mean gain over {len(rows)} combinations: "
              f"z {100 * np.mean(gains['z']):+.1f} AUC, "
              f"rotation {100 * np.mean(gains['rot']):+.1f}, "
              f"lateral {100 * np.mean(gains['lat']):+.1f}   "
              f"(paper: +10.9 / +0.6 / -0.1)")
    else:
        print("  No keypoint data available.")


# ---------------------------------------------------------------------------
# Section G: native-signal AUROC (all models)
# ---------------------------------------------------------------------------

def _native_signals(run: Run) -> dict[str, tuple[np.ndarray, bool]]:
    """{name: (scores, invert)} - the confidence each estimator already emits."""
    out: dict[str, list] = {}
    def collect(name, fn):
        vals = []
        for q in run.queries:
            d = q.get("diagnostics") or {}
            vals.append(fn(d, q))
        arr = np.asarray(vals, dtype=float)
        if not np.all(np.isnan(arr)):
            out[name] = arr

    if run.model == "megapose":
        collect("best_pose_score", lambda d, q: d.get("best_pose_score", np.nan))
    elif run.model == "gigapose":
        collect("best_template_score", lambda d, q: d.get("best_template_score", np.nan))
    if run.model in ("foundpose", "gigapose"):
        collect("n_total_corr", lambda d, q: d.get("n_total_corr", np.nan))
    # PnP stats exist for every model that solves via PnP.
    collect("pnp_inlier_ratio", lambda d, q: (d.get("pnp") or {}).get("inlier_ratio", np.nan))
    collect("pnp_reproj_px", lambda d, q: (d.get("pnp") or {}).get("mean_reproj_error_px", np.nan))
    if run.model.startswith("nemo"):
        s = extract_signals(run.queries)
        out["conf_50"] = s["conf_50"]
        out["conf_count_50"] = s["conf_count_50"]
        out["anc_r"] = s["anc_r"]

    invert = {"pnp_reproj_px", "anc_r"}
    return {k: ((-v if k in invert else v), k in invert) for k, v in out.items()}


def section_native_auroc(runs: list[Run], thr: float) -> None:
    print_header(f"G. Native-Signal AUROC  (label: ADD < {thr:.0f}mm, all models)")

    rows = []
    for r in sorted(runs, key=lambda r: (r.model, r.dataset, r.variant)):
        f = frame_arrays(r)
        add = f["add_mm"]
        labels = (add < thr).astype(int)
        for name, (scores, _) in _native_signals(r).items():
            valid = ~np.isnan(scores) & ~np.isnan(add)
            if valid.sum() < 10 or labels[valid].sum() == 0 \
                    or (1 - labels[valid]).sum() == 0:
                continue
            auroc = compute_auroc(scores[valid], labels[valid])
            rows.append([r.label, DISPLAY.get(r.dataset, r.dataset),
                         name, f"{auroc:.3f}", int(valid.sum())])

    if rows:
        print_table(["Run", "Dataset", "Signal", "AUROC", "N"], rows, col_width=20)
        print("  (paper: native confidences AUROC 0.55-0.99, markedly better after"
              " refinement -- e.g. MegaPose 0.62 -> 0.85 on panda-orb)")
    else:
        print("  No native confidence signals found.")


# ---------------------------------------------------------------------------
# Section H: failure-taxonomy exemplars (manifest for fig:failure-taxonomy)
# ---------------------------------------------------------------------------

# The failure taxonomy names four stages. This section
# picks ONE exemplar frame per stage and writes a manifest that the overlay
# renderer turns into the figure's panels. Stages 2-4 are recoverable from the
# stored per-frame outputs; stage 1 (detection miss) is not -- it needs the
# externally known confound frame ids (see --confound-frames).
#
#   detection_miss        CNOS picks the wrong instance -> pose on a distractor.
#                         NOT derivable from stored poses; supplied externally.
#   correspondence_starved too few matches -> NeMO abstains (pnp_failed).
#   depth_axis            translation right in-plane, wrong in range:
#                         high z-share AND a material total error.
#   symmetry_flip         confident, plausible, wrong: low in-plane (lateral)
#                         error but a large oracle rotation correction.

_STAGES = ["detection_miss", "correspondence_starved", "depth_axis", "symmetry_flip"]


def _pick_depth_axis(run: Run, skip_fk: bool) -> dict | None:
    """Frame with the strongest depth dominance that is still a real error.

    Rank by pooled z-share restricted to frames whose mean |dz| is at least the
    run's median |dz| (so we surface a frame that is both depth-dominated and
    visibly off, not a trivially-good one).
    """
    kd = keypoint_depth(run, skip_fk)
    tot = np.asarray(kd["tot2"], float)
    z2 = np.asarray(kd["z2"], float)
    dmm = np.asarray(kd["depth_mm"], float)
    # Exclude abstained (pnp_failed) frames explicitly: they have no stored
    # pose, so their overlay panel would render empty -- meaningless here.
    failed = np.array([bool(q.get("pnp_failed")) for q in run.queries])
    ok = ~failed & ~np.isnan(tot) & (tot > 0) & ~np.isnan(dmm)
    if ok.sum() < 5:
        return None
    share = np.where(ok, z2 / np.where(tot > 0, tot, np.nan), np.nan)
    floor = np.nanmedian(dmm[ok])
    cand = ok & (dmm >= floor)
    if cand.sum() == 0:
        cand = ok
    idx = int(np.nanargmax(np.where(cand, share, -np.inf)))
    return {"frame_id": kd["frame_id"][idx],
            "z_share": round(float(share[idx]), 3),
            "mean_abs_dz_mm": round(float(dmm[idx]), 1)}


def _pick_symmetry_flip(run: Run, skip_fk: bool) -> dict | None:
    """Frame that is laterally right but rotationally wrong (silent flip).

    The flip renders a near-identical silhouette, so the in-plane (lateral, x/y)
    keypoint error stays small while the oracle rotation (Sec. F) recovers a lot.
    Rank by rotation gain among frames in the low-lateral-error half.
    """
    kp = keypoint_arrays(run, skip_fk)
    if kp is None:
        return None
    pred, gt = kp
    # keypoint_arrays keeps only frames with usable keypoints, so pred/gt can be
    # shorter than run.queries (e.g. CRAVES 420 vs 428). Rebuild the SAME kept
    # subset here so every per-frame array below aligns with pred/gt.
    kept = [q for q in run.queries if _query_keypoints(run, q, skip_fk) is not None]
    # Exclude abstained (pnp_failed) frames explicitly: no stored pose -> the
    # overlay panel would render empty, and a flip must be VISIBLE.
    failed = np.array([bool(q.get("pnp_failed")) for q in kept])
    lat = np.linalg.norm((pred - gt)[:, :, :2], axis=2).mean(axis=1)  # mm
    rot_fixed = _best_rotation(pred, gt)
    base_err = np.linalg.norm(pred - gt, axis=2).mean(axis=1)
    rot_err = np.linalg.norm(rot_fixed - gt, axis=2).mean(axis=1)
    rot_gain = base_err - rot_err  # large when a rotation recovers the pose
    low_lat = lat <= np.nanmedian(lat)
    cand = ~failed & low_lat & (rot_gain > 0)
    if cand.sum() == 0:
        cand = ~failed & (rot_gain > 0)
    if cand.sum() == 0:
        return None
    idx = int(np.nanargmax(np.where(cand, rot_gain, -np.inf)))
    return {"frame_id": str(kept[idx].get("frame_id", idx)),
            "lateral_err_mm": round(float(lat[idx]), 1),
            "rot_gain_mm": round(float(rot_gain[idx]), 1)}


def _pick_correspondence_starved(run: Run) -> dict | None:
    """A NeMO abstention (pnp_failed), or the lowest-inlier PnP frame."""
    f = frame_arrays(run)
    fid = f["frame_id"]
    failed = np.array([bool(q.get("pnp_failed")) for q in run.queries])
    if failed.any():
        idx = int(np.argmax(failed))
        return {"frame_id": fid[idx], "reason": "pnp_failed (abstention)"}
    # Fallback: lowest inlier ratio among returned poses.
    inliers = np.array([((q.get("diagnostics") or {}).get("pnp") or {})
                        .get("inlier_ratio", np.nan) for q in run.queries], float)
    if np.all(np.isnan(inliers)):
        return None
    idx = int(np.nanargmin(inliers))
    return {"frame_id": fid[idx],
            "reason": f"lowest pnp inlier ratio ({inliers[idx]:.2f})"}


def section_failure_exemplars(runs: list[Run], skip_fk: bool,
                              confound_frames: dict | None) -> dict:
    """Select one exemplar per failure stage; return the manifest dict.

    Stage 3/4 are mined across ALL runs and we keep the single strongest
    exemplar per stage (highest z-share / highest rotation gain respectively).
    Stage 2 comes from a NeMO run (the only estimator that abstains), preferring
    the coverage-poorest dataset. Stage 1 is taken verbatim from
    --confound-frames because it cannot be recovered from stored poses.
    """
    manifest: dict[str, dict] = {}

    # Stage 2: correspondence starvation -> a NeMO abstention.
    nemo = [r for r in runs if r.model.startswith("nemo")]
    stage2 = None
    for r in sorted(nemo, key=lambda r: DATASET_ORDER.index(r.dataset)
                    if r.dataset in DATASET_ORDER else 99):
        pick = _pick_correspondence_starved(r)
        if pick and "pnp_failed" in pick.get("reason", ""):
            stage2 = {"model": r.model, "dataset": r.dataset,
                      "variant": r.variant, **pick}
            break
        if pick and stage2 is None:
            stage2 = {"model": r.model, "dataset": r.dataset,
                      "variant": r.variant, **pick}
    if stage2:
        manifest["correspondence_starved"] = stage2

    # Stages 3 + 4: strongest exemplar across every run with keypoints.
    best3, best4 = None, None
    for r in runs:
        if r.dataset == "baxter":
            continue
        d = _pick_depth_axis(r, skip_fk)
        if d and (best3 is None or d["z_share"] > best3["z_share"]):
            best3 = {"model": r.model, "dataset": r.dataset,
                     "variant": r.variant, **d}
        s = _pick_symmetry_flip(r, skip_fk)
        if s and (best4 is None or s["rot_gain_mm"] > best4["rot_gain_mm"]):
            best4 = {"model": r.model, "dataset": r.dataset,
                     "variant": r.variant, **s}
    if best3:
        manifest["depth_axis"] = best3
    if best4:
        manifest["symmetry_flip"] = best4

    # Stage 1: detection miss -- external confound frame ids.
    if confound_frames:
        manifest["detection_miss"] = dict(confound_frames)

    print_header("H. Failure-Taxonomy Exemplars  (manifest for fig:failure-taxonomy)")
    for stage in _STAGES:
        e = manifest.get(stage)
        if e is None:
            print(f"  {stage:24s}  -- none selected --")
        else:
            print(f"  {stage:24s}  {e.get('model', '?'):16s} "
                  f"{e.get('dataset', '?'):10s} frame {e.get('frame_id', '?')}"
                  f"   ({', '.join(f'{k}={v}' for k, v in e.items() if k not in ('model', 'dataset', 'variant', 'frame_id'))})")
    return manifest


# ---------------------------------------------------------------------------
# Plot: paper Fig. 2  (a) depth-share radar  (b) depth-error dumbbell
# ---------------------------------------------------------------------------

# Fig. 2's four coarse estimators, in legend order (nemo_bootstrap is omitted,
# as in the paper). Colour is fixed per model across both panels. Order matches
# the canonical order (NeMO, MegaPose, GigaPose, FoundPose), the same as
# the result tables and the pipeline overview.
_FIG2_MODELS = ["nemo", "megapose", "gigapose", "foundpose"]
_MODEL_NAME = {"megapose": "MegaPose", "gigapose": "GigaPose",
               "nemo": "NeMO", "foundpose": "FoundPose"}
# Canonical model->color map (shared with risk_coverage via analysis_utils.MODEL_COLOR):
# NeMO=blue, GigaPose=orange, MegaPose=aqua, FoundPose=gold. Kept identical across
# every figure so the same estimator is never a different color between chapters.
_FIG2_COLOR = {"megapose": AQUA, "gigapose": ORANGE,
               "nemo": BLUE, "foundpose": YELLOW}
# Figure uses the paper's platform names (CRAVES arm = OWI, hydra prefix dropped).
_PLATFORM = {"panda_orb": "Panda", "baxter": "Baxter", "craves": "OWI",
             "hydra_lbr": "LBR", "hydra_xarm": "xArm", "hydra_meca": "Meca"}


def _mean_abs_dz(run: Run, skip_fk: bool) -> float:
    """Mean over frames and ADD points of the depth error |z_pred - z_gt|, mm.

    This is the paper's Fig. 2(b) quantity: each ADD keypoint's camera-z error,
    averaged over all keypoints and frames. Baxter has no link keypoints, so it
    uses its single end-effector depth error. Needs FK for hydra/craves keypoints
    (run without --skip-fk), like Sections E/F.
    """
    if run.dataset == "baxter":
        f = frame_arrays(run)
        d = f["depth_mm"][f["ok"]]
        return float(d.mean()) if len(d) else np.nan
    kp = keypoint_arrays(run, skip_fk)
    if kp is None:
        return np.nan
    pred, gt = kp  # (n_frames, n_kp, 3) mm, camera coords
    return float(np.abs(pred[:, :, 2] - gt[:, :, 2]).mean())


def _plot_depth_figure(runs: list[Run], out_dir: Path, skip_fk: bool) -> None:
    import matplotlib.pyplot as plt

    coarse = [r for r in runs if not r.refined and r.model in _FIG2_MODELS]
    datasets = [d for d in DATASET_ORDER if any(r.dataset == d for r in coarse)]
    models = [m for m in _FIG2_MODELS if any(r.model == m for r in coarse)]
    if not datasets or not models:
        return
    color = _FIG2_COLOR

    med_share = {}
    for r in coarse:
        kd = keypoint_depth(r, skip_fk)
        denom = float(np.nansum(kd["tot2"]))
        if denom > 0:
            med_share[(r.model, r.dataset)] = float(np.nansum(kd["z2"]) / denom)
    refined = {(c.model, c.dataset): r for c, r in pair_runs(runs)}

    fig = plt.figure(figsize=(13, 6))
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax2 = fig.add_subplot(1, 2, 2)

    # (a) radar: median depth share per platform, one polygon per coarse model.
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    loop = angles + angles[:1]
    ax1.set_theta_offset(np.pi / 2)  # first platform at the top
    ax1.set_theta_direction(-1)      # clockwise, as in the paper
    for m in models:
        vals = [med_share.get((m, d), np.nan) for d in datasets]
        ax1.plot(loop, vals + vals[:1], color=color[m], lw=1.9)
        ax1.fill(loop, vals + vals[:1], color=color[m], alpha=0.06)
    ax1.plot(loop, [1 / 3] * len(datasets) + [1 / 3], color=INK["muted"],
             lw=1.2, ls="--")
    ax1.set_xticks(angles)
    ax1.set_xticklabels([_PLATFORM.get(d, d) for d in datasets])
    ax1.tick_params(axis="x", pad=16)        # push platform labels off the ring
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0.5, 1.0])               # radial guides, labelled inside
    ax1.set_yticklabels(["0.5", "1.0"], fontsize=9, color=INK["secondary"])
    ax1.set_rlabel_position(20)
    ax1.set_title("(a)", pad=24)

    # (b) dumbbell: mean |dz| per ADD point, coarse (dot) -> refined (arrowhead),
    # grouped by platform on a log axis, alternate groups shaded. Refined =
    # native where it exists, else the shared MegaPose refiner (see select_runs).
    y = 0.0
    yticks, ylabels = [], []
    for gi, d in enumerate(datasets):
        start = y
        for m in reversed(models):  # paper stacks MegaPose at the bottom of each group
            cs = [r for r in coarse if r.model == m and r.dataset == d]
            if not cs:
                continue
            zc = _mean_abs_dz(cs[0], skip_fk)
            r = refined.get((m, d))
            zr = _mean_abs_dz(r, skip_fk) if r is not None else np.nan
            if np.isfinite(zc) and np.isfinite(zr):
                ax2.plot([zc, zr], [y, y], color=color[m], lw=1.7,
                         solid_capstyle="round", zorder=2)
                ax2.scatter([zr], [y], marker=("<" if zr < zc else ">"),
                            color=color[m], s=34, zorder=3)
            if np.isfinite(zc):
                ax2.scatter([zc], [y], color=color[m], s=30, zorder=3)
            y += 1
        if y > start:
            if gi % 2 == 0:                  # alternate-shade, first group shaded
                ax2.axhspan(start - 0.5, y - 0.5, color=INK["muted"],
                            alpha=0.08, zorder=0)
            yticks.append((start + y - 1) / 2)
            ylabels.append(_PLATFORM.get(d, d))
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels)
    ax2.set_ylim(y - 0.5, -0.5)              # top platform first, tight margins
    ax2.set_xscale("log")
    # Plain integer ticks (10, 20, 50, ...) instead of 10^1/10^2, minor ticks unlabelled.
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
    ax2.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200]))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.xaxis.set_minor_locator(LogLocator(subs=range(2, 10)))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel(r"mean $|\Delta z|$ per point (mm)")
    ax2.set_title("(b)")

    # one shared legend under both panels
    handles = [plt.Line2D([0], [0], color=color[m], lw=2.4) for m in models]
    labels = [_MODEL_NAME.get(m, m) for m in models]
    handles.append(plt.Line2D([0], [0], color=INK["muted"], lw=1.3, ls="--"))
    labels.append("isotropic (1/3)")
    handles.append(plt.Line2D([0], [0], color=INK["secondary"], marker="o",
                              markersize=6, lw=0))
    labels.append("coarse")
    handles.append(plt.Line2D([0], [0], color=INK["secondary"], marker=">",
                              markersize=7, lw=0))
    labels.append("refined")
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = out_dir / "depth_dominance.png"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] {p}  (+ .pdf)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", type=Path, nargs="?",
                    default=Path(__file__).resolve().parent.parent / "results",
                    help="results tree root (default: ROBOP/results)")
    ap.add_argument("--failure-threshold", type=float, default=100.0,
                    help="ADD threshold in mm for failure labels (default: 100)")
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated model filter")
    ap.add_argument("--datasets", type=str, default=None,
                    help="comma-separated dataset filter (normalised names)")
    ap.add_argument("--skip-fk", action="store_true",
                    help="skip FK keypoint reconstruction for hydra/craves "
                         "(sections E/F then cover panda_orb only)")
    ap.add_argument("--plots", type=Path, default=None,
                    help="directory to save figures (optional)")
    ap.add_argument("--exemplars", type=Path, default=None,
                    help="write the failure-taxonomy exemplar manifest (JSON) "
                         "here; consumed by the overlay renderer for "
                         "fig:failure-taxonomy")
    ap.add_argument("--confound-frames", type=str, default=None,
                    help="detection-miss exemplar(s), since they are not "
                         "recoverable from stored poses. Format: "
                         "model:dataset:frame_id[,frame_id...] e.g. "
                         "'nemo:hydra_lbr:00000012'")
    args = ap.parse_args()

    models = set(args.models.split(",")) if args.models else None
    datasets = set(args.datasets.split(",")) if args.datasets else None

    print(f"Scanning {args.results_root} ...")
    runs = discover(args.results_root, models, datasets)
    runs = select_runs(runs)
    print(f"\n{len(runs)} runs: "
          f"{sorted({(r.model, r.dataset, r.variant) for r in runs})}")

    section_depth_dominance(runs, args.skip_fk)
    section_refinement_pairs(runs, args.skip_fk)
    section_translation_metric(runs)
    section_keypoint_chain(runs, args.skip_fk)
    section_oracle(runs, args.skip_fk)
    section_native_auroc(runs, args.failure_threshold)

    if args.exemplars:
        confound = None
        if args.confound_frames:
            try:
                model, dataset, fids = args.confound_frames.split(":", 2)
                confound = {"model": model, "dataset": dataset,
                            "frame_id": fids.split(",")[0],
                            "extra_frame_ids": fids.split(","),
                            "note": "externally supplied (not in stored poses)"}
            except ValueError:
                print("  [warn] --confound-frames must be model:dataset:frame_id"
                      f"[, ...]; got {args.confound_frames!r}")
        manifest = section_failure_exemplars(runs, args.skip_fk, confound)
        args.exemplars.parent.mkdir(parents=True, exist_ok=True)
        args.exemplars.write_text(json.dumps(manifest, indent=2))
        print(f"  [exemplars] {args.exemplars}")

    if args.plots:
        args.plots.mkdir(parents=True, exist_ok=True)
        setup_matplotlib()
        _plot_depth_figure(runs, args.plots, args.skip_fk)


if __name__ == "__main__":
    main()
