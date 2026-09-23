"""CPU-only viewsphere template-fit check.

Projects FK-posed mesh vertices through every GigaPose or FoundPose template
view. Exit code 1 indicates clipping or invalid depth in any state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

from robot_renderer.registry import get_robot_entry
from robot_renderer.robot_renderer import _resolve_mesh_files

_ROBOP_ROOT = Path(__file__).resolve().parents[1]


def _import_estimator_utils():
    """Load the numpy-only helper without importing estimator dependencies."""
    try:
        from robop import estimator_utils as module
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "estimator_utils", _ROBOP_ROOT / "src" / "robop" / "estimator_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


estimator_utils = _import_estimator_utils()

# GigaPose's fixed template camera (call_panda3d.py).
_GIGAPOSE_K = np.array(
    [572.4114, 0.0, 320.0, 0.0, 573.57043, 240.0, 0.0, 0.0, 1.0], dtype=np.float64
).reshape(3, 3)
_GIGAPOSE_WH = (640, 480)


def load_foundpose_view_rotations(num_views: int, num_inplane: int) -> np.ndarray:
    """Load FoundPose's Fibonacci views and optional in-plane rotations."""
    fp_root = _ROBOP_ROOT / "external" / "foundpose"
    if not (fp_root / "utils").is_dir():
        raise FileNotFoundError(
            f"{fp_root} not found. Clone the submodule: git -C ROBOP submodule update --init external/foundpose"
        )
    sys.path.insert(0, str(fp_root))
    from utils import misc as fp_misc

    views, _ = fp_misc.sample_views(min_n_views=num_views, radius=1000.0, mode="fibonacci")
    t = np.array([np.asarray(v["t"]).flatten() for v in views])
    if np.abs(t[:, :2]).max() > 1e-6:
        raise AssertionError("sample_views produced off-axis views")
    rotations = [np.asarray(v["R"], dtype=np.float64) for v in views]
    if num_inplane > 1:
        inplane_angle = 2.0 * np.pi / num_inplane
        expanded = []
        for R in rotations:
            for k in range(num_inplane):
                a = inplane_angle * k
                ca, sa = np.cos(a), np.sin(a)
                R_ip = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
                expanded.append(R_ip @ R)
        rotations = expanded
    return np.array(rotations)


def load_view_rotations(level: int) -> np.ndarray:
    """Object-to-camera rotations of GigaPose's predefined viewsphere poses.

    All predefined poses put the object on the optical axis (t = (0, 0, 1 m)),
    so only the rotations matter; the caller supplies the distance.
    """
    path = (
        _ROBOP_ROOT / "external" / "gigapose" / "src" / "lib3d"
        / "predefined_poses" / f"obj_poses_level{level}.npy"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Clone the submodule: git -C ROBOP submodule update --init external/gigapose"
        )
    poses = np.load(path)
    t = poses[:, :3, 3]
    off_axis = np.abs(t[:, :2]).max()
    if off_axis > 1e-6:
        raise AssertionError(
            f"predefined poses are expected on the optical axis, got |t_xy| up to {off_axis}"
        )
    return poses[:, :3, :3].astype(np.float64)


def load_joint_states(paths: list, num_joints: int) -> np.ndarray:
    """Load joint states from .npy / JSON files into one (N, num_joints) array.

    JSON files may be a list of joint lists or a Hydra ground_truth.json
    ({"0": {"joints": [...]}, ...}).
    """
    states = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(f"joints file not found: {p}")
        if p.suffix == ".npy":
            arr = np.load(p)
        elif p.suffix == ".json":
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                items = sorted(data.items(), key=lambda kv: int(kv[0]))
                arr = np.array([v["joints"] for _, v in items], dtype=np.float64)
            else:
                arr = np.array(data, dtype=np.float64)
        else:
            raise ValueError(f"unsupported joints file type: {p} (use .npy or .json)")
        arr = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        if arr.shape[1] != num_joints:
            raise ValueError(f"{p}: expected {num_joints} joints per state, got shape {arr.shape}")
        states.append(arr)
    return np.concatenate(states, axis=0)


def load_mesh_groups(entry: dict) -> list:
    """Per-link vertex arrays (metres, link frame) from the registry entry."""
    mesh_files = entry.get("mesh_files") or _resolve_mesh_files(entry["mesh_dir"])
    if not mesh_files:
        raise FileNotFoundError(f"no mesh files found in {entry['mesh_dir']}")
    groups = []
    for group in mesh_files:
        if isinstance(group, (str, Path)):
            group = [group]
        parts = []
        for path in group:
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"mesh file missing: {path} (are the robot assets downloaded?)")
            mesh = trimesh.load(path, force="mesh", process=False)
            parts.append(np.asarray(mesh.vertices, dtype=np.float64))
        groups.append(np.concatenate(parts, axis=0))
    return groups


def posed_centered_vertices_mm(adapter, groups: list, q: np.ndarray) -> np.ndarray:
    """FK-pose the link vertex groups, scale to mm, centre at the bbox centre.

    Mirrors renderer_pt3d._assemble_mesh_tensors + _build_posed_trimesh.
    """
    R, t = adapter.get_joint_R_t(q)
    if len(R) != len(groups):
        raise ValueError(f"FK returned {len(R)} link poses but {len(groups)} mesh groups are registered")
    verts = np.concatenate([g @ Ri.T + ti for g, Ri, ti in zip(groups, R, t)], axis=0)
    verts = verts * 1000.0
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    return verts - center


def convex_hull_or_all(verts: np.ndarray) -> np.ndarray:
    """Reduce to convex-hull vertices when scipy is available (projection
    extrema of a convex set are attained at its extreme points)."""
    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        return verts
    return verts[ConvexHull(verts).vertices]


def check_state(
    verts_mm: np.ndarray,
    view_R: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    distance_mm: float,
    near_mm: float,
) -> dict:
    """Project one posed state through all views; return worst-case statistics."""
    hull = convex_hull_or_all(verts_mm)
    t = np.array([0.0, 0.0, distance_mm])
    min_margin = np.inf
    min_depth = np.inf
    bad_views = 0
    for R in view_R:
        cam = hull @ R.T + t
        z = cam[:, 2]
        depth_min = float(z.min())
        min_depth = min(min_depth, depth_min)
        if depth_min <= near_mm:
            bad_views += 1
            min_margin = min(min_margin, -np.inf)
            continue
        uv = cam @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        margin = float(
            np.min([uv[:, 0].min(), (width - 1) - uv[:, 0].max(),
                    uv[:, 1].min(), (height - 1) - uv[:, 1].max()])
        )
        min_margin = min(min_margin, margin)
        if margin < 0:
            bad_views += 1
    return {"min_margin_px": min_margin, "min_depth_mm": min_depth, "bad_views": bad_views}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True, help="registered robot name (e.g. xarm7, lbr_med7)")
    parser.add_argument("--joints", nargs="+", default=[],
                        help=".npy (N, dof) or JSON joint files (Hydra ground_truth.json supported)")
    parser.add_argument("--zero", action="store_true", help="check the all-zero joint state")
    parser.add_argument("--camera", choices=["gigapose", "foundpose"], default="gigapose",
                        help="template camera to check against")
    parser.add_argument("--template-level", type=int, default=1, help="GigaPose viewsphere level")
    parser.add_argument("--K-npy", default=None,
                        help="foundpose mode: dataset camera K as a (3, 3) .npy file")
    parser.add_argument("--image-size", default=None,
                        help="foundpose mode: dataset image size as WxH, e.g. 1280x720")
    parser.add_argument("--num-views", type=int, default=57,
                        help="foundpose mode: Fibonacci viewpoints (estimator default 57)")
    parser.add_argument("--inplane", type=int, default=1,
                        help="foundpose mode: in-plane rotations per viewpoint")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="relative border margin for the auto distance (estimator default 0.05)")
    parser.add_argument("--distance-mm", type=float, default=None,
                        help="check one fixed camera distance instead of the auto distance")
    parser.add_argument("--near-mm", type=float, default=10.0,
                        help="minimum acceptable camera-frame depth")
    args = parser.parse_args()

    entry = get_robot_entry(args.robot)
    adapter = entry["adapter_cls"](entry["urdf"])
    fixed_mm = args.distance_mm

    if args.camera == "gigapose":
        K, (width, height) = _GIGAPOSE_K, _GIGAPOSE_WH
        view_R = load_view_rotations(args.template_level)
    else:
        if args.K_npy is None or args.image_size is None:
            raise SystemExit("--camera foundpose needs --K-npy and --image-size WxH")
        w, h = (int(v) for v in args.image_size.lower().split("x"))
        K, side = estimator_utils.square_camera_from_K(np.load(args.K_npy), w, h)
        width = height = side
        view_R = load_foundpose_view_rotations(args.num_views, args.inplane)

    if args.zero and args.joints:
        raise SystemExit("pass either --zero or --joints, not both")
    if args.zero:
        states = np.zeros((1, type(adapter).NUM_JOINTS))
    elif args.joints:
        states = load_joint_states(args.joints, type(adapter).NUM_JOINTS)
    else:
        raise SystemExit("no joint states: pass --joints file(s) or --zero")

    groups = load_mesh_groups(entry)
    dist_desc = (
        f"fixed {fixed_mm:.0f} mm" if fixed_mm is not None
        else f"auto (margin {args.margin:.2f})"
    )
    print(f"robot={args.robot} camera={args.camera} ({width}x{height}) "
          f"views={len(view_R)} states={len(states)} distance={dist_desc}")

    radii, distances = [], []
    worst_margin = np.inf
    worst_depth = np.inf
    total_bad = 0
    for idx, q in enumerate(states):
        verts = posed_centered_vertices_mm(adapter, groups, q)
        radius = estimator_utils.bounding_radius_mm(verts)
        if fixed_mm is not None:
            distance = fixed_mm
        else:
            distance = estimator_utils.viewsphere_distance_mm(radius, K, width, height, margin=args.margin)
        radii.append(radius)
        distances.append(distance)
        stats = check_state(verts, view_R, K, width, height, distance, args.near_mm)
        worst_margin = min(worst_margin, stats["min_margin_px"])
        worst_depth = min(worst_depth, stats["min_depth_mm"])
        total_bad += stats["bad_views"]
        flag = "" if stats["bad_views"] == 0 else f"  <-- {stats['bad_views']} view(s) violate"
        print(f"  state {idx:3d}: radius {radius:7.1f} mm  distance {distance:7.1f} mm  "
              f"min margin {stats['min_margin_px']:8.1f} px  min depth {stats['min_depth_mm']:7.1f} mm{flag}")

    print(f"summary: radius {min(radii):.1f}-{max(radii):.1f} mm, "
          f"distance {min(distances):.1f}-{max(distances):.1f} mm, "
          f"worst margin {worst_margin:.1f} px, worst depth {worst_depth:.1f} mm, "
          f"violating views {total_bad}/{len(states) * len(view_R)}")
    if total_bad > 0:
        print("FAIL: some views clip the robot or cross the near plane.")
        return 1
    print("PASS: every view of every state fits the template frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
