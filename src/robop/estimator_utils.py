"""Shared checkpoint, template-fit, cache and coordinate helpers.

Module-level dependencies stay numpy/stdlib-only so template-fit checks run
without the GPU stack.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Released GigaPose and NeMO loaders use strict=False, which can leave mismatched
# parameters randomly initialized. Evaluation therefore requires an exact match.


class CheckpointMismatch(RuntimeError):
    """A checkpoint does not match the model it was loaded into."""


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, for recording which weights a run used."""
    h = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_state_dict_matches(model, state_dict, label: str) -> None:
    """Raise unless state_dict covers exactly the model's parameters."""
    model_keys = set(model.state_dict())
    ckpt_keys = set(state_dict)
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        raise CheckpointMismatch(
            f"{label}: checkpoint does not match the model "
            f"({len(missing)} missing, {len(unexpected)} unexpected). "
            f"Missing: {missing[:10]}. Unexpected: {unexpected[:10]}. "
            f"Check that the checkpoint matches the configured architecture."
        )
    shape_mismatches = [
        f"{k}: checkpoint {tuple(state_dict[k].shape)} vs model {tuple(v.shape)}"
        for k, v in model.state_dict().items()
        if k in state_dict and hasattr(state_dict[k], "shape") and state_dict[k].shape != v.shape
    ]
    if shape_mismatches:
        raise CheckpointMismatch(f"{label}: parameter shape mismatch: {shape_mismatches[:10]}")


# Template cameras must contain the complete FK-posed robot in every view.

MANIFEST_NAME = "template_manifest.json"


class TemplateFitError(RuntimeError):
    """A rendered template set failed fit validation."""


def bounding_radius_mm(vertices_mm) -> float:
    """Radius of the origin-centred bounding sphere of a posed mesh (mm).

    Viewsphere poses rotate the object about its origin, so the maximum
    vertex norm bounds the projected extent for every view.
    """
    v = np.asarray(vertices_mm, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] == 0:
        raise ValueError(f"expected nonempty (N, 3) vertices, got shape {v.shape}")
    return float(np.linalg.norm(v, axis=1).max())


def viewsphere_distance_mm(
    radius_mm: float,
    K,
    width: int,
    height: int,
    margin: float = 0.05,
) -> float:
    """Minimum camera distance (mm) at which a sphere of radius_mm centred on
    the optical axis projects fully inside a width x height image, keeping a
    relative border margin.

    The upstream FoundPose and GigaPose generators use fixed dataset/object
    distances, which clip metre-scale arms. ROBOP derives the distance per joint
    state instead.

    The silhouette of a sphere at distance d on the optical axis is a circle
    of radius f * r / sqrt(d^2 - r^2) around the principal point. Requiring
    that radius <= (1 - margin) * (available half extent) per image axis
    gives d >= r * sqrt(1 + (f / rho)^2); the tighter axis wins.
    """
    if not radius_mm > 0:
        raise ValueError(f"radius_mm must be positive, got {radius_mm}")
    if not 0.0 <= margin < 1.0:
        raise ValueError(f"margin must be in [0, 1), got {margin}")
    K = np.asarray(K, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    rho_x = (1.0 - margin) * min(cx, (width - 1) - cx)
    rho_y = (1.0 - margin) * min(cy, (height - 1) - cy)
    if rho_x <= 0 or rho_y <= 0:
        raise ValueError(
            f"principal point ({cx}, {cy}) leaves no usable extent inside {width}x{height}"
        )
    factor = max(
        float(np.sqrt(1.0 + (fx / rho_x) ** 2)),
        float(np.sqrt(1.0 + (fy / rho_y) ** 2)),
    )
    return float(radius_mm) * factor


def square_camera_from_K(K, width: int, height: int, patch_size: int = 14):
    """Square template camera from a dataset camera, as FoundPose's template
    generator builds it (gen_templates.py): viewport side = patch_size *
    floor(max side / patch_size), focal lengths unchanged, principal point
    shifted into the square viewport.

    Returns (K_square, side).
    """
    if width <= 0 or height <= 0 or patch_size <= 0:
        raise ValueError(f"invalid dimensions: {width}x{height}, patch {patch_size}")
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"expected (3, 3) K, got shape {K.shape}")
    side = patch_size * int(max(width, height) / patch_size)
    K_sq = K.copy()
    K_sq[0, 2] -= 0.5 * (width - side)
    K_sq[1, 2] -= 0.5 * (height - side)
    return K_sq, side


def check_mask_fit(mask, min_pixels: int) -> dict:
    """Fit statistics for one rendered template mask ((H, W), nonzero = foreground).

    This explicit check prevents clipped template sets from silently degrading
    retrieval.

    Returns a dict with n_pixels, bbox_xyxy ([x0, y0, x1, y1] with EXCLUSIVE
    maxima, matching Python slice semantics; None when empty), touches_border,
    and ok (nonempty, at least min_pixels, and not touching any image border).
    """
    m = np.asarray(mask) > 0
    if m.ndim != 2:
        raise ValueError(f"expected (H, W) mask, got shape {m.shape}")
    # cv2 single-pass C ops; identical stats to np.nonzero/min/max without the
    # index-array allocation (matters at multi-MP template resolutions).
    import cv2
    mu = np.ascontiguousarray(m, dtype=np.uint8)
    n = int(cv2.countNonZero(mu))
    if n == 0:
        return {"n_pixels": 0, "bbox_xyxy": None, "touches_border": False, "ok": False}
    x, y, w, h = cv2.boundingRect(mu)
    touches = bool(x == 0 or y == 0 or x + w == mu.shape[1] or y + h == mu.shape[0])
    return {
        "n_pixels": n,
        "bbox_xyxy": [x, y, x + w, y + h],
        "touches_border": touches,
        "ok": (n >= min_pixels) and not touches,
    }


def write_template_manifest(render_dir, meta: dict, views: list) -> Path:
    """Write the manifest recording the render policy (meta) and per-view fit stats.

    Callers write the manifest only after a complete render and validation
    pass, so its presence marks a cache directory as complete.
    """
    path = Path(render_dir) / MANIFEST_NAME
    payload = {"meta": meta, "views": views}
    path.write_text(json.dumps(payload, indent=1, allow_nan=False))
    return path


def load_template_manifest(render_dir) -> Optional[dict]:
    """Return the parsed manifest, or None when the directory has none
    (not rendered or interrupted render)."""
    path = Path(render_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def manifest_problems(manifest: dict, render_dir) -> list:
    """Return completeness and fit problems for a cached template set.

    Render policy is encoded in the cache directory key and is not rechecked.
    """
    render_dir = Path(render_dir)
    problems = []
    meta = manifest.get("meta", {})
    views = manifest.get("views", [])
    n_views = meta.get("n_views")
    if n_views is not None and len(views) != n_views:
        problems.append(f"manifest records {len(views)} views, expected {n_views}")
    for view in views:
        name = view.get("file")
        if not name or not (render_dir / name).is_file():
            problems.append(f"{name or '(unnamed view)'}: file missing")
        if not view.get("ok", False):
            problems.append(
                f"{name}: failed fit checks (n_pixels={view.get('n_pixels')}, "
                f"touches_border={view.get('touches_border')})"
            )
    return problems


def handle_fit_problems(
    problems,
    model_name: str,
    auto_distance_knob: str,
    distance_mm: float,
    radius_mm: float,
    dist_mode: str,
    what: str,
) -> None:
    """Raise on any template fit problem.

    model_name and auto_distance_knob differ per estimator: the knob is the
    config field that switches that model back to the auto camera distance,
    which is the fix when a fixed distance clipped the templates (empty when the
    model has no fixed-distance override).
    """
    if not problems:
        return
    head = (
        f"{model_name}: {len(problems)} template fit problem(s) in {what} "
        f"(distance={distance_mm:.0f} mm [{dist_mode}], bounding radius={radius_mm:.0f} mm)"
    )
    shown = problems[:20]
    detail = "\n  ".join(shown)
    if len(problems) > len(shown):
        detail += f"\n  ... {len(problems) - len(shown)} more"
    hint = (
        f"\n  Set estimator.{auto_distance_knob}=null for the auto distance."
        if dist_mode == "fixed" and auto_distance_knob else ""
    )
    raise TemplateFitError(f"{head}:\n  {detail}{hint}")


# Process / environment plumbing for the vendored GigaPose + MegaPose code


@contextlib.contextmanager
def suppress_c_stdout_stderr():
    """Redirect C-level stdout+stderr to /dev/null.

    Swallows panda3d's "Known pipe types"/"0 0" chatter - including its forked
    render workers, which inherit the redirected fds at fork - the torch.load
    FutureWarning emitted while building the MegaPose pipeline, and faiss's
    k-means progress output.

    Set ROBOP_SHOW_RENDER_OUTPUT=1 to disable the redirect when a forked render
    worker is failing silently (its EGL/panda3d error would otherwise be lost).
    """
    if os.environ.get("ROBOP_SHOW_RENDER_OUTPUT"):
        yield
        return
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


_GP_ROOT: Optional[Path] = None


def ensure_gigapose_path() -> Path:
    """Put external/gigapose AND external/gigapose/src on sys.path.

    Both the GigaPose and the vendored MegaPose code mix two import styles:
    `import src.models...` / `import src.custom_megapose...` (needs the repo
    root, external/gigapose) and `from megapose...` / `from models...`
    (top-level packages under src/, per setup.cfg package_dir=src - needs
    external/gigapose/src). poses.py uses the top-level form, so both are
    required.
    """
    global _GP_ROOT
    if _GP_ROOT is not None:
        return _GP_ROOT
    here = Path(__file__).resolve().parent          # src/robop/
    gp = here.parent.parent / "external" / "gigapose"
    if not gp.is_dir():
        raise RuntimeError(
            f"GigaPose (which also vendors MegaPose) not found at {gp}. "
            "Clone it: git -C ROBOP submodule update --init external/gigapose"
        )
    for p in (str(gp / "src"), str(gp)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # MegaPose config.py requires CONDA_PREFIX even for direct env-python calls.
    os.environ.setdefault("CONDA_PREFIX", sys.prefix)
    # MegaPose's panda3d_scene_renderer asserts CUDA_VISIBLE_DEVICES names a
    # single device; outside a scheduler allocation it is typically unset, and
    # the forked render workers then die on the assert while the parent waits on
    # the render queue forever. Same fallback call_panda3d gets passed.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    _GP_ROOT = gp
    return gp


def export_ply_with_normals(mesh, path) -> None:
    """Export ASCII PLY with finite angle-weighted vertex normals.

    MegaPose's panda3d normals channel reads normals from the loaded geometry;
    ``export_posed_trimesh`` otherwise leaves them absent.
    """
    normals = np.asarray(mesh.vertex_normals)   # compute + cache (angle-weighted)
    if normals.shape != (len(mesh.vertices), 3) or not np.isfinite(normals).all():
        raise ValueError(
            f"Mesh for {path} has invalid vertex normals (shape {normals.shape}, "
            f"{int((~np.isfinite(normals)).any(axis=1).sum())} non-finite)."
        )
    # ASCII, matching BOP's shipped models
    mesh.export(str(path), encoding="ascii")


def joint_hash(q) -> str:
    """Cache key for one joint configuration."""
    return hashlib.md5(np.asarray(q, dtype=np.float64).tobytes()).hexdigest()


def reseed_all(seed: int) -> None:
    """Reseed Python, numpy, torch and OpenCV for one reproducible frame.

    Pending CUDA work is synchronized before changing RNG state.
    """
    import random

    import cv2
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed)


# Convert a centered millimeter mesh pose back to the reported URDF base frame.
#   p_cam = R @ (p_urdf - center) + t   =>   t_base = t - R @ center


def center_offset_m(mesh_transform) -> np.ndarray:
    """Centred-mesh origin offset in METRES from a robot-renderer mesh_transform."""
    return (-np.asarray(mesh_transform, dtype=np.float64)[:3, 3]) / 1000.0


def centred_to_base(T_obj_cam, center_m) -> np.ndarray:
    """Centred-mesh->camera pose (metres) -> URDF base->camera pose (metres)."""
    T_obj_cam = np.asarray(T_obj_cam, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = T_obj_cam[:3, :3]
    T[:3, 3] = T_obj_cam[:3, 3] - T_obj_cam[:3, :3] @ np.asarray(center_m, dtype=np.float64)
    return T


def base_to_centred(T_base_cam, center_m) -> np.ndarray:
    """URDF base->camera pose (metres) -> centred-mesh->camera pose (metres).

    Inverse of centred_to_base; used to hand a coarse pose to MegaPose's
    refiner, which works in the centred-mesh frame.
    """
    T_base_cam = np.asarray(T_base_cam, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = T_base_cam[:3, :3]
    T[:3, 3] = T_base_cam[:3, 3] + T_base_cam[:3, :3] @ np.asarray(center_m, dtype=np.float64)
    return T
