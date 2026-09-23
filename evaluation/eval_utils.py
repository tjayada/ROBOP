"""Shared evaluation setup, metrics, detections and result serialization.

Evaluation scripts import this before torch and cv2 so OpenMP and cuBLAS read
the deterministic environment settings below during initialization.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")   # forces single-threaded OpenCV thread pool

import gzip
import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

try:  # package import (`from evaluation.eval_utils import ...`)
    from . import joint_noise
except ImportError:  # script-dir execution (`python evaluation/run_eval_*.py`)
    import joint_noise

log = logging.getLogger(__name__)


def _to_np(x) -> np.ndarray:
    """Convert a torch.Tensor or array-like to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def project_points(K, points_3d) -> np.ndarray:
    """Project (N, 3) camera-frame points through a 3×3 intrinsic -> (N, 2) pixels."""
    homog = np.asarray(K) @ np.asarray(points_3d).T
    return (homog[:2] / (homog[2:3] + 1e-8)).T


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)  # single-threaded OpenCV makes RANSAC fully deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def select_eval_frame_indices(
    n_total: int,
    num_eval_frames: Optional[int],
    covered_indices: Optional[List[int]] = None,
    shard: Optional[str] = None,
) -> List[int]:
    """Select all or evenly spaced frames, then optionally take shard ``k/M``.

    For a finite sample, ``covered_indices`` makes sampling operate over frames
    listed in the detections file. Full evaluation always uses the full dataset.
    Contiguous shards partition the post-sampling selection without overlap.
    """
    if num_eval_frames is None:
        indices = list(range(n_total))
    else:
        universe = covered_indices if covered_indices is not None else list(range(n_total))
        if num_eval_frames >= len(universe):
            indices = sorted(universe)
        else:
            pos = np.linspace(0, len(universe) - 1, num_eval_frames).round().astype(int)
            indices = sorted({universe[p] for p in pos})
    return _apply_shard(indices, shard)


def _apply_shard(indices: List[int], shard: Optional[str]) -> List[int]:
    """Return contiguous block k of M from an ordered list (shard='k/M', 0-based k).

    Uses np.array_split, so the M blocks concatenated in k-order reproduce
    `indices` exactly (disjoint, full cover). None returns the list unchanged.
    """
    if shard is None:
        return indices
    try:
        k, m = (int(x) for x in str(shard).split("/"))
    except ValueError:
        raise ValueError(f"frame_shard must look like 'k/M', got {shard!r}.") from None
    if not (m >= 1 and 0 <= k < m):
        raise ValueError(f"frame_shard 'k/M' needs M >= 1 and 0 <= k < M, got {shard!r}.")
    if m > len(indices):
        raise ValueError(
            f"frame_shard M={m} exceeds the {len(indices)} selected frames; use fewer shards."
        )
    return [int(i) for i in np.array_split(np.asarray(indices), m)[k]]


# np.trapz was renamed np.trapezoid in NumPy 2.0 (the old name still works but
# warns). Bind whichever exists so the AUC below stays warning-clean either way.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def add_auc(errors_mm, ceiling_mm: float) -> float:
    """ADD-AUC over [0, ceiling_mm], matching DREAM's `dream/analysis.py::pnp_metrics`
    (the convention CtRNet/RoboPose report against): trapezoidal integral of the
    fraction-below curve on a 0.01 mm grid, normalised to [0, 1]; `searchsorted`
    side="right" is DREAM's inclusive `<=`. PnP failures must arrive at a sentinel
    >= ceiling so they never fall below a threshold, pulling the ceiling below 1.

    Duplicated (not imported) in analysis/analysis_utils.add_auc so the analysis
    stack stays torch-free; keep the two in sync.
    """
    delta = 0.01  # mm  (DREAM delta_threshold = 0.00001 m)
    thresholds = np.arange(0.0, float(ceiling_mm), delta)
    add_sorted = np.sort(np.asarray(errors_mm, dtype=np.float64).ravel())
    if add_sorted.size == 0:
        return float("nan")
    frac_below = np.searchsorted(add_sorted, thresholds, side="right") / add_sorted.size
    return float(_TRAPZ(frac_below, dx=delta) / float(ceiling_mm))


def compute_add_metrics(per_frame_errors_mm: np.ndarray) -> dict:
    """
    Compute unified ADD metrics from per-frame mean errors in millimetres.

    Args:
        per_frame_errors_mm: (N,) array of per-frame mean ADD errors in mm.
            PnP failures should be assigned a large sentinel value (e.g. 1000 mm)
            so they are counted as failures at every threshold.

    Returns:
        Dict with keys: add_mean_mm, add_median_mm, add_p90_mm, add_p95_mm,
        add_auc_100mm, add_auc_400mm, add_at_100mm, add_at_50mm.

        AUCs use `add_auc` at 100 mm (DREAM/CtRNet Panda) and 400 mm (CtRNet Baxter).
    """
    arr = np.asarray(per_frame_errors_mm, dtype=np.float64).flatten()
    return {
        "add_mean_mm":   float(np.mean(arr)),
        "add_median_mm": float(np.median(arr)),
        "add_p90_mm":    float(np.percentile(arr, 90)),
        "add_p95_mm":    float(np.percentile(arr, 95)),
        "add_auc_100mm": add_auc(arr, 100.0),
        "add_auc_400mm": add_auc(arr, 400.0),
        "add_at_100mm":  float(np.mean(arr < 100)),
        "add_at_50mm":   float(np.mean(arr < 50)),
    }


def collect_model_stats(estimator, cfg) -> dict:
    """
    Collect fixed model properties for the comparison table.

    Returns a dict with: num_params, checkpoint_size_mb, num_templates.
    Gracefully omits keys that cannot be determined (e.g. no checkpoint path).
    """
    stats: dict = {}

    # NeMO exposes nemo_model; FoundPose exposes _extractor._model.
    model_obj = (
        getattr(estimator, "nemo_model", None)
        or getattr(getattr(estimator, "_extractor", None), "_model", None)
    )
    if model_obj is not None:
        stats["num_params"] = sum(p.numel() for p in model_obj.parameters())

    checkpoint = cfg.get("checkpoint", None)
    if checkpoint and os.path.isfile(str(checkpoint)):
        from robop import estimator_utils

        stats["checkpoint_size_mb"] = round(os.path.getsize(str(checkpoint)) / 1024 ** 2, 1)
        # which weights produced these numbers
        stats["checkpoint_sha256"] = estimator_utils.sha256_file(str(checkpoint))

    # MegaPose-family weights live under megapose_models_root, not cfg.checkpoint
    # (a directory would fail the isfile check above). Hash the downloaded
    # checkpoint.pth.tar for the coarse and refiner models so a swapped or stale
    # models directory is visible in the results JSON.
    models_root = str(cfg.estimator.get("megapose_models_root", "") or "")
    model_name = cfg.estimator.get("model_name") or cfg.estimator.get("refiner_model_name")
    if models_root and os.path.isdir(models_root) and model_name:
        from robop import estimator_utils
        from src.megapose.utils.load_model import NAMED_MODELS

        info = NAMED_MODELS.get(model_name)
        if info is not None:
            for label, run_id in (("coarse", info["coarse_run_id"]),
                                  ("refiner", info["refiner_run_id"])):
                p = os.path.join(models_root, run_id, "checkpoint.pth.tar")
                if os.path.isfile(p):
                    stats[f"megapose_{label}_sha256"] = estimator_utils.sha256_file(p)

    # Template counts must reflect each model's renderer: NeMO uses the PT3D
    # viewsphere, GigaPose its object-pose viewset, FoundPose views times in-plane
    # rotations, and MegaPose has no persistent template set.
    est_type = str(cfg.estimator.get("type", "nemo")).lower()
    num_views = cfg.estimator.get("num_views", None)  # NeMO template count
    if est_type == "megapose":
        pass  # render-and-compare over a per-frame SO(3) grid - no fixed template set
    elif est_type == "gigapose":
        poses = getattr(estimator, "_template_obj_poses_1m", None)
        if poses is not None:
            stats["num_templates"] = int(len(poses))
    elif est_type == "foundpose":
        stats["num_templates"] = (
            int(cfg.estimator.get("num_views", 57)) * int(cfg.estimator.get("num_inplane_rotations", 14))
        )
    elif num_views is not None:
        stats["num_templates"] = int(num_views)

    return stats


# Per-phase timing keys the estimators emit into diagnostics["timings_ms"].
# Shared by aggregate_frame_diagnostics (which computes timing_<phase>_median_ms)
# and log_diagnostics_summary (which prints them) so the two cannot drift apart.
TIMING_PHASE_KEYS = ("preprocess", "render", "encode", "decode", "pnp", "coarse", "postprocess")


def aggregate_frame_diagnostics(diag_list: list) -> dict:
    """
    Aggregate per-frame diagnostics (from _last_frame_diagnostics) into summary stats.

    Refiner accounting covers every frame; timing/GPU/PnP/runtime skip the first
    5 frames as warm-up, consistent with the existing runtime stats.
    Returns a dict that can be merged directly into the eval metrics dict.
    """
    all_diags = [d for d in diag_list if d is not None]
    diags = [d for d in diag_list[5:] if d is not None]

    result: dict = {}

    # Refinement accounting: a full-method run must not silently be coarse
    # Counted over ALL frames (no warm-up skip): a refiner failure on any of the
    # first 5 frames must not be hidden, and short runs must still report.
    refine = [d["refinement"] for d in all_diags if isinstance(d.get("refinement"), dict)]
    enabled = [r for r in refine if r.get("enabled")]
    if enabled:
        succeeded = sum(1 for r in enabled if r.get("succeeded"))
        result["refiner_frames"] = len(enabled)
        result["refiner_succeeded"] = succeeded
        result["refiner_failed"] = len(enabled) - succeeded
        result["refiner_success_rate"] = round(succeeded / len(enabled), 4)

    if not diags:
        return result

    # Phase timing
    # Keys the estimators actually emit into diagnostics["timings_ms"]: NeMO emits
    # preprocess/render/encode/decode/pnp/postprocess, GigaPose emits coarse.
    # log_diagnostics_summary logs this same list - keep the two in sync.
    for key in TIMING_PHASE_KEYS:
        vals = [d["timings_ms"][key] for d in diags if "timings_ms" in d and key in d["timings_ms"]]
        if vals:
            result[f"timing_{key}_median_ms"] = round(float(np.median(vals)), 1)

    # GPU peak memory
    gpu_vals = [d["gpu_peak_mb"] for d in diags if d.get("gpu_peak_mb") is not None]
    if gpu_vals:
        result["gpu_peak_mb_median"] = round(float(np.median(gpu_vals)), 1)
        result["gpu_peak_mb_max"]    = round(float(np.max(gpu_vals)), 1)

    # PnP correspondence quality (successful frames only)
    pnp_entries = [d["pnp"] for d in diags if d.get("pnp") is not None]
    if pnp_entries:
        result["pnp_mean_correspondences"]  = round(
            float(np.mean([e["total_correspondences"] for e in pnp_entries])), 1
        )
        result["pnp_median_inlier_ratio"]   = round(float(np.median([e["inlier_ratio"] for e in pnp_entries])), 4)
        # reproj error may be None (e.g. GigaPose recovers pose analytically, no PnP reprojection)
        reproj_vals = [e["mean_reproj_error_px"] for e in pnp_entries if e.get("mean_reproj_error_px") is not None]
        if reproj_vals:
            result["pnp_median_reproj_err_px"] = round(float(np.median(reproj_vals)), 3)

    # Runtime stats (total frame time)
    total_vals = [d["frame_time_ms"] for d in diags if "frame_time_ms" in d]
    if total_vals:
        result["runtime_median_ms"] = round(float(np.median(total_vals)), 1)
        result["runtime_mean_ms"]   = round(float(np.mean(total_vals)), 1)
        result["runtime_p90_ms"]    = round(float(np.percentile(total_vals, 90)), 1)

    return result


def scale_K_to_render(
    K_np: np.ndarray,
    render_size: int,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """Scale intrinsics for square viewsphere template rendering.

    This camera affects template appearance only. Query PnP intrinsics are
    adjusted independently for each frame and crop mode.
    """
    K = K_np.astype(np.float32).copy()
    s = min(image_h, image_w)
    crop_x = (image_w - s) // 2
    crop_y = (image_h - s) // 2
    scale = render_size / s
    K[0, 0] *= scale
    K[0, 2] = (K[0, 2] - crop_x) * scale
    K[1, 1] *= scale
    K[1, 2] = (K[1, 2] - crop_y) * scale
    return K


# One-shot guard so the units check logs at most once per process, not per frame.
_UNITS_CHECK_DONE: set = set()


def assert_translation_units_consistent(t_pred, t_gt, tag: str = "") -> None:
    """Reject an unambiguous meter/millimeter mismatch on the first valid frame.

    Predicted and ground-truth base translations must have a norm ratio within
    ``[0.01, 100]``; a unit mismatch is approximately 1000.
    """
    if tag in _UNITS_CHECK_DONE:
        return
    _UNITS_CHECK_DONE.add(tag)
    n_pred = float(np.linalg.norm(np.asarray(t_pred, dtype=np.float64).ravel()))
    n_gt = float(np.linalg.norm(np.asarray(t_gt, dtype=np.float64).ravel()))
    if n_gt < 1e-9 or n_pred < 1e-9:
        log.warning("[units:%s] degenerate translation (‖pred‖=%.4g ‖gt‖=%.4g) - "
                    "units check skipped.", tag, n_pred, n_gt)
        return
    ratio = n_pred / n_gt
    if not (0.01 <= ratio <= 100.0):
        raise AssertionError(
            f"[units:{tag}] predicted base translation ‖t_pred‖={n_pred:.4g} vs "
            f"GT ‖t_gt‖={n_gt:.4g} differ by {ratio:.4g}× - likely a metres/millimetres "
            f"mismatch between the rendered mesh units and the dataset GT. The ADD->mm "
            f"reporting assumes the mesh (URDF) is in metres."
        )
    log.info("[units:%s] OK - ‖t_pred‖=%.4g m, ‖t_gt‖=%.4g m (ratio %.3g×).",
             tag, n_pred, n_gt, ratio)


def build_renderer(cfg: DictConfig, K_render: torch.Tensor, device: torch.device):
    """Build the robot renderer; only NeMO consumes its template viewsphere."""
    import robot_renderer as rr
    from robot_renderer import ViewConfig

    est = cfg.estimator
    est_type = str(est.get("type", "nemo")).lower()
    if est_type in ("nemo", "nemo_bank"):
        view_cfg = ViewConfig(
            render_size=int(est.get("render_size", 448)),
            viewset=str(est.get("viewset", "fibonacci_256")),
            num_views=int(est.get("num_views", 32)),
            sphere_distance_factor=float(est.get("sphere_distance_factor", 2.25)),
            elevation_range=tuple(est.get("elevation_range", (-90.0, 90.0))),
            orientation=str(est.get("orientation", "simple_upright")),
            diverse_selection=bool(est.get("diverse_selection", True)),
            anchor_elevation_range=(
                tuple(est.anchor_elevation_range)
                if est.get("anchor_elevation_range") is not None else None
            ),
            backgrounds=(
                list(est.backgrounds) if est.get("backgrounds") is not None else None
            ),
            fill_frame=bool(est.get("fill_frame", False)),
            fill_frame_pad=float(est.get("fill_frame_pad", 0.1)),
            fill_frame_min_px=int(est.get("fill_frame_min_px", 64)),
        )
    else:
        # Geometry-source-only models: viewsphere unused (no template rendering).
        view_cfg = ViewConfig()
    return rr.RobotRenderer(
        cfg.robot.name, K=K_render, config=view_cfg, device=device
    )


def build_estimator(cfg: DictConfig, renderer, device: torch.device):
    """
    Construct and return the configured pose estimator.

    Supported types (cfg.estimator.type):
      - "nemo" (default)  - requires cfg.checkpoint
      - "foundpose"       - training-free; checkpoint not used
      - "gigapose"        - requires cfg.checkpoint
      - "megapose"        - requires estimator.megapose_models_root
      - "megapose_refiner" - refines the poses in estimator.init_results_path
    """
    estimator_type = str(cfg.estimator.get("type", "nemo")).lower()

    if estimator_type == "foundpose":
        from robop.foundpose_estimator import FoundPoseConfig, FoundPoseRobotEstimator

        _cache = cfg.estimator.get("cache_dir", None)
        fp_cfg = FoundPoseConfig(
            dino_model=str(cfg.estimator.get("dino_model", "dinov2_vitl14_reg")),
            dino_layer=int(cfg.estimator.get("dino_layer", 18)),
            render_size=int(cfg.estimator.get("render_size", 420)),
            num_views=int(cfg.estimator.get("num_views", 57)),
            num_inplane_rotations=int(cfg.estimator.get("num_inplane_rotations", 14)),
            grid_cell_size=float(cfg.estimator.get("grid_cell_size", 14.0)),
            apply_pca=bool(cfg.estimator.get("apply_pca", True)),
            pca_components=int(cfg.estimator.get("pca_components", 256)),
            cluster_num=int(cfg.estimator.get("cluster_num", 2048)),
            match_top_n_templates=int(cfg.estimator.get("match_top_n_templates", 5)),
            match_top_k_buddies=int(cfg.estimator.get("match_top_k_buddies", 300)),
            pnp_ransac_iter=int(cfg.estimator.get("pnp_ransac_iter", 400)),
            pnp_reproj_error=float(cfg.estimator.get("pnp_reproj_error", 10.0)),
            pnp_confidence=float(cfg.estimator.get("pnp_confidence", 0.99)),
            pnp_refine_lm=bool(cfg.estimator.get("pnp_refine_lm", True)),
            min_correspondences=int(cfg.estimator.get("min_correspondences", 6)),
            seed=int(cfg.get("seed", 0)),
            cache_dir=str(_cache) if _cache is not None else None,
            sphere_distance_mm=(
                float(cfg.estimator.sphere_distance_mm)
                if cfg.estimator.get("sphere_distance_mm") is not None
                else None
            ),
            ssaa_factor=float(cfg.estimator.get("ssaa_factor", 4.0)),
            template_fit_margin=float(cfg.estimator.get("template_fit_margin", 0.05)),
            min_template_pixels=int(cfg.estimator.get("min_template_pixels", 100)),
            crop_rel_pad=float(cfg.estimator.get("crop_rel_pad", 0.2)),
            verbose=bool(cfg.estimator.get("verbose", False)),
            debug_dir=(
                str(cfg.estimator.debug_dir)
                if cfg.estimator.get("debug_dir") is not None else None
            ),
        )
        estimator = FoundPoseRobotEstimator(
            renderer=renderer, config=fp_cfg, device=device,
        )
        log.info("Estimator: FoundPose (DINOv2 %s) - internal pyrenderer: "
                 "%d views × %d in-plane rot = %d templates @ %dpx.",
                 fp_cfg.dino_model, fp_cfg.num_views, fp_cfg.num_inplane_rotations,
                 fp_cfg.num_views * fp_cfg.num_inplane_rotations, fp_cfg.render_size)

    elif estimator_type == "gigapose":
        from robop.gigapose_estimator import GigaPoseConfig, GigaPoseRobotEstimator

        _cache = cfg.estimator.get("cache_dir", None)
        # Reuse the top-level `checkpoint` key (same as NeMO) so existing eval
        # configs work without a `+estimator.checkpoint_path=...` override.
        if cfg.get("checkpoint") is None:
            raise ValueError("checkpoint is required for the GigaPose estimator (set checkpoint=...).")
        gp_cfg = GigaPoseConfig(
            checkpoint_path=str(cfg.checkpoint),
            template_level=int(cfg.estimator.get("template_level", 1)),
            pose_distribution=str(cfg.estimator.get("pose_distribution", "all")),
            template_fit_margin=float(cfg.estimator.get("template_fit_margin", 0.05)),
            min_template_pixels=int(cfg.estimator.get("min_template_pixels", 100)),
            k=int(cfg.estimator.get("k", 5)),
            sim_threshold=float(cfg.estimator.get("sim_threshold", 0.5)),
            patch_threshold=int(cfg.estimator.get("patch_threshold", 3)),
            ransac_pixel_threshold=int(cfg.estimator.get("ransac_pixel_threshold", 14)),
            ae_config=str(cfg.estimator.get("ae_config", "dinov2_l")),
            ist_config=str(cfg.estimator.get("ist_config", "resnet")),
            refine=bool(cfg.estimator.get("refine", False)),
            refiner_multi_hypothesis=bool(cfg.estimator.get("refiner_multi_hypothesis", True)),
            refiner_model_name=str(cfg.estimator.get("refiner_model_name", "megapose-1.0-RGB-multi-hypothesis")),
            n_refiner_iterations=int(cfg.estimator.get("n_refiner_iterations", 5)),
            megapose_models_root=str(cfg.estimator.get("megapose_models_root", "")),
            refiner_num_workers=int(cfg.estimator.get("refiner_num_workers", 4)),
            seed=int(cfg.get("seed", 0)),
            cache_dir=str(_cache) if _cache is not None else None,
            debug_save_templates=(
                str(cfg.estimator.debug_save_templates)
                if cfg.estimator.get("debug_save_templates") is not None else None
            ),
            debug_reproject_dir=(
                str(cfg.estimator.debug_reproject_dir)
                if cfg.estimator.get("debug_reproject_dir") is not None else None
            ),
            n_reproject_frames=int(cfg.estimator.get("n_reproject_frames", 20)),
            verbose=bool(cfg.estimator.get("verbose", False)),
        )
        estimator = GigaPoseRobotEstimator(
            renderer=renderer, config=gp_cfg, device=device,
        )
        # GigaPose alone (coarse retrieval) vs GigaPose + MegaPose refiner are
        # effectively two different methods - name them distinctly in the log
        # (the run directory is differentiated via the estimator config choice:
        # estimator=gigapose vs estimator=gigapose_megapose).
        _gp_method = (
            "GigaPose + MegaPose refiner (coarse+refine, %s)" % (
                "multi-hypothesis" if gp_cfg.refiner_multi_hypothesis else "single-hypothesis")
            if gp_cfg.refine else "GigaPose (coarse only)"
        )
        log.info("Estimator: %s - internal panda3d renderer (distance=auto, level=%d).",
                 _gp_method, gp_cfg.template_level)

    elif estimator_type == "megapose":
        from robop.megapose_estimator import MegaPoseConfig, MegaPoseRobotEstimator

        mp_cfg = MegaPoseConfig(
            model_name=str(cfg.estimator.get("model_name", "megapose-1.0-RGB-multi-hypothesis")),
            megapose_models_root=str(cfg.estimator.get("megapose_models_root", "")),
            n_refiner_iterations=(
                int(cfg.estimator.n_refiner_iterations)
                if cfg.estimator.get("n_refiner_iterations") is not None else None
            ),
            n_pose_hypotheses=(
                int(cfg.estimator.n_pose_hypotheses)
                if cfg.estimator.get("n_pose_hypotheses") is not None else None
            ),
            coarse_only=bool(cfg.estimator.get("coarse_only", False)),
            num_workers=int(cfg.estimator.get("num_workers", 4)),
            batch_size_objects=int(cfg.estimator.get("batch_size_objects", 8)),
            batch_size_images=int(cfg.estimator.get("batch_size_images", 128)),
            debug_reproject_dir=(
                str(cfg.estimator.debug_reproject_dir)
                if cfg.estimator.get("debug_reproject_dir") is not None else None
            ),
            debug_render_dir=(
                str(cfg.estimator.debug_render_dir)
                if cfg.estimator.get("debug_render_dir") is not None else None
            ),
            n_reproject_frames=int(cfg.estimator.get("n_reproject_frames", 20)),
            seed=int(cfg.get("seed", 0)),
            verbose=bool(cfg.estimator.get("verbose", False)),
        )
        estimator = MegaPoseRobotEstimator(renderer=renderer, config=mp_cfg, device=device)
        log.info("Estimator: MegaPose %s (%s) - internal render-and-compare "
                 "(per-frame SO(3) hypothesis grid via panda3d, %d pose hypotheses).",
                 "coarse-only" if mp_cfg.coarse_only else "standalone", mp_cfg.model_name,
                 estimator._n_pose_hypotheses)

    elif estimator_type == "megapose_refiner":
        from robop.megapose_refiner_estimator import (
            MegaPoseRefinerConfig, MegaPoseRefinerEstimator,
        )

        ref_cfg = MegaPoseRefinerConfig(
            init_results_path=str(cfg.estimator.get("init_results_path", "")),
            model_name=str(cfg.estimator.get("model_name", "megapose-1.0-RGB-multi-hypothesis")),
            megapose_models_root=str(cfg.estimator.get("megapose_models_root", "")),
            n_refiner_iterations=(
                int(cfg.estimator.n_refiner_iterations)
                if cfg.estimator.get("n_refiner_iterations") is not None else None
            ),
            joints_atol=float(cfg.estimator.get("joints_atol", 1e-5)),
            score_refined=bool(cfg.estimator.get("score_refined", False)),
            num_workers=int(cfg.estimator.get("num_workers", 4)),
            batch_size_objects=int(cfg.estimator.get("batch_size_objects", 8)),
            batch_size_images=int(cfg.estimator.get("batch_size_images", 128)),
            debug_reproject_dir=(
                str(cfg.estimator.debug_reproject_dir)
                if cfg.estimator.get("debug_reproject_dir") is not None else None
            ),
            n_reproject_frames=int(cfg.estimator.get("n_reproject_frames", 20)),
            seed=int(cfg.get("seed", 0)),
            verbose=bool(cfg.estimator.get("verbose", False)),
        )
        estimator = MegaPoseRefinerEstimator(renderer=renderer, config=ref_cfg, device=device)
        log.info("Estimator: MegaPose REFINER - refining init poses from %s "
                 "(%d refiner iterations).",
                 ref_cfg.init_results_path, estimator._n_refiner_iterations)

    else:  # nemo (default)
        if cfg.get("checkpoint") is None:
            raise ValueError("checkpoint is required for the NeMO estimator.")

        from nemolib.model import Model
        from robop import estimator_utils
        import torch as _torch
        _ckpt = _torch.load(cfg.checkpoint, weights_only=False, map_location="cpu")
        nemo_model = Model.from_checkpoint(_ckpt, device=device)
        # The released loader ignores key mismatches (model.py load_state_dict
        # strict=False), so check the same state dict it built here instead.
        estimator_utils.assert_state_dict_matches(
            nemo_model,
            {k[len("model."):]: v for k, v in _ckpt["state_dict"].items()
             if k.startswith("model.")},
            f"NeMO checkpoint {cfg.checkpoint}",
        )

        from robop import NeMORobotPoseEstimator, NeMORobotConfig
        # The .get fallbacks below are the demo-regime defaults (see
        # NeMORobotConfig); the shipped configs/estimator/nemo.yaml always sets
        # every key, so a real eval run never falls back to them.
        est_cfg = NeMORobotConfig(
            seed=int(cfg.get("seed", 0)),
            conf_threshold=cfg.estimator.conf_threshold,
            mask_threshold=float(cfg.estimator.get("mask_threshold", 0.5)),
            crop_mode=str(cfg.estimator.get("crop_mode", "center")),
            bbox_crop_pad=float(cfg.estimator.get("bbox_crop_pad", 0.0)),
            bootstrap_mask_pad=float(cfg.estimator.get("bootstrap_mask_pad", 0.1)),
            bootstrap_min_mask_pixels=int(cfg.estimator.get("bootstrap_min_mask_pixels", 64)),
            alignment_method=str(cfg.estimator.get("alignment_method", "extent")),
            alignment_k_best=int(cfg.estimator.get("alignment_k_best", 5)),
            alignment_max_iter=int(cfg.estimator.get("alignment_max_iter", 10000)),
            alignment_lr=float(cfg.estimator.get("alignment_lr", 0.05)),
            alignment_huber_delta=float(cfg.estimator.get("alignment_huber_delta", 0.001)),
            alignment_differentiable_axis=bool(
                cfg.estimator.get("alignment_differentiable_axis", False)),
            include_query_in_templates=cfg.estimator.include_query_in_templates,
            use_mesh_surface_sampling=cfg.estimator.get("use_mesh_surface_sampling", True),
            use_blurred_query_background=cfg.estimator.use_blurred_query_background,
            blur_sigma=cfg.estimator.blur_sigma,
            nemo_encoding_size=cfg.estimator.nemo_encoding_size,
            num_sample_points=int(cfg.estimator.get("num_sample_points", 1500)),
            pnp_min_correspondences=int(cfg.estimator.get("pnp_min_correspondences", 32)),
            pnp_min_inlier_ratio=float(cfg.estimator.get("pnp_min_inlier_ratio", 0.0)),
            pnp_initial_iterations=cfg.estimator.pnp_initial_iterations,
            pnp_initial_reproj_error=cfg.estimator.pnp_initial_reproj_error,
            pnp_initial_confidence=cfg.estimator.pnp_initial_confidence,
            debug_save_decoder=cfg.estimator.get("debug_save_decoder", None),
            debug_save_failures=cfg.estimator.get("debug_save_failures", None),
        )
        if estimator_type == "nemo_bank":
            # Same NeMO, with the render+encode phase served from a config-keyed bank
            # (reuse_bank.py). nemo_estimator.py itself is untouched.
            from robop.nemo_bank_estimator import NeMOBankEstimator

            _bank = cfg.estimator.get("reuse_bank", None)
            _tol = None if _bank is None else _bank.get("tolerance_mm", None)
            estimator = NeMOBankEstimator(
                nemo_model=nemo_model, renderer=renderer, config=est_cfg, device=device,
                robot_name=cfg.robot.name,
                tolerance_mm=None if _tol is None else float(_tol),
            )
        else:
            estimator = NeMORobotPoseEstimator(
                nemo_model=nemo_model, renderer=renderer, config=est_cfg, device=device,
            )
        estimator.eval()
        log.info("Estimator: %s (checkpoint: %s) - %d robot-renderer PT3D templates @ %dpx.",
                 estimator_type, cfg.checkpoint, int(cfg.estimator.get("num_views", 32)),
                 int(cfg.estimator.get("render_size", 448)))

    return estimator


def build_eval_stack(cfg: DictConfig, device: torch.device, K_eval_np: np.ndarray,
                     image_hw: tuple) -> tuple:
    """Build the renderer -> estimator -> adapter chain shared by every run_eval_*.py.

    `K_eval_np` is the 3×3 intrinsic at the resolution the images are evaluated
    at, and `image_hw` that resolution - together they give the template-render
    K (see scale_K_to_render). Each script derives those two differently (from a
    camera file, from dataset constants, from the first sample), which is why
    they stay with the caller.

    Seeding happens before this function: `seed_everything` is the first statement
    of each main(), before dataset construction.

    Returns (adapter, model_stats).
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    K_render = torch.tensor(
        scale_K_to_render(np.asarray(K_eval_np, dtype=np.float32),
                          int(cfg.estimator.get("render_size", 448)), h, w),
        dtype=torch.float32, device=device,
    )
    renderer = build_renderer(cfg, K_render, device)
    estimator = build_estimator(cfg, renderer, device)
    K_eval_t = torch.tensor(np.asarray(K_eval_np, dtype=np.float32),
                            dtype=torch.float32, device=device)
    adapter = EstimatorAdapter(estimator, K_eval_t,
                               joint_noise_cfg=cfg.get("joint_noise", None))
    model_stats = collect_model_stats(estimator, cfg)
    if model_stats:
        log.info(f"Model stats: {model_stats}")
    return adapter, model_stats


def pose_output_problems(pose) -> list:
    """Validate an estimator's 4x4 base-to-camera pose before it reaches
    metrics and serialization.

    Checks shape, finiteness, a homogeneous last row, a right-handed
    near-orthonormal rotation, and that the base origin lies in front of the
    camera. Behind-camera poses are failures; CtRNet's sign-flip rule for
    negative-depth keypoints is not used.

    Returns a list of problem strings; empty means valid.
    """
    T = pose.detach().cpu().numpy() if isinstance(pose, torch.Tensor) else np.asarray(pose)
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return [f"pose shape {tuple(T.shape)}, expected (4, 4)"]
    if not np.isfinite(T).all():
        return ["pose contains non-finite values"]
    problems = []
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        problems.append(f"last row {T[3].tolist()} is not [0, 0, 0, 1]")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
        problems.append("rotation is not orthonormal")
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-3:
        problems.append(f"rotation determinant {det:.6f}, expected +1")
    if T[2, 3] <= 0.0:
        problems.append(f"base origin depth {T[2, 3]:.4f} is not in front of the camera")
    return problems


class EstimatorAdapter:
    """Adapts any ROBOP pose estimator to the shared evaluate_* interface.

    Estimators declare which optional per-frame inputs they consume via a
    class-level INPUT_CONTRACT dict (keys: mask, bbox_xyxy, frame_key,
    mask_3x3_opening); the adapter passes exactly those. Each declared input must
    be an explicit parameter of inference_single_image; **kwargs does not count.
    """

    _CONTRACT_KEYS = ("mask", "bbox_xyxy", "frame_key", "mask_3x3_opening")

    def __init__(self, estimator, K: torch.Tensor, joint_noise_cfg=None):
        self.estimator = estimator
        self._K = K
        contract = getattr(type(estimator), "INPUT_CONTRACT", None)
        if not isinstance(contract, dict) or set(contract) != set(self._CONTRACT_KEYS):
            raise TypeError(
                f"{type(estimator).__name__} must declare a class-level INPUT_CONTRACT "
                f"dict with exactly the keys {self._CONTRACT_KEYS}, got {contract!r}."
            )
        sig = inspect.signature(estimator.inference_single_image).parameters
        for name in ("mask", "bbox_xyxy", "frame_key"):
            if contract[name] and name not in sig:
                raise TypeError(
                    f"{type(estimator).__name__}.INPUT_CONTRACT accepts {name!r} but "
                    f"inference_single_image has no explicit parameter of that name."
                )
        self._contract = contract
        # A zero-magnitude perturbation must keep the normal inference path exact.
        self._jn_magnitude_deg = float(joint_noise_cfg["magnitude_deg"]) if joint_noise_cfg else 0.0
        if self._jn_magnitude_deg < 0.0:
            raise ValueError(
                f"joint_noise.magnitude_deg must be >= 0, got {self._jn_magnitude_deg}. "
                "A negative value would silently run as an unlabeled baseline."
            )
        self._jn_seed = int(joint_noise_cfg["seed"]) if joint_noise_cfg else 0
        self._jn_qlim = None  # resolved lazily on first perturbed call

    @property
    def apply_mask_opening(self) -> bool:
        """Whether decode_detection should run the FoundPose-style 3x3 opening
        for this estimator (INPUT_CONTRACT["mask_3x3_opening"])."""
        return bool(self._contract["mask_3x3_opening"])

    def inference_single_image(self, img, joint_angles, mask=None, bbox_xyxy=None,
                               frame_key=None, K=None):
        jn_record = None
        if self._jn_magnitude_deg > 0.0:
            joint_angles, jn_record = self._perturb_joint_angles(joint_angles, frame_key)
        kwargs = {"K": self._frame_K(K)}
        if self._contract["mask"]:
            kwargs["mask"] = mask
        if self._contract["bbox_xyxy"]:
            kwargs["bbox_xyxy"] = bbox_xyxy
        # frame identity (image path / hydra meas-key) - needed by refiner-style
        # estimators that look up per-frame init poses from a results JSON
        if self._contract["frame_key"]:
            kwargs["frame_key"] = frame_key
        result = self.estimator.inference_single_image(img, joint_angles, **kwargs)
        if jn_record is not None:
            diag = getattr(self.estimator, "_last_frame_diagnostics", None)
            if isinstance(diag, dict):
                diag["joint_noise"] = jn_record
            else:
                log.warning("joint_noise active but the estimator exposes no per-frame "
                            "diagnostics dict - perturbation record dropped.")
        return self._validated(result, frame_key)

    def _validated(self, result, frame_key):
        """Turn an invalid returned pose into an explicit failure (None) with a
        recorded reason. Without this, one NaN entry can make whole aggregate
        metrics NaN while the run looks normal."""
        pose = result[0] if isinstance(result, tuple) else result
        if pose is None:
            return result
        problems = pose_output_problems(pose)
        if not problems:
            return result
        reason = "; ".join(problems)
        log.warning("Frame %s: estimator returned an invalid pose (%s) - recorded as failure.",
                    frame_key, reason)
        diag = getattr(self.estimator, "_last_frame_diagnostics", None)
        if isinstance(diag, dict):
            diag["invalid_pose_reason"] = reason
        if isinstance(result, tuple):
            return (None,) + tuple(result[1:])
        return None

    def _frame_K(self, K):
        """Resolve the K for this frame; never silently frame zero's.

        Loaders with per-frame intrinsics pass K per call; every estimator
        consumes it. A call without a K falls back to the construction K.
        """
        if K is None:
            return self._K
        K_np = K.cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
        if isinstance(self._K, torch.Tensor):
            return torch.as_tensor(K_np, dtype=self._K.dtype, device=self._K.device)
        return K_np

    def _perturb_joint_angles(self, joint_angles, frame_key):
        if frame_key is None:
            raise ValueError("joint_noise is active but no frame_key was passed - "
                             "deterministic per-frame seeding requires it.")
        q_np = _to_np(joint_angles).astype(np.float64).reshape(-1)
        if self._jn_qlim is None:
            self._jn_qlim = joint_noise.resolve_qlim(self.estimator.robot_kin, q_np.shape[0])
            log.info("joint_noise: magnitude=%.3f°, qlim resolved for %s (%d joints).",
                     self._jn_magnitude_deg, type(self.estimator.robot_kin).__name__,
                     q_np.shape[0])
        q_tilde, record = joint_noise.perturb(
            q_np, self._jn_magnitude_deg, self._jn_seed, str(frame_key), self._jn_qlim)
        if isinstance(joint_angles, torch.Tensor):
            q_tilde = torch.as_tensor(q_tilde, dtype=joint_angles.dtype,
                                      device=joint_angles.device)
        return q_tilde, record


# Detection loading - robot-detector run_detect.py output format

def _rle_to_mask(rle: dict) -> np.ndarray:
    """Decode robot-detector custom column-major RLE -> uint8 (H, W) mask."""
    H, W = rle["size"]
    flat = np.zeros(H * W, dtype=np.uint8)
    idx, val = 0, 0
    for count in rle["counts"]:
        flat[idx:idx + count] = val
        idx += count
        val = 1 - val
    return flat.reshape(H, W, order="F")


def _detection_entry_problems(det: dict) -> list:
    """Schema checks for one found=true detection entry.

    A malformed entry crashing or decoding to an empty mask mid-run would
    otherwise surface as an unrelated estimator error hundreds of frames in.
    Returns a list of problem strings; empty means valid.
    """
    seg = det.get("segmentation")
    if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
        return ["segmentation missing size/counts"]
    size = seg["size"]
    if len(size) != 2 or not all(isinstance(v, int) and v > 0 for v in size):
        return [f"segmentation size invalid: {size}"]
    H, W = size
    problems = []
    counts = seg["counts"]
    if not counts or not all(isinstance(c, int) and c >= 0 for c in counts):
        problems.append("RLE counts must be nonnegative integers")
    elif sum(counts) != H * W:
        problems.append(f"RLE counts sum {sum(counts)} != H*W = {H * W}")
    elif sum(counts[1::2]) == 0:   # runs alternate starting with background
        problems.append("mask is empty (no foreground run)")
    bbox = det.get("bbox_xyxy")
    if bbox is None or len(bbox) != 4 or not np.all(np.isfinite(np.asarray(bbox, dtype=np.float64))):
        problems.append(f"bbox_xyxy invalid: {bbox}")
    else:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            problems.append(f"bbox {bbox} out of bounds for the {W}x{H} mask")
    return problems


def load_robot_detections(path):
    """Load detections keyed by stable dataset frame ID.

    Returns valid detections, every listed frame ID, and file provenance. Keeping
    both ID sets distinguishes a recorded miss from a frame absent from the file.
    ``None`` explicitly selects detector-free evaluation; invalid paths raise.
    """
    if path is None:
        return {}, set(), None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Detections file not found: {p}. Point detections_path at a file "
            f"produced by robot-detector, or set it to null to run without detections."
        )
    # Read .json or .json.gz (robot-detector v1.1.0 ships gzipped); hash the
    # decompressed bytes so the sha256 stays comparable to plain-JSON runs.
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    data = json.loads(raw.decode())
    entries = data.get("detections", [])
    if not entries:
        raise ValueError(
            f"Detections file {p} contains no detection entries. An explicitly "
            f"supplied file must cover the evaluated frames; set detections_path "
            f"to null for a detector-free run."
        )
    found, all_ids = {}, set()
    for det in entries:
        if "frame_id" not in det:
            raise ValueError(f"Detections file {p}: entry without frame_id.")
        fid = str(det["frame_id"])
        if fid in all_ids:
            raise ValueError(f"Detections file {p}: duplicate frame_id {fid!r}.")
        all_ids.add(fid)
        if det.get("found"):
            problems = _detection_entry_problems(det)
            if problems:
                raise ValueError(f"Detections file {p}, frame {fid}: " + "; ".join(problems))
            found[fid] = det
    meta = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(),
            "n_entries": len(all_ids), "n_found": len(found)}
    log.info("Loaded %d detections from %s (%d entries, %d with no detection, sha256 %s).",
             len(found), p, len(all_ids), len(all_ids) - len(found), meta["sha256"][:12])
    return found, all_ids, meta


def verify_detection_coverage(found: dict, all_ids: set, frame_ids, label: str = "") -> None:
    """Check that every frame about to be evaluated has an entry in the detections file.

    A frame missing from the file would otherwise fall through to a center-crop
    fallback and quietly change the results, so it raises. Frames whose entry
    says found=false are fine: they are counted here and take the estimator's
    documented no-detection path.
    """
    if not all_ids:
        return
    frame_ids = [str(f) for f in frame_ids]
    missing = [f for f in frame_ids if f not in all_ids]
    n_no_det = sum(1 for f in frame_ids if f in all_ids and f not in found)
    log.info("%sdetections: %d/%d covered, %d with no detection.",
             f"{label} " if label else "",
             len(frame_ids) - len(missing), len(frame_ids), n_no_det)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(frame_ids)} evaluated frames have no entry in "
            f"the detections file (examples: {missing[:5]}). The file does not cover "
            f"this frame selection: regenerate it for this dataset, or set "
            f"num_eval_frames - an int subsamples within the frames the file covers."
        )


def decode_detection(det: dict, img_H: int, img_W: int, apply_opening: bool = False):
    """Decode a single detection entry into (mask_uint8, bbox_xyxy).

    apply_opening runs a 3x3 morphological opening on the decoded mask. That
    step mirrors FoundPose's infer_pose_util.get_instances_for_pose_estimation
    and is applied only for estimators whose upstream does it (the estimator's
    INPUT_CONTRACT["mask_3x3_opening"]); upstream GigaPose and MegaPose
    consume the decoded mask directly.

    A pure resolution scale between the detector image and the eval image
    (same aspect ratio) is resolved by nearest-neighbor resize; an
    aspect-ratio mismatch is an error, because the transform between the two
    croppings cannot be inferred from image sizes alone.

    Returns (mask np.uint8 H×W, bbox_xyxy np.float32 [x1,y1,x2,y2]).
    """
    mask = _rle_to_mask(det["segmentation"])

    if apply_opening:
        # mirrors infer_pose_util.py lines 87-91 (upstream FoundPose)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )

    H_mask, W_mask = mask.shape
    if H_mask != img_H or W_mask != img_W:
        scale_y = img_H / H_mask
        scale_x = img_W / W_mask
        if abs(scale_y - scale_x) >= 0.05:
            raise ValueError(
                f"Detection mask ({W_mask}x{H_mask}) and eval image ({img_W}x{img_H}) "
                f"have different aspect ratios. The transform between them cannot be "
                f"inferred from sizes alone; regenerate the detections on the "
                f"evaluated footage."
            )
        # same aspect ratio -> the eval ran on a scaled version of the detector image
        mask = cv2.resize(mask, (img_W, img_H), interpolation=cv2.INTER_NEAREST)
        bbox = np.array(det["bbox_xyxy"], dtype=np.float32)
        bbox[[0, 2]] *= scale_x
        bbox[[1, 3]] *= scale_y
    else:
        bbox = np.array(det["bbox_xyxy"], dtype=np.float32)

    return mask, bbox


def detection_for_frame(found: dict, all_ids: set, frame_id, img_H: int, img_W: int,
                        apply_opening: bool):
    """Look up one frame's detection under the benchmark-wide miss policy.

    A found=false entry is an evaluated detection-miss failure for every
    estimator: the caller records the frame as failed without running
    inference, instead of silently switching that frame to a detector-free
    regime (which would mix two methods inside one run).

    Returns (mask, bbox_xyxy, miss). miss=True means the detector reported no
    detection for this frame.
    """
    fid = str(frame_id)
    if fid in found:
        mask, bbox = decode_detection(found[fid], img_H, img_W, apply_opening=apply_opening)
        return mask, bbox, False
    if fid in all_ids:
        return None, None, True
    raise KeyError(
        f"Frame {fid!r} is not in the detections file at all - "
        f"verify_detection_coverage should have rejected this run."
    )


def load_detections_for_run(cfg: DictConfig) -> tuple:
    """Load the configured detections file and log what came back.

    Returns load_robot_detections' (found, all_ids, meta) unchanged.
    """
    detections, detection_ids, det_meta = load_robot_detections(cfg.get("detections_path", None))
    if det_meta is None:
        log.info("No detections loaded - running without segmentation masks.")
    return detections, detection_ids, det_meta


def run_frame_inference(model, image, joint_angles, frame_id, detections,
                        detection_ids, frame_key, K=None) -> tuple:
    """Resolve one frame's detection, apply the miss policy, and run inference.

    Returns (pose, diagnostics). A detector miss short-circuits to
    (None, {"detection_miss": True}) without running the estimator: a miss is an
    evaluated failure, never a silent switch to a detector-free regime for just
    that frame (which would mix two methods inside one run).

    `K` is the per-frame intrinsic for loaders that have one; None lets the
    adapter fall back to its construction K.
    """
    mask_np, bbox_xyxy = None, None
    if detection_ids:
        img_H, img_W = image.shape[-2], image.shape[-1]
        mask_np, bbox_xyxy, det_miss = detection_for_frame(
            detections, detection_ids, frame_id, img_H, img_W, model.apply_mask_opening)
        if det_miss:
            return None, {"detection_miss": True}

    pose, *_ = model.inference_single_image(
        image, joint_angles, mask=mask_np, bbox_xyxy=bbox_xyxy,
        frame_key=frame_key, K=K,
    )
    return pose, getattr(model.estimator, "_last_frame_diagnostics", None)


def log_diagnostics_summary(logger, metrics: dict) -> None:
    """Log runtime, phase timing, VRAM, and PnP stats from aggregate_frame_diagnostics output."""
    if "runtime_median_ms" in metrics:
        parts = [f"median={metrics['runtime_median_ms']:.1f} ms"]
        if "runtime_mean_ms" in metrics:
            parts.append(f"mean={metrics['runtime_mean_ms']:.1f} ms")
        if "runtime_p90_ms" in metrics:
            parts.append(f"P90={metrics['runtime_p90_ms']:.1f} ms")
        logger.info(f"  Runtime:    {'  '.join(parts)}")
    for phase in TIMING_PHASE_KEYS:
        key = f"timing_{phase}_median_ms"
        if key in metrics:
            logger.info(f"    {phase:<11}: {metrics[key]:.1f} ms")
    if "gpu_peak_mb_median" in metrics:
        logger.info(
            f"  GPU VRAM:   median={metrics['gpu_peak_mb_median']:.1f} MB"
            f"  max={metrics['gpu_peak_mb_max']:.1f} MB"
        )
    if "pnp_mean_correspondences" in metrics:
        line = (
            f"  PnP corr:   mean={metrics['pnp_mean_correspondences']:.0f}"
            f"  inlier_ratio(median)={metrics['pnp_median_inlier_ratio']:.3f}"
        )
        # reproj error is absent for estimators without PnP reprojection (e.g. GigaPose)
        if "pnp_median_reproj_err_px" in metrics:
            line += f"  reproj_err(median)={metrics['pnp_median_reproj_err_px']:.2f} px"
        logger.info(line)


def _git_revision(repo_dir) -> Optional[dict]:
    """Commit and modified-files flag for a git working tree, or None when
    unavailable. Untracked files are excluded so build artifacts such as
    __pycache__ do not mark a clean checkout as modified."""
    import subprocess

    repo_dir = Path(repo_dir)
    try:
        rev = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10,
        )
        return {"commit": rev.stdout.strip(),
                "modified": bool(status.stdout.strip()) if status.returncode == 0 else None}
    except (OSError, subprocess.SubprocessError):
        return None


def run_provenance(cfg: DictConfig) -> dict:
    """What produced this result: code revisions, environment, seed, timestamp.

    Recorded next to every result so a number can be traced back to the exact
    code and settings that produced it. Checkpoint and detection hashes are
    recorded by collect_model_stats and load_robot_detections.
    """
    import platform
    from datetime import datetime, timezone

    repo_root = Path(__file__).resolve().parent.parent
    externals = {}
    external_dir = repo_root / "external"
    if external_dir.is_dir():
        for sub in sorted(p for p in external_dir.iterdir() if p.is_dir()):
            rev = _git_revision(sub)
            if rev is not None:
                externals[sub.name] = rev
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "robop": _git_revision(repo_root),
        "externals": externals,
        "seed": int(cfg.get("seed", 0)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }


def _json_safe(obj):
    """Recursively convert to strict-JSON values: numpy scalars/arrays become
    Python types, non-finite floats become None (JSON has no NaN/Infinity;
    null is the JSON encoding of a missing value)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_results_json(save_path, results_json: dict, cfg: DictConfig) -> None:
    """Write per-frame results as strict JSON and the resolved config as YAML
    alongside it. allow_nan=False backstops the sanitizer: a non-finite value
    that slips through raises instead of writing non-standard NaN tokens."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        # Compact separators: at the full 32k split an indented results JSON is
        # ~128 MB/model vs ~69 MB compact; readers use json.load, whitespace-agnostic.
        json.dump(_json_safe(results_json), f, separators=(",", ":"), allow_nan=False)
    log.info(f"Saved per-frame results to {save_path}")

    config_path = save_path.with_suffix(".config.yaml")
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f"Saved resolved config to {config_path}")


def save_eval_results(save_path, cfg: DictConfig, *, dataset: str, metrics: dict,
                      model_stats: dict, det_meta, queries: list,
                      extra_summary: Optional[dict] = None) -> None:
    """Assemble and write the standard results JSON for one evaluation run.

    `extra_summary` is spliced between the metrics and the model stats so each
    dataset keeps its established summary key order - results files are diffed
    across runs.
    """
    save_results_json(save_path, {
        "summary": {"dataset": dataset, **metrics, **(extra_summary or {}), **model_stats},
        "detections": det_meta,
        "provenance": run_provenance(cfg),
        "queries": queries,
    }, cfg)
