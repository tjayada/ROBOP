from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nemolib.model import Model

from robop import estimator_utils
from robop.mask_utils import binary_mask_to_rle

from robot_renderer import RobotRenderer, gaussian_blur_chw
from robot_renderer.types import RenderResult

log = logging.getLogger(__name__)


def _rotation_error_rad(R_gt: torch.Tensor, R_pred: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation error in radians. Accepts (3,3) or (N,3,3)."""
    if R_gt.ndim == 2:
        R_gt = R_gt.unsqueeze(0)
    if R_pred.ndim == 2:
        R_pred = R_pred.unsqueeze(0)
    R_diff = torch.matmul(R_pred.transpose(1, 2), R_gt)
    trace_val = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(dim=-1)
    cos_angle = (trace_val - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos_angle)


def rotation_error_normalized(R_gt: torch.Tensor, R_pred: torch.Tensor) -> torch.Tensor:
    """Geodesic rotation error divided by pi - the convention the similarity
    alignment objective below is written in."""
    return _rotation_error_rad(R_gt, R_pred) / torch.pi


# Similarity alignment from NeMO's learned frame to the CAD frame, following the
# paper's supplement 7.4, Algorithm 1.


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber loss, quadratic below delta and linear above."""
    abs_x = torch.abs(x)
    quadratic = torch.minimum(abs_x, torch.tensor(delta, device=x.device, dtype=x.dtype))
    linear = abs_x - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def axis_angle_to_matrix(r_vec: torch.Tensor, differentiable_axis: bool = False) -> torch.Tensor:
    """Rodrigues formula for a (3,) axis-angle vector.

    differentiable_axis=False detaches the skew-symmetric matrix, so gradients
    reach the parameter only through the angle and the axis stays at its
    initialisation, matching the published alignment. True enables the optional
    larger search over the axis as well.
    """
    theta = torch.norm(r_vec)
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    A = torch.where(theta < 1e-9, torch.ones_like(theta), sin_theta / theta)
    B = torch.where(theta < 1e-9, 0.5 * torch.ones_like(theta), (1 - cos_theta) / (theta ** 2))

    kx, ky, kz = r_vec[0], r_vec[1], r_vec[2]
    zero = torch.zeros((), device=r_vec.device, dtype=r_vec.dtype)
    K = torch.stack([
        torch.stack([zero, -kz, ky]),
        torch.stack([kz, zero, -kx]),
        torch.stack([-ky, kx, zero]),
    ])
    if not differentiable_axis:
        K = K.detach()
    return torch.eye(3, device=r_vec.device, dtype=r_vec.dtype) + A * K + B * (K @ K)


def optimize_similarity(
    gt_poses: torch.Tensor,
    est_poses: torch.Tensor,
    k_best: int = 5,
    max_iter: int = 10000,
    lr: float = 0.05,
    delta: float = 0.001,
    differentiable_axis: bool = False,
    seed: Optional[int] = None,
) -> dict:
    """Fit scale, rotation and center offset mapping estimated template poses
    onto their ground truth.

    gt_poses/est_poses are (N,4,4) object-to-camera matrices for the same
    templates. The correction acts in the object frame,
    pose_corrected = est_pose @ [R | t], with translations scaled by s.

    Returns scale, rotation, center_offset and per-template residuals.
    """
    if gt_poses.shape != est_poses.shape or gt_poses.ndim != 3 or gt_poses.shape[1:] != (4, 4):
        raise ValueError(
            f"expected matching (N, 4, 4) pose batches, got {tuple(gt_poses.shape)} "
            f"and {tuple(est_poses.shape)}"
        )
    n = gt_poses.shape[0]
    if n == 0:
        raise ValueError("no template poses to align against")
    if seed is not None:
        torch.manual_seed(seed)

    device = gt_poses.device
    gt_poses = gt_poses.to(torch.float32)
    est_poses = est_poses.to(device=device, dtype=torch.float32)

    # scale starts from the average translation-norm ratio, slightly low so the
    # optimizer approaches it from below
    s_init = torch.mean(
        torch.norm(gt_poses[:, :3, 3], dim=1) / (torch.norm(est_poses[:, :3, 3], dim=1) + 1e-6)
    ) * 0.9
    s_param = torch.nn.Parameter(s_init.reshape(1).clone().to(device))
    r = torch.nn.Parameter(torch.ones(3, device=device) * 1e-2)
    t = torch.nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam([s_param, r, t], lr=lr)

    # keep the templates whose initial rotation is closest to ground truth
    initial_angle_errors = rotation_error_normalized(gt_poses[:, :3, :3], est_poses[:, :3, :3])
    keep = torch.topk(initial_angle_errors, min(k_best, n), largest=False).indices

    best = {"loss": float("inf")}
    for _ in range(max_iter):
        optimizer.zero_grad()
        s_val = torch.abs(s_param)
        R_corr = axis_angle_to_matrix(r, differentiable_axis=differentiable_axis)
        T_corr = torch.eye(4, device=device)
        T_corr[:3, :3] = R_corr
        T_corr[:3, 3] = t

        pred = torch.matmul(est_poses, T_corr.unsqueeze(0))
        angle_errors = rotation_error_normalized(gt_poses[:, :3, :3], pred[:, :3, :3])
        trans_errors = torch.norm(s_val * pred[:, :3, 3] - gt_poses[:, :3, 3], dim=1)

        loss = (huber(angle_errors[keep], delta).mean()
                + huber(trans_errors[keep] / 10.0, delta).mean())
        loss_value = float(loss.detach())
        if loss_value < best["loss"]:
            best = {
                "loss": loss_value,
                "scale": s_val.detach().clone(),
                "rotation": R_corr.detach().clone(),
                "center_offset": t.detach().clone(),
                "angle_errors": angle_errors.detach().clone(),
                "trans_errors": trans_errors.detach().clone(),
            }
        loss.backward()
        optimizer.step()

    if "scale" not in best:
        raise RuntimeError(
            "optimize_similarity did not find a finite loss in "
            f"{max_iter} iterations (loss was NaN or inf throughout). The "
            "alignment inputs are likely degenerate (check the kept template "
            "poses and the learning rate)."
        )

    return {
        "scale": best["scale"],
        "rotation": best["rotation"],
        "center_offset": best["center_offset"],
        "loss": best["loss"],
        "kept_template_ids": keep.detach().cpu().tolist(),
        "initial_rotation_errors_deg": (initial_angle_errors * 180.0).detach().cpu().tolist(),
        "final_rotation_errors_deg": (best["angle_errors"] * 180.0).cpu().tolist(),
        "final_translation_errors": best["trans_errors"].cpu().tolist(),
    }


# The one threshold analysis/analysis_utils.py reads; the key name "thr_0.50" is
# part of the results-JSON contract.
_CONF_STAT_THRESHOLDS: Tuple[float, ...] = (0.50,)

# Bbox extent the sampled surface points are normalised to before encoding: the
# paper scales the NeMO point cloud so the anchor image's surface points "take 1/3
# of the volume [-1, 1]^3" (Finding NeMO, arXiv:2602.04343, supplement 7.3), i.e.
# 1/3 of the range 2. Sets what the encoder sees; the metric scale is recovered
# separately from the encoder's output extent (see _setup_nemo_alignment).
_SURFACE_EXTENT_TARGET = 2.0 / 3.0


def _pnp_pose_estimation_with_diagnostics(
    decoder_output: dict,
    bbox_tensor: torch.Tensor,
    K_array: np.ndarray,
    nemo_scale_factor: float,
    conf_threshold: float,
    iterationsCount: int,
    reprojectionError: float,
    confidence: float,
    min_correspondences: int = 32,
    min_inlier_ratio: float = 0.0,
) -> Tuple[List, dict]:
    """Local replacement for nemolib._pnp_pose_estimation that also returns diagnostics.

    Mirrors nemolib.model._pnp_pose_estimation + _run_pnp_on_multiple_images exactly,
    without modifying the submodule.  Additionally computes per-threshold confidence
    stats and RANSAC inlier / reprojection-error diagnostics.

    Returns
    -------
    poses       : list[np.ndarray | None] - one (4,4) pose per (b*t) image.
    diagnostics : dict with 'conf_stats' and 'pnp' entries for the first image only
                  (b*t == 1 in all current call sites).
    """
    pts3d = decoder_output["pts3d"]  # (b, t, h, w, 3)
    conf  = decoder_output["conf"]   # (b, t, h, w)

    b, t, h, w, _ = pts3d.shape

    # confidence stats over all diagnostic thresholds
    conf_flat = conf.detach().cpu().float().reshape(-1).numpy()
    conf_stats: dict = {}
    for thr in _CONF_STAT_THRESHOLDS:
        above = conf_flat > thr
        count = int(above.sum())
        conf_stats[f"thr_{thr:.2f}"] = {
            "count":     count,
            "mean_conf": round(float(conf_flat[above].mean()), 4) if count > 0 else 0.0,
        }

    # preprocessing (mirrors _pnp_pose_estimation)
    new_xmin = bbox_tensor[..., 0].cpu().numpy()  # (b, t)
    new_ymin = bbox_tensor[..., 1].cpu().numpy()
    new_xmax = bbox_tensor[..., 2].cpu().numpy()
    new_ymax = bbox_tensor[..., 3].cpu().numpy()
    new_width = new_xmax - new_xmin
    assert (np.abs(new_width - (new_ymax - new_ymin)) < 1e-3).all(), \
        "Bounding box is not rectangular!"
    bbox_scale = w / new_width  # (b, t)

    conf_2d_3d = rearrange(conf.cpu().numpy(), "b t h w -> (b t) (h w)")
    indices    = np.repeat(np.indices((h, w))[np.newaxis, ...], b * t, axis=0)
    points_2d  = rearrange(indices, "bt c h w -> bt (h w) c").astype(np.float32)
    points_2d  = points_2d[..., [1, 0]]           # (row, col) -> (x, y)
    points_2d[..., 0] += (new_xmin * bbox_scale).reshape(-1, 1)
    points_2d[..., 1] += (new_ymin * bbox_scale).reshape(-1, 1)

    points_3d = (
        rearrange(pts3d, "b t h w c -> (b t) (h w) c").cpu().numpy().astype(np.float32)
    )
    points_3d *= float(nemo_scale_factor)

    K_bt     = K_array.reshape(-1, 3, 3)
    K_scaled = K_bt * bbox_scale.reshape(-1, 1, 1)
    K_scaled[:, 2, 2] = 1.0

    valid = conf_2d_3d > conf_threshold  # (b*t, h*w) bool

    # PnP loop (mirrors _run_pnp_on_multiple_images)
    poses: List[Optional[np.ndarray]] = []
    pnp_stats_list: List[Optional[dict]] = []

    for i in range(b * t):
        valid_2d   = points_2d[i][valid[i]]
        valid_3d   = points_3d[i][valid[i]]
        total_corr = int(valid[i].sum())

        if total_corr < min_correspondences:
            poses.append(None)
            pnp_stats_list.append(None)
            continue

        K_i = K_scaled[i]
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                valid_3d, valid_2d, K_i, None,
                flags=cv2.SOLVEPNP_SQPNP,
                iterationsCount=iterationsCount,
                reprojectionError=reprojectionError,
                confidence=confidence,
            )
        except Exception as exc:
            log.warning("PnP raised: %s", exc)
            success = False

        if not success or inliers is None:
            poses.append(None)
            pnp_stats_list.append(None)
            continue

        inlier_idx   = inliers.flatten()
        inlier_count = len(inlier_idx)

        if min_inlier_ratio > 0.0 and inlier_count / max(total_corr, 1) < min_inlier_ratio:
            poses.append(None)
            pnp_stats_list.append({
                "total_correspondences": total_corr,
                "inlier_count":          inlier_count,
                "inlier_ratio":          round(inlier_count / max(total_corr, 1), 4),
                "mean_reproj_error_px":  None,
                "rejected_min_inlier_ratio": True,
            })
            continue

        proj_pts, _ = cv2.projectPoints(valid_3d[inlier_idx], rvec, tvec, K_i, None)
        mean_reproj = float(
            np.linalg.norm(proj_pts.reshape(-1, 2) - valid_2d[inlier_idx], axis=1).mean()
        )

        estT = np.eye(4)
        estT[:3, :3] = cv2.Rodrigues(rvec)[0]
        estT[:3, 3]  = tvec.flatten()
        poses.append(estT)
        pnp_stats_list.append({
            "total_correspondences": total_corr,
            "inlier_count":          inlier_count,
            "inlier_ratio":          round(inlier_count / max(total_corr, 1), 4),
            "mean_reproj_error_px":  round(mean_reproj, 4),
        })

    diagnostics = {
        "conf_stats": conf_stats,
        "pnp":        pnp_stats_list[0] if pnp_stats_list else None,
    }
    return poses, diagnostics


def _farthest_point_sampling(points: torch.Tensor, n: int) -> torch.Tensor:
    V = points.shape[0]
    if n >= V:
        return torch.arange(V, device=points.device, dtype=torch.long)
    selected = torch.empty(n, dtype=torch.long, device=points.device)
    centroid = points.mean(dim=0)
    current = int(torch.argmin(torch.sum((points - centroid) ** 2, dim=1)).item())
    selected[0] = current
    distances = torch.sum((points - points[current]) ** 2, dim=1)
    for i in range(1, n):
        current = int(torch.argmax(distances).item())
        selected[i] = current
        dist_to_new = torch.sum((points - points[current]) ** 2, dim=1)
        distances = torch.minimum(distances, dist_to_new)
    return selected


@dataclass
class NeMORobotConfig:
    """
    NeMO inference parameters.

    Rendering parameters (render_size, viewset, num_views, orientation, etc.)
    live in robot_renderer.ViewConfig and are passed to RobotRenderer at init.

    These defaults are the detector-free demo regime, so the README quick-start
    runs without a detection. The benchmark regime is configs/estimator/nemo.yaml,
    which every eval script loads.
    """
    # Confidence threshold for decoder xyz predictions. Range is checkpoint-dependent
    # (this one: exact 0 for background, up to ~1.6 for foreground).
    conf_threshold: float = 0.5
    # Threshold on the decoder's binary segmentation mask output (sigmoid, range [0, 1]).
    mask_threshold: float = 0.5
    # center uses a square center crop; bbox requires a detection; bootstrap uses
    # the supplement 7.5 patch scan and reruns on its largest decoded component.
    crop_mode: str = "center"
    # crop_mode="bootstrap" knobs.
    bootstrap_mask_pad: float = 0.1        # relative pad added around the mask bbox before squarify
    bootstrap_min_mask_pixels: int = 64    # min foreground px to trust the mask; else fall back to center
    # Relative pad around the detection bbox before squarification (crop_mode="bbox"),
    # guarding against a tight box clipping the arm (FoundPose uses 0.2 for this).
    # Counter-indicated where the arm already fills the frame: the square then runs
    # past the image edge.
    bbox_crop_pad: float = 0.0
    # How the learned NeMO frame is aligned to the metric CAD frame:
    #   "extent" - measured from the known metric URDF mesh extent.
    #   "bundle" - the paper's fitted similarity alignment (supplement 7.4, Algorithm 1).
    alignment_method: str = "extent"
    alignment_k_best: int = 5
    alignment_max_iter: int = 10000
    alignment_lr: float = 0.05
    alignment_huber_delta: float = 0.001
    # The published optimizer rebuilds the rotation's skew matrix as a constant,
    # so only the angle about the initial axis is optimized. True frees the axis,
    # which optimizes more than the published method does.
    alignment_differentiable_axis: bool = False
    use_mesh_surface_sampling: bool = True
    seed: int = 0
    include_query_in_templates: bool = True
    use_blurred_query_background: bool = False
    blur_sigma: float = 2.0
    # Encoding
    nemo_encoding_size: int = 224
    # Initial PnP
    pnp_min_correspondences: int = 32
    # Minimum fraction of valid correspondences that must be RANSAC inliers.
    # 0.0 = accept any result (original behaviour). The BOP evaluation uses 0.3.
    pnp_min_inlier_ratio: float = 0.0
    pnp_initial_iterations: int = 10_000
    pnp_initial_reproj_error: float = 10.0
    pnp_initial_confidence: float = 0.99999
    # Number of surface points sampled from the mesh for NeMO encoding.
    num_sample_points: int = 1500
    # Debug: save decoder output image and stop after the first frame.
    # Set to a directory path (e.g. "debug_decoder") to enable, null to disable.
    debug_save_decoder: Optional[str] = None
    # Debug: per-failure dump (query crop, confidence heatmap, full image with the
    # detection box, and a stats sidecar) for every frame whose PnP returns None.
    # Does not stop the run. null = disabled.
    debug_save_failures: Optional[str] = None


class NeMORobotPoseEstimator(nn.Module):
    """
    NeMO-based robot pose estimator.

    Args:
        nemo_model: Trained NeMO Model instance.
        renderer:   Initialised RobotRenderer (already configured with ViewConfig,
                    intrinsics, and robot).  Call renderer.to(device) before passing.
        config:     NeMORobotConfig for inference parameters.
        device:     Torch device.  Defaults to CUDA if available.
    """

    # Input contract read by evaluation's EstimatorAdapter: NeMO consumes an
    # optional detection bbox; the modal mask is predicted internally.
    INPUT_CONTRACT = {"mask": False, "bbox_xyxy": True, "frame_key": False,
                      "mask_3x3_opening": False}

    def __init__(
        self,
        nemo_model: Model,
        renderer: RobotRenderer,
        config: Optional[NeMORobotConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.nemo_model = nemo_model.to(self.device)
        self.nemo_model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)

        self.renderer = renderer.to(self.device)
        self.config = config or NeMORobotConfig()
        if self.config.alignment_method not in ("extent", "bundle"):
            raise ValueError(
                f"alignment_method must be 'extent' or 'bundle', "
                f"got {self.config.alignment_method!r}"
            )
        if self.config.crop_mode not in ("bootstrap", "bbox", "center"):
            raise ValueError(
                f"crop_mode must be 'bootstrap', 'bbox' or 'center', "
                f"got {self.config.crop_mode!r}"
            )

        # center_offset in metric units (computed by _setup_nemo_alignment).
        # Used in _apply_alignment_to_pose to correct the O-matrix translation.
        self._nemo_center_offset_metric: Optional[torch.Tensor] = None

        # Scale factor derived from mesh surface sampling (set by _sample_surface_points_from_mesh).
        # None means fall back to _nemo_scale_factor.
        self._mesh_surface_scale: Optional[float] = None
        # Raw metric bbox extent of the sampled surface points (set during sampling).
        # _cad_size_metric is the anchor-frame extent (used for input normalisation);
        # _cad_size_metric_aligned is the same sample's extent in the aligned-mesh frame
        # (the geo_scale-consistent one). Per-axis extents are kept for anisotropy diag.
        self._cad_size_metric: Optional[float] = None
        self._cad_size_metric_aligned: Optional[float] = None
        self._cad_axis_extent_anchor: Optional[Tuple[float, float, float]] = None
        self._cad_axis_extent_aligned: Optional[Tuple[float, float, float]] = None
        # Fallback scale from anchor bbox (computed from RenderResult in estimate_pose_from_image).
        self._nemo_scale_factor: float = 1.0

        # Populated each call with confidence stats and PnP diagnostics.
        self._last_frame_diagnostics: Optional[dict] = None
        self._debug_call_idx: int = 0
        self._frame_counter: int = 0   # global per-call index, for failure-dump filenames
        # Populated by _setup_nemo_alignment with anchor PnP diagnostics (rotation/translation error).
        self._last_alignment_diagnostics: Optional[dict] = None

        # (crop_x, crop_y, crop_w, crop_h) set by _preprocess_query_np each call.
        # Used by _prepare_pnp_inputs to adjust the K matrix for any spatial crop.
        self._last_crop_params: Optional[Tuple[int, int, int, int]] = None

        # Bbox (xyxy, original-image px) of the decoder's modal mask from the last
        # decode, mapped back through the query crop. Populated every frame; consumed
        # by crop_mode="bootstrap" pass 1 to localise the arm without a detector.
        self._last_decoder_mask_bbox_orig: Optional[np.ndarray] = None
        # Bootstrap pass-1 evidence, kept so a bad second-pass crop is explainable.
        self._last_bootstrap_components: Optional[dict] = None

    @property
    def robot_kin(self):
        """Expose kinematics for FK-based external evaluators."""
        return self.renderer._kinematics

    def estimate_pose_from_image(
        self,
        image: Union[str, torch.Tensor],
        joint_angles: Union[np.ndarray, torch.Tensor, Sequence[float]],
        K: Union[np.ndarray, torch.Tensor],
        bbox_xyxy: Optional[Union[np.ndarray, torch.Tensor, Sequence[float]]] = None,
        num_sample_points: Optional[int] = None,
        use_mesh_surface_sampling: Optional[bool] = None,
        _crop_mode_override: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """
        Full NeMO pipeline for a single real image and joint configuration.

        Returns:
            (4, 4) base-to-camera transform in URDF frame, or None if PnP fails.
        """
        if num_sample_points is None:
            num_sample_points = self.config.num_sample_points

        # Resolve the crop mode for THIS call. `_crop_mode_override` is the internal
        # channel the "bootstrap" two-pass uses to drive its sub-calls with concrete modes.
        crop_mode = _crop_mode_override or self.config.crop_mode

        # Self-bootstrap: detector-free localisation via the decoder mask
        # Scan square patches to find the arm, then re-crop it (Finding NeMO,
        # arXiv:2602.04343, supplement 7.5). Only the entry call dispatches; the
        # sub-calls below pass concrete modes and skip this.
        if crop_mode == "bootstrap":
            scan = None
            if bbox_xyxy is None:
                scan = self._bootstrap_scan(
                    image, joint_angles, K, num_sample_points, use_mesh_surface_sampling,
                )
                bbox_xyxy = scan["bbox"]
                if bbox_xyxy is None:
                    # No patch found the arm: the scan's own first pose is the result.
                    return scan["pose"]
            out = self.estimate_pose_from_image(
                image, joint_angles, K, bbox_xyxy=bbox_xyxy,
                num_sample_points=num_sample_points,
                use_mesh_surface_sampling=use_mesh_surface_sampling,
                _crop_mode_override="bbox",
            )
            # Each sub-call overwrites _last_frame_diagnostics, so the scan's cost
            # has to be folded back in; runtime is a headline benchmark metric.
            self._merge_bootstrap_pass_diagnostics(scan)
            return out
        estimator_utils.reseed_all(self.config.seed)

        device = self.device
        _frame_idx = self._frame_counter
        self._frame_counter += 1
        _t_wall = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def _sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        imgs = self._load_query_image(
            image, self.config.nemo_encoding_size, device,
            crop_mode=crop_mode,
            bbox_xyxy=bbox_xyxy,
        )
        images_tensor = rearrange(imgs, "t c h w -> 1 t c h w")  # (1,1,C,H,W)

        background_image = None
        if self.config.use_blurred_query_background:
            query_chw = imgs[0].to(device)
            blurred = gaussian_blur_chw(query_chw, sigma=self.config.blur_sigma)
            render_size = self.renderer.config.render_size
            if blurred.shape[-2] != render_size or blurred.shape[-1] != render_size:
                blurred = F.interpolate(
                    blurred.unsqueeze(0),
                    size=(render_size, render_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            background_image = blurred.permute(1, 2, 0)  # (H, W, 3)
        _sync()
        _t_a = time.perf_counter()  # end of preprocess phase

        result = self.renderer.render_templates(
            joint_angles, background_image=background_image
        )
        gt_poses = [v.gt_pose for v in result.views]

        # Compute fallback NeMO scale from anchor bbox (used by _setup_nemo_alignment).
        self._nemo_scale_factor = self._compute_anchor_bbox_scale(result)

        templates_hr = torch.stack([v.image for v in result.views], dim=0)  # (T,3,H,W) on device
        # antialias=True matches the query downsample (_load_query_image): both feed the
        # same encoder, so they must be filtered identically. Also matches nemolib's
        # released image_to_tensor; the training loader's no-AA grid_sample does not.
        templates = F.interpolate(
            templates_hr,
            size=(self.config.nemo_encoding_size, self.config.nemo_encoding_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0.0, 1.0)

        if self.config.include_query_in_templates:
            query_t = images_tensor[0].to(templates.device)  # (1,C,H,W)
            templates_for_nemo = torch.cat([templates, query_t], dim=0)
        else:
            templates_for_nemo = templates
        _sync()
        _t_b = time.perf_counter()  # end of render phase

        self._mesh_surface_scale = None  # reset before each call
        use_mesh_surface_sampling = (
            self.config.use_mesh_surface_sampling
            if use_mesh_surface_sampling is None
            else use_mesh_surface_sampling
        )
        nemo = self._build_nemo_from_templates(
            templates_for_nemo, num_sample_points, result,
            use_mesh_surface_sampling=use_mesh_surface_sampling,
        )
        _sync()

        self._setup_nemo_alignment(
            nemo=nemo, templates=templates, gt_poses=gt_poses, result=result,
        )
        _sync()
        _t_c = time.perf_counter()  # end of encode phase (build + alignment; may include anchor PnP diagnostic)

        with torch.no_grad():
            decoder_output = self.nemo_model.decode_images(images_tensor, nemo["features_3d_updated"])

        if "mask" in decoder_output and decoder_output["mask"] is not None:
            decoder_output["conf"] = (
                decoder_output["conf"].clone() * (decoder_output["mask"] > self.config.mask_threshold)
            )

        _sync()
        _t_d = time.perf_counter()  # end of decode phase

        # Localise the arm from the decoder's own output (modal mask, else confidence),
        # mapped back to original-image px. Consumed by crop_mode="bootstrap" pass 1.
        self._last_decoder_mask_bbox_orig = self._decoder_mask_bbox_orig(decoder_output)

        if self.config.debug_save_decoder is not None:
            from nemolib.visualization import get_prediction_image_from_decoder_output
            os.makedirs(self.config.debug_save_decoder, exist_ok=True)
            frame_id = f"{self._debug_call_idx:06d}"
            self._debug_call_idx += 1

            # Decoder output
            dec_vis = get_prediction_image_from_decoder_output(images_tensor, decoder_output)[0]
            cv2.imwrite(
                os.path.join(self.config.debug_save_decoder, f"decoder_{frame_id}.png"),
                cv2.cvtColor(dec_vis, cv2.COLOR_RGB2BGR),
            )

            # Template grid - save each rendered view as an individual image
            tpl_dir = os.path.join(self.config.debug_save_decoder, f"templates_{frame_id}")
            os.makedirs(tpl_dir, exist_ok=True)
            for ti, tpl in enumerate(templates_hr):
                # tpl: (3, H, W) float [0,1] on device
                tpl_np = (tpl.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(tpl_dir, f"template_{ti:03d}.png"),
                    cv2.cvtColor(tpl_np, cv2.COLOR_RGB2BGR),
                )

            log.info(f"[debug] Saved decoder + {len(templates_hr)} templates to {self.config.debug_save_decoder}")
            raise RuntimeError(f"debug_save_decoder: stopping after frame {self._debug_call_idx - 1}")

        H_img, W_img = imgs.shape[-2], imgs.shape[-1]
        # The PnP box is always the full decoder frame: _load_query_image stores the
        # active crop in _last_crop_params and _prepare_pnp_inputs replays it into K,
        # so the crop geometry is fully accounted for in EVERY crop mode.
        bbox_tensor, K_array, K_np = self._prepare_pnp_inputs(
            K, H_img, W_img, self.config.nemo_encoding_size, device
        )

        cv2.setRNGSeed(self.config.seed)
        pose_estimations, _pnp_diag = _pnp_pose_estimation_with_diagnostics(
            decoder_output, bbox_tensor, K_array,
            nemo_scale_factor=float(nemo["scale_factor"].item()),
            conf_threshold=self.config.conf_threshold,
            iterationsCount=self.config.pnp_initial_iterations,
            reprojectionError=self.config.pnp_initial_reproj_error,
            confidence=self.config.pnp_initial_confidence,
            min_correspondences=self.config.pnp_min_correspondences,
            min_inlier_ratio=self.config.pnp_min_inlier_ratio,
        )
        _t_done = time.perf_counter()
        _gpu_peak_mb = (
            round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
            if device.type == "cuda" else None
        )
        _timings_ms = {
            "preprocess": round((_t_a - _t_wall) * 1000, 1),
            "render":     round((_t_b - _t_a)    * 1000, 1),
            "encode":     round((_t_c - _t_b)    * 1000, 1),
            "decode":     round((_t_d - _t_c)    * 1000, 1),
            "pnp":        round((_t_done - _t_d)  * 1000, 1),
            "total":      round((_t_done - _t_wall) * 1000, 1),
        }

        def _finalize_timing() -> None:
            """Close the timing at the real end of the call. The phases above stop
            after PnP, so mask serialization, the IoU diagnostic and the final pose
            composition would otherwise be missing from the reported runtime."""
            now = time.perf_counter()
            d = self._last_frame_diagnostics
            d["timings_ms"]["postprocess"] = round((now - _t_done) * 1000, 1)
            d["timings_ms"]["total"] = round((now - _t_wall) * 1000, 1)
            d["frame_time_ms"] = d["timings_ms"]["total"]

        self._last_frame_diagnostics = {
            **_pnp_diag,
            "alignment":    self._last_alignment_diagnostics,
            "timings_ms":   _timings_ms,
            "frame_time_ms": _timings_ms["total"],  # common interface key shared across estimator adapters
            "gpu_peak_mb":  _gpu_peak_mb,
        }

        _modal = decoder_output.get("mask")
        _amodal = decoder_output.get("mask_full")
        if _modal is not None:
            self._last_frame_diagnostics["modal_mask_rle"] = binary_mask_to_rle(
                (_modal[0, 0].detach().cpu().float().numpy() > self.config.mask_threshold).astype(np.uint8)
            )
        if _amodal is not None:
            self._last_frame_diagnostics["amodal_mask_rle"] = binary_mask_to_rle(
                (_amodal[0, 0].detach().cpu().float().numpy() > self.config.mask_threshold).astype(np.uint8)
            )

        if not pose_estimations or pose_estimations[0] is None:
            if self.config.debug_save_failures is not None:
                self._dump_failure(
                    self.config.debug_save_failures, _frame_idx, image, imgs,
                    decoder_output, bbox_xyxy,
                )
            _finalize_timing()
            return None

        T_chi_c = torch.tensor(pose_estimations[0], dtype=torch.float32, device=device)

        self._last_frame_diagnostics["iou"] = self._compute_mask_iou_diagnostics(
            T_chi_c, decoder_output, bbox_tensor, K_array, result, nemo
        )

        out = self._apply_alignment_to_pose(T_chi_c, result)
        _finalize_timing()
        return out

    def inference_single_image(
        self,
        img: torch.Tensor,
        joint_angles: torch.Tensor,
        K=None,
        bbox_xyxy=None,
        use_mesh_surface_sampling: Optional[bool] = None,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None]:
        """CtRNet-compatible wrapper around estimate_pose_from_image."""
        if K is None:
            raise ValueError("K (camera intrinsics) is required.")
        if isinstance(joint_angles, torch.Tensor) and joint_angles.ndim == 2:
            joint_angles = joint_angles[0]
        pose = self.estimate_pose_from_image(
            image=img,
            joint_angles=joint_angles,
            K=K,
            bbox_xyxy=bbox_xyxy,
            use_mesh_surface_sampling=use_mesh_surface_sampling,
        )
        return pose, None, None, None, None

    @staticmethod
    def _image_hw(image) -> Tuple[int, int]:
        """(H, W) of a query image given as a path or an array/tensor."""
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read query image: {image}")
            return int(img.shape[0]), int(img.shape[1])
        return int(image.shape[-2]), int(image.shape[-1])

    def _bootstrap_patch_origins(self, h: int, w: int) -> List[Tuple[int, int]]:
        """Square patch origins for the detector-free scan: side = smallest image
        side, stride 1/3 of it (supplement 7.5), plus a flush-edge patch so the scan
        covers the full frame (the stride grid alone leaves a strip uncovered)."""
        s = min(h, w)
        stride = max(1, s // 3)
        xs = list(range(0, w - s + 1, stride))
        ys = list(range(0, h - s + 1, stride))
        if xs[-1] != w - s:
            xs.append(w - s)
        if ys[-1] != h - s:
            ys.append(h - s)
        return [(x, y) for y in ys for x in xs]

    def _bootstrap_scan(self, image, joint_angles, K, num_sample_points,
                        use_mesh_surface_sampling) -> dict:
        """Decode every patch and keep the largest foreground component.

        Supplement 7.5 accumulates patch predictions and clusters them to separate
        instances; with one arm per frame that reduces to the largest component.
        The paper's second, half-size patch scale is skipped: it exists to find
        small or multiple objects, and the arm spans a large part of the frame.
        """
        h, w = self._image_hw(image)
        s = min(h, w)
        scan: dict = {"pose": None, "bbox": None, "components": None,
                      "crop_params": None, "diags": []}
        best_area = -1
        for i, (x0, y0) in enumerate(self._bootstrap_patch_origins(h, w)):
            pose = self.estimate_pose_from_image(
                image, joint_angles, K,
                bbox_xyxy=(float(x0), float(y0), float(x0 + s), float(y0 + s)),
                num_sample_points=num_sample_points,
                use_mesh_surface_sampling=use_mesh_surface_sampling,
                _crop_mode_override="bbox",
            )
            scan["diags"].append(self._last_frame_diagnostics)
            if i == 0:
                scan["pose"] = pose   # fallback when no patch localises the arm
            if self._last_decoder_mask_bbox_orig is None:
                continue
            area = int((self._last_bootstrap_components or {}).get("selected_area", 0))
            if area > best_area:
                best_area = area
                scan["bbox"] = self._last_decoder_mask_bbox_orig
                scan["components"] = self._last_bootstrap_components
                scan["crop_params"] = self._last_crop_params
        return scan

    def _merge_bootstrap_pass_diagnostics(self, scan: Optional[dict]) -> None:
        """Fold the scan's cost into the final (bbox-pass) diagnostics.

        Accuracy fields (conf_stats, pnp, alignment, iou, masks) stay the final
        pass's - that is the pose that gets returned. Only cost is combined: phase
        timings summed over every patch, GPU peak the max. No-op when the scan was
        skipped (external detection bbox supplied).
        """
        diag = self._last_frame_diagnostics
        if diag is None or not scan or not scan["diags"]:
            return
        diags = [d for d in scan["diags"] if d] + [diag]
        keys = set().union(*(set(d.get("timings_ms", {})) for d in diags))
        if keys:
            merged = {
                k: round(sum(d.get("timings_ms", {}).get(k, 0.0) for d in diags), 1)
                for k in keys
            }
            diag["timings_ms"] = merged
            diag["frame_time_ms"] = merged.get("total", diag.get("frame_time_ms"))
        gpu = [d["gpu_peak_mb"] for d in diags if d.get("gpu_peak_mb") is not None]
        if gpu:
            diag["gpu_peak_mb"] = max(gpu)
        diag["bootstrap_passes"] = len(diags)
        # Enough of the scan to reconstruct a failed localisation from the result
        # file: without it, a bad final crop cannot be explained. These are the
        # WINNING patch's values, captured during the scan (the instance attributes
        # hold the final pass's by the time this runs).
        first = scan["diags"][0] or {}
        diag["bootstrap_scan"] = {
            "n_patches": len(scan["diags"]),
            "conf_stats": first.get("conf_stats"),
            "pnp": first.get("pnp"),
            "mask_components": scan["components"],
            "localized_bbox_xyxy": (
                [round(float(v), 2) for v in scan["bbox"]]
                if scan["bbox"] is not None else None
            ),
            "crop_params": list(scan["crop_params"] or ()) or None,
        }

    def _compute_anchor_bbox_scale(self, result: RenderResult) -> float:
        """Fallback NeMO scale from the anchor-view bbox: max_dim /
        _SURFACE_EXTENT_TARGET. Used when the encoder reports no usable
        extent; inverts the normalisation applied in _sample_surface_points_from_mesh."""
        verts = result.mesh_verts
        center = result.orientation_center.to(device=verts.device, dtype=torch.float32)
        R_anchor = result.anchor_gt_pose[:3, :3].to(device=verts.device, dtype=torch.float32)
        verts_centered = verts - center
        verts_anchor_cam = (R_anchor @ verts_centered.T).T
        bbox_cam = verts_anchor_cam.max(dim=0)[0] - verts_anchor_cam.min(dim=0)[0]
        return float(bbox_cam.max().item()) / _SURFACE_EXTENT_TARGET

    def _build_nemo_from_templates(
        self,
        templates: torch.Tensor,
        num_sample_points: int,
        result: RenderResult,
        use_mesh_surface_sampling: bool = True,
    ) -> dict:
        """Build the NeMO representation from template views (nemolib
        Model.encode_images), then express its surface points in the aligned-mesh
        frame the pose chain uses."""
        device = self.device
        templates = templates.to(device)
        images_tensor = rearrange(templates, "t c h w -> 1 t c h w")

        if use_mesh_surface_sampling:
            sample_points = self._sample_surface_points_from_mesh(num_sample_points, result)
        else:
            sample_points = torch.rand(1, num_sample_points, 3, device=device) * 2 - 1

        with torch.no_grad():
            nemo = self.nemo_model.encode_images(images_tensor, sample_points)

            # Rotate surface_points from anchor camera frame back to aligned-mesh frame.
            # features_3d_updated is intentionally NOT set here - _setup_nemo_alignment
            # subtracts center_offset and builds the final features_3d_updated immediately after.
            R_anchor = result.anchor_gt_pose[:3, :3].to(device=device, dtype=torch.float32)
            sp = nemo["surface_points"]
            nemo["surface_points"] = (R_anchor.T @ sp[0].T).T.unsqueeze(0)
            nemo["scale_factor"] = torch.tensor([1.0], device=device, dtype=torch.float32)

        return nemo

    def _sample_surface_points_from_mesh(
        self,
        num_sample_points: int,
        result: RenderResult,
    ) -> torch.Tensor:
        """
        Sample surface points from the rendered, aligned robot mesh.

        FPS on the CAD surface is the paper's own better-performing branch
        (Finding NeMO, arXiv:2602.04343, supplement 7.6, Tab. 7/8) and is available
        to us because FK gives direct mesh access. Uses visibility-filtered face
        centroids (pooled across all template views) as the pool when available,
        falling back to all mesh vertices; points are centred, rotated to the anchor
        camera frame, and scaled to NeMO space.

        Returns:
            (1, N, 3) float tensor on self.device, in anchor camera frame, with the
            max bbox extent normalised to _SURFACE_EXTENT_TARGET.
        """
        use_vis = (
            result.visible_surface_points is not None
            and result.visible_surface_points.shape[0] > 0
        )
        if use_vis:
            verts = result.visible_surface_points.to(dtype=torch.float32, device=self.device)
        else:
            verts = result.mesh_verts.to(dtype=torch.float32, device=self.device)

        V = verts.shape[0]
        if V == 0:
            return torch.zeros(1, num_sample_points, 3, device=self.device, dtype=torch.float32)

        idx = _farthest_point_sampling(verts.to(dtype=torch.float32), num_sample_points)
        points = verts[idx].to(dtype=torch.float32)

        centroid = result.orientation_center.to(dtype=torch.float32, device=points.device)
        points = points - centroid

        # Aligned-mesh-frame extent (BEFORE the anchor rotation). This is the frame the
        # encoder output surface_points live in after they are rotated back, so it is the
        # geo_scale-consistent numerator. Measured on the same sample as the anchor extent.
        ext_aligned = points.max(dim=0).values - points.min(dim=0).values
        cad_size_aligned = ext_aligned.max()
        self._cad_axis_extent_aligned = tuple(float(v) for v in ext_aligned.tolist())

        R_anchor = result.anchor_gt_pose[:3, :3].to(device=points.device, dtype=torch.float32)
        points = (R_anchor @ points.T).T

        ext_anchor = points.max(dim=0).values - points.min(dim=0).values
        cad_size = ext_anchor.max()   # anchor frame - used for input normalisation
        self._cad_axis_extent_anchor = tuple(float(v) for v in ext_anchor.tolist())
        if cad_size > 1e-8:
            self._mesh_surface_scale = float(cad_size.item()) / _SURFACE_EXTENT_TARGET
            # Raw metric extent of the sampled points, before normalisation. Used by
            # _setup_nemo_alignment to compute the metres-per-NeMO-unit scale from the
            # encoder's *actual* output extent rather than assuming the input target.
            self._cad_size_metric = float(cad_size.item())
            self._cad_size_metric_aligned = float(cad_size_aligned.item())
            points = points / cad_size * _SURFACE_EXTENT_TARGET
        else:
            self._mesh_surface_scale = self._nemo_scale_factor
            self._cad_size_metric = None
            self._cad_size_metric_aligned = None

        return points.unsqueeze(0)

    def _template_camera(self, H: int, W: int):
        """Template-view K (scaled to the model's input size) and full-frame
        bbox, shared by the bundle-alignment fit and the anchor-PnP error check."""
        s_img = float(H) / float(self.renderer.config.render_size)
        K_model = self.renderer._K.to(device=self.device, dtype=torch.float32).clone()
        K_model[0, 0] *= s_img
        K_model[1, 1] *= s_img
        K_model[0, 2] *= s_img
        K_model[1, 2] *= s_img
        bbox = torch.tensor([0.0, 0.0, float(W), float(H)], device=self.device)
        return K_model, bbox

    def _fit_bundle_alignment(self, nemo: dict, templates: torch.Tensor, gt_poses: List[torch.Tensor]):
        """Fit the paper's similarity alignment on the template views.

        Estimates a pose for every template from the current representation,
        then optimizes scale, rotation and center offset against the template
        ground-truth poses (supplement section 7.4, algorithm 1). Returns None
        when too few templates yield a pose.
        """
        n_t, _, H, W = templates.shape
        if len(gt_poses) < n_t:
            log.warning("NeMO alignment: %d templates but %d ground-truth poses - "
                        "skipping bundle alignment.", n_t, len(gt_poses))
            return None
        if bool(getattr(self.renderer.config, "fill_frame", False)):
            raise ValueError(
                "alignment_method='bundle' needs template images that match the "
                "template camera, but fill_frame crops and resizes them. Set "
                "estimator.fill_frame=false for the bundle alignment."
            )

        K_model, bbox = self._template_camera(H, W)
        K_array = K_model.expand(1, n_t, 3, 3).detach().cpu().numpy()
        bbox_tensor = bbox.expand(1, n_t, 4)

        with torch.no_grad():
            decoded = self.nemo_model.decode_images(
                rearrange(templates.to(device=self.device, dtype=torch.float32),
                          "t c h w -> 1 t c h w"),
                nemo["features_3d_updated"],
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        cv2.setRNGSeed(self.config.seed)
        poses, _ = _pnp_pose_estimation_with_diagnostics(
            decoded, bbox_tensor, K_array,
            nemo_scale_factor=1.0,
            conf_threshold=self.config.conf_threshold,
            iterationsCount=self.config.pnp_initial_iterations,
            reprojectionError=self.config.pnp_initial_reproj_error,
            confidence=self.config.pnp_initial_confidence,
            min_correspondences=self.config.pnp_min_correspondences,
            min_inlier_ratio=self.config.pnp_min_inlier_ratio,
        )

        # Ground truth is the template pose of the mesh centroid, which is where
        # the origin of the learned frame sits.
        est_list, gt_list, template_ids = [], [], []
        for i, pose in enumerate(poses):
            if pose is None:
                continue
            gt = gt_poses[i].to(device=self.device, dtype=torch.float32)
            est_list.append(torch.as_tensor(pose, dtype=torch.float32, device=self.device))
            gt_list.append(gt)
            template_ids.append(i)

        if len(est_list) < 2:
            log.warning("NeMO alignment: only %d template poses recovered - "
                        "skipping bundle alignment.", len(est_list))
            self._last_alignment_diagnostics["bundle"] = {
                "fitted": False, "n_template_poses": len(est_list),
            }
            return None

        result = optimize_similarity(
            torch.stack(gt_list), torch.stack(est_list),
            k_best=self.config.alignment_k_best,
            max_iter=self.config.alignment_max_iter,
            lr=self.config.alignment_lr,
            delta=self.config.alignment_huber_delta,
            differentiable_axis=self.config.alignment_differentiable_axis,
            seed=self.config.seed,
        )
        self._last_alignment_diagnostics["bundle"] = {
            "fitted": True,
            "n_template_poses": len(est_list),
            "template_ids": template_ids,
            "kept_template_ids": [template_ids[i] for i in result["kept_template_ids"]],
            "scale": round(float(result["scale"]), 6),
            "rotation": [[round(float(v), 6) for v in row] for row in result["rotation"].tolist()],
            "center_offset": [round(float(v), 6) for v in result["center_offset"].tolist()],
            "loss": round(result["loss"], 8),
            "initial_rotation_errors_deg": [round(v, 3) for v in result["initial_rotation_errors_deg"]],
            "final_rotation_errors_deg": [round(v, 3) for v in result["final_rotation_errors_deg"]],
            "final_translation_errors_m": [round(v, 6) for v in result["final_translation_errors"]],
        }
        log.info("NeMO bundle alignment: scale=%.5f from %d/%d template poses (loss %.3e).",
                 float(result["scale"]), len(est_list), n_t, result["loss"])
        return result

    def _setup_nemo_alignment(
        self,
        nemo: dict,
        templates: torch.Tensor,
        gt_poses: List[torch.Tensor],
        result: RenderResult,
    ) -> None:
        """Compute scale + center-offset alignment and mutate nemo in-place.

        The known template mesh provides metric scale directly rather than through
        fitting (the paper's Algorithm 1 fit is for the
        BOP setting, where the GT object frame is unknown; supplement 7.4).

        After this call nemo["surface_points"] has center_offset subtracted (bbox
        centre at 0), nemo["scale_factor"] holds the metric scale, and
        nemo["features_3d_updated"] is rebuilt.
        """
        device = self.device
        _, _, H, W = templates.shape
        self._last_alignment_diagnostics = None

        # Center offset: bbox centre of the encoder's surface_points output.
        sp = nemo["surface_points"][0]
        center_offset = (sp.min(dim=0).values + sp.max(dim=0).values) / 2.0

        # Metres per NeMO unit, measured against the encoder's OUTPUT extent, not the
        # _SURFACE_EXTENT_TARGET input normalisation: the UDF contracts the points by a
        # few percent, which would otherwise show up as a constant depth bias.
        sp_axis_extent = (sp.max(dim=0).values - sp.min(dim=0).values)
        sp_extent = float(sp_axis_extent.max().item())
        # cad_aligned shares sp_extent's frame, so geo_scale is a pure scale;
        # cad_anchor feeds the diagnostics only.
        cad_anchor  = self._cad_size_metric
        cad_aligned = self._cad_size_metric_aligned
        cad_used = cad_aligned
        if cad_used is not None and sp_extent > 1e-8:
            geo_scale = cad_used / sp_extent
        else:
            geo_scale = (
                self._mesh_surface_scale
                if self._mesh_surface_scale is not None
                else self._nemo_scale_factor
            )
        log.debug(f"[Alignment] geo_scale={geo_scale:.5f}  sp_extent={sp_extent:.5f}")

        # Scale-chain diagnostics (always recorded; flow into results JSON)
        cad_ratio = (cad_anchor / cad_aligned) if (cad_anchor and cad_aligned) else None
        self._last_alignment_diagnostics = {
            "geo_scale":            round(float(geo_scale), 6),
            "sp_extent":            round(sp_extent, 6),
            "sp_axis_extent":       [round(float(v), 6) for v in sp_axis_extent.tolist()],
            "cad_size_anchor":      round(float(cad_anchor), 6) if cad_anchor else None,
            "cad_size_aligned":     round(float(cad_aligned), 6) if cad_aligned else None,
            # >1 means the anchor-frame extent would overestimate scale and place
            # the arm too far. This factor predicts that depth-scale error.
            "cad_anchor_over_aligned": round(float(cad_ratio), 6) if cad_ratio else None,
            "cad_axis_extent_anchor":  [round(v, 6) for v in self._cad_axis_extent_anchor]
                                       if self._cad_axis_extent_anchor else None,
            "cad_axis_extent_aligned": [round(v, 6) for v in self._cad_axis_extent_aligned]
                                       if self._cad_axis_extent_aligned else None,
        }

        # The paper's alignment replaces both the center offset and the scale by
        # a similarity transform fitted against the template ground-truth poses.
        if self.config.alignment_method == "bundle":
            # The fit decodes the template images against the current (rotated,
            # not yet centred) representation, so the updated features must be
            # built first; they are rebuilt again after the centre offset is
            # applied below.
            with torch.no_grad():
                nemo["features_3d_updated"] = (
                    nemo["features_3d"] + self.nemo_model.point_encoder(nemo["surface_points"])
                )
            fitted = self._fit_bundle_alignment(nemo, templates, gt_poses)
            if fitted is not None:
                # Algorithm 1 returns (s, R, t) but does not specify how R enters
                # the pose. Applying it only to the center offset is exact near
                # identity; larger corrections trigger a warning.
                r_corr_deg = math.degrees(float(_rotation_error_rad(
                    torch.eye(3, device=fitted["rotation"].device,
                              dtype=fitted["rotation"].dtype),
                    fitted["rotation"],
                )[0]))
                if r_corr_deg > 1.0:
                    log.warning(
                        "[Alignment] bundle fit's R_corr deviates %.2f° from "
                        "identity; the reference implementation applies it only "
                        "to the center offset, so every pose carries this "
                        "rotation error (see bundle.rotation in the diagnostics).",
                        r_corr_deg,
                    )
                center_offset = fitted["center_offset"] @ fitted["rotation"]
                geo_scale = float(fitted["scale"])

        # Apply to nemo dict.
        self._nemo_center_offset_metric = (center_offset * geo_scale).to(
            device=self.device, dtype=torch.float32
        )
        nemo["surface_points"] = nemo["surface_points"] - center_offset
        nemo["scale_factor"] = torch.tensor([geo_scale], device=device, dtype=torch.float32)
        with torch.no_grad():
            nemo["features_3d_updated"] = (
                nemo["features_3d"] + self.nemo_model.point_encoder(nemo["surface_points"])
            )

        # Anchor PnP is diagnostic only and is skipped when fill_frame makes the
        # renderer intrinsics inconsistent with the cropped template.
        fill_frame_on = bool(getattr(self.renderer.config, "fill_frame", False))
        if fill_frame_on:
            self._last_alignment_diagnostics["anchor_pnp_skipped"] = "fill_frame"
        if len(gt_poses) > 0 and not fill_frame_on:
            K_model, bbox = self._template_camera(H, W)
            K_array_anchor = K_model.expand(1, 1, 3, 3).detach().cpu().numpy()
            bbox_tensor_anchor = bbox.expand(1, 1, 4)

            anchor_img = templates[0].to(device=device, dtype=torch.float32)
            imgs_anchor = rearrange(anchor_img, "c h w -> 1 1 c h w")
            with torch.no_grad():
                decoder_anchor = self.nemo_model.decode_images(imgs_anchor, nemo["features_3d_updated"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            cv2.setRNGSeed(self.config.seed)
            anchor_poses, _ = _pnp_pose_estimation_with_diagnostics(
                decoder_anchor, bbox_tensor_anchor, K_array_anchor,
                nemo_scale_factor=1.0,
                conf_threshold=self.config.conf_threshold,
                iterationsCount=self.config.pnp_initial_iterations,
                reprojectionError=self.config.pnp_initial_reproj_error,
                confidence=self.config.pnp_initial_confidence,
                min_correspondences=self.config.pnp_min_correspondences,
                min_inlier_ratio=self.config.pnp_min_inlier_ratio,
            )
            if anchor_poses and anchor_poses[0] is not None:
                anchor_pose_mat = torch.tensor(anchor_poses[0], dtype=torch.float32, device=device)
                t_anchor_est = anchor_pose_mat[:3, 3]
                R_anchor_est = anchor_pose_mat[:3, :3]
                R_anchor_gt = gt_poses[0][:3, :3].to(device=device, dtype=torch.float32)
                t_anchor_gt = gt_poses[0][:3, 3].to(device=device, dtype=torch.float32)
                # GT camera position of mesh centroid (= NeMO p=0 in metric units).
                c_center = result.orientation_center.to(device=device, dtype=torch.float32)
                t_anchor_gt_centroid = R_anchor_gt @ c_center + t_anchor_gt
                # Rotation error.
                rot_err_deg = math.degrees(float(
                    _rotation_error_rad(
                        R_anchor_gt.unsqueeze(0), R_anchor_est.unsqueeze(0)
                    )[0].item()
                ))
                # Translation error: scale t_anchor_est to metric (geo_scale),
                # then compare to GT centroid position. No correction term.
                trans_err_m = float(
                    torch.norm(geo_scale * t_anchor_est - t_anchor_gt_centroid).item()
                )
                # Merge into the scale-chain diagnostics set earlier (don't overwrite).
                self._last_alignment_diagnostics.update({
                    "anchor_rotation_error_deg": round(rot_err_deg, 4),
                    "anchor_translation_error_m": round(trans_err_m, 4),
                })
                log.debug(
                    f"[Alignment] anchor diagnostic: rot={rot_err_deg:.2f}°  "
                    f"trans={trans_err_m * 100:.2f} cm  "
                    f"geo_scale={geo_scale:.4f}"
                )
            else:
                log.warning("[Alignment] Anchor PnP failed (diagnostic only).")

        log.debug(
            f"[Alignment] scale={geo_scale:.4f}  "
            f"center_offset={[f'{v:.4f}' for v in center_offset.tolist()]}"
        )

    def _apply_alignment_to_pose(
        self,
        T_chi_c: torch.Tensor,
        result: RenderResult,
    ) -> torch.Tensor:
        """
        Convert T_chi_c (NeMO aligned frame -> camera) to T_base_cam (URDF frame -> camera).

        NeMO has no URDF, so this chain is specific to the robot reframing.
        Uses result.mesh_transform (R_align) and result.orientation_center (c) from
        robot-renderer to map the oriented frame back to the URDF base frame.
        """
        R_align = result.mesh_transform[:3, :3].to(dtype=T_chi_c.dtype, device=T_chi_c.device)
        c = result.orientation_center.to(dtype=T_chi_c.dtype, device=T_chi_c.device)

        # T_align maps URDF base -> NeMO aligned frame:
        #   p_nemo = R_align @ (p_urdf - c)   (surface points were centred by subtracting c)
        # After _setup_nemo_alignment the origin is also shifted by center_offset_metric:
        #   T_align_t = -(R_align @ c) - center_offset_metric
        t_align = -(R_align @ c)
        if self._nemo_center_offset_metric is not None:
            t_align = t_align - self._nemo_center_offset_metric.to(
                dtype=T_chi_c.dtype, device=T_chi_c.device
            )

        T_align = torch.eye(4, dtype=T_chi_c.dtype, device=T_chi_c.device)
        T_align[:3, :3] = R_align
        T_align[:3, 3] = t_align
        # (NeMO frame -> cam) @ (URDF base -> NeMO frame) = (URDF base -> cam)
        return T_chi_c @ T_align

    def _compute_mask_iou_diagnostics(
        self,
        T_chi_c: torch.Tensor,
        decoder_output: dict,
        bbox_tensor: torch.Tensor,
        K_array: np.ndarray,
        result: RenderResult,
        nemo: dict,
    ) -> dict:
        """
        Compute IoU between decoder masks and the projected mesh silhouette.

        All masks are in decoder output space (model_input_size × model_input_size).
        The rendered mask is built by projecting all mesh vertices through T_chi_c
        (NeMO-frame pose, before alignment) and dilating to fill projection gaps.

        Returns a dict with:
            conf_vs_render  : {thr_X.XX: float} - IoU at each confidence threshold
            modal_vs_render : float - decoder mask (> 0.5) vs render mask
            amodal_vs_render: float - decoder mask_full (> 0.5) vs render mask
        """
        conf_map = decoder_output["conf"][0, 0].detach().cpu().float().numpy()
        H, W = conf_map.shape

        # Build mesh in NeMO metric space - identical transform to mesh-snap.
        mesh_verts = result.mesh_verts.to(dtype=torch.float32, device=self.device)
        center = result.orientation_center.to(dtype=torch.float32, device=self.device)
        scale_factor = float(nemo["scale_factor"].item())
        mesh_nemo = (mesh_verts - center) / scale_factor if scale_factor > 1e-8 else (mesh_verts - center)
        if self._nemo_center_offset_metric is not None and scale_factor > 1e-8:
            mesh_nemo = mesh_nemo - (
                self._nemo_center_offset_metric.to(dtype=torch.float32, device=self.device)
                / scale_factor
            )
        mesh_metric = mesh_nemo.detach().cpu().numpy().astype(np.float64) * scale_factor

        # Project mesh through estimated pose.
        R = T_chi_c[:3, :3].detach().cpu().numpy().astype(np.float64)
        t = T_chi_c[:3, 3].detach().cpu().numpy().astype(np.float64)
        pts_cam = (R @ mesh_metric.T).T + t  # (N, 3)

        # Camera matrix already scaled to H×W by _prepare_pnp_inputs; account for bbox.
        bb = bbox_tensor[0, 0].detach().cpu().numpy().astype(np.float64)
        xmin, ymin, xmax, _ = bb
        bbox_scale = float(W) / max(xmax - xmin, 1e-8)
        K_proj = K_array[0, 0].astype(np.float64) * bbox_scale
        K_proj[2, 2] = 1.0

        valid_z = pts_cam[:, 2] > 1e-6
        pts_valid = pts_cam[valid_z]
        if len(pts_valid) == 0:
            return {}

        raw = (K_proj @ pts_valid.T).T
        proj_x = raw[:, 0] / raw[:, 2]
        proj_y = raw[:, 1] / raw[:, 2]
        # Shift from PnP coordinate space back to decoder pixel indices.
        col = (proj_x - xmin * bbox_scale).astype(np.int32)
        row = (proj_y - ymin * bbox_scale).astype(np.int32)
        in_bounds = (col >= 0) & (col < W) & (row >= 0) & (row < H)

        render_mask = np.zeros((H, W), dtype=np.uint8)
        render_mask[row[in_bounds], col[in_bounds]] = 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        render_mask = cv2.dilate(render_mask, kernel).astype(bool)

        def _iou(a: np.ndarray, b: np.ndarray) -> float:
            inter = int((a & b).sum())
            union = int((a | b).sum())
            return round(inter / union, 4) if union > 0 else 0.0

        # Conf IoU at each threshold.
        conf_iou = {
            f"thr_{thr:.2f}": _iou(conf_map > thr, render_mask)
            for thr in _CONF_STAT_THRESHOLDS
        }

        result_dict: dict = {"conf_vs_render": conf_iou}

        m = decoder_output.get("mask")
        if m is not None:
            result_dict["modal_vs_render"] = _iou(
                m[0, 0].detach().cpu().float().numpy() > self.config.mask_threshold, render_mask
            )

        mf = decoder_output.get("mask_full")
        if mf is not None:
            result_dict["amodal_vs_render"] = _iou(
                mf[0, 0].detach().cpu().float().numpy() > self.config.mask_threshold, render_mask
            )

        return result_dict

    def _dump_failure(self, out_dir, frame_idx, image, imgs, decoder_output, bbox_xyxy) -> None:
        """Save crop, confidence, bbox and parameters for a failed PnP frame."""
        try:
            os.makedirs(out_dir, exist_ok=True)
            tag = f"fail_{frame_idx:04d}"

            crop = (imgs[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{tag}_crop.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            conf = decoder_output["conf"][0, 0].detach().cpu().float().numpy()
            cmax = float(conf.max()) if conf.size else 0.0
            conf_u8 = (conf / cmax * 255).astype(np.uint8) if cmax > 1e-9 else np.zeros(conf.shape, np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{tag}_conf_max{cmax:.3f}.png"),
                        cv2.applyColorMap(conf_u8, cv2.COLORMAP_JET))

            if isinstance(image, str):
                full = cv2.imread(image, cv2.IMREAD_COLOR)
            else:
                t = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
                t = t[0] if t.ndim == 4 else t
                full = cv2.cvtColor(
                    (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
            if full is not None:
                if bbox_xyxy is not None:
                    x0, y0, x1, y1 = (int(round(float(v))) for v in bbox_xyxy[:4])
                    cv2.rectangle(full, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.imwrite(os.path.join(out_dir, f"{tag}_full.png"), full)

            # 4) sidecar: conf_stats + bbox + crop params
            cs = (self._last_frame_diagnostics or {}).get("conf_stats", {})
            with open(os.path.join(out_dir, f"{tag}.txt"), "w") as fh:
                fh.write(f"frame_idx={frame_idx}\n")
                fh.write(f"bbox_xyxy={None if bbox_xyxy is None else [float(v) for v in bbox_xyxy[:4]]}\n")
                fh.write(f"crop_params(x,y,w,h)={self._last_crop_params}\n")
                fh.write(f"conf_max={cmax:.4f}\n")
                for k, v in cs.items():
                    fh.write(f"  {k}: count={v['count']} mean_conf={v['mean_conf']}\n")
            log.info("[debug] NeMO: dumped PnP-failure artifacts for frame %d -> %s", frame_idx, out_dir)
        except Exception as e:  # never crash the eval over a debug dump
            log.warning("NeMO failure-dump for frame %d failed: %s", frame_idx, e)

    def _decoder_mask_bbox_orig(self, decoder_output: dict) -> Optional[np.ndarray]:
        """Bbox (xyxy, original-image px) of the decoder's foreground, mapped back
        through the current query crop (`_last_crop_params`).

        Prefers the decoder's modal segmentation mask; falls back to the confidence
        map (conf > conf_threshold) when no mask head is present. Returns None if the
        foreground is absent or smaller than `bootstrap_min_mask_pixels`.

        This is the detector-free localiser behind crop_mode="bootstrap": the first
        (coarse) decode tells us where the arm is, so the second pass can bbox-crop it.
        """
        if self._last_crop_params is None:
            return None

        m = decoder_output.get("mask")
        if m is not None:
            fg = (m[0, 0].detach().cpu().float().numpy() > self.config.mask_threshold)
        else:
            conf = decoder_output.get("conf")
            if conf is None:
                return None
            fg = (conf[0, 0].detach().cpu().float().numpy() > self.config.conf_threshold)

        # The largest connected component is the single-instance counterpart to
        # supplement 7.5's clustering and small-cluster removal.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg.astype(np.uint8), connectivity=8
        )
        if n_labels <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        component_area = int(areas[best - 1])
        self._last_bootstrap_components = {
            "n_components": int(n_labels - 1),
            "component_areas": sorted((int(a) for a in areas), reverse=True)[:10],
            "selected_area": component_area,
        }
        # the floor applies to the selected component, not to the sum of islands
        if component_area < int(self.config.bootstrap_min_mask_pixels):
            return None

        x, y, w_box, h_box = (
            int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP]),
            int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT]),
        )
        h_dec, w_dec = fg.shape
        crop_x, crop_y, crop_w, crop_h = self._last_crop_params
        # Decoder grid (w_dec×h_dec) spans the crop region; map component extent -> crop px -> original px.
        x0 = x / w_dec * crop_w + crop_x
        x1 = (x + w_box) / w_dec * crop_w + crop_x
        y0 = y / h_dec * crop_h + crop_y
        y1 = (y + h_box) / h_dec * crop_h + crop_y

        pad = max(0.0, float(self.config.bootstrap_mask_pad))
        bw, bh = x1 - x0, y1 - y0
        return np.array(
            [x0 - bw * pad, y0 - bh * pad, x1 + bw * pad, y1 + bh * pad],
            dtype=np.float32,
        )

    def _load_query_image(
        self, image, nemo_encoding_size, device,
        crop_mode: str = "center", bbox_xyxy=None,
    ) -> torch.Tensor:
        # Load to (1, 3, H, W) float32 in [0, 1].
        if isinstance(image, str):
            img_bgr = cv2.imread(image, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not read query image: {image}")
            img_t = (
                torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                .float().div(255.0).permute(2, 0, 1).unsqueeze(0)
            )
        else:
            img_t0 = image if isinstance(image, torch.Tensor) else torch.tensor(image)
            if img_t0.ndim == 3:
                img_t0 = img_t0.unsqueeze(0)
            img_t = img_t0.detach().float().clamp(0.0, 1.0)

        _, _, h, w = img_t.shape
        if crop_mode == "bbox" and bbox_xyxy is None:
            raise ValueError(
                "crop_mode='bbox' needs a detection bbox, but none was supplied. "
                "Pass detections_path in the eval config, or select crop_mode="
                "'bootstrap' (detector-free two-pass) or 'center'. "
                "Falling back to a center crop here would silently evaluate a "
                "different regime than the run is named after."
            )
        if crop_mode == "bbox" and bbox_xyxy is not None:
            x0, y0, x1, y1 = (float(v) for v in bbox_xyxy[:4])
            bw, bh = x1 - x0, y1 - y0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            # Pad before squarifying so an imperfect box does not clip the arm
            # (mirrors FoundPose's crop_rel_pad).
            pad = max(0.0, float(getattr(self.config, "bbox_crop_pad", 0.0)))
            s = max(bw, bh) * (1.0 + 2.0 * pad)
            # Shift-window square crop (BOP `square_crop`, utils.py:309): slide an
            # overflowing square back inside rather than clamping its far edge, so the
            # crop stays square and the arm never reaches the encoder stretched.
            s = min(s, float(w), float(h))
            s_i = int(s)
            bx0 = max(0, min(int(cx - s / 2), w - s_i))
            by0 = max(0, min(int(cy - s / 2), h - s_i))
            bx1 = bx0 + s_i
            by1 = by0 + s_i
            img_t = img_t[:, :, by0:by1, bx0:bx1]
            self._last_crop_params = (bx0, by0, bx1 - bx0, by1 - by0)
        else:  # "center" (default)
            s = min(h, w)
            crop_x = (w - s) // 2
            crop_y = (h - s) // 2
            img_t = img_t[:, :, crop_y:crop_y + s, crop_x:crop_x + s]
            self._last_crop_params = (crop_x, crop_y, s, s)

        # Antialiasing avoids native-resolution crop aliasing (loader
        # scale=1.0): crops come from the full-res image, so they reach the 224 encoder
        # through a large decimation (~2.75x on xArm) that aliases without it. On crops
        # below 224 this is an upsample and the flag is a no-op.
        img_t = F.interpolate(
            img_t, size=(nemo_encoding_size, nemo_encoding_size),
            mode="bicubic", align_corners=False, antialias=True,
        ).clamp(0.0, 1.0)

        return img_t.to(device)

    def _prepare_pnp_inputs(self, K, H_img, W_img, nemo_encoding_size, device):
        """Replay the query crop into K so the PnP intrinsics match what the decoder
        saw. Every crop mode bakes its crop into K here rather than passing a
        bbox down to nemolib's PnP, which assumes the decoder input WAS the bbox crop."""
        K_np = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K, dtype=np.float32)

        bbox_tensor = torch.tensor(
            [[0.0, 0.0, float(W_img), float(H_img)]], device=device
        ).unsqueeze(1)

        # Rescale K to match the cropped/padded, resized NeMO input.
        # _load_query_image always runs first and stores the crop parameters.
        crop_x, crop_y, crop_w, crop_h = self._last_crop_params  # type: ignore[misc]

        # Per-axis scale.  For square crops (center/bbox) crop_w == crop_h so
        # scale_x == scale_y; kept per-axis so K stays exact for any crop shape.
        scale_x = float(nemo_encoding_size) / max(float(crop_w), 1.0)
        scale_y = float(nemo_encoding_size) / max(float(crop_h), 1.0)
        K_np = K_np.copy()
        K_np[0, 0] *= scale_x
        K_np[0, 2] = (K_np[0, 2] - crop_x) * scale_x
        K_np[1, 1] *= scale_y
        K_np[1, 2] = (K_np[1, 2] - crop_y) * scale_y

        K_array = torch.tensor(K_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).numpy()
        return bbox_tensor, K_array, K_np
