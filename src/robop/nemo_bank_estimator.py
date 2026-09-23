"""NeMO with a config-keyed representation bank (online reuse).

`estimate_pose_from_image` is a copy of NeMORobotPoseEstimator's, with one change:
the render+encode block (its _t_a.._t_c phases, ~98.7% of frame time on panda-orb)
is served from `ReuseBank` when a stored config is within tolerance. Everything
else - alignment, PnP, masks, diagnostics - is inherited, so the numerically subtle
code stays single-source in nemo_estimator.py, which this file does not modify.

Caching is only valid while the representation is a pure function of the joint
config, so `include_query_in_templates` and `use_blurred_query_background` must be
off (enforced in __init__), and crop_mode="bootstrap" is refused because its
two-pass recursion re-enters the pipeline per patch.

Set reuse_bank.tolerance_mm = 0.0 for the equivalence tripwire: the bank is live
and exercises the copied path, but only exact-duplicate configs hit (d_surf = 0),
which by construction have an identical representation - so results must match the
plain `nemo` baseline bit-for-bit.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from robop import estimator_utils
from robop.mask_utils import binary_mask_to_rle
from robop.nemo_estimator import (
    NeMORobotPoseEstimator,
    _pnp_pose_estimation_with_diagnostics,
)
from robop.reuse_bank import ReuseBank

log = logging.getLogger(__name__)

# Instance state written by the build phase that decode/PnP/alignment later read.
_CACHED_ATTRS = (
    "_nemo_scale_factor",
    "_mesh_surface_scale",
    "_cad_axis_extent_aligned",
    "_cad_axis_extent_anchor",
    "_cad_size_metric",
    "_cad_size_metric_aligned",
    "_nemo_center_offset_metric",
    "_last_alignment_diagnostics",
)


class _CachedRender:
    """The only RenderResult fields used after the build phase.

    _apply_alignment_to_pose reads mesh_transform + orientation_center;
    _compute_mask_iou_diagnostics reads mesh_verts. The rendered template images are
    never read downstream, and dropping them is what keeps an entry ~10 MB instead
    of ~80 MB.
    """

    __slots__ = ("mesh_transform", "orientation_center", "mesh_verts")

    def __init__(self, result):
        self.mesh_transform = result.mesh_transform
        self.orientation_center = result.orientation_center
        self.mesh_verts = result.mesh_verts


class NeMOBankEstimator(NeMORobotPoseEstimator):

    def __init__(self, *, nemo_model, renderer, config, device,
                 robot_name: str, tolerance_mm: Optional[float] = None):
        super().__init__(nemo_model=nemo_model, renderer=renderer,
                         config=config, device=device)
        self._bank: Optional[ReuseBank] = None
        if tolerance_mm is None:
            return
        if config.include_query_in_templates or config.use_blurred_query_background:
            raise ValueError(
                "reuse_bank needs a query-independent representation: set "
                "include_query_in_templates=false and use_blurred_query_background=false."
            )
        if config.crop_mode == "bootstrap":
            raise ValueError("reuse_bank does not support crop_mode='bootstrap'.")
        self._bank = ReuseBank(robot_name, self.robot_kin, tolerance_mm)
        log.info("NeMO reuse bank active: tolerance %.3f mm d_surf.", tolerance_mm)

    # Cache payload -------------------------------------------------------------

    def _snapshot(self, nemo, render: _CachedRender) -> dict:
        return {"nemo": nemo, "render": render,
                "attrs": {a: getattr(self, a, None) for a in _CACHED_ATTRS}}

    def _restore(self, payload: dict):
        for name, value in payload["attrs"].items():
            setattr(self, name, value)
        return payload["nemo"], payload["render"]

    # ---------------------------------------------------------------------------

    def estimate_pose_from_image(
        self,
        image: Union[str, torch.Tensor],
        joint_angles: Union[np.ndarray, torch.Tensor, Sequence[float]],
        K: Union[np.ndarray, torch.Tensor],
        bbox_xyxy=None,
        num_sample_points: Optional[int] = None,
        use_mesh_surface_sampling: Optional[bool] = None,
        _crop_mode_override: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        if self._bank is None:                      # caching off: parent, untouched
            return super().estimate_pose_from_image(
                image=image, joint_angles=joint_angles, K=K, bbox_xyxy=bbox_xyxy,
                num_sample_points=num_sample_points,
                use_mesh_surface_sampling=use_mesh_surface_sampling,
                _crop_mode_override=_crop_mode_override,
            )

        if num_sample_points is None:
            num_sample_points = self.config.num_sample_points
        crop_mode = _crop_mode_override or self.config.crop_mode
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
            crop_mode=crop_mode, bbox_xyxy=bbox_xyxy,
        )
        images_tensor = rearrange(imgs, "t c h w -> 1 t c h w")
        _sync()
        _t_a = time.perf_counter()      # end of preprocess phase

        # Build phase, served from the bank when a stored config is within tolerance.
        cached, d_surf_mm, entry_idx = self._bank.lookup(joint_angles)
        if cached is not None:
            nemo, result = self._restore(cached)
            _sync()
            _t_b = _t_c = time.perf_counter()
        else:
            result = self.renderer.render_templates(joint_angles, background_image=None)
            gt_poses = [v.gt_pose for v in result.views]
            self._nemo_scale_factor = self._compute_anchor_bbox_scale(result)
            templates_hr = torch.stack([v.image for v in result.views], dim=0)
            templates = F.interpolate(
                templates_hr,
                size=(self.config.nemo_encoding_size, self.config.nemo_encoding_size),
                mode="bicubic", align_corners=False, antialias=True,
            ).clamp(0.0, 1.0)
            _sync()
            _t_b = time.perf_counter()  # end of render phase

            self._mesh_surface_scale = None
            _use_mss = (
                self.config.use_mesh_surface_sampling
                if use_mesh_surface_sampling is None else use_mesh_surface_sampling
            )
            nemo = self._build_nemo_from_templates(
                templates, num_sample_points, result, use_mesh_surface_sampling=_use_mss,
            )
            _sync()
            self._setup_nemo_alignment(
                nemo=nemo, templates=templates, gt_poses=gt_poses, result=result,
            )
            _sync()
            _t_c = time.perf_counter()  # end of encode phase

            result = _CachedRender(result)   # drop template images before storing
            self._bank.insert(joint_angles, self._snapshot(nemo, result))
            entry_idx = len(self._bank) - 1   # a miss is served by its own new entry

        with torch.no_grad():
            decoder_output = self.nemo_model.decode_images(
                images_tensor, nemo["features_3d_updated"])

        if "mask" in decoder_output and decoder_output["mask"] is not None:
            decoder_output["conf"] = (
                decoder_output["conf"].clone()
                * (decoder_output["mask"] > self.config.mask_threshold)
            )
        _sync()
        _t_d = time.perf_counter()      # end of decode phase

        self._last_decoder_mask_bbox_orig = self._decoder_mask_bbox_orig(decoder_output)

        H_img, W_img = imgs.shape[-2], imgs.shape[-1]
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
            "pnp":        round((_t_done - _t_d) * 1000, 1),
            "total":      round((_t_done - _t_wall) * 1000, 1),
        }

        def _finalize_timing() -> None:
            now = time.perf_counter()
            d = self._last_frame_diagnostics
            d["timings_ms"]["postprocess"] = round((now - _t_done) * 1000, 1)
            d["timings_ms"]["total"] = round((now - _t_wall) * 1000, 1)
            d["frame_time_ms"] = d["timings_ms"]["total"]

        self._last_frame_diagnostics = {
            **_pnp_diag,
            "alignment":    self._last_alignment_diagnostics,
            "timings_ms":   _timings_ms,
            "frame_time_ms": _timings_ms["total"],
            "gpu_peak_mb":  _gpu_peak_mb,
            # Per-frame reuse record: lets the analysis plot ADD against the realized
            # distance to the bank entry and compare it with the perturbation curve. entry_idx
            # attributes every frame to the entry that served it (misses included), so
            # failures can be checked for clustering on a few entries.
            "reuse_bank": {
                "hit": cached is not None,
                "d_surf_mm": d_surf_mm,
                "bank_size": len(self._bank),
                "entry_idx": entry_idx,
            },
        }

        _modal = decoder_output.get("mask")
        _amodal = decoder_output.get("mask_full")
        if _modal is not None:
            self._last_frame_diagnostics["modal_mask_rle"] = binary_mask_to_rle(
                (_modal[0, 0].detach().cpu().float().numpy()
                 > self.config.mask_threshold).astype(np.uint8)
            )
        if _amodal is not None:
            self._last_frame_diagnostics["amodal_mask_rle"] = binary_mask_to_rle(
                (_amodal[0, 0].detach().cpu().float().numpy()
                 > self.config.mask_threshold).astype(np.uint8)
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
