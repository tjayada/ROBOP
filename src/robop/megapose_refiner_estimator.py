"""MegaPose refinement of poses from an existing results JSON.

This ROBOP extension applies MegaPose's render-and-compare refiner to poses from
any coarse model. Frames are matched by image path and joint angles; missing
coarse poses remain failures. MegaPose Sec. 4.4 notes that large initial errors
may not converge.

"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from robop import estimator_utils
from robop.megapose_estimator import (
    MegaPoseConfig,
    MegaPoseRobotEstimator,
    megapose_pose_estimator,
)

log = logging.getLogger(__name__)


@dataclass
class MegaPoseRefinerConfig(MegaPoseConfig):
    """MegaPose configuration with an initial-results path."""

    init_results_path: str = ""     # required coarse-results JSON
    joints_atol: float = 1e-5       # frame-pairing sanity check tolerance (rad)
    score_refined: bool = False     # score the refined pose with MegaPose's
                                    # coarse net (diagnostic pose_logit only)


class _InitPoseStore:
    """Per-frame pose initialisations loaded from a coarse results JSON.

    MegaPose normally refines detections within one pipeline; this adapter reads
    initial poses from another model's results file.
    """

    def __init__(self, results_path: str, joints_atol: float) -> None:
        path = Path(results_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"init_results_path not found: {path} - point it at the coarse "
                f"model's results JSON."
            )
        data = json.loads(path.read_text())
        queries = data["queries"] if isinstance(data, dict) else data

        self._atol = joints_atol
        self._exact: Dict[str, dict] = {}
        self._tails: Dict[str, Optional[dict]] = {}  # None => ambiguous tail

        n_no_pose = 0
        for q in queries:
            if "image_path" in q:
                key = str(q["image_path"])
            elif "measurement" in q and "frame_idx" in q:  # Hydra
                key = f"meas{int(q['measurement'])}_{int(q['frame_idx']):03d}"
            else:
                raise KeyError(
                    "Query entry has neither 'image_path' nor "
                    "'measurement'/'frame_idx' - cannot build a frame key."
                )
            entry = {
                "pose": (np.asarray(q["est_pose"], dtype=np.float64)
                         if q.get("est_pose") is not None else None),
                "joints": np.asarray(q["joints_rad"], dtype=np.float64)
                          if q.get("joints_rad") is not None else None,
            }
            if entry["pose"] is None:
                n_no_pose += 1
            self._exact[key] = entry
            # Hydra entries also carry frame_id (meas{M}_{F:03d}); index under it
            # too. The eval loop forwards image_path as frame_key, whose 2-part
            # tail (images/NNN.png) collides across measurements, so the
            # frame_id is the only unambiguous cross-machine key for Hydra.
            if q.get("frame_id") is not None:
                self._exact[str(q["frame_id"])] = entry
            # tail index for cross-machine path differences (last 2 components)
            if "/" in key:
                tail = "/".join(Path(key).parts[-2:])
                self._tails[tail] = None if tail in self._tails else entry

        log.info(
            "MegaPoseRefiner: loaded %d init poses from %s (%d without a pose "
            "- coarse failures, will be propagated).",
            len(self._exact), path.name, n_no_pose,
        )

    def lookup(self, frame_key: str, joint_angles: np.ndarray) -> Optional[np.ndarray]:
        """Return the init T_base_cam (4,4) or None (coarse failure).

        Raises on unknown frames or joint mismatches - both indicate the refine
        run is iterating a different frame set than the coarse run.
        """
        entry = self._exact.get(str(frame_key))
        if entry is None and "/" in str(frame_key):
            tail = "/".join(Path(str(frame_key)).parts[-2:])
            entry = self._tails.get(tail)
        if entry is None:
            # Hydra frame_key is an image path (measurement_M/images/NNN.png) whose
            # 2-part tail collides across measurements; the meas{M}_{F:03d} id
            # derived from the path is the unambiguous key those queries were
            # registered under (see __init__).
            parts = Path(str(frame_key)).parts
            if len(parts) >= 3 and parts[-2] == "images" and parts[-3].startswith("measurement_"):
                m = parts[-3].split("measurement_", 1)[1]
                stem = Path(parts[-1]).stem
                if m.isdigit() and stem.isdigit():
                    entry = self._exact.get(f"meas{int(m)}_{int(stem):03d}")
        if entry is None and "/" in str(frame_key):
            tail = "/".join(Path(str(frame_key)).parts[-2:])
            if tail in self._tails:
                raise KeyError(
                    f"Frame key {frame_key!r}: path tail {tail!r} is ambiguous in "
                    f"the init results JSON - paths collide; run the refine eval "
                    f"with the same data_folder as the coarse run."
                )
        if entry is None:
            raise KeyError(
                f"Frame {frame_key!r} not found in the init results JSON. The "
                f"refine run must iterate the same frames as the coarse run "
                f"(same dataset config / num_eval_frames)."
            )
        if entry["joints"] is not None:
            q_in = np.asarray(joint_angles, dtype=np.float64).ravel()
            q_st = entry["joints"].ravel()
            if q_in.shape != q_st.shape or not np.allclose(q_in, q_st, atol=self._atol):
                raise ValueError(
                    f"Frame {frame_key!r}: joint angles do not match the init "
                    f"results JSON (max diff "
                    f"{np.abs(q_in - q_st).max() if q_in.shape == q_st.shape else 'shape mismatch'}) "
                    f"- wrong frame pairing or wrong init file."
                )
        return entry["pose"]


class MegaPoseRefinerEstimator(MegaPoseRobotEstimator):
    """Refine per-frame init poses (from a results JSON) with MegaPose's refiner.

    Mirrors ``gigapose_estimator._refine_pose`` exactly, with a single
    hypothesis per frame: full-image observation, ``forward_refiner`` over
    ``n_refiner_iterations``, optional coarse-net scoring for diagnostics.
    """

    # Input contract override: the refiner looks up its init pose by frame_key
    # and crops pose-conditionally, so detections are not consumed.
    INPUT_CONTRACT = {"mask": False, "bbox_xyxy": False, "frame_key": True,
                      "mask_3x3_opening": False}

    def __init__(self, renderer, config: Optional[MegaPoseRefinerConfig] = None,
                 device: Optional[torch.device] = None) -> None:
        config = config or MegaPoseRefinerConfig()
        super().__init__(renderer, config, device)
        if not config.init_results_path:
            raise ValueError(
                "MegaPoseRefinerEstimator needs estimator.init_results_path "
                "(the coarse model's results JSON)."
            )
        self._init_store = _InitPoseStore(config.init_results_path, config.joints_atol)

    @torch.no_grad()
    def inference_single_image(
        self,
        img: torch.Tensor,        # (3, H, W) float [0,1] - full eval image
        joint_angles,
        K=None,                   # (3, 3) real camera intrinsics
        mask=None,                # unused
        bbox_xyxy=None,           # unused - the refiner crops pose-conditionally
        frame_key: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None]:
        import pandas as pd
        from megapose.utils.tensor_collection import PandasTensorCollection
        from src.custom_megapose.refiner_utils import load_pretrained_refiner

        _t = time.perf_counter()
        frame_idx = self._frame_idx
        self._frame_idx += 1
        self._last_frame_diagnostics = None

        estimator_utils.reseed_all(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()   # baseline for the per-frame VRAM metric

        if frame_key is None:
            raise ValueError(
                "MegaPoseRefinerEstimator needs frame_key - update the eval "
                "script to pass it through EstimatorAdapter."
            )
        K_np = (K.cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)).astype(np.float64)

        init_T_base_cam = self._init_store.lookup(frame_key, joint_angles)
        if init_T_base_cam is None:
            # coarse failure - nothing to refine; propagate as a failure
            self._last_frame_diagnostics = {
                "frame_time_ms": round((time.perf_counter() - _t) * 1000, 1),
                "timings_ms": {"total": 0.0},
                "gpu_peak_mb": None,
                "pnp": None,
                "refined": False,
                "init_missing": True,
                # refinement dict mirrors gigapose_estimator so the refiner_* run
                # counters (aggregate_frame_diagnostics) count this frame: no
                # init pose means refinement was never attempted.
                "refinement": {"enabled": True, "attempted": False,
                               "succeeded": False, "n_hypotheses": 0,
                               "error": "init_missing"},
            }
            return None, None, None, None, None

        # FK-posed arm mesh (mm, centred) + frame transform
        q = np.asarray(joint_angles, dtype=np.float64)
        mesh, mesh_transform = self.renderer.export_posed_trimesh(q)
        center_m = estimator_utils.center_offset_m(mesh_transform)
        # the refiner works in the centred-mesh frame MegaPose renders from
        TCO_init = estimator_utils.base_to_centred(init_T_base_cam, center_m)

        pose_logit = None
        with megapose_pose_estimator(
            mesh, img, K_np, self.device,
            model_name=self.config.model_name,
            models_root=self.config.megapose_models_root,
            num_workers=self.config.num_workers,
            batch_size_objects=self.config.batch_size_objects,
            batch_size_images=self.config.batch_size_images,
            n_iterations=self._n_refiner_iterations,
            move_coarse_model=self.config.score_refined,
            loader=load_pretrained_refiner,
        ) as (pose_estimator, observation, obj_label):
            poses_init = torch.from_numpy(TCO_init[None].astype(np.float32)).to(self.device)
            data_TCO = PandasTensorCollection(
                infos=pd.DataFrame({
                    "label": [obj_label],
                    "batch_im_id": [0],
                    "instance_id": [0],
                    "matching_score": [1.0],
                }),
                poses=poses_init,
            )
            preds, _ = pose_estimator.forward_refiner(
                observation=observation,
                data_TCO_input=data_TCO,
                n_iterations=self._n_refiner_iterations,
                keep_all_outputs=False,
                cuda_timer=None,
            )
            data_TCO_ref = preds[f"iteration={self._n_refiner_iterations}"]
            if self.config.score_refined and pose_estimator.coarse_model is not None:
                data_TCO_scored, _ = pose_estimator.forward_scoring_model(
                    observation, data_TCO_ref
                )
                if "pose_logit" in data_TCO_scored.infos:
                    pose_logit = float(data_TCO_scored.infos["pose_logit"].iloc[0])
                data_TCO_ref = data_TCO_scored
            T_obj_cam_ref = data_TCO_ref.poses[0].detach().cpu().numpy().astype(np.float64)

        T_base_cam = estimator_utils.centred_to_base(T_obj_cam_ref, center_m)

        frame_time_ms = round((time.perf_counter() - _t) * 1000, 1)
        gpu_peak_mb = (
            round(torch.cuda.max_memory_allocated() / 1e6, 1)
            if torch.cuda.is_available() else None
        )
        self._last_frame_diagnostics = {
            "frame_time_ms": frame_time_ms,
            "timings_ms": {"total": frame_time_ms},
            "gpu_peak_mb": gpu_peak_mb,
            "pnp": None,   # refiner has no PnP stage
            "refined": True,
            "n_refiner_iterations": self._n_refiner_iterations,
            "pose_logit": round(pose_logit, 4) if pose_logit is not None else None,
            # refinement dict mirrors gigapose_estimator (single hypothesis).
            "refinement": {"enabled": True, "attempted": True,
                           "succeeded": True, "n_hypotheses": 1},
        }

        if (self.config.debug_reproject_dir is not None
                and frame_idx < self.config.n_reproject_frames):
            self._save_reprojection_overlay(img, T_base_cam, K_np, joint_angles, frame_idx)

        return torch.from_numpy(T_base_cam.astype(np.float32)), None, None, None, None
