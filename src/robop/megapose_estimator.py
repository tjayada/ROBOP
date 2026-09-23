"""Standalone MegaPose adapter (Labbe et al., CoRL 2022).

Uses the official ``NAMED_MODELS`` pipeline vendored by GigaPose. MegaPose uses
the detection bbox, while robot-renderer supplies the FK-posed mesh. The
PoseEstimator is rebuilt because the articulated object changes each frame.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from robop import estimator_utils
from robop.estimator_utils import (
    ensure_gigapose_path,
    export_ply_with_normals,
    suppress_c_stdout_stderr,
)

log = logging.getLogger(__name__)


@contextlib.contextmanager
def megapose_pose_estimator(
    mesh,
    img: torch.Tensor,
    K_np: np.ndarray,
    device,
    *,
    model_name: str,
    models_root: str,
    num_workers: int,
    batch_size_objects: int,
    batch_size_images: int,
    n_iterations: int,
    move_coarse_model: bool,
    loader=None,
):
    """Build MegaPose's PoseEstimator around one FK-posed arm mesh, then tear it down.

    Shared by the three call sites that drive MegaPose's networks: standalone
    MegaPose, GigaPose's refinement phase and the MegaPose-as-refiner estimator.
    Yields (pose_estimator, observation, obj_label), inside the C-level output
    suppression (panda3d's forked workers write to fd 1/2) and the try whose
    finally stops the renderer.

    move_coarse_model: the coarse net is loaded either way; move it to the device
    only when this call path scores with it.
    loader: load_pretrained_refiner when the caller already validated it at setup.
    """
    from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
    from megapose.inference.types import ObservationTensor
    from omegaconf import OmegaConf

    if loader is None:
        from src.custom_megapose.refiner_utils import load_pretrained_refiner as loader

    with tempfile.TemporaryDirectory() as tmp:
        mesh_ply = Path(tmp) / "arm.ply"
        export_ply_with_normals(mesh, mesh_ply)
        obj_label = "obj_000001"
        object_dataset = RigidObjectDataset(
            [RigidObject(label=obj_label, mesh_path=mesh_ply, mesh_units="mm")]
        )
        cfg = OmegaConf.create({
            "model_name": model_name,
            "depth_refiner": None,
            "models_root": models_root,
            "num_workers": num_workers,
            "batch_size_objects": batch_size_objects,
            "batch_size_images": batch_size_images,
            "n_iterations": n_iterations,
        })
        with suppress_c_stdout_stderr():
            pose_estimator = loader(cfg, object_dataset)
            # mirrors Refiner.move_to_device
            pose_estimator.refiner_model.mesh_db.to(device)
            pose_estimator.refiner_model.to(device).eval()
            if move_coarse_model and pose_estimator.coarse_model is not None:
                pose_estimator.coarse_model.mesh_db.to(device)
                pose_estimator.coarse_model.to(device).eval()
            pose_estimator.to(device)

            rgb_hwc = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            observation = ObservationTensor.from_numpy(
                rgb_hwc, depth=None, K=K_np.astype(np.float32)
            )
            observation.images = observation.images.to(device)
            observation.K = observation.K.to(device)

            try:
                yield pose_estimator, observation, obj_label
            finally:
                try:
                    pose_estimator.refiner_model.renderer.stop()
                except Exception:
                    pass


@dataclass
class MegaPoseConfig:
    """Configuration for MegaPoseRobotEstimator (standalone MegaPose baseline)."""

    # NAMED_MODELS key (src/megapose/utils/load_model.py). The "multi-hypothesis"
    # variant is n_pose_hypotheses=5, the headline BOP config. Weights come from
    # megapose_models_root, not the GigaPose checkpoint.
    model_name: str = "megapose-1.0-RGB-multi-hypothesis"
    megapose_models_root: str = ""   # path to pretrained/megapose-models/

    # None = the model's published inference_parameters.
    n_refiner_iterations: Optional[int] = None
    n_pose_hypotheses: Optional[int] = None

    # Return the top-1 SO(3)-grid hypothesis by coarse_logit and skip the refiner
    # and scoring. Set by the `megapose_coarse` config.
    coarse_only: bool = False

    num_workers: int = 4
    batch_size_objects: int = 8
    batch_size_images: int = 128   # load_named_model's default

    # Debug: project the FK-posed arm mesh through the final T_base_cam + the real
    # camera K onto the eval image for the first n_reproject_frames frames.
    debug_reproject_dir: Optional[str] = None
    n_reproject_frames: int = 20
    # Dumps MegaPose's own render-and-compare panels [observation | rendered RGB |
    # rendered normals], i.e. exactly what the scoring net matches against.
    debug_render_dir: Optional[str] = None

    seed: int = 0
    verbose: bool = False


class MegaPoseRobotEstimator:
    """Standalone MegaPose adapter (CUDA required: the SO(3) grid is .cuda())."""

    # Input contract read by evaluation's EstimatorAdapter: MegaPose crops by
    # detection bbox; masks are not consumed.
    INPUT_CONTRACT = {"mask": False, "bbox_xyxy": True, "frame_key": False,
                      "mask_3x3_opening": False}

    def __init__(
        self,
        renderer,
        config: Optional[MegaPoseConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        ensure_gigapose_path()
        # The panda3d batch renderer passes tensors from forked workers by FD,
        # which the 576-rotation coarse render overruns under the default
        # strategy. MegaPose's own train_megapose.py uses 'file_system' too.
        torch.multiprocessing.set_sharing_strategy("file_system")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.renderer = renderer.to(device)
        self.config = config or MegaPoseConfig()
        if self.config.verbose:
            log.setLevel(logging.DEBUG)

        # Resolve the inference parameters from the published model entry, with
        # config overrides (mirrors run_inference_on_example's model_info).
        from src.megapose.utils.load_model import NAMED_MODELS
        if self.config.model_name not in NAMED_MODELS:
            raise ValueError(
                f"Unknown MegaPose model_name {self.config.model_name!r}. "
                f"Options: {list(NAMED_MODELS)}"
            )
        # This adapter builds an RGB observation with depth=None and never runs
        # ICP, so the depth-requiring entries (RGBD, ICP) would either fail on a
        # missing channel or quietly run without their refinement stage.
        if NAMED_MODELS[self.config.model_name].get("requires_depth"):
            rgb_only = [k for k, v in NAMED_MODELS.items() if not v.get("requires_depth")]
            raise ValueError(
                f"model_name {self.config.model_name!r} requires depth, which this "
                f"RGB-only pipeline does not provide. Options: {rgb_only}"
            )
        params = NAMED_MODELS[self.config.model_name]["inference_parameters"]
        self._validate_model_files(NAMED_MODELS[self.config.model_name])
        self._n_refiner_iterations = (
            self.config.n_refiner_iterations
            if self.config.n_refiner_iterations is not None
            else int(params["n_refiner_iterations"])
        )
        self._n_pose_hypotheses = (
            self.config.n_pose_hypotheses
            if self.config.n_pose_hypotheses is not None
            else int(params["n_pose_hypotheses"])
        )
        # Phrased for both this class and MegaPoseRefinerEstimator, which inherits
        # this __init__ but uses only the refiner (n_pose_hypotheses is logged by
        # build_estimator, which knows which path is running).
        log.info("MegaPose: model=%s, n_refiner_iterations=%d.",
                 self.config.model_name, self._n_refiner_iterations)

        self._last_frame_diagnostics: Optional[dict] = None
        self._frame_idx = 0

    def _validate_model_files(self, model_info: dict) -> None:
        """Check the run directories load_pretrained_refiner opens, so a missing
        or partial download does not surface only at frame 0.
        """
        root = self.config.megapose_models_root
        if not root or not Path(root).is_dir():
            raise FileNotFoundError(
                f"megapose_models_root is not a valid directory: {root!r}. "
                "Run `make download-megapose` and pass "
                "estimator.megapose_models_root=/abs/path/pretrained/megapose-models."
            )
        missing = [
            str(Path(root) / run_id / f)
            for run_id in (model_info["coarse_run_id"], model_info["refiner_run_id"])
            for f in ("config.yaml", "checkpoint.pth.tar")
            if not (Path(root) / run_id / f).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"MegaPose model files missing under {root}: {missing}. "
                "Run `make download-megapose`."
            )

    @property
    def robot_kin(self):
        return self.renderer._kinematics

    # Debug: reproject the recovered pose onto the eval image
    def _save_reprojection_overlay(self, img, T_base_cam, K_np, joint_angles, frame_idx) -> None:
        """Project the FK-posed mesh through T_base_cam and the real K onto the
        eval image. A frame convention bug lands the dots off the arm."""
        try:
            from PIL import Image, ImageDraw

            q = np.asarray(joint_angles, dtype=np.float64)
            mesh, mt = self.renderer.export_posed_trimesh(q)
            center_mm = -np.asarray(mt, dtype=np.float64)[:3, 3]
            verts_base_m = (np.asarray(mesh.vertices, dtype=np.float64) + center_mm) / 1000.0
            if len(verts_base_m) > 3000:
                verts_base_m = verts_base_m[np.linspace(0, len(verts_base_m) - 1, 3000, dtype=int)]

            T = np.asarray(T_base_cam, dtype=np.float64)
            verts_cam = verts_base_m @ T[:3, :3].T + T[:3, 3]
            verts_cam = verts_cam[verts_cam[:, 2] > 1e-6]
            if len(verts_cam) == 0:
                return
            uv = (np.asarray(K_np, dtype=np.float64) @ verts_cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]

            img_np = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            H, W = img_np.shape[:2]
            out = Image.fromarray(img_np.copy())
            draw = ImageDraw.Draw(out)
            for x, y in uv:
                if 0 <= x < W and 0 <= y < H:
                    draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(0, 200, 255))
            draw.text((4, 4), f"frame {frame_idx} - megapose", fill=(0, 200, 255))

            out_dir = Path(self.config.debug_reproject_dir)
            if not out_dir.is_absolute():
                try:
                    from hydra.core.hydra_config import HydraConfig
                    from hydra.utils import get_original_cwd
                    if HydraConfig.initialized():
                        out_dir = Path(get_original_cwd()) / out_dir
                except Exception:
                    pass
            out_dir = out_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"frame_{frame_idx:05d}_megapose.png"
            out.save(str(save_path))
            log.info("[debug] MegaPose: saved reprojection for frame %d -> %s", frame_idx, save_path)
        except Exception as e:
            log.warning("MegaPose frame %d: reprojection overlay failed: %s", frame_idx, e)

    def _save_render_debug(self, pose_estimator, observation, data_TCO, frame_idx) -> None:
        """Dump the render-and-compare panels the scoring net compares against:
        [observation crop | rendered RGB | rendered normals]. Re-runs the scoring
        model with return_debug_data, so it must run before the renderer stops.
        """
        try:
            from PIL import Image

            _, extra = pose_estimator.forward_scoring_model(
                observation, data_TCO, return_debug_data=True
            )
            dbg = extra.get("debug", {})
            images_crop = dbg.get("images_crop")
            renders = dbg.get("renders")
            if images_crop is None or renders is None:
                return
            crop = images_crop[0].detach().cpu().float()          # (C, H, W) [0,1]
            rend = renders[0].detach().cpu().float()              # (n_ch, H, W)
            rgb_dims = list(pose_estimator.coarse_model.render_rgb_dims)
            panels = [crop[:3].clamp(0, 1), rend[rgb_dims].clamp(0, 1)]
            normal_dims = list(getattr(pose_estimator.coarse_model, "_render_normal_dims", []))
            if normal_dims:
                nrm = rend[normal_dims]
                panels.append((nrm * 0.5 + 0.5).clamp(0, 1))      # [-1,1] -> [0,1] for display
            strip = torch.cat(panels, dim=2)                      # side-by-side (width)
            arr = (strip.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            out_dir = Path(self.config.debug_render_dir)
            if not out_dir.is_absolute():
                try:
                    from hydra.core.hydra_config import HydraConfig
                    from hydra.utils import get_original_cwd
                    if HydraConfig.initialized():
                        out_dir = Path(get_original_cwd()) / out_dir
                except Exception:
                    pass
            out_dir = out_dir.resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"frame_{frame_idx:05d}_rendercompare.png"
            Image.fromarray(arr).save(str(save_path))
            log.info(
                "[debug] MegaPose: saved render-compare for frame %d -> %s "
                "(panels: observation | render RGB%s)",
                frame_idx, save_path, " | render normals" if normal_dims else "",
            )
        except Exception as e:
            log.warning("MegaPose frame %d: render-compare debug failed: %s", frame_idx, e)

    # Per-frame inference
    @torch.no_grad()
    def inference_single_image(
        self,
        img: torch.Tensor,        # (3, H, W) float [0,1] - full eval image
        joint_angles,
        K=None,                   # (3, 3) real camera intrinsics
        mask=None,                # (H, W) - UNUSED (MegaPose crops by bbox)
        bbox_xyxy=None,           # [x1, y1, x2, y2] - detection bbox (eval space)
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None]:
        from megapose.datasets.scene_dataset import ObjectData
        from megapose.inference.utils import add_instance_id, make_detections_from_object_data

        _t = time.perf_counter()
        frame_idx = self._frame_idx
        self._frame_idx += 1
        self._last_frame_diagnostics = None

        estimator_utils.reseed_all(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()   # baseline for the per-frame VRAM metric

        if bbox_xyxy is None:
            raise ValueError(
                "MegaPose requires a detection bbox (run robot-detector CNOS and "
                "pass detections_path in the eval config)."
            )
        K_np = (K.cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)).astype(np.float64)

        # FK-posed arm mesh (mm, centred) + frame transform
        q = np.asarray(joint_angles, dtype=np.float64)
        mesh, mesh_transform = self.renderer.export_posed_trimesh(q)
        center_m = estimator_utils.center_offset_m(mesh_transform)

        bbox = np.asarray(bbox_xyxy, dtype=np.float32).reshape(4)

        # load_pretrained_refiner loads both nets; run_inference_pipeline scores
        # with the coarse one. coarse_only stops after the coarse stage.
        with megapose_pose_estimator(
            mesh, img, K_np, self.device,
            model_name=self.config.model_name,
            models_root=self.config.megapose_models_root,
            num_workers=self.config.num_workers,
            batch_size_objects=self.config.batch_size_objects,
            batch_size_images=self.config.batch_size_images,
            n_iterations=self._n_refiner_iterations,
            move_coarse_model=True,
        ) as (pose_estimator, observation, obj_label):
            detections = make_detections_from_object_data(
                [ObjectData(label=obj_label, bbox_modal=bbox)]
            ).to(self.device)

            if self.config.coarse_only:
                # The coarse steps of run_inference_pipeline, truncated: the coarse
                # net's own top-1 by coarse_logit.
                detections = add_instance_id(detections)
                data_TCO_coarse, _ = pose_estimator.forward_coarse_model(
                    observation=observation, detections=detections,
                )
                coarse_top1 = pose_estimator.filter_pose_estimates(
                    data_TCO_coarse, top_K=1, filter_field="coarse_logit",
                )
                T_obj_cam = coarse_top1.poses[0].detach().cpu().numpy().astype(np.float64)
                best_score = float(coarse_top1.infos["coarse_logit"].iloc[0])
            else:
                output, _ = pose_estimator.run_inference_pipeline(
                    observation=observation,
                    detections=detections,
                    run_detector=False,
                    n_refiner_iterations=self._n_refiner_iterations,
                    n_pose_hypotheses=self._n_pose_hypotheses,
                )
                T_obj_cam = output.poses[0].detach().cpu().numpy().astype(np.float64)
                best_score = (
                    float(output.infos["pose_score"].iloc[0])
                    if "pose_score" in output.infos else None
                )
                # render-compare debug must run while the renderer is still alive
                if (self.config.debug_render_dir is not None
                        and frame_idx < self.config.n_reproject_frames):
                    self._save_render_debug(pose_estimator, observation, output, frame_idx)

        T_base_cam = estimator_utils.centred_to_base(T_obj_cam, center_m)

        frame_time_ms = round((time.perf_counter() - _t) * 1000, 1)
        gpu_peak_mb = (
            round(torch.cuda.max_memory_allocated() / 1e6, 1)
            if torch.cuda.is_available() else None
        )
        self._last_frame_diagnostics = {
            "frame_time_ms": frame_time_ms,
            "timings_ms": {"total": frame_time_ms},
            "gpu_peak_mb": gpu_peak_mb,
            # coarse_only: this is the coarse net's coarse_logit; full: the refined pose_score.
            "best_pose_score": round(best_score, 4) if best_score is not None else None,
            "coarse_only": self.config.coarse_only,
            "pnp": None,   # MegaPose has no PnP stage
        }

        if (self.config.debug_reproject_dir is not None
                and frame_idx < self.config.n_reproject_frames):
            self._save_reprojection_overlay(img, T_base_cam, K_np, joint_angles, frame_idx)

        return torch.from_numpy(T_base_cam.astype(np.float32)), None, None, None, None
