"""GigaPose adapter for articulated robot pose estimation.

Mirrors ``src/models/gigaPose.py:eval_retrieval`` (Nguyen et al., CVPR 2024).
GigaPose's panda3d renderer preserves the trained appearance distribution. The
robot-arm adaptation derives a no-clip template distance per joint state.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from robop import estimator_utils
from robop.estimator_utils import ensure_gigapose_path, export_ply_with_normals
from robop.megapose_estimator import megapose_pose_estimator

log = logging.getLogger(__name__)


# GigaPose's fixed template-render intrinsics (call_panda3d.py) - 640×480, the
# camera the appearance networks were trained against.  K_render only affects
# how templates *look*; the recovered pose uses the real per-frame query K via
# the focal-ratio term in ObjectPoseRecovery._forward_recovery.
_TEMPLATE_K = np.array(
    [572.4114, 0.0, 320.0, 0.0, 573.57043, 240.0, 0.0, 0.0, 1.0], dtype=np.float64
).reshape(3, 3)
_TEMPLATE_W, _TEMPLATE_H = 640, 480

# CLIP normalization (configs/data/transform.yaml) - NOT ImageNet.
_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
_NORM_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class GigaPoseConfig:
    """Configuration for GigaPoseRobotEstimator (defaults mirror GigaPose's large model)."""

    checkpoint_path: str = ""  # path to gigaPose_v1.ckpt (required)

    # Template viewsphere: level 1 is the paper's 162 views (Sec. 3.1). ist_net
    # regresses in-plane rotation, so templates carry no in-plane copies.
    template_level: int = 1
    pose_distribution: str = "all"   # "all" or "upper" (upper hemisphere only)

    # Robot-arm framing: see the module docstring. The sphere bound is exact
    # because the predefined view poses rotate the object about its own origin.
    template_fit_margin: float = 0.05
    min_template_pixels: int = 100

    # Matching (configs/model/large.yaml: testing_metric).
    k: int = 5                       # top-k templates retrieved per detection
    sim_threshold: float = 0.5
    patch_threshold: int = 3

    # 2D-affine RANSAC (ObjectPoseRecovery default; paper Sec. 3.3: one patch).
    ransac_pixel_threshold: int = 14

    # Backbone selection (must match the checkpoint).
    ae_config: str = "dinov2_l"      # configs/model/ae_net/<ae_config>.yaml
    ist_config: str = "resnet"       # configs/model/ist_net/<ist_config>.yaml

    image_size: int = 224
    patch_size: int = 14

    # Persistence. Default: templates are rendered, encoded and discarded in
    # memory every call. cache_dir opts into a disk cache of the panda3d renders
    # only, keyed by robot name + joint hash; features are always rebuilt.
    cache_dir: Optional[str] = None
    # Renders the first frame's templates (raw + 224 crops) into
    # <dir>/templates_000000/ and raises, for eyeballing arm framing.
    debug_save_templates: Optional[str] = None
    # Projects the FK-posed mesh through the final T_base_cam and the real K onto
    # the eval image for the first n_reproject_frames frames. A frame-convention
    # bug shows up as the dots landing off the arm.
    debug_reproject_dir: Optional[str] = None
    n_reproject_frames: int = 20

    # Published GigaPose adds the MegaPose RGB refiner; false selects the paper's
    # coarse ablation. Articulation requires rebuilding the refiner per frame.
    refine: bool = False
    # test.yaml use_multiple=true: refine all coarse top-k poses and keep the
    # best-scoring one (Table 1 row 9, 57.8 AR). False = single hypothesis (54.7).
    refiner_multi_hypothesis: bool = True
    refiner_model_name: str = "megapose-1.0-RGB-multi-hypothesis"
    n_refiner_iterations: int = 5
    megapose_models_root: str = ""   # path to pretrained/megapose-models/
    refiner_num_workers: int = 4

    seed: int = 0
    verbose: bool = False


class GigaPoseRobotEstimator:
    """GigaPose adapter for articulated robot pose estimation."""

    # Input contract read by evaluation's EstimatorAdapter. No mask opening:
    # upstream consumes the decoded mask directly (dataloader/train.py:process_real).
    INPUT_CONTRACT = {"mask": True, "bbox_xyxy": True, "frame_key": False,
                      "mask_3x3_opening": False}

    def __init__(
        self,
        renderer,
        config: Optional[GigaPoseConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        ensure_gigapose_path()
        # The refiner's panda3d batch renderer passes tensors from forked workers
        # by FD, which overruns the default strategy under multi-hypothesis
        # scoring. MegaPose's own train_megapose.py uses 'file_system' too.
        torch.multiprocessing.set_sharing_strategy("file_system")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.renderer = renderer.to(device)
        self.config = config or GigaPoseConfig()
        if self.config.verbose:
            log.setLevel(logging.DEBUG)

        self._build_networks()

        # template viewpoints (object->camera, OpenCV convention, translations mm)
        # at the nominal 1000 mm distance; _build_templates rescales the
        # translations per joint state.
        from src.lib3d.template_transform import get_obj_poses_from_template_level
        self._template_obj_poses_1m = get_obj_poses_from_template_level(
            level=self.config.template_level,
            pose_distribution=self.config.pose_distribution,
        ).astype(np.float64)                              # (N, 4, 4)

        self._last_frame_diagnostics: Optional[dict] = None
        self._frame_idx = 0
        self._refiner_failures = 0   # publishable full-method runs require zero
        self._render_logged = False   # one-shot guard for the per-frame render line

        # Fail fast: otherwise the run completes and saves "gigapose_megapose"
        # results that are actually coarse GigaPose.
        self._refiner_loader = None
        if self.config.refine:
            self._validate_refiner_setup()

    def _validate_refiner_setup(self) -> None:
        """Check the model name, the files load_pretrained_refiner opens and the
        refiner import, so a setup problem cannot be absorbed per frame.
        """
        root = self.config.megapose_models_root
        if not root or not Path(root).is_dir():
            raise FileNotFoundError(
                f"refine=True but megapose_models_root is not a valid directory: {root!r}. "
                "The MegaPose refiner needs the pretrained models - run "
                "`make download-megapose` and pass "
                "estimator.megapose_models_root=/abs/path/pretrained/megapose-models."
            )
        from src.megapose.utils.load_model import NAMED_MODELS

        name = self.config.refiner_model_name
        if name not in NAMED_MODELS:
            raise ValueError(
                f"Unknown refiner_model_name {name!r}. Available: {sorted(NAMED_MODELS)}."
            )
        info = NAMED_MODELS[name]
        if info.get("requires_depth"):
            raise ValueError(
                f"refiner_model_name {name!r} requires depth, which this RGB-only "
                "pipeline does not provide."
            )
        missing = [
            str(Path(root) / run_id / f)
            for run_id in (info["coarse_run_id"], info["refiner_run_id"])
            for f in ("config.yaml", "checkpoint.pth.tar")
            if not (Path(root) / run_id / f).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"MegaPose refiner files missing under {root}: {missing}. "
                "Run `make download-megapose`."
            )
        # Import at setup: an import error must end the run, not turn every
        # frame into a silent coarse fallback.
        from src.custom_megapose.refiner_utils import load_pretrained_refiner

        self._refiner_loader = load_pretrained_refiner
        log.info("GigaPose refiner: %s validated (coarse %s, refiner %s).",
                 name, info["coarse_run_id"], info["refiner_run_id"])

    @property
    def robot_kin(self):
        return self.renderer._kinematics

    # Network construction - instantiate from GigaPose's own configs, then
    # load checkpoint weights manually (PL-version-agnostic).
    def _build_networks(self) -> None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from src.models.matching import LocalSimilarity

        gp = ensure_gigapose_path()
        ae_yaml = OmegaConf.load(gp / "configs" / "model" / "ae_net" / f"{self.config.ae_config}.yaml")
        ist_yaml = OmegaConf.load(gp / "configs" / "model" / "ist_net" / f"{self.config.ist_config}.yaml")
        # Provide the interpolation parents the yamls reference
        # (${model.ae_net.model_name}, ${model.ist_net.descriptor_size}).
        cfg = OmegaConf.create({"model": {"ae_net": ae_yaml, "ist_net": ist_yaml}})

        log.info("GigaPose: instantiating ae_net (%s) + ist_net (%s).",
                 self.config.ae_config, self.config.ist_config)
        self.ae_net = instantiate(cfg.model.ae_net).to(self.device).eval()
        self.ist_net = instantiate(cfg.model.ist_net).to(self.device).eval()

        self.testing_metric = LocalSimilarity(
            k=self.config.k,
            sim_threshold=self.config.sim_threshold,
            patch_threshold=self.config.patch_threshold,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
        )

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        ckpt_path = self.config.checkpoint_path
        if not ckpt_path or not Path(ckpt_path).is_file():
            raise FileNotFoundError(
                f"GigaPose checkpoint not found: {ckpt_path!r}. "
                "Download via external/gigapose/src/scripts/download_gigapose.py "
                "and set estimator.checkpoint_path."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        # GigaPose LightningModule holds submodules `ae_net.*` and `ist_net.*`.
        ae_sd = {k[len("ae_net."):]: v for k, v in state.items() if k.startswith("ae_net.")}
        ist_sd = {k[len("ist_net."):]: v for k, v in state.items() if k.startswith("ist_net.")}
        if not ae_sd or not ist_sd:
            raise RuntimeError(
                "Checkpoint has no ae_net.*/ist_net.* keys; layout differs from "
                f"expected GigaPose state_dict (got {len(state)} keys)."
            )
        # A wrong checkpoint/config pairing must fail here rather than leave
        # parameters randomly initialized and still produce poses.
        estimator_utils.assert_state_dict_matches(self.ae_net, ae_sd, "GigaPose ae_net")
        estimator_utils.assert_state_dict_matches(self.ist_net, ist_sd, "GigaPose ist_net")
        self.ae_net.load_state_dict(ae_sd, strict=True)
        self.ist_net.load_state_dict(ist_sd, strict=True)
        self.checkpoint_sha256 = estimator_utils.sha256_file(ckpt_path)
        log.info("GigaPose: loaded weights (ae %d + ist %d parameters, sha256 %s).",
                 len(ae_sd), len(ist_sd), self.checkpoint_sha256[:12])

    # Image helpers
    def _normalize(self, rgb_chw: torch.Tensor) -> torch.Tensor:
        """Apply GigaPose's CLIP normalization to a (B,3,H,W) float[0,1] tensor."""
        mean = torch.tensor(_NORM_MEAN, device=rgb_chw.device).view(1, 3, 1, 1)
        std = torch.tensor(_NORM_STD, device=rgb_chw.device).view(1, 3, 1, 1)
        return (rgb_chw - mean) / std

    def _crop_resize_pad(self, rgba_chw: torch.Tensor, box_xyxy: np.ndarray):
        """Run GigaPose's CropResizePad on a single (4,H,W) rgba tensor + xyxy box.

        Returns (rgb_224 [3,224,224] float[0,1], mask_224 [224,224] float, M [3,3]).
        """
        from src.utils.crop import CropResizePad
        crop = CropResizePad(target_size=self.config.image_size, patch_size=self.config.patch_size)
        # CropResizePad uses the box entries as slice indices -> must be integer.
        boxes = torch.as_tensor(box_xyxy, device=self.device).round().long().view(1, 4)
        out = crop(boxes, rgba_chw.unsqueeze(0).to(self.device))
        img4 = out["images"][0]            # (4, 224, 224)
        M = out["M"][0]                    # (3, 3)
        return img4[:3], img4[3], M

    # Stage 1: template onboarding
    def _joint_hash(self, q: np.ndarray) -> str:
        return hashlib.md5(q.tobytes()).hexdigest()

    def _template_config_hash(self) -> str:
        """Hash of everything that changes the rendered images for a given joint
        config (mirrors FoundPose's _config_hash). Without it, cached renders
        from a different distance would produce scale-wrong poses.
        """
        c = self.config
        fields = (c.template_level, c.pose_distribution, c.template_fit_margin)
        return hashlib.md5(str(fields).encode()).hexdigest()[:12]

    def _render_templates_panda3d(self, mesh_ply: Path, render_dir: Path, obj_poses: np.ndarray) -> None:
        """Render the viewsphere with GigaPose's own call_panda3d (subprocess).

        Writes NNNNNN.png (rgba) + NNNNNN_depth.png into `render_dir`.
        Mirrors render_custom_templates.call_render.
        """
        # Use ABSOLUTE paths: the subprocess runs with cwd=external/gigapose, so
        # relative paths (resolved against the eval's Hydra cwd) would not be found.
        mesh_ply = Path(mesh_ply).resolve()
        render_dir = Path(render_dir).resolve()
        render_dir.mkdir(parents=True, exist_ok=True)
        pose_path = render_dir / "object_poses.npy"
        np.save(pose_path, obj_poses)

        # call_panda3d overwrites CUDA_VISIBLE_DEVICES and EGL_VISIBLE_DEVICES with
        # this argument, so pass what the parent already has: hard-coding "0" would
        # select the first physical device instead of the one a scheduler allocated.
        gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0") or "0"
        cmd = [
            sys.executable, "-m", "src.custom_megapose.call_panda3d",
            str(mesh_ply), str(pose_path), str(render_dir), gpu_devices,
            "true",   # disable_output
            "true",   # scale_translation to meter (mesh is in mm)
        ]
        # Every frame is a fresh joint config, so log the render once at INFO.
        if not self._render_logged:
            log.info("GigaPose: rendering %d templates/frame via internal panda3d renderer "
                     "(further per-frame renders logged at DEBUG only).",
                     len(obj_poses))
            self._render_logged = True
        else:
            log.debug("GigaPose: rendering %d templates via call_panda3d.", len(obj_poses))
        # call_panda3d uses `from megapose...` (top-level), so the subprocess
        # needs external/gigapose/src on PYTHONPATH (+ the repo root for src.*).
        gp_root = ensure_gigapose_path()
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(gp_root / "src"), str(gp_root), env.get("PYTHONPATH", "")]
        )
        # Capture panda3d's startup chatter; surface it only on failure.
        result = subprocess.run(
            cmd, cwd=str(gp_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"GigaPose call_panda3d failed (exit {result.returncode}):\n{result.stdout}"
            )

    def _finalize_templates(self, rgb_list, mask_list, M_list, pose_list, template_K, conv) -> dict:
        """Shared tail: encode ae/ist features + build ObjectPoseRecovery + cache.

        conv carries the obj->base conversion info used in inference_single_image:
        {"conv_type": "panda3d", "mesh_transform": (4,4) ndarray}.
        """
        from src.models.poses import ObjectPoseRecovery

        rgb = torch.stack(rgb_list).to(self.device)        # (N, 3, 224, 224)
        mask = torch.stack(mask_list).to(self.device)      # (N, 224, 224)
        M = torch.stack(M_list).to(self.device)            # (N, 3, 3)
        poses = torch.stack(pose_list).to(self.device)     # (N, 4, 4)
        K = (
            torch.from_numpy(template_K).float()
            if isinstance(template_K, np.ndarray)
            else template_K.float()
        ).to(self.device)

        with torch.no_grad():
            ae_features = self.ae_net(self._normalize(rgb))            # (N, C, 16, 16)
            ist_features = self.ist_net.forward_by_chunk(self._normalize(rgb))

        pose_recovery = ObjectPoseRecovery(
            template_K=K.unsqueeze(0),                # (1, 3, 3)
            template_Ms=M.unsqueeze(0),               # (1, N, 3, 3)
            template_poses=poses.unsqueeze(0),        # (1, N, 4, 4)
            pixel_threshold=self.config.ransac_pixel_threshold,
        )
        return {
            "ae_features": ae_features.unsqueeze(0),  # (1, N, C, 16, 16)
            "ist_features": ist_features.unsqueeze(0),
            "mask": mask.unsqueeze(0),                # (1, N, 224, 224)
            "pose_recovery": pose_recovery,
            "n_templates": rgb.shape[0],
            **conv,
        }

    def _template_distance_mm(self, mesh) -> Tuple[float, float, str]:
        """Template-camera distance (mm) for one posed mesh + its bounding radius.

        Derives the distance from the mesh's origin-centred bounding sphere so
        every viewsphere render fits the 640×480 template frame with
        template_fit_margin of border room.
        """
        radius_mm = estimator_utils.bounding_radius_mm(np.asarray(mesh.vertices))
        distance_mm = estimator_utils.viewsphere_distance_mm(
            radius_mm, _TEMPLATE_K, _TEMPLATE_W, _TEMPLATE_H,
            margin=self.config.template_fit_margin,
        )
        return distance_mm, radius_mm, "auto"

    def _handle_fit_problems(self, problems, distance_mm, radius_mm, dist_mode, what) -> None:
        # The distance is always derived, so no override hint is needed.
        estimator_utils.handle_fit_problems(
            problems, "GigaPose", "",
            distance_mm, radius_mm, dist_mode, what,
        )

    def _build_templates(self, joint_angles) -> dict:
        """Render + encode templates with GigaPose's own panda3d renderer.

        In-memory by default. A cache dir counts as reusable only when its
        manifest passes validation, so an interrupted or stale-policy render is
        redone instead of silently truncating the template set.
        """
        from PIL import Image

        q = np.asarray(joint_angles, dtype=np.float64)
        jhash = self._joint_hash(q)

        # export the FK-posed arm mesh (mm, centred, vertex colours) and its frame transform
        mesh, mesh_transform = self.renderer.export_posed_trimesh(q)

        distance_mm, radius_mm, dist_mode = self._template_distance_mm(mesh)
        n_views = len(self._template_obj_poses_1m)
        # Recorded in the manifest, not revalidated against it on load: every
        # field here is already part of _template_config_hash()/_joint_hash(),
        # which key the cache directory.
        manifest_meta = {
            "estimator": "gigapose",
            "n_views": n_views,
            "width": _TEMPLATE_W,
            "height": _TEMPLATE_H,
            "template_level": self.config.template_level,
            "pose_distribution": self.config.pose_distribution,
            "distance_mode": dist_mode,
            "distance_mm": distance_mm,
        }

        # decide where renders live
        tmp_ctx = None
        manifest = None
        if self.config.cache_dir is not None:
            render_dir = (
                Path(self.config.cache_dir) / self.renderer.name
                / self._template_config_hash() / jhash
            )
            manifest = estimator_utils.load_template_manifest(render_dir)
        elif self.config.debug_save_templates is not None:
            render_dir = Path(self.config.debug_save_templates) / f"templates_{self._frame_idx:06d}"
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            render_dir = Path(tmp_ctx.name)
        cached = manifest is not None

        if cached:
            problems = estimator_utils.manifest_problems(manifest, render_dir)
            self._handle_fit_problems(problems, distance_mm, radius_mm, dist_mode,
                                      f"cached renders at {render_dir}")
            # reuse the manifest's recorded distance so the poses fed to
            # ObjectPoseRecovery match the cached renders exactly (recomputing
            # may drift sub-mm across mesh/library versions; larger drift
            # fails the validation above)
            distance_mm = float(manifest.get("meta", {}).get("distance_mm", distance_mm))

        obj_poses = self._template_obj_poses_1m.copy()
        obj_poses[:, :3, 3] *= distance_mm / 1000.0

        try:
            if not cached:
                render_dir.mkdir(parents=True, exist_ok=True)
                mesh_ply = render_dir / "arm.ply"
                export_ply_with_normals(mesh, mesh_ply)
                self._render_templates_panda3d(mesh_ply, render_dir, obj_poses)
            else:
                log.debug("GigaPose: reusing cached renders at %s", render_dir)

            rgb_list, mask_list, M_list, pose_list = [], [], [], []
            problems, view_records = [], []
            for i in range(n_views):
                png = render_dir / f"{i:06d}.png"
                if not png.is_file():
                    # call_panda3d writes every view unconditionally, so a
                    # missing file means the render died partway
                    problems.append(f"{png.name}: render missing")
                    view_records.append({"file": png.name, "n_pixels": 0, "bbox_xyxy": None,
                                         "touches_border": False, "ok": False})
                    continue
                rgba = np.array(Image.open(png).convert("RGBA"))            # (H,W,4), writable copy
                stats = estimator_utils.check_mask_fit(rgba[:, :, 3], self.config.min_template_pixels)
                record = {"file": png.name, **stats}
                if not cached and not (render_dir / f"{i:06d}_depth.png").is_file():
                    record["ok"] = False
                    problems.append(f"{i:06d}_depth.png: depth render missing")
                view_records.append(record)
                if not record["ok"]:
                    if stats["n_pixels"] < self.config.min_template_pixels or stats["touches_border"]:
                        problems.append(
                            f"{png.name}: n_pixels={stats['n_pixels']}, "
                            f"touches_border={stats['touches_border']}"
                        )
                    continue
                rgba_chw = torch.from_numpy(rgba).float().permute(2, 0, 1) / 255.0
                # bbox is max-EXCLUSIVE, matching CropResizePad's slice semantics
                box = np.array(stats["bbox_xyxy"], dtype=np.float32)
                rgb_224, mask_224, M = self._crop_resize_pad(rgba_chw, box)
                rgb_list.append(rgb_224)
                mask_list.append(mask_224)
                M_list.append(M)
                # template pose in METRES (geometry was rendered at metres -> /1000)
                pose_m = obj_poses[i].copy()
                pose_m[:3, 3] /= 1000.0
                pose_list.append(torch.from_numpy(pose_m).float())

            self._handle_fit_problems(problems, distance_mm, radius_mm, dist_mode,
                                      f"renders at {render_dir}")
            if tmp_ctx is None and not cached:
                estimator_utils.write_template_manifest(
                    render_dir,
                    meta={**manifest_meta,
                          "radius_mm": radius_mm,
                          "template_fit_margin": self.config.template_fit_margin,
                          "min_template_pixels": self.config.min_template_pixels,
                          "K": _TEMPLATE_K.tolist()},
                    views=view_records,
                )

            # debug: dump the cropped 224 templates the network actually sees
            if self.config.debug_save_templates is not None and rgb_list:
                crop_dir = render_dir / "cropped"
                crop_dir.mkdir(parents=True, exist_ok=True)
                for i, rgb_224 in enumerate(rgb_list):
                    arr = (rgb_224.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(arr).save(crop_dir / f"crop_{i:03d}.png")
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        if not rgb_list:
            raise RuntimeError(
                "GigaPose: no templates passed fit acceptance - check the render log and the mesh."
            )

        return self._finalize_templates(
            rgb_list, mask_list, M_list, pose_list,
            template_K=_TEMPLATE_K,
            conv={"conv_type": "panda3d", "mesh_transform": mesh_transform},
        )

    # Second stage: MegaPose RGB render-and-compare refinement
    def _obj_cam_to_base(self, T_obj_cam, cache) -> np.ndarray:
        """Recovered object-to-camera pose (metres) to the URDF base frame."""
        center_m = estimator_utils.center_offset_m(cache["mesh_transform"])
        return estimator_utils.centred_to_base(T_obj_cam, center_m)

    def _refine_pose(self, hypotheses, img, K_np, joint_angles):
        """Refine coarse hypotheses [(T_base_cam, matching_score), best-first].

        Mirrors src/models/refiner.py Refiner.test_step: refine all hypotheses
        with forward_refiner, then score each with MegaPose's coarse net and keep
        the best (filter_pose_estimates top_K=1 by pose_logit). With one
        hypothesis that filter is the identity, so the scoring pass is skipped.
        Returns the best refined T_base_cam.
        """
        import pandas as pd
        from megapose.utils.tensor_collection import PandasTensorCollection

        device = self.device
        q = np.asarray(joint_angles, dtype=np.float64)
        mesh, mesh_transform = self.renderer.export_posed_trimesh(q)
        center_m = estimator_utils.center_offset_m(mesh_transform)
        multi = len(hypotheses) > 1

        # the refiner works in the centred-mesh frame MegaPose renders from
        TCO_list, match_scores = [], []
        for T_base_cam, mscore in hypotheses:
            TCO_list.append(estimator_utils.base_to_centred(T_base_cam, center_m))
            match_scores.append(float(mscore))

        with megapose_pose_estimator(
            mesh, img, K_np, device,
            model_name=self.config.refiner_model_name,
            models_root=self.config.megapose_models_root,
            num_workers=self.config.refiner_num_workers,
            batch_size_objects=8,
            batch_size_images=512,
            n_iterations=self.config.n_refiner_iterations,
            # the scorer (MegaPose coarse net) is only needed for multi-hyp selection
            move_coarse_model=multi,
            loader=self._refiner_loader,   # imported and checked at setup
        ) as (pose_estimator, observation, obj_label):
            n_hyp = len(TCO_list)
            poses_init = torch.from_numpy(np.stack(TCO_list).astype(np.float32)).to(device)  # (n_hyp,4,4)
            data_TCO = PandasTensorCollection(
                infos=pd.DataFrame({
                    "label": [obj_label] * n_hyp,
                    "batch_im_id": [0] * n_hyp,
                    "instance_id": [0] * n_hyp,   # same detection -> one group for the filter
                    "matching_score": match_scores,
                }),
                poses=poses_init,
            )
            preds, _ = pose_estimator.forward_refiner(
                observation=observation,
                data_TCO_input=data_TCO,
                n_iterations=self.config.n_refiner_iterations,
                keep_all_outputs=False,
                cuda_timer=None,
            )
            data_TCO_ref = preds[f"iteration={self.config.n_refiner_iterations}"]
            if multi:
                # score each refined hypothesis with MegaPose's coarse net and
                # keep the best - mirrors refiner.py test_step (use_multiple=True).
                data_TCO_scored, _ = pose_estimator.forward_scoring_model(
                    observation, data_TCO_ref
                )
                best = pose_estimator.filter_pose_estimates(
                    data_TCO_scored, top_K=1, filter_field="pose_logit"
                )
                T_obj_cam_ref = best.poses[0].detach().cpu().numpy().astype(np.float64)
            else:
                T_obj_cam_ref = data_TCO_ref.poses[0].detach().cpu().numpy().astype(np.float64)

        return estimator_utils.centred_to_base(T_obj_cam_ref, center_m)

    # Debug: reproject the recovered pose onto the eval image
    def _save_reprojection_overlay(
        self, img, T_base_cam, K_np, joint_angles, frame_idx, tag, bbox_xyxy=None
    ) -> None:
        """Project the FK-posed mesh through T_base_cam and the real K onto the
        eval image. Tests the exact pose the ADD metric consumes: a frame
        convention bug shows up as the dots landing off the arm.
        """
        # Debug-only: never let an overlay failure crash an eval run.
        try:
            from PIL import Image, ImageDraw

            q = np.asarray(joint_angles, dtype=np.float64)
            mesh, mt = self.renderer.export_posed_trimesh(q)
            # export mesh is centred-mm; mt[:3,3] = -center_mm -> un-centre to URDF base (m)
            center_mm = -np.asarray(mt, dtype=np.float64)[:3, 3]
            verts_base_m = (np.asarray(mesh.vertices, dtype=np.float64) + center_mm) / 1000.0
            if len(verts_base_m) > 3000:                   # subsample for speed
                verts_base_m = verts_base_m[np.linspace(0, len(verts_base_m) - 1, 3000, dtype=int)]

            T = np.asarray(T_base_cam, dtype=np.float64)
            verts_cam = verts_base_m @ T[:3, :3].T + T[:3, 3]  # URDF base -> camera (m)
            verts_cam = verts_cam[verts_cam[:, 2] > 1e-6]      # keep points in front
            if len(verts_cam) == 0:
                return
            uv = (np.asarray(K_np, dtype=np.float64) @ verts_cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]

            img_np = (img.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            H, W = img_np.shape[:2]
            out = Image.fromarray(img_np.copy())
            draw = ImageDraw.Draw(out)
            if bbox_xyxy is not None:
                x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy).flatten()[:4]]
                draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=2)
            color = (0, 255, 80) if tag == "coarse" else (60, 140, 255)
            for x, y in uv:
                if 0 <= x < W and 0 <= y < H:
                    draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)
            draw.text((4, 4), f"frame {frame_idx} - {tag}", fill=color)

            # Hydra chdir's into its run dir at runtime, so a relative path would
            # land there (invisible from the launch dir). Resolve against the
            # original launch cwd and log the absolute path so it's findable.
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
            save_path = out_dir / f"frame_{frame_idx:05d}_{tag}.png"
            out.save(str(save_path))
            log.info("[debug] GigaPose: saved %s reprojection for frame %d -> %s", tag, frame_idx, save_path)
        except Exception as e:
            log.warning("GigaPose frame %d: reprojection overlay (%s) failed: %s", frame_idx, tag, e)

    # Stage 2/3: per-frame inference
    @torch.no_grad()
    def inference_single_image(
        self,
        img: torch.Tensor,        # (3, H, W) float [0,1] - full eval image
        joint_angles,
        K=None,                   # (3, 3) real camera intrinsics
        mask=None,                # (H, W) uint8/bool - detection mask (eval space)
        bbox_xyxy=None,           # [x1, y1, x2, y2] - detection bbox (eval space)
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None]:

        _t = time.perf_counter()
        frame_idx = self._frame_idx
        self._frame_idx += 1
        self._last_frame_diagnostics = None

        estimator_utils.reseed_all(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()   # baseline for the per-frame VRAM metric

        if bbox_xyxy is None or mask is None:
            raise ValueError(
                "GigaPose requires a detection bbox + mask (run robot-detector "
                "CNOS and pass detections_path in the eval config)."
            )
        K_np = (K.cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)).astype(np.float64)

        cache = self._build_templates(joint_angles)
        pose_recovery = cache["pose_recovery"]
        n_templates = cache["n_templates"]

        # NeMO-style debug: save templates on the first frame, then stop.
        if self.config.debug_save_templates is not None:
            log.info(
                "[debug] GigaPose: saved %d templates (+ cropped crops) to %s/templates_%06d - "
                "stopping after frame %d.",
                n_templates, self.config.debug_save_templates, frame_idx, frame_idx,
            )
            raise RuntimeError(f"debug_save_templates: stopping after frame {frame_idx}")

        # process_real masks the query background before applying CropResizePad,
        # matching the object-on-black templates.
        mask_t = (torch.from_numpy(np.asarray(mask)).to(self.device) > 0).float()  # (H, W) binary
        m_rgb = img.to(self.device) * mask_t.unsqueeze(0)   # zero background
        rgba = torch.cat([m_rgb, mask_t.unsqueeze(0)], dim=0)   # (4, H, W)
        box = np.asarray(bbox_xyxy, dtype=np.float32)
        tar_rgb, tar_mask_img, tar_M = self._crop_resize_pad(rgba, box)
        tar_img = self._normalize(tar_rgb.unsqueeze(0))     # (1, 3, 224, 224)
        tar_mask = (tar_mask_img > 0.5).float().unsqueeze(0)  # (1, 224, 224)
        tar_K = torch.from_numpy(K_np).float().unsqueeze(0).to(self.device)
        tar_M = tar_M.unsqueeze(0)

        # appearance NN retrieval
        tar_ae = self.ae_net(tar_img)                       # (1, C, 16, 16)
        predictions = self.testing_metric.test(
            src_feats=cache["ae_features"],                 # (1, N, C, 16, 16)
            tar_feat=tar_ae,
            src_masks=cache["mask"],                         # (1, N, 224, 224)
            tar_mask=tar_mask,
        )
        # id_src: (1, k); src_pts/tar_pts: (1, k, P, 2)

        # in-plane + scale regression per retrieved template
        k = self.config.k
        B = 1
        device = self.device
        num_patches = predictions.src_pts.shape[2]
        pred_scales = torch.zeros(B, k, num_patches, device=device)
        pred_cosSin = torch.zeros(B, k, num_patches, 2, device=device)
        idx_sample = torch.arange(B, device=device)
        tar_ist = self.ist_net.forward_by_chunk(tar_img)
        for idx_k in range(k):
            src_ist = cache["ist_features"][[0]][idx_sample, predictions.id_src[:, idx_k]]
            pred_scales[:, idx_k], pred_cosSin[:, idx_k] = self.ist_net.inference(
                src_feat=src_ist,
                tar_feat=tar_ist,
                src_pts=predictions.src_pts[:, idx_k],
                tar_pts=predictions.tar_pts[:, idx_k],
            )
        predictions.register_tensor("relScale", pred_scales)
        predictions.register_tensor("relInplane", pred_cosSin)

        # 2D-affine RANSAC -> analytic pose recovery
        predictions = pose_recovery.forward_ransac(predictions=predictions)
        score = torch.sum(predictions.ransac_scores, dim=2) / num_patches
        predictions.register_tensor("scores", score)

        tar_label = torch.ones(B, dtype=torch.long, device=device)  # single object -> id 1
        pred_poses = pose_recovery.forward_recovery(
            tar_label=tar_label,
            tar_K=tar_K,
            tar_M=tar_M,
            pred_src_views=predictions.id_src,
            pred_M=predictions.M.clone(),
        )                                                    # (1, k, 4, 4), object->camera, metres

        best_k = int(torch.argmax(score[0]).item())
        best_score = float(score[0, best_k].item())
        T_obj_cam = pred_poses[0, best_k].cpu().numpy().astype(np.float64)

        coarse_time_ms = round((time.perf_counter() - _t) * 1000, 1)
        # forward_ransac calls RANSAC without weights, so ransac_scores is a raw
        # inlier indicator. Valid correspondences use upstream's own test
        # (ransac.py:forward, src_pts[:, 0] != -1), not the full 16x16 grid.
        n_inliers = int((predictions.ransac_scores[0, best_k] > 0).sum().item())
        n_corr = int((predictions.src_pts[0, best_k, :, 0] != -1).sum().item())
        diag = {
            # closed after refinement below, so the reported cost is the full method
            "frame_time_ms": coarse_time_ms,
            "timings_ms": {"coarse": coarse_time_ms, "total": coarse_time_ms},
            "gpu_peak_mb": None,
            "refinement": {"enabled": bool(self.config.refine), "attempted": False,
                           "succeeded": False, "n_hypotheses": 0},
            "n_templates": n_templates,
            # upstream's own template score: inliers over the full patch grid
            # (gigaPose.py:eval_retrieval).
            "best_template_score": round(best_score, 4),
            # GigaPose has no PnP: 2D-affine RANSAC stats go in the pnp slot so
            # aggregate_frame_diagnostics still works.
            "pnp": {
                "total_correspondences": n_corr,
                "inlier_count": n_inliers,
                "inlier_ratio": round(n_inliers / max(n_corr, 1), 4),
                "mean_reproj_error_px": None,
            },
        }
        self._last_frame_diagnostics = diag

        def _close_timing() -> None:
            """Cost for the whole call, so refine runs are not timed as coarse."""
            diag["timings_ms"]["total"] = round((time.perf_counter() - _t) * 1000, 1)
            diag["frame_time_ms"] = diag["timings_ms"]["total"]
            diag["gpu_peak_mb"] = (
                round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if torch.cuda.is_available() else None
            )

        if best_score <= 0.0:
            log.warning("GigaPose frame %d: best template score <= 0 - retrieval likely failed.", frame_idx)
            diag["zero_score_best"] = True

        # object -> URDF base (metres) - best coarse pose
        T_base_cam = self._obj_cam_to_base(T_obj_cam, cache)

        # debug: reproject the coarse pose onto the eval image
        _do_reproject = (
            self.config.debug_reproject_dir is not None
            and frame_idx < self.config.n_reproject_frames
        )
        if _do_reproject:
            self._save_reprojection_overlay(
                img, T_base_cam, K_np, joint_angles, frame_idx, "coarse", bbox_xyxy
            )

        # second stage: MegaPose render-and-compare refinement (faithful full method)
        if self.config.refine:
            try:
                if self.config.refiner_multi_hypothesis and k > 1:
                    # GigaPose use_multiple=True: refine ALL coarse top-k poses,
                    # score each with MegaPose's coarse net, keep the best.
                    order = torch.argsort(score[0], descending=True)
                    hypotheses = []
                    for i in order:
                        i = int(i)
                        s = float(score[0, i].item())
                        hypotheses.append(
                            (self._obj_cam_to_base(pred_poses[0, i].cpu().numpy(), cache), s)
                        )
                    if not hypotheses:
                        hypotheses = [(T_base_cam, best_score)]
                else:
                    hypotheses = [(T_base_cam, best_score)]
                diag["refinement"].update(attempted=True, n_hypotheses=len(hypotheses))
                T_base_cam = self._refine_pose(hypotheses, img, K_np, joint_angles)
                diag["refinement"]["succeeded"] = True
                if _do_reproject:
                    self._save_reprojection_overlay(
                        img, T_base_cam, K_np, joint_angles, frame_idx, "refined", bbox_xyxy
                    )
            except Exception as e:
                # Setup problems are rejected at construction, so what reaches
                # here is a per-frame failure: keep the coarse pose and count it.
                self._refiner_failures += 1
                diag["refinement"].update(succeeded=False, error=f"{type(e).__name__}: {e}")
                log.warning("GigaPose frame %d: refinement failed (%s) - using coarse pose "
                            "(%d of %d frames so far).",
                            frame_idx, e, self._refiner_failures, self._frame_idx)

        _close_timing()
        return torch.from_numpy(T_base_cam.astype(np.float32)), None, None, None, None
