"""FoundPose adapter for articulated robot pose estimation.

Follows upstream ``gen_templates.py``, ``gen_repre.py`` and ``infer.py``.
Templates may be cached by joint configuration; feature representations are
rebuilt in memory because serializing one per configuration is impractical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from robop import estimator_utils
from robop.estimator_utils import suppress_c_stdout_stderr

log = logging.getLogger(__name__)

# DINOv2 patch size; the raw template viewport side is snapped down to a
# multiple of it, as in the upstream generator (features_patch_size).
_DINO_PATCH_SIZE = 14

# Silence faiss's AVX512/AVX2 fallback INFO messages (not errors - faiss loads fine).
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


# FoundPose's absl import replaces Hydra's root handler. Restore this snapshot
# after lazy imports.
_HYDRA_HANDLERS: Optional[list] = None
_HYDRA_LEVEL:    Optional[int]  = None


def _capture_hydra_handlers() -> None:
    """Snapshot the root logger's handler list.  Call once before any absl import."""
    global _HYDRA_HANDLERS, _HYDRA_LEVEL
    if _HYDRA_HANDLERS is None:
        _root = logging.getLogger()
        _HYDRA_HANDLERS = list(_root.handlers)
        _HYDRA_LEVEL    = _root.level


def _restore_logging() -> None:
    """Restore pre-absl root handlers and suppress absl INFO messages."""
    if _HYDRA_HANDLERS is not None:
        _root = logging.getLogger()
        for _h in list(_root.handlers):
            if _h not in _HYDRA_HANDLERS:
                _root.removeHandler(_h)
        for _h in _HYDRA_HANDLERS:
            if _h not in _root.handlers:
                _root.addHandler(_h)
        if _HYDRA_LEVEL is not None:
            _root.setLevel(min(_root.level, _HYDRA_LEVEL))
    try:
        import absl.logging as _absl_log
        _absl_log.set_verbosity(_absl_log.WARNING)
    except ImportError:
        pass
    for name in ("template_util", "repre_util", "cluster_util", "feature_util",
                 "corresp_util", "knn_util", "projector_util", "pnp_util",
                 "acceleratesupport", "OpenGL"):
        logging.getLogger(name).setLevel(logging.WARNING)

# Lazy FoundPose path injection
_FP_ROOT: Optional[Path] = None


def _ensure_foundpose_path() -> None:
    global _FP_ROOT
    if _FP_ROOT is not None:
        return
    here = Path(__file__).resolve().parent          # src/robop/
    fp = here.parent.parent / "external" / "foundpose"
    if not fp.is_dir():
        raise RuntimeError(
            f"FoundPose not found at {fp}. "
            "Clone it: git submodule update --init external/foundpose"
        )
    # external/dinov2 too: upstream's dinov2_utils imports dinov2.hub.backbones
    # from the DINOv2 source shipped inside FoundPose.
    for p in (str(fp / "external" / "dinov2"), str(fp)):
        if p not in sys.path:
            sys.path.insert(0, p)
    _FP_ROOT = fp
    _restore_logging()


# DINOv2 wrapper using the source bundled with FoundPose.

class _DINOv2Extractor(torch.nn.Module):
    """
    torch-hub DINOv2 -> {"feature_maps": (B, D, pH, pW)}.

    Replicates FoundPose's DinoFeatureExtractor with facet=token, norm=True:
    hooks into an intermediate transformer block, removes CLS + register tokens,
    applies LayerNorm. The layer default here is the released LMO config's 9; the
    shipped foundpose.yaml passes the paper's 18 (see its comment).
    """

    PATCH_SIZE = 14

    def __init__(self, model_name: str, device: torch.device, layer: int = 9) -> None:
        super().__init__()
        # Build from the DINOv2 source shipped inside FoundPose, as upstream's
        # dinov2_utils.DinoFeatureExtractor does. torch.hub.load would resolve
        # the implementation from a remote default branch that moves
        # independently of the pinned submodule.
        _ensure_foundpose_path()
        import dinov2.hub.backbones as dinov2_backbones

        hub_name = model_name.replace("-", "_")
        if hub_name not in dinov2_backbones.__dict__:
            raise ValueError(
                f"Unknown DINOv2 model {model_name!r}. Available: "
                f"{sorted(k for k in dinov2_backbones.__dict__ if k.startswith('dinov2_'))}"
            )
        # the vendored builder loads its pretrained weights with strict=True
        self._model = dinov2_backbones.__dict__[hub_name](pretrained=True).eval().to(device)
        self._device = device
        self._layer = layer
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std",  std)

    def forward(self, images: torch.Tensor) -> dict:
        x = (images.to(self._device).clamp(0.0, 1.0) - self._mean) / self._std

        captured: list = []
        handle = self._model.blocks[self._layer].register_forward_hook(
            lambda _m, _inp, out: captured.append(out)
        )
        with torch.no_grad():
            self._model(x)
        handle.remove()

        tokens = captured[0]  # (B, 1 + num_reg + P, D)

        # Strip CLS + register tokens; apply the model's final LayerNorm.
        num_reg = getattr(self._model, "num_register_tokens", 0)
        cls_tok   = tokens[:, :1, :]
        patch_tok = tokens[:, 1 + num_reg:, :]
        normed    = self._model.norm(torch.cat([cls_tok, patch_tok], dim=1))
        patch_tok = normed[:, 1:, :]  # (B, P, D)

        B, P, D = patch_tok.shape
        H, W = x.shape[2], x.shape[3]
        pH, pW = H // self.PATCH_SIZE, W // self.PATCH_SIZE
        return {"feature_maps": patch_tok.reshape(B, pH, pW, D).permute(0, 3, 1, 2)}


# Config

@dataclass
class FoundPoseConfig:
    """Configuration for FoundPoseRobotEstimator (mirrors FoundPose's original defaults)."""

    # DINOv2 backbone - paper uses ViT-L/14 with register tokens.
    # (The open-source LMO release uses the weaker dinov2_vits14_reg + layer 9.)
    dino_model: str = "dinov2_vitl14_reg"
    # Intermediate transformer block to extract features from (0-indexed).
    # Paper: layer 18 of ViT-L/14 (24 blocks, 0-indexed).
    dino_layer: int = 18

    # Template rendering resolution (DINOv2-aligned: 420 = 14 × 30 patches).
    render_size: int = 420

    # Fibonacci viewsphere viewpoints (original: 57).
    num_views: int = 57
    # In-plane rotations per viewpoint -> total templates = num_views × num_inplane_rotations.
    # Original: 14 rotations -> ~57 × 14 = 798 templates.
    num_inplane_rotations: int = 14

    # Grid sampling density in px - one DINOv2 patch per cell.
    # Paper §4: "30×30 patch descriptors from each template/crop" -> 420/14 = 30.
    # The open-source infer.py uses 1.0 (dense) + CNOS mask filtering.  With a
    # CNOS mask available the patch-aligned 14.0 grid matches the paper.
    grid_cell_size: float = 14.0

    # Padding fraction around the object bounding box for the query crop.
    # Mirrors infer.py crop_rel_pad=0.2 (object fills ~83 % of the crop).
    crop_rel_pad: float = 0.2

    apply_pca: bool = True
    pca_components: int = 256

    cluster_num: int = 2048

    match_top_n_templates: int = 5
    match_top_k_buddies: int = 300

    # Paper §4: "PnP-RANSAC running for up to 400 iterations".
    # The open-source infer.py uses 1000, but the paper value is 400.
    pnp_ransac_iter: int = 400
    pnp_reproj_error: float = 10.0
    pnp_confidence: float = 0.99
    pnp_refine_lm: bool = True
    min_correspondences: int = 6

    # None derives a no-clip distance per joint state; a number pins the
    # dataset-wide distance used by the upstream generator.
    sphere_distance_mm: Optional[float] = None
    # Supersampling factor for template rendering; upstream gen_templates.py
    # renders at 4x and downsamples (INTER_AREA for color, NEAREST otherwise).
    ssaa_factor: float = 4.0
    # Border room used by the auto distance (fraction of the usable half extent).
    template_fit_margin: float = 0.05
    # Minimum foreground pixels per template at the final render_size scale.
    min_template_pixels: int = 100

    # Random seed for PnP RANSAC and any other stochastic ops.  Mirrors NeMO's
    # per-frame re-seeding pattern so results are reproducible across runs.
    seed: int = 0

    # Disk cache root.  None = render + build fresh every call (no persistence).
    cache_dir: Optional[str] = None

    # Worker processes for the template warp stage (per-view CPU numpy/cv2).
    # None = one per allocated CPU (SLURM affinity); 1 = sequential in-process.
    warp_workers: Optional[int] = None

    # Directory to save per-frame debug images (query crop, retrieved templates,
    # correspondences).  None = disabled.  Images are written as
    # {debug_dir}/frame_{N:05d}_{stage}.png.
    debug_dir: Optional[str] = None

    # Enable DEBUG-level logging for this estimator.
    # Equivalent to: hydra.job_logging.loggers.robop.foundpose_estimator.level=DEBUG
    verbose: bool = False


# Main estimator

class FoundPoseRobotEstimator:
    """
    FoundPose adapter for articulated robot pose estimation.

    Args:
        renderer:  A :class:`robot_renderer.RobotRenderer` instance - used only
                   for :meth:`export_posed_trimesh`.
        config:    :class:`FoundPoseConfig`.
        device:    Torch device.
        backbone:  Optional pre-loaded DINOv2 backbone (torch-hub style) to reuse.
    """

    # Input contract read by evaluation's EstimatorAdapter. The raw template
    # camera pins to the first frame's K; the query path uses K per frame.
    # mask_3x3_opening mirrors upstream infer_pose_util's mask preprocessing.
    INPUT_CONTRACT = {"mask": True, "bbox_xyxy": True, "frame_key": False,
                      "mask_3x3_opening": True}

    def __init__(
        self,
        renderer,
        config: Optional[FoundPoseConfig] = None,
        device: Optional[torch.device] = None,
        backbone: Optional[torch.nn.Module] = None,
    ) -> None:
        _capture_hydra_handlers()  # must be first - snapshot before any absl-importing import
        _ensure_foundpose_path()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device   = device
        self.renderer = renderer.to(device)
        self.config   = config or FoundPoseConfig()

        if self.config.verbose:
            log.setLevel(logging.DEBUG)

        if backbone is not None:
            log.info("FoundPose: wrapping pre-loaded backbone.")
            ext = _DINOv2Extractor.__new__(_DINOv2Extractor)
            torch.nn.Module.__init__(ext)
            ext._model     = backbone.eval().to(device)
            ext._device    = device
            ext._layer     = self.config.dino_layer
            ext.PATCH_SIZE = 14
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            ext.register_buffer("_mean", mean)
            ext.register_buffer("_std",  std)
            self._extractor = ext
        else:
            log.info("FoundPose: loading DINOv2 %s (layer %d).", self.config.dino_model, self.config.dino_layer)
            self._extractor = _DINOv2Extractor(self.config.dino_model, device, layer=self.config.dino_layer)

        log.info("FoundPose: backbone on %s.",
                 next(self._extractor._model.parameters()).device)

        self._fp_renderer = None        # lazy PyrenderRasterizer
        self._warp_pool = None          # lazy multiprocessing pool (template warp)
        self._repre_cache = None        # (key, cache) for the most recent joint config
        self._template_K: Optional[np.ndarray] = None    # set on first inference call
        self._template_side: Optional[int] = None        # raw square viewport side (px)
        self._last_frame_diagnostics: Optional[dict] = None
        self._frame_idx: int = 0

    # Public properties

    @property
    def robot_kin(self):
        return self.renderer._kinematics

    # Cache helpers

    def _config_hash(self) -> str:
        c = self.config
        # The template render K is dataset-dependent (square_camera_from_K on the
        # first query), so it belongs in the cache key: a shared cache_dir would
        # otherwise serve dataset A's templates to dataset B. Rounded so fp noise
        # cannot split the cache.
        k_fields = (
            None if self._template_K is None
            else tuple(np.round(self._template_K, 6).flatten().tolist())
        )
        fields = (
            c.dino_model, c.dino_layer, c.render_size,
            c.num_views, c.num_inplane_rotations,
            c.sphere_distance_mm,   # changes camera distance -> different template images
            c.ssaa_factor,          # changes render supersampling -> different template images
            c.template_fit_margin,  # changes the auto camera distance -> different template images
            c.min_template_pixels,
            c.crop_rel_pad,         # changes crop padding -> different template images
            c.grid_cell_size, c.apply_pca, c.pca_components, c.cluster_num,
            k_fields,               # template render K -> different template images
        )
        return hashlib.md5(str(fields).encode()).hexdigest()[:12]

    def _joint_dir(self, joint_hash: str) -> Optional[Path]:
        if self.config.cache_dir is None:
            return None
        return (
            Path(self.config.cache_dir)
            / self.renderer.name
            / self._config_hash()
            / joint_hash
        )

    # Lazy renderer init

    def _ensure_fp_renderer(self) -> None:
        if self._fp_renderer is not None:
            return
        _ensure_foundpose_path()
        if "PYOPENGL_PLATFORM" not in os.environ:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        from utils.renderer import PyrenderRasterizer
        self._fp_renderer = PyrenderRasterizer()
        log.info("FoundPose: pyrender backend PYOPENGL_PLATFORM=%s "
                 "(osmesa = CPU rendering, egl = GPU).",
                 os.environ["PYOPENGL_PLATFORM"])

    def _get_warp_pool(self):
        """Lazy worker pool for the template warp stage (per-view CPU numpy/cv2).

        Forked before the GL context exists; workers only run cv2/numpy, never
        pyrender or CUDA. None means sequential in-process (warp_workers=1)."""
        if self._warp_pool is not None:
            return self._warp_pool
        n = self.config.warp_workers
        if n is None:
            try:
                n = len(os.sched_getaffinity(0))   # respects SLURM cpus-per-task
            except AttributeError:
                n = os.cpu_count() or 1
        if n <= 1:
            return None
        import multiprocessing as mp
        self._warp_pool = mp.Pool(processes=n, initializer=_warp_worker_init)
        log.info("FoundPose: template warp pool with %d workers.", n)
        return self._warp_pool

    def _render_view_raw(self, scene, camera):
        """Render the shared scene once from ``camera``.

        Mirrors upstream PyrenderRasterizer._render_scene (utils/renderer.py),
        including the hardcoded SpotLight, but returns color as uint8:
        upstream's uint8 -> float32/255 followed by our *255 -> uint8 is
        bit-neutral (verified 0/256 values altered), so the round-trip and its
        two full-array passes are skipped.
        """
        import pyrender
        from utils.renderer import get_opencv_to_opengl_camera_trans

        r = self._fp_renderer
        if r.renderer is None:
            r.im_size = (camera.width, camera.height)
            r.renderer = pyrender.OffscreenRenderer(r.im_size[0], r.im_size[1])

        # OpenCV to OpenGL camera frame; translation mm -> m (upstream).
        trans_c2w = camera.T_world_from_eye.dot(get_opencv_to_opengl_camera_trans())
        trans_c2w[:3, 3] *= 0.001
        camera_node = pyrender.Node(
            camera=pyrender.IntrinsicsCamera(
                fx=camera.f[0], fy=camera.f[1],
                cx=camera.c[0], cy=camera.c[1],
                znear=0.1, zfar=3000.0,
            ),
            matrix=trans_c2w,
        )
        light_node = pyrender.Node(
            light=pyrender.SpotLight(
                color=np.ones(3), intensity=2.4,
                innerConeAngle=np.pi / 16.0, outerConeAngle=np.pi / 6.0,
            ),
            matrix=trans_c2w,
        )
        scene.add_node(camera_node)
        scene.add_node(light_node)
        rgb, depth = r.renderer.render(scene, flags=r.renderer_flags)
        scene.remove_node(camera_node)
        scene.remove_node(light_node)

        depth *= 1000.0          # m -> mm (upstream)
        return rgb, depth, depth > 0

    # Stage 1: template rendering

    def _render_templates(
        self,
        mesh,                       # trimesh.Trimesh, vertices in mm, centred
        sphere_radius_mm: float,
        K_raw: np.ndarray,          # (3,3) float64, raw square template camera
        side: int,                  # raw square viewport side (px)
    ) -> Tuple[List[dict], List[dict], List[str]]:
        """
        Render Fibonacci-viewsphere templates with FoundPose's PyrenderRasterizer.

        Mirrors the upstream generator: render the full raw viewport at
        ssaa_factor supersampling, check that the silhouette stays inside the
        raw viewport, crop, then downsample to render_size.

        Returns (templates, view_records, problems):
            templates    : dicts with cameras (PinholePlaneCameraModel),
                           rgb (H,W,3 uint8), depth (H,W float32 mm), mask (bool)
            view_records : per-view fit statistics for the cache manifest
            problems     : fit violations (empty when every view passed)
        """
        _ensure_foundpose_path()
        from utils import misc as fp_misc
        from utils.structs import AlignedBox2f, PinholePlaneCameraModel
        from utils.misc import calc_crop_box, construct_crop_camera
        _restore_logging()

        self._ensure_fp_renderer()

        rs = self.config.render_size
        ssaa = float(self.config.ssaa_factor)
        raw_side = int(round(side * ssaa))
        fx, fy = float(K_raw[0, 0]) * ssaa, float(K_raw[1, 1]) * ssaa
        cx, cy = float(K_raw[0, 2]) * ssaa, float(K_raw[1, 2]) * ssaa
        # min_template_pixels is configured at render_size scale; the raw check
        # runs at the supersampled raw scale
        min_px_raw = int(self.config.min_template_pixels * ssaa * ssaa)

        views, _ = fp_misc.sample_views(
            min_n_views=self.config.num_views,
            radius=sphere_radius_mm,
            mode="fibonacci",
        )

        # Expand each viewpoint with in-plane rotations (original: 14 per viewpoint).
        if self.config.num_inplane_rotations > 1:
            inplane_angle = 2.0 * np.pi / self.config.num_inplane_rotations
            expanded = []
            for view in views:
                for k in range(self.config.num_inplane_rotations):
                    a = inplane_angle * k
                    ca, sa = np.cos(a), np.sin(a)
                    R_ip = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
                    expanded.append({
                        "R": R_ip @ view["R"],
                        "t": R_ip @ view["t"],
                    })
            views = expanded

        # The auto distance assumes the object sits on the optical axis in every
        # view; sample_views builds look-at cameras, so it does. Guard it anyway.
        t_all = np.array([np.asarray(v["t"]).flatten() for v in views])
        if np.abs(t_all[:, :2]).max() > 1e-6 * sphere_radius_mm:
            raise AssertionError("sample_views produced off-axis template views")

        log.info("FoundPose: rendering %d templates (radius=%.1f mm, raw %d px, ssaa %.1fx).",
                 len(views), sphere_radius_mm, raw_side, ssaa)

        templates: List[dict] = []
        view_records: List[dict] = []
        problems: List[str] = []

        # Upstream add_object_model/render_object_model reuse one scene per object.
        import pyrender
        mesh.vertices /= 1000.0          # mm -> m, as render_meshes does
        pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
        mesh.vertices *= 1000.0          # restore the caller's units
        scene = pyrender.Scene(bg_color=np.zeros(4),
                               ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
        scene.add(pyr_mesh)

        pool = self._get_warp_pool()
        chunk = 2 * (pool._processes if pool is not None else 1)  # bounds raw views in flight
        pending: list = []   # raw views awaiting warp (order = view order)

        _t_render_start = time.perf_counter()  # DEBUG
        _t_last = _t_render_start
        _t_gl = _t_warp = _t_fit = 0.0

        def _flush_warps() -> None:
            nonlocal _t_warp
            _t0 = time.perf_counter()
            if pool is None:
                warped = [_warp_template_view(a) for a in pending]
            else:
                warped = pool.map(_warp_template_view, pending)   # ordered
            templates.extend(warped)
            pending.clear()
            _t_warp += time.perf_counter() - _t0

        for i, view in enumerate(views):
            # Interval progress: this loop dominates runtime and is otherwise silent
            # until it finishes. A rising interval means per-view cost grows.
            if i and i % 100 == 0:
                _now = time.perf_counter()
                log.debug("pyrender loop: %d/%d views, %.0f ms/view (last 100)",
                          i, len(views), (_now - _t_last) * 10)
                _t_last = _now

            # T_m2c: model (mm, centred) -> camera (mm)
            T_m2c = np.eye(4, dtype=np.float64)
            T_m2c[:3, :3] = view["R"]
            T_m2c[:3, 3]  = view["t"].flatten()
            T_c2m = np.linalg.inv(T_m2c)   # = T_world_from_eye

            camera = PinholePlaneCameraModel(
                width=raw_side, height=raw_side,
                f=(fx, fy), c=(cx, cy),
                T_world_from_eye=T_c2m,
            )

            # Mirrors upstream _render_scene's camera/light handling; the scene
            # stays reusable across views (camera/light nodes removed per call).
            _t0 = time.perf_counter()
            rgb, depth, mask = self._render_view_raw(scene, camera)
            _t_gl += time.perf_counter() - _t0

            # Raw-viewport fit check, before cropping (the upstream generator
            # raises here on border contact: "The model does not fit the viewport.")
            _t0 = time.perf_counter()
            stats = estimator_utils.check_mask_fit(mask, min_px_raw)
            _t_fit += time.perf_counter() - _t0
            record = {"file": f"template_{len(templates) + len(pending):04d}_rgb.png",
                      **stats}
            view_records.append(record)
            if not stats["ok"]:
                problems.append(
                    f"view {len(view_records) - 1}: n_pixels={stats['n_pixels']}, "
                    f"touches_border={stats['touches_border']}"
                )
                record["file"] = None
                continue

            # Mirror gen_templates.py's square crop and warped RGB/depth path.
            b = stats["bbox_xyxy"]      # max-exclusive
            box = AlignedBox2f(
                left=float(b[0]), top=float(b[1]),
                right=float(b[2] - 1), bottom=float(b[3] - 1),
            )
            crop_box = calc_crop_box(box=box, make_square=True)
            crop_camera = construct_crop_camera(
                box=crop_box, camera_model_c2w=camera,
                viewport_size=(int(round(rs * ssaa)), int(round(rs * ssaa))),
                viewport_rel_pad=self.config.crop_rel_pad,
            )
            interp = cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
            # Warp + downsample happen in the worker pool (per-view CPU work,
            # ~90% of loop time); payload order is preserved by pool.map.
            pending.append((rgb, depth, mask, camera, crop_camera, interp,
                            float(ssaa), rs))
            if len(pending) >= chunk:
                _flush_warps()

        _flush_warps()   # remaining raw views

        _render_elapsed = time.perf_counter() - _t_render_start
        log.debug("pyrender loop: %d views in %.1fs (%.1f ms/view)",
                  len(views), _render_elapsed, _render_elapsed / max(len(views), 1) * 1000)
        log.debug("  stage split: GL %.1fs | warp+resize %.1fs (%s) | fit-check %.1fs | other %.1fs",
                  _t_gl, _t_warp,
                  "sequential" if pool is None else f"pool x{pool._processes}",
                  _t_fit, _render_elapsed - _t_gl - _t_warp - _t_fit)
        log.debug("  gpu after render loop: %s", _gpu_mem_str())
        log.info("FoundPose: %d/%d templates passed fit acceptance.", len(templates), len(views))
        return templates, view_records, problems

    def _handle_fit_problems(self, problems, distance_mm, radius_mm, dist_mode, what) -> None:
        estimator_utils.handle_fit_problems(
            problems, "FoundPose", "sphere_distance_mm",
            distance_mm, radius_mm, dist_mode, what,
        )

    # Disk I/O for templates

    def _save_templates(
        self,
        templates: List[dict],
        tpl_dir: Path,
        mesh_transform: np.ndarray,
        manifest_meta: Optional[dict] = None,
        view_records: Optional[List[dict]] = None,
    ) -> None:
        """Persist templates by joint configuration with a fit manifest."""
        tpl_dir.mkdir(parents=True, exist_ok=True)
        metadata = []
        for i, tpl in enumerate(templates):
            rgb_path   = tpl_dir / f"template_{i:04d}_rgb.png"
            depth_path = tpl_dir / f"template_{i:04d}_depth.png"
            mask_path  = tpl_dir / f"template_{i:04d}_mask.png"

            Image.fromarray(tpl["rgb"]).save(rgb_path)
            # BOP convention: 16-bit PNG, values in mm rounded to nearest integer.
            Image.fromarray(np.round(tpl["depth"]).astype(np.uint16)).save(depth_path)
            Image.fromarray(tpl["mask"].astype(np.uint8) * 255).save(mask_path)

            cam = tpl["cameras"]
            metadata.append({
                "template_id": i,
                "cameras": {
                    "ImageSizeX": int(cam.width),
                    "ImageSizeY": int(cam.height),
                    "fx": float(cam.f[0]), "fy": float(cam.f[1]),
                    "cx": float(cam.c[0]), "cy": float(cam.c[1]),
                    "T_WorldFromCamera": cam.T_world_from_eye.tolist(),
                },
                "rgb_image_path":   str(rgb_path),
                "depth_map_path":   str(depth_path),
                "binary_mask_path": str(mask_path),
            })

        (tpl_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        np.save(str(tpl_dir / "mesh_transform.npy"), mesh_transform)
        # written last: manifest presence marks the cache as complete and validated
        if manifest_meta is not None and view_records is not None:
            estimator_utils.write_template_manifest(tpl_dir, manifest_meta, view_records)
        log.info("FoundPose: saved %d templates to %s.", len(templates), tpl_dir)

    def _load_templates(self, tpl_dir: Path) -> Tuple[List[dict], np.ndarray]:
        _ensure_foundpose_path()
        from utils.structs import PinholePlaneCameraModel

        metadata = json.loads((tpl_dir / "metadata.json").read_text())
        mesh_transform = np.load(str(tpl_dir / "mesh_transform.npy"))
        templates = []
        for m in metadata:
            cam_d = m["cameras"]
            cam = PinholePlaneCameraModel(
                width=cam_d["ImageSizeX"], height=cam_d["ImageSizeY"],
                f=(cam_d["fx"], cam_d["fy"]),
                c=(cam_d["cx"], cam_d["cy"]),
                T_world_from_eye=np.array(cam_d["T_WorldFromCamera"]),
            )
            rgb   = np.array(Image.open(m["rgb_image_path"]))
            depth = np.array(Image.open(m["depth_map_path"])).astype(np.float32)
            mask  = np.array(Image.open(m["binary_mask_path"])).astype(bool)
            templates.append({"cameras": cam, "rgb": rgb, "depth": depth, "mask": mask})

        log.info("FoundPose: loaded %d templates from %s.", len(templates), tpl_dir)
        return templates, mesh_transform

    # Stage 2: representation building  (gen_repre.py logic)

    def _build_repre(
        self,
        templates: List[dict],
        mesh_transform: np.ndarray,
    ) -> dict:
        """
        Build FeatureBasedObjectRepre from rendered templates.

        Mirrors gen_repre.py: DINOv2 features -> PCA -> k-means -> TF-IDF.
        All 3D coordinates are in mm (FoundPose convention).
        """
        _ensure_foundpose_path()
        from utils import (
            cluster_util, feature_util, projector_util, repre_util, template_util,
        )
        from utils.structs import PinholePlaneCameraModel
        _restore_logging()

        feat_vecs_list:   List[torch.Tensor] = []
        vertices_list:    List[torch.Tensor] = []
        feat_to_vtx_list: List[torch.Tensor] = []
        feat_to_tpl_list: List[torch.Tensor] = []
        template_cameras: List[PinholePlaneCameraModel] = []

        _t_repre_start = time.perf_counter()  # DEBUG
        for tpl in templates:
            cam = tpl["cameras"]

            image_chw = (
                torch.from_numpy(tpl["rgb"]).float().permute(2, 0, 1) / 255.0
            ).to(self.device)
            depth_hw = torch.from_numpy(tpl["depth"]).float().to(self.device)
            mask_hw  = torch.from_numpy(tpl["mask"]).float().to(self.device)

            # T_model_from_camera: camera (mm) -> model frame (mm, centred)
            T_c2m = torch.tensor(
                cam.T_world_from_eye, dtype=torch.float32, device=self.device
            )

            feat_vecs, vertex_ids, vertices = feature_util.get_visual_features_registered_in_3d(
                image_chw=image_chw,
                depth_image_hw=depth_hw,
                object_mask=mask_hw,
                camera=cam,
                T_model_from_camera=T_c2m,
                extractor=self._extractor,
                grid_cell_size=self.config.grid_cell_size,
            )

            if feat_vecs.shape[0] == 0:
                continue

            valid_tpl_id = len(template_cameras)  # contiguous counter of non-empty templates
            feat_vecs_list.append(feat_vecs)
            vertices_list.append(vertices)
            feat_to_vtx_list.append(vertex_ids.to(self.device))
            feat_to_tpl_list.append(
                valid_tpl_id * torch.ones(feat_vecs.shape[0], dtype=torch.int32, device=self.device)
            )
            template_cameras.append(cam)

        log.debug("DINOv2 feature extraction: %d templates in %.1fs",
                  len(templates), time.perf_counter() - _t_repre_start)

        if not feat_vecs_list:
            raise RuntimeError("FoundPose: no templates yielded foreground features.")

        feat_vectors = torch.cat(feat_vecs_list)
        log.debug("template features on %s; gpu: %s", feat_vectors.device, _gpu_mem_str())
        vertices     = torch.cat(vertices_list)
        feat_to_vtx  = torch.cat(feat_to_vtx_list)
        feat_to_tpl  = torch.cat(feat_to_tpl_list)
        n_templates  = len(template_cameras)

        # PCA
        _t0 = time.perf_counter()
        feat_raw_projectors = []
        if self.config.apply_pca:
            pca = projector_util.PCAProjector(n_components=self.config.pca_components, whiten=False)
            pca.fit(feat_vectors, max_samples=100_000)
            feat_raw_projectors.append(pca)
            feat_vectors = pca.transform(feat_vectors)
        log.debug("PCA (%d components): %.1fs", self.config.pca_components, time.perf_counter() - _t0)

        # K-means visual words
        log.info("FoundPose: clustering %d features into %d visual words...",
                 feat_vectors.shape[0], self.config.cluster_num)
        _t0 = time.perf_counter()
        with suppress_c_stdout_stderr():
            centroids, cluster_ids, _ = cluster_util.kmeans(
                samples=feat_vectors, num_centroids=self.config.cluster_num, verbose=False,
            )
        log.debug("k-means (%d clusters, %d features): %.1fs",
                  self.config.cluster_num, feat_vectors.shape[0], time.perf_counter() - _t0)
        # Upstream cluster_util.kmeans selects faiss-GPU iff the samples are on
        # cuda; with a cpu-only faiss build that silently means CPU. Log which.
        try:
            import faiss
            log.debug("faiss GPUs visible: %d (k-means ran on %s)",
                      faiss.get_num_gpus(),
                      "GPU" if (feat_vectors.is_cuda and faiss.get_num_gpus() > 0) else "CPU")
        except ImportError:
            pass

        # The released FoundPose configuration uses hard TF-IDF assignment.
        _t0 = time.perf_counter()
        tpl_desc_opts = repre_util.TemplateDescOpts(
            desc_type="tfidf",
            tfidf_soft_assign=False,
        )
        template_descs, feat_cluster_idfs = template_util.calc_tfidf_descriptors(
            feat_vectors=feat_vectors,
            feat_words=centroids,
            feat_to_word_ids=cluster_ids,
            feat_to_template_ids=feat_to_tpl,
            num_templates=n_templates,
            tfidf_knn_k=tpl_desc_opts.tfidf_knn_k,
            tfidf_soft_assign=tpl_desc_opts.tfidf_soft_assign,
            tfidf_soft_sigma_squared=tpl_desc_opts.tfidf_soft_sigma_squared,
        )
        log.debug("TF-IDF descriptors: %.1fs", time.perf_counter() - _t0)

        repre = repre_util.FeatureBasedObjectRepre(
            vertices=vertices,
            feat_vectors=feat_vectors,
            feat_opts=repre_util.FeatureOpts(extractor_name=self.config.dino_model),
            feat_to_vertex_ids=feat_to_vtx,
            feat_to_template_ids=feat_to_tpl,
            feat_to_cluster_ids=cluster_ids,
            feat_cluster_centroids=centroids,
            feat_cluster_idfs=feat_cluster_idfs,
            templates=None,
            template_cameras_cam_from_model=template_cameras,
            template_descs=template_descs,
            template_desc_opts=tpl_desc_opts,
            feat_raw_projectors=feat_raw_projectors,
        )

        log.info(
            "FoundPose: representation built - %d templates, %d features, %d clusters.",
            n_templates, feat_vectors.shape[0], self.config.cluster_num,
        )
        return {"repre": repre, "mesh_transform": mesh_transform}

    # KNN index construction  (infer.py logic - fast, from loaded repre)

    def _build_knn_indices(self, repre) -> Tuple[list, object]:
        _ensure_foundpose_path()
        from utils import knn_util

        n_templates = len(repre.template_cameras_cam_from_model)
        template_knn_indices = []
        for tpl_id in range(n_templates):
            mask     = repre.feat_to_template_ids == tpl_id
            ids      = torch.nonzero(mask).flatten()
            feats    = repre.feat_vectors[ids].cpu()
            idx      = knn_util.KNN(k=1, metric="l2")
            idx.fit(feats)
            template_knn_indices.append(idx)

        vw_knn = knn_util.KNN(
            k=repre.template_desc_opts.tfidf_knn_k,
            metric=repre.template_desc_opts.tfidf_knn_metric,
        )
        vw_knn.fit(repre.feat_cluster_centroids)

        return template_knn_indices, vw_knn

    # Representation size diagnostics

    def _log_repre_sizes(self, repre) -> None:
        """Log the in-memory size of each significant field in the representation."""
        fields = [
            "feat_vectors", "vertices", "feat_to_vertex_ids",
            "feat_to_template_ids", "feat_to_cluster_ids",
            "feat_cluster_centroids", "feat_cluster_idfs",
            "template_descs", "templates",
        ]
        total = 0.0
        lines = []
        for key in fields:
            val = getattr(repre, key, None)
            if val is None:
                lines.append(f"  {key:<35s} None")
                continue
            if isinstance(val, torch.Tensor):
                mb = val.nbytes / 1e6
                total += mb
                lines.append(
                    f"  {key:<35s} {str(tuple(val.shape)):<25s} {mb:6.1f} MB  ({val.dtype})"
                )

        for label, projectors in (
            ("feat_raw_projectors", repre.feat_raw_projectors),
            ("feat_vis_projectors", repre.feat_vis_projectors),
        ):
            for i, p in enumerate(projectors or []):
                if hasattr(p, "pca") and hasattr(p.pca, "components_"):
                    mb = p.pca.components_.nbytes / 1e6
                    total += mb
                    lines.append(
                        f"  {label}[{i}] PCA          "
                        f"{str(p.pca.components_.shape):<25s} {mb:6.1f} MB"
                    )

        lines.append(f"  {'TOTAL':<60s} {total:6.1f} MB")
        # DEBUG, not INFO: build_representation runs on every frame, so this
        # would otherwise put a dozen lines per frame on the default log stream.
        log.debug("[FoundPose] Representation field sizes:")
        for line in lines:
            log.debug(line)

    # Build representation (templates cached on disk; repre always fresh)

    def build_representation(self, joint_angles) -> dict:
        """
        Build the representation for the given joint configuration.

        Templates are loaded from disk if available (cache_dir is set);
        otherwise they are rendered and saved.  The representation itself
        is always built fresh - it is never written to disk.

        Returns a dict with keys:
            repre, template_knn_indices, vw_knn, mesh_transform
        """
        if self._template_K is None:
            raise RuntimeError(
                "build_representation() called before template K is known. "
                "Call inference_single_image() at least once first, or set "
                "self._template_K manually."
            )

        q          = np.asarray(joint_angles, dtype=np.float64)
        joint_hash = estimator_utils.joint_hash(q)
        joint_dir  = self._joint_dir(joint_hash)

        # Reuse the representation when consecutive frames share a joint
        # configuration (Baxter has 20 configs x 5 camera views). The per-frame
        # reseed uses a fixed seed and faiss k-means is seeded upstream, so a
        # rebuild would return exactly this object -- a cache, not an approximation.
        repre_key = (joint_hash, self._template_side, self._template_K.tobytes())
        if self._repre_cache is not None and self._repre_cache[0] == repre_key:
            log.debug("FoundPose: reusing representation for joint hash %s.", joint_hash[:8])
            return self._repre_cache[1]

        # Viewsphere distance for this joint state: the fixed per-dataset value
        # (upstream derives one radius per dataset from its test depth range) or
        # the auto minimum-fit distance from the posed mesh's bounding sphere.
        mesh, mesh_transform_fresh = self.renderer.export_posed_trimesh(q)
        radius_mm = estimator_utils.bounding_radius_mm(np.asarray(mesh.vertices))
        if self.config.sphere_distance_mm is not None:
            sphere_radius_mm, dist_mode = float(self.config.sphere_distance_mm), "fixed"
        else:
            sphere_radius_mm = estimator_utils.viewsphere_distance_mm(
                radius_mm, self._template_K, self._template_side, self._template_side,
                margin=self.config.template_fit_margin,
            )
            dist_mode = "auto"
        n_expected = self.config.num_views * max(1, self.config.num_inplane_rotations)
        # Recorded in the manifest, not revalidated against it on load: every
        # field here is already part of _config_hash()/joint_hash(), which key
        # the cache directory.
        manifest_meta = {
            "estimator": "foundpose",
            "n_views": n_expected,
            "render_size": self.config.render_size,
            "raw_side": self._template_side,
            "ssaa_factor": self.config.ssaa_factor,
            "distance_mode": dist_mode,
            "distance_mm": sphere_radius_mm,
        }

        # Level 1: templates on disk (reusable only with a valid manifest)
        templates: Optional[List[dict]] = None
        mesh_transform: Optional[np.ndarray] = None

        if joint_dir is not None:
            tpl_dir = joint_dir / "templates"
            manifest = estimator_utils.load_template_manifest(tpl_dir)
            if manifest is not None:
                problems = estimator_utils.manifest_problems(manifest, tpl_dir)
                self._handle_fit_problems(problems, sphere_radius_mm, radius_mm, dist_mode,
                                          f"cached templates at {tpl_dir}")
                templates, mesh_transform = self._load_templates(tpl_dir)

        # Level 2: render from scratch
        if templates is None:
            log.info("FoundPose: rendering templates for joint config %s "
                     "(distance %.1f mm [%s], bounding radius %.1f mm).",
                     q, sphere_radius_mm, dist_mode, radius_mm)
            mesh_transform = mesh_transform_fresh
            templates, view_records, problems = self._render_templates(
                mesh, sphere_radius_mm, self._template_K, self._template_side
            )
            self._handle_fit_problems(problems, sphere_radius_mm, radius_mm, dist_mode,
                                      "freshly rendered templates")
            if joint_dir is not None:
                self._save_templates(
                    templates, joint_dir / "templates", mesh_transform,
                    manifest_meta={**manifest_meta, "radius_mm": radius_mm,
                                   "template_fit_margin": self.config.template_fit_margin,
                                   "min_template_pixels": self.config.min_template_pixels,
                                   "K_raw": np.round(self._template_K, 6).tolist()},
                    view_records=view_records,
                )

        # Build representation (always fresh - not persisted to disk)
        _t0 = time.perf_counter()
        cache = self._build_repre(templates, mesh_transform)
        log.debug("_build_repre total: %.1fs", time.perf_counter() - _t0)

        _t0 = time.perf_counter()
        tki, vw = self._build_knn_indices(cache["repre"])
        log.debug("KNN index build: %.1fs", time.perf_counter() - _t0)
        cache["template_knn_indices"] = tki
        cache["vw_knn"] = vw

        self._log_repre_sizes(cache["repre"])
        self._repre_cache = (repre_key, cache)
        return cache

    # Online phase: inference  (infer.py logic)

    def inference_single_image(
        self,
        img: torch.Tensor,
        joint_angles,
        K=None,
        mask=None,       # (H, W) uint8 or bool ndarray - segmentation mask in original image space
        bbox_xyxy=None,  # [x1, y1, x2, y2] in original image space - used for crop box when provided
        **kwargs,
    ) -> Tuple[Optional[torch.Tensor], None, None, None, None]:
        _t = time.perf_counter()
        frame_idx = self._frame_idx
        self._frame_idx += 1
        self._last_frame_diagnostics = None

        estimator_utils.reseed_all(self.config.seed)

        _ensure_foundpose_path()
        from utils import corresp_util, feature_util, pnp_util, projector_util
        from utils.structs import PinholePlaneCameraModel
        from utils.misc import tensor_to_array
        _restore_logging()

        # Derive template K from the first query image (full-image K, fixed)
        K_np = (
            K.cpu().numpy().astype(np.float64)
            if isinstance(K, torch.Tensor)
            else np.asarray(K, dtype=np.float64)
        )
        if self._template_K is None:
            H = img.shape[-2] if img.ndim >= 3 else img.shape[0]
            W = img.shape[-1] if img.ndim >= 3 else img.shape[1]
            # Upstream raw template camera (gen_templates.py): square viewport of
            # the MAX image side snapped to the patch grid, dataset focal lengths
            # unchanged. A min-side viewport would crop the field of view and
            # clip large arms before the object crop.
            self._template_K, self._template_side = estimator_utils.square_camera_from_K(
                K_np, W, H, patch_size=_DINO_PATCH_SIZE
            )
            log.info(
                "FoundPose: raw template camera from first query (H=%d W=%d -> %d×%d square).",
                H, W, self._template_side, self._template_side,
            )

        # Load / build representation
        q     = np.asarray(joint_angles, dtype=np.float64)
        cache = self.build_representation(q)
        repre               = cache["repre"]
        template_knn_indices = cache["template_knn_indices"]
        vw_knn              = cache["vw_knn"]
        mesh_transform      = cache["mesh_transform"]   # (4,4) float64, URDF-m -> centred-mm
        n_templates = len(repre.template_cameras_cam_from_model)

        # Crop and prepare query image
        rs = self.config.render_size
        if mask is not None:
            query_chw, K_query, query_mask_rs, T_crop_to_orig = _bbox_crop_resize(
                img, K_np, mask, rs, self.config.crop_rel_pad, self.device,
                bbox_xyxy=bbox_xyxy,
            )
        else:
            query_chw, K_query = _center_crop_resize(img, K_np, rs, self.device)
            query_mask_rs  = None
            T_crop_to_orig = np.eye(4, dtype=np.float64)

        query_camera = PinholePlaneCameraModel(
            width=rs, height=rs,
            f=(float(K_query[0, 0]), float(K_query[1, 1])),
            c=(float(K_query[0, 2]), float(K_query[1, 2])),
            T_world_from_eye=np.eye(4),
        )

        # Feature extraction
        extractor_out   = self._extractor(query_chw.unsqueeze(0))
        feature_map_chw = extractor_out["feature_maps"][0]   # (D, pH, pW)

        grid_points = feature_util.generate_grid_points(
            grid_size=(rs, rs), cell_size=self.config.grid_cell_size,
        ).to(self.device)
        n_grid_total = int(grid_points.shape[0])

        if query_mask_rs is not None:
            query_points = feature_util.filter_points_by_mask(
                grid_points,
                torch.from_numpy(query_mask_rs.astype(np.float32)).to(self.device),
            )
            if query_points.shape[0] == 0:
                log.warning("FoundPose frame %d: mask filtering left zero query points - "
                            "falling back to all grid points.", frame_idx)
                query_points = grid_points
        else:
            query_points = grid_points

        n_query_pts = int(query_points.shape[0])
        mask_coverage = round(n_query_pts / max(n_grid_total, 1), 3)
        log.debug("frame %d | query: %d/%d grid pts in mask (%.0f%%), templates: %d",
                  frame_idx, n_query_pts, n_grid_total, mask_coverage * 100, n_templates)
        if n_query_pts < self.config.min_correspondences:
            log.warning(
                "FoundPose frame %d: only %d query points in mask - too few for any correspondence.",
                frame_idx, n_query_pts,
            )

        query_features = feature_util.sample_feature_map_at_points(
            feature_map_chw=feature_map_chw,
            points=query_points,
            image_size=(rs, rs),
        ).contiguous()

        if (
            query_features.shape[1] != repre.feat_vectors.shape[1]
            and repre.feat_raw_projectors
        ):
            query_features = projector_util.project_features(
                feat_vectors=query_features,
                projectors=repre.feat_raw_projectors,
            ).contiguous()

        # 2D-3D correspondences
        corresps = corresp_util.establish_correspondences(
            query_points=query_points,
            query_features=query_features,
            object_repre=repre,
            template_matching_type="tfidf",
            feat_matching_type="cyclic_buddies",
            top_n_templates=self.config.match_top_n_templates,
            top_k_buddies=self.config.match_top_k_buddies,
            visual_words_knn_index=vw_knn,
            template_knn_indices=template_knn_indices,
        )

        # Per-template correspondence diagnostics.
        corresp_diag = []
        for c in corresps:
            n_c = int(len(c["coord_2d"]))
            corresp_diag.append({
                "template_id":    int(c["template_id"]),
                "template_score": round(float(c["template_score"]), 4),
                "n_correspondences": n_c,
                "below_min":      n_c < self.config.min_correspondences,
            })

        n_total_corr = sum(d["n_correspondences"] for d in corresp_diag)
        tfidf_scores = [d["template_score"] for d in corresp_diag]
        score_spread = round(max(tfidf_scores) - min(tfidf_scores), 4) if tfidf_scores else 0.0
        log.debug(
            "frame %d | TF-IDF retrieved %d templates  "
            "corr: %s  scores: %s  spread=%.4f  total_corr: %d",
            frame_idx, len(corresps),
            [d["n_correspondences"] for d in corresp_diag],
            [round(s, 4) for s in tfidf_scores],
            score_spread, n_total_corr,
        )
        if score_spread < 0.01:
            log.debug("frame %d | TF-IDF scores nearly uniform (spread=%.4f) - retrieval may be random.",
                      frame_idx, score_spread)
        if n_total_corr == 0:
            log.warning("FoundPose frame %d: zero correspondences from cyclic buddies - PnP will fail.", frame_idx)

        # PnP RANSAC - best by inlier count
        best_pose    = None
        best_quality = -1.0
        best_inliers = 0
        best_n_corr  = 0
        best_reproj  = None
        pnp_attempts = []

        for corresp in corresps:
            n_c = int(len(corresp["coord_2d"]))
            if n_c < self.config.min_correspondences:
                pnp_attempts.append({
                    "template_id": int(corresp["template_id"]),
                    "n_correspondences": n_c,
                    "skipped": True,
                    "success": False,
                })
                continue

            success, R_m2c, t_m2c, inliers, quality = pnp_util.estimate_pose(
                corresp=corresp,
                camera_c2w=query_camera,
                pnp_type="opencv",
                pnp_ransac_iter=self.config.pnp_ransac_iter,
                pnp_inlier_thresh=self.config.pnp_reproj_error,
                pnp_required_ransac_conf=self.config.pnp_confidence,
                pnp_refine_lm=self.config.pnp_refine_lm,
            )

            n_inliers = int(len(inliers)) if (inliers is not None) else 0
            inlier_ratio = round(n_inliers / max(n_c, 1), 3)

            reproj_err = None
            if success and R_m2c is not None and inliers is not None and len(inliers) > 0:
                pts3d = tensor_to_array(corresp["coord_3d"]).astype(np.float32)
                pts2d = tensor_to_array(corresp["coord_2d"]).astype(np.float32)
                inlier_ids = np.asarray(inliers).flatten()
                rvec = cv2.Rodrigues(R_m2c)[0]
                proj, _ = cv2.projectPoints(
                    pts3d[inlier_ids], rvec, t_m2c,
                    K_query.astype(np.float32), None,
                )
                reproj_err = float(
                    np.linalg.norm(proj.reshape(-1, 2) - pts2d[inlier_ids], axis=1).mean()
                )

            # 3D spatial spread of inliers - measures how well-distributed they are
            # along the arm. Low spread (< arm_length/4) -> degenerate PnP geometry.
            inlier_spread_mm = None
            if success and R_m2c is not None and inliers is not None and len(inliers) > 1:
                inlier_pts3d = pts3d[np.asarray(inliers).flatten()]  # (N, 3) in mm
                span = inlier_pts3d.max(axis=0) - inlier_pts3d.min(axis=0)
                inlier_spread_mm = round(float(np.linalg.norm(span)), 1)

            pnp_attempts.append({
                "template_id":    int(corresp["template_id"]),
                "n_correspondences": n_c,
                "skipped":        False,
                "success":        bool(success and R_m2c is not None),
                "n_inliers":      n_inliers,
                "inlier_ratio":   inlier_ratio,
                "reproj_err_px":  round(reproj_err, 2) if reproj_err is not None else None,
                "inlier_spread_mm": inlier_spread_mm,
            })

            if success and R_m2c is not None and (quality is not None) and quality > best_quality:
                best_quality = quality
                best_pose    = (R_m2c, t_m2c)
                best_inliers = n_inliers
                best_n_corr  = n_c
                best_reproj  = reproj_err

        # Log PnP summary.
        for att in pnp_attempts:
            if att.get("skipped"):
                log.debug("frame %d | tpl %d: SKIPPED (only %d corr < min %d)",
                          frame_idx, att["template_id"], att["n_correspondences"],
                          self.config.min_correspondences)
            else:
                log.debug(
                    "frame %d | tpl %d: PnP %s  %d corr -> %d inliers (%.0f%%)  "
                    "reproj=%.1f px  spread=%s mm",
                    frame_idx, att["template_id"],
                    "OK  " if att["success"] else "FAIL",
                    att["n_correspondences"], att["n_inliers"],
                    att["inlier_ratio"] * 100,
                    att["reproj_err_px"] if att["reproj_err_px"] is not None else float("nan"),
                    att.get("inlier_spread_mm", "n/a"),
                )

        frame_time_ms = round((time.perf_counter() - _t) * 1000, 1)
        log.debug("frame %d | total %.0f ms | gpu: %s", frame_idx, frame_time_ms, _gpu_mem_str())

        # Build full diagnostics dict.
        diag: dict = {
            "frame_time_ms": frame_time_ms,
            "n_query_points": n_query_pts,
            "n_grid_points":  n_grid_total,
            "mask_coverage":  mask_coverage,
            "n_templates":    n_templates,
            "corresp_per_template": corresp_diag,
            "n_total_corr":   n_total_corr,
            "pnp_attempts":   pnp_attempts,
            "pnp": None,
        }

        if best_pose is None:
            log.warning(
                "FoundPose frame %d: ALL PnP attempts failed. "
                "query_pts=%d  total_corr=%d  attempts=%d",
                frame_idx, n_query_pts, n_total_corr, len(pnp_attempts),
            )
            self._last_frame_diagnostics = diag
            if self.config.debug_dir is not None:
                _save_debug_images(
                    self.config.debug_dir, frame_idx, query_chw,
                    query_mask_rs, corresps, repre, None, None, K_query,
                    orig_img_chw=img if isinstance(img, torch.Tensor) else None,
                    orig_mask_np=mask, orig_bbox_xyxy=bbox_xyxy,
                )
            return None, None, None, None, None

        R_m2c, t_m2c = best_pose
        diag["pnp"] = {
            # correspondences of the template that won, not the sum over all
            # retrieved templates: the ratio describes the selected pose
            "total_correspondences": best_n_corr,
            "inlier_count":          best_inliers,
            "inlier_ratio":          round(best_inliers / max(best_n_corr, 1), 4),
            "mean_reproj_error_px":  round(best_reproj, 4) if best_reproj is not None else None,
            "corr_all_templates":    n_total_corr,
        }
        self._last_frame_diagnostics = diag

        if self.config.debug_dir is not None:
            _save_debug_images(
                self.config.debug_dir, frame_idx, query_chw,
                query_mask_rs, corresps, repre, R_m2c, t_m2c, K_query,
                orig_img_chw=img if isinstance(img, torch.Tensor) else None,
                orig_mask_np=mask, orig_bbox_xyxy=bbox_xyxy,
            )

        # Mirror infer.py 661-663, then convert the centered-mm mesh pose to the
        # URDF-base pose; mesh_transform includes scale and is not a rigid pose.
        T_m2c_4x4          = np.eye(4, dtype=np.float64)
        T_m2c_4x4[:3, :3]  = np.asarray(R_m2c, dtype=np.float64)
        T_m2c_4x4[:3, 3]   = np.asarray(t_m2c, dtype=np.float64).flatten()
        T_m2c_orig         = T_crop_to_orig @ T_m2c_4x4   # model -> original camera frame
        T_m2c_orig[:3, 3] /= 1000.0                        # mm -> metres before un-centring

        T_base_cam = estimator_utils.centred_to_base(
            T_m2c_orig, estimator_utils.center_offset_m(mesh_transform)
        )

        return torch.from_numpy(T_base_cam.astype(np.float32)), None, None, None, None


# Helpers

def _calc_warp_maps(src_camera, dst_camera):
    """dst<-src pixel maps for cv2.remap, computed once per view.

    Mirrors the map computation inside upstream warp_image (utils/misc.py) -
    identical inputs -> identical maps. Upstream recomputes them inside every
    warp_image call (3x per view: rgb, mask, depth).
    """
    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))
    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts   = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)
    src_win_pts[src_eye_pts[:, 2] < 0] = -1   # upstream depth_check
    src_win_pts = src_win_pts.astype(np.float32)
    return src_win_pts[:, 0].reshape((H, W)), src_win_pts[:, 1].reshape((H, W))


def _depth_in_dst_camera(src_camera, dst_camera, depth):
    """Depth values re-expressed in the dst camera frame.

    Mirrors the point-cloud branch of upstream warp_depth_image (utils/misc.py).
    """
    depth_image = np.array(depth)
    if not np.allclose(src_camera.T_world_from_eye, dst_camera.T_world_from_eye):
        valid_mask = depth_image > 0
        ys, xs = np.nonzero(valid_mask)
        pts_in_src = src_camera.window_to_eye(np.vstack([xs, ys]).T)
        pts_in_src *= np.expand_dims(depth_image[valid_mask] / pts_in_src[:, 2], axis=1)
        pts_in_w   = src_camera.eye_to_world(pts_in_src)
        pts_in_trg = dst_camera.world_to_eye(pts_in_w)
        depth_image[valid_mask] = pts_in_trg[:, 2]
    return depth_image


def _warp_worker_init():
    # The pool forks after the parent already ran threaded cv2/numpy; GCD/OpenMP
    # thread pools do not survive fork and remap would deadlock in the child.
    # Single-threaded workers avoid that, and intra-op threads are redundant
    # next to the process pool. Output is thread-count independent.
    cv2.setNumThreads(1)


def _warp_template_view(args):
    """Warp + downsample one raw template view. Worker for _render_templates:
    the same cv2/numpy calls as the former inline loop body, so pooled and
    sequential runs produce identical templates."""
    rgb, depth, mask, camera, crop_camera, interp, ssaa, rs = args
    # One warp map per view, reused for rgb/mask/depth. Mirrors upstream
    # warp_image/warp_depth_image (utils/misc.py), which recompute the
    # identical map on every call; same maps + cv2.remap -> same output.
    map_x, map_y = _calc_warp_maps(src_camera=camera, dst_camera=crop_camera)
    rgb   = cv2.remap(rgb, map_x, map_y, interp)
    mask  = cv2.remap(mask.astype(np.uint8), map_x, map_y,
                      cv2.INTER_NEAREST).astype(bool)
    depth = cv2.remap(_depth_in_dst_camera(camera, crop_camera, depth),
                      map_x, map_y, cv2.INTER_NEAREST)
    if ssaa != 1.0:
        # sys.path was set up in the parent before fork (utils.structs imported
        # there), so this import resolves from sys.modules in the child.
        from utils.structs import PinholePlaneCameraModel
        # upstream SSAA downsample: INTER_AREA for color, NEAREST otherwise
        rgb   = cv2.resize(rgb, (rs, rs), interpolation=cv2.INTER_AREA)
        mask  = cv2.resize(mask.astype(np.uint8), (rs, rs),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        depth = cv2.resize(depth, (rs, rs), interpolation=cv2.INTER_NEAREST)
        sf = rs / float(crop_camera.width)
        crop_camera = PinholePlaneCameraModel(
            width=rs, height=rs,
            f=(crop_camera.f[0] * sf, crop_camera.f[1] * sf),
            c=(crop_camera.c[0] * sf, crop_camera.c[1] * sf),
            T_world_from_eye=crop_camera.T_world_from_eye,
        )
    # Match the uint16-mm precision the disk cache round-trips through, so a
    # fresh render and a cache reload give identical 3D points (and identical PnP).
    depth = np.round(depth).astype(np.uint16).astype(np.float32)
    return {"cameras": crop_camera, "rgb": rgb, "depth": depth, "mask": mask}


def _gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return "cuda unavailable"
    return (f"{torch.cuda.memory_allocated() / 1e6:.0f} MB allocated, "
            f"{torch.cuda.max_memory_allocated() / 1e6:.0f} MB peak")


def _save_debug_images(
    debug_dir: str,
    frame_idx: int,
    query_chw: torch.Tensor,
    query_mask_rs,          # (H, W) bool ndarray or None
    corresps: list,
    repre,                  # FeatureBasedObjectRepre
    R_m2c,                  # (3,3) ndarray or None
    t_m2c,                  # (3,) or (3,1) ndarray or None
    K_query: np.ndarray,    # (3,3) crop-camera intrinsics
    orig_img_chw: torch.Tensor = None,   # (3, H, W) float [0,1] - original eval image
    orig_mask_np = None,                 # (H, W) uint8 - raw detection mask in eval space
    orig_bbox_xyxy = None,               # [x1,y1,x2,y2] in eval space
) -> None:
    """Save per-frame debug images to debug_dir.

    Writes up to three images per frame:
      frame_NNNNN_detection.png     - original eval image with raw CNOS mask + bbox overlaid
      frame_NNNNN_query.png         - query crop with mask overlay and corresp dots
      frame_NNNNN_reprojection.png  - reprojected 3D vertices if pose was estimated
    """
    from pathlib import Path
    from PIL import Image, ImageDraw

    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(out_dir / f"frame_{frame_idx:05d}")

    # detection mask on original eval image
    if orig_img_chw is not None and orig_mask_np is not None:
        orig_np = (orig_img_chw.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        det_img = Image.fromarray(orig_np).convert("RGBA")

        mask_bin = np.asarray(orig_mask_np).astype(bool)
        overlay = Image.new("RGBA", det_img.size, (0, 200, 0, 0))
        alpha   = Image.fromarray((mask_bin.astype(np.uint8) * 120), mode="L")
        overlay.putalpha(alpha)
        det_img = Image.alpha_composite(det_img, overlay)

        if orig_bbox_xyxy is not None:
            draw = ImageDraw.Draw(det_img)
            x1, y1, x2, y2 = [float(v) for v in orig_bbox_xyxy]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 0, 255), width=2)

        det_img.convert("RGB").save(f"{prefix}_detection.png")

    # query crop (float CHW [0,1] -> uint8 HWC)
    query_np = (query_chw.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    query_img = Image.fromarray(query_np).convert("RGBA")

    # Mask overlay - semi-transparent green tint over foreground pixels.
    if query_mask_rs is not None:
        overlay = Image.new("RGBA", query_img.size, (0, 200, 0, 0))
        alpha   = Image.fromarray((query_mask_rs.astype(np.uint8) * 70), mode="L")
        overlay.putalpha(alpha)
        query_img = Image.alpha_composite(query_img, overlay)

    # Draw correspondence dots per retrieved template (different hue each).
    COLORS = [(255, 60, 60), (60, 220, 60), (60, 120, 255), (255, 220, 0), (220, 60, 255)]
    draw = ImageDraw.Draw(query_img)
    for ci, corresp in enumerate(corresps):
        color = COLORS[ci % len(COLORS)]
        pts2d = corresp["coord_2d"]
        if hasattr(pts2d, "cpu"):
            pts2d = pts2d.cpu().numpy()
        pts2d = np.asarray(pts2d)
        n_c   = len(pts2d)
        label = f"tpl{corresp['template_id']} n={n_c}"
        draw.text((4, 4 + ci * 14), label, fill=color + (255,))
        for pt in pts2d[:100]:            # cap at 100 dots per template
            x, y = float(pt[0]), float(pt[1])
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color + (200,))

    query_img.convert("RGB").save(f"{prefix}_query.png")

    # reprojection
    if R_m2c is None or t_m2c is None:
        return

    verts = repre.vertices                 # (N, 3) mm, centred-model space
    if hasattr(verts, "cpu"):
        verts_np = verts.cpu().numpy().astype(np.float32)
    else:
        verts_np = np.asarray(verts, dtype=np.float32)

    # Sub-sample to keep projection fast.
    if len(verts_np) > 2000:
        idx      = np.linspace(0, len(verts_np) - 1, 2000, dtype=int)
        verts_np = verts_np[idx]

    try:
        rvec = cv2.Rodrigues(np.asarray(R_m2c, dtype=np.float64))[0]
        tvec = np.asarray(t_m2c, dtype=np.float64).flatten().reshape(3, 1)
        proj, _ = cv2.projectPoints(verts_np, rvec, tvec, K_query.astype(np.float64), None)
        proj = proj.reshape(-1, 2)
    except Exception:
        return

    H, W   = query_np.shape[:2]
    rp_img = Image.fromarray(query_np.copy())
    draw   = ImageDraw.Draw(rp_img)
    for pt in proj:
        x, y = float(pt[0]), float(pt[1])
        if 0 <= x < W and 0 <= y < H:
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 255, 80))
    rp_img.save(f"{prefix}_reprojection.png")


def _scale_K_to_square(K: np.ndarray, H: int, W: int, size: int) -> np.ndarray:
    """Centre-crop to square then scale to size×size - adjust K accordingly.

    Query-side helper for the detection-less center-crop fallback below; the
    template camera uses estimator_utils.square_camera_from_K (upstream max-side
    construction) instead.
    """
    S  = min(H, W)
    x0 = (W - S) // 2
    y0 = (H - S) // 2
    K_out = K.copy()
    K_out[0, 2] -= x0
    K_out[1, 2] -= y0
    scale = size / S
    K_out[0, 0] *= scale
    K_out[0, 2] *= scale
    K_out[1, 1] *= scale
    K_out[1, 2] *= scale
    return K_out


def _center_crop_resize(
    img: torch.Tensor,
    K: np.ndarray,
    size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Centre-crop to square, resize to size×size, return (img_chw, K_adjusted)."""
    if img.ndim == 4:
        img = img[0]
    _, H, W = img.shape
    S  = min(H, W)
    x0 = (W - S) // 2
    y0 = (H - S) // 2
    img_crop = img[:, y0:y0 + S, x0:x0 + S]
    # antialias=True mirrors FoundPose's own INTER_AREA-on-downscale convention
    # (utils/misc.py); essential at native resolution where this center crop is large.
    img_out  = F.interpolate(
        img_crop.unsqueeze(0).float().to(device),
        size=(size, size), mode="bilinear", align_corners=False, antialias=True,
    ).squeeze(0)
    K_out = _scale_K_to_square(K, H, W, size)
    return img_out, K_out


def _bbox_crop_resize(
    img: torch.Tensor,
    K: np.ndarray,
    mask,
    size: int,
    crop_rel_pad: float,
    device: torch.device,
    bbox_xyxy=None,   # [x1,y1,x2,y2] in original image space - uses detection bbox when provided
) -> Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Apply FoundPose's square object crop and virtual crop camera.

    A supplied bbox takes precedence over the mask extent. The returned transform
    maps a crop-camera PnP pose back to the original camera. Empty masks use the
    center-crop fallback with an identity transform.
    """
    if img.ndim == 4:
        img = img[0]
    _, H, W = img.shape

    mask_np = np.asarray(mask).astype(bool)
    ys, xs  = mask_np.nonzero()
    if len(xs) == 0:
        img_out, K_out = _center_crop_resize(img, K, size, device)
        return img_out, K_out, None, np.eye(4, dtype=np.float64)

    # Original square crop + look-at virtual camera + warp.
    from utils.misc import calc_crop_box, construct_crop_camera, warp_image
    from utils.structs import AlignedBox2f, PinholePlaneCameraModel

    orig_camera = PinholePlaneCameraModel(
        width=W, height=H,
        f=(float(K[0, 0]), float(K[1, 1])), c=(float(K[0, 2]), float(K[1, 2])),
        T_world_from_eye=np.eye(4),
    )
    if bbox_xyxy is not None:
        bx1, by1, bx2, by2 = [float(v) for v in bbox_xyxy]
    else:
        bx1, by1 = float(xs.min()), float(ys.min())
        bx2, by2 = float(xs.max()), float(ys.max())
    box = AlignedBox2f(left=bx1, top=by1, right=bx2, bottom=by2)
    crop_box = calc_crop_box(box=box, make_square=True)
    crop_camera = construct_crop_camera(
        box=crop_box, camera_model_c2w=orig_camera,
        viewport_size=(size, size), viewport_rel_pad=crop_rel_pad,
    )
    interp = cv2.INTER_AREA if crop_box.width >= crop_camera.width else cv2.INTER_LINEAR
    img_hwc = img.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    warped  = warp_image(src_camera=orig_camera, dst_camera=crop_camera,
                         src_image=img_hwc, interpolation=interp)
    img_out = torch.from_numpy(np.ascontiguousarray(warped)).permute(2, 0, 1).float().to(device)
    mask_warped = warp_image(src_camera=orig_camera, dst_camera=crop_camera,
                             src_image=mask_np.astype(np.uint8),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    K_out = np.array([
        [crop_camera.f[0], 0.0, crop_camera.c[0]],
        [0.0, crop_camera.f[1], crop_camera.c[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    T_crop_to_orig = np.asarray(crop_camera.T_world_from_eye, dtype=np.float64)
    return img_out, K_out, mask_warped, T_crop_to_orig
