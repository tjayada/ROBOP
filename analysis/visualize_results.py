#!/usr/bin/env python3
"""Render query, ground-truth and predicted overlays from an evaluation JSON.

Image paths, joint angles, poses and intrinsics come from each query entry.
Robot configuration is read from the sibling results config unless overridden.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from eval_utils import select_eval_frame_indices  # noqa: E402

from robot_renderer.config import ViewConfig  # noqa: E402
from robot_renderer.robot_renderer import RobotRenderer  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d.io.obj")

# OpenCV/BOP -> PyTorch3D axis flip.
_C3 = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

RED   = np.array([1.0, 0.0, 0.0], dtype=np.float32)
GREEN = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _pose_to_R_T(T_m2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    R = (T_m2c[:3, :3].T @ _C3).astype(np.float32)
    T = (_C3 @ T_m2c[:3, 3]).astype(np.float32)
    return R, T


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def _render_contour(
    pt3d, robot_meshes, T_m2c: np.ndarray, device: torch.device,
    color: np.ndarray, contour_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render the arm at T_m2c; returns (rgb overlay with colored contour ring, alpha)."""
    R_np, T_np = _pose_to_R_T(T_m2c)
    R = torch.from_numpy(R_np).unsqueeze(0).to(device)
    T = torch.from_numpy(T_np).unsqueeze(0).to(device)
    rgba_t, fragments = pt3d.phong_render_batched(robot_meshes, R, T)
    mask     = (fragments.pix_to_face[0, ..., 0] >= 0).cpu().numpy().astype(np.float32)
    rgb_np   = rgba_t[0, ..., :3].cpu().numpy().clip(0.0, 1.0)
    dilated  = _dilate_mask(mask, contour_width)
    ring     = np.clip(dilated - mask, 0.0, 1.0)[..., np.newaxis]
    overlay  = rgb_np * (1.0 - ring) + color * ring
    alpha    = np.clip(dilated, 0.0, 1.0)[..., np.newaxis]
    return overlay, alpha


def _composite(base: np.ndarray, overlay_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.broadcast_to(alpha, overlay_rgb.shape)
    return (base * (1.0 - alpha) + overlay_rgb * alpha).clip(0.0, 1.0)


def _load_rgb(path) -> np.ndarray:
    img = Image.open(str(path)).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    pil = Image.fromarray((img * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def _put_label(img: np.ndarray, text: str, color) -> None:
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_marker(img: np.ndarray, xy, color, native_hw=None) -> None:
    x, y = float(xy[0]), float(xy[1])
    if native_hw is not None:
        # The marker was projected at native resolution (e.g. Baxter 2048x1536);
        # scale it to the eval-resolution panel it is drawn on.
        nh, nw = float(native_hw[0]), float(native_hw[1])
        x *= img.shape[1] / nw
        y *= img.shape[0] / nh
    x, y = int(round(x)), int(round(y))
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)


# Per-query field access (dataset-agnostic, key probing)

def _get_K(q: dict) -> Optional[np.ndarray]:
    for key in ("K", "K_eval"):
        if q.get(key) is not None:
            return np.asarray(q[key], dtype=np.float32)
    return None


def _get_frame_label(q: dict) -> str:
    if q.get("frame_id") is not None:
        return str(q["frame_id"])
    return Path(q["image_path"]).stem


def _resolve_image_path(q: dict, old_root: Optional[str], data_folder: Optional[Path]) -> Optional[Path]:
    """Use the stored path if it exists; otherwise re-root it onto --data-folder."""
    p = Path(q["image_path"])
    if p.exists():
        return p
    if data_folder is not None and old_root:
        try:
            return data_folder / p.relative_to(old_root)
        except ValueError:
            pass
    return None


# Main

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", required=True,
                        help="Results JSON from any run_eval_* script")
    parser.add_argument("--robot", default=None,
                        help="robot-renderer registry name; default: robot.name from the "
                             "sibling <results>.config.yaml")
    parser.add_argument("--data-folder", default=None,
                        help="Dataset root on THIS machine - only needed to re-root the "
                             "results' absolute image paths when they came from another machine")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: <results dir>/<results stem>_overlays)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Render N evenly-spaced frames; >= total (or omitted) renders all")
    parser.add_argument("--frames", default=None,
                        help="Comma-separated frame labels to render (frame_id or image "
                             "stem; digit labels are zero-padded to 6 for matching). "
                             "Overrides --num-samples. Used by the failure-taxonomy "
                             "exemplar renders (see analyze_compare.py --exemplars).")
    parser.add_argument("--contour-width", type=int, default=3)
    parser.add_argument("--skip-failed", action="store_true",
                        help="Skip frames where PnP failed")
    parser.add_argument("--labels", action="store_true",
                        help="Draw text labels (GT/Pred, ADD mm, frame id) on the panels")
    parser.add_argument("--no-gt-arm", action="store_true",
                        help="Skip the GT panel: output becomes [query | pred]")
    args = parser.parse_args()

    results_path = Path(args.results)
    with open(results_path) as f:
        results = json.load(f)
    queries = results["queries"]

    # Sibling resolved config: robot name + the data_folder the run used.
    robot_name, old_root = args.robot, None
    config_path = results_path.with_suffix(".config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            run_cfg = yaml.safe_load(f)
        robot_name = robot_name or run_cfg.get("robot", {}).get("name")
        old_root = (run_cfg.get("dataset", {}) or {}).get("data_folder")
    if robot_name is None:
        sys.exit(f"ERROR: --robot not given and no sibling config at {config_path}")

    data_folder = Path(args.data_folder) if args.data_folder else None

    if args.skip_failed:
        queries = [q for q in queries if not q.get("pnp_failed", False)]
    if args.frames:
        def _norm(s: str) -> str:
            return s.zfill(6) if s.isdigit() else s
        wanted = {_norm(x.strip()) for x in args.frames.split(",") if x.strip()}
        queries = [q for q in queries if _norm(_get_frame_label(q)) in wanted]
        missing = wanted - {_norm(_get_frame_label(q)) for q in queries}
        if missing:
            print(f"WARNING: frames not found in results: {sorted(missing)}")
    else:
        indices = select_eval_frame_indices(len(queries), args.num_samples)
        if len(indices) < len(queries):
            print(f"Rendering {len(indices)} / {len(queries)} frames (evenly spaced)")
        queries = [queries[i] for i in indices]
    if not queries:
        sys.exit("ERROR: no frames to render")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir) if args.out_dir else \
        results_path.parent / f"{results_path.stem}_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Renderer + PT3D cameras are built once, from the first renderable frame
    # (all current datasets have static intrinsics + image size within a run).
    rr = None
    pt3d = None
    pt3d_built_for = None   # (K bytes, H, W) the pt3d cameras were last built for

    n_saved = 0
    for i, q in enumerate(queries):
        label = _get_frame_label(q)

        img_path = _resolve_image_path(q, old_root, data_folder)
        if img_path is None:
            print(f"[skip] {label}: image not found ({q['image_path']}); pass --data-folder to re-root")
            continue
        query_rgb = _load_rgb(img_path)

        H_eval, W_eval = int(q["image_size_hw"][0]), int(q["image_size_hw"][1])
        query_rgb = _resize(query_rgb, W_eval, H_eval)

        K = _get_K(q)
        if K is None:
            print(f"[skip] {label}: no intrinsics in the results JSON - re-run the eval")
            continue

        if pt3d is None:
            rr = RobotRenderer(
                name=robot_name, K=K,
                config=ViewConfig(render_size=max(H_eval, W_eval)),
                device=device,
            )
            rr.to(device)
            pt3d = rr._renderer
        # Rebuild the cameras whenever K or (H, W) changes: Hydra pools three
        # measurements with different intrinsics, so a single build from the
        # first frame would render measurements 1/2 with measurement 0's camera.
        frame_key = (K.tobytes(), H_eval, W_eval)
        if frame_key != pt3d_built_for:
            pt3d._build_pt3d_renderers(K, (H_eval, W_eval))
            pt3d_built_for = frame_key

        joints = np.asarray(q["joints_rad"], dtype=np.float32)
        try:
            robot_meshes = pt3d.transform_mesh2robot_config(joints)
        except Exception as e:
            print(f"[warn] {label}: mesh build failed: {e}")
            continue

        # GT panel (green)
        gt_panel = None
        if not args.no_gt_arm:
            gt_panel = query_rgb.copy()
            gt_pose = np.asarray(q["gt_pose"], dtype=np.float32) if q.get("gt_pose") is not None else None
            if gt_pose is not None:
                try:
                    gt_rgb, gt_alpha = _render_contour(
                        pt3d, robot_meshes, gt_pose, device, GREEN, args.contour_width,
                    )
                    gt_panel = _composite(gt_panel, gt_rgb, gt_alpha)
                except Exception as e:
                    print(f"[warn] {label}: GT render failed: {e}")
                if args.labels:
                    _put_label(gt_panel, "GT (green)", (0.2, 1.0, 0.2))
            elif q.get("ee_2d_gt_full_res") is not None:
                # Baxter: no full GT pose in the results - mark the GT end effector.
                _draw_marker(gt_panel, q["ee_2d_gt_full_res"], (0.2, 1.0, 0.2),
                             native_hw=q.get("native_size_hw"))
                if args.labels:
                    _put_label(gt_panel, "GT EE (green)", (0.2, 1.0, 0.2))
            elif args.labels:
                _put_label(gt_panel, "GT n/a", (0.7, 0.7, 0.7))

        # Predicted panel (red)
        pred_panel = query_rgb.copy()
        pnp_failed = q.get("pnp_failed", False)
        if not pnp_failed and q.get("est_pose") is not None:
            try:
                est_pose = np.asarray(q["est_pose"], dtype=np.float32)
                est_rgb, est_alpha = _render_contour(
                    pt3d, robot_meshes, est_pose, device, RED, args.contour_width,
                )
                pred_panel = _composite(pred_panel, est_rgb, est_alpha)
            except Exception as e:
                print(f"[warn] {label}: pred render failed: {e}")
        if q.get("ee_2d_pred") is not None:
            _draw_marker(pred_panel, q["ee_2d_pred"], (1.0, 0.4, 0.4),
                         native_hw=q.get("native_size_hw"))

        add_mm = q.get("add_m_mm")
        if pnp_failed:
            pred_label = "Pred (red)  [PnP failed]"
        elif add_mm is not None:
            pred_label = f"Pred (red)  ADD:{add_mm:.1f}mm"
        else:
            pred_label = "Pred (red)"
        if args.labels:
            _put_label(pred_panel, pred_label, (1.0, 0.4, 0.4))

        # Query panel + side-by-side
        header_panel = query_rgb.copy()
        if args.labels:
            _put_label(header_panel, label, (1.0, 1.0, 1.0))

        panels = [header_panel] + ([gt_panel] if gt_panel is not None else []) + [pred_panel]
        sbs = np.concatenate(panels, axis=1)

        out_path = out_dir / f"{label}.jpg"
        Image.fromarray((np.clip(sbs, 0.0, 1.0) * 255).astype(np.uint8)).save(str(out_path))
        n_saved += 1
        print(f"[{i + 1}/{len(queries)}] {out_path}")

    print(f"\nDone. {n_saved} overlays saved to {out_dir}/")


if __name__ == "__main__":
    main()
