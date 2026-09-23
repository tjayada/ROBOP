"""CRAVES-Lab dataset loader and camera helpers.

Keypoints and boxes remain in native 1280x720 coordinates. Projection follows
RoboPose's ``third_party/craves/get_2d_gt.py`` and ``d3.py:CameraPose``; a fixed
rotation below converts Unreal Engine world coordinates to the URDF frame.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from loaders.utils import scale_K_to_size, scaled_size


# 17 keypoints = 3 joint keypoints (Base, Elbow, Wrist) + 14 vertex-group kps.
# The `Rotation` joint is skipped (see get_2d_gt.py: `joints2d[1:]`).

CRAVES_JOINT_KP_NAMES = ["Rotation", "Base", "Elbow", "Wrist"]  # first one is dropped
CRAVES_VERTEX_SEQ = [
    [0], [1], [2], [3, 4],
    [5, 5, 6, 6, 7, 8], [5, 6, 7, 7, 8, 8],
    [9, 9, 10, 10, 11, 12], [9, 10, 11, 11, 12, 12],
    [13, 14], [15, 16], [17, 18], [19, 20], [21], [22, 23],
]
CRAVES_ACTOR_NAME = "RobotArmActor_3"
CRAVES_NUM_KEYPOINTS = 17   # 3 joint + 14 vertex
CRAVES_NUM_JOINT_KP = 3     # Base, Elbow, Wrist - correspond to FK link origins


def fov2f(fov_deg: float, width: int) -> float:
    """Horizontal-FoV (UE4 convention) to focal length in pixels."""
    return width / (2.0 * np.tan(fov_deg * np.pi / 180.0 / 2.0))


def _make_translation(X: float, Y: float, Z: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = -X
    T[1, 3] = -Y
    T[2, 3] = -Z
    return T


def _make_rotation_ue(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
    """UE4 rotation: yaw·pitch·roll, with pitch and roll negated (UE convention)."""
    pitch = -pitch_deg / 180.0 * np.pi
    yaw   =  yaw_deg   / 180.0 * np.pi
    roll  = -roll_deg  / 180.0 * np.pi
    ryaw = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                     [np.sin(yaw),  np.cos(yaw), 0, 0],
                     [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    rpitch = np.array([[ np.cos(pitch), 0, np.sin(pitch), 0],
                       [0, 1, 0, 0],
                       [-np.sin(pitch), 0, np.cos(pitch), 0],
                       [0, 0, 0, 1]], dtype=np.float64)
    rroll = np.array([[1, 0, 0, 0],
                      [0, np.cos(roll), -np.sin(roll), 0],
                      [0, np.sin(roll),  np.cos(roll), 0],
                      [0, 0, 0, 1]], dtype=np.float64)
    return ryaw @ rpitch @ rroll


def _make_rearrange() -> np.ndarray:
    """UE world axes (Xf, Yr, Zu) -> OpenCV cam axes (x right, y down, z forward)."""
    return np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)


class CRAVESCameraPose:
    """Mirrors robopose/third_party/craves/d3.py:CameraPose for GT 2D keypoint construction."""

    def __init__(self, x, y, z, pitch, yaw, roll, width, height, f):
        self.x, self.y, self.z = x, y, z
        self.pitch, self.yaw, self.roll = pitch, yaw, roll
        self.width, self.height, self.f = width, height, f

    def project_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        points_3d = np.asarray(points_3d, dtype=np.float64)
        N = points_3d.shape[0]
        homog = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
        world_to_cam = _make_rotation_ue(self.pitch, self.yaw, self.roll).T @ _make_translation(self.x, self.y, self.z)
        pts_cam = (_make_rearrange() @ world_to_cam @ homog.T).T
        px, py = self.width / 2.0, self.height / 2.0
        pts_2d = np.zeros((N, 2), dtype=np.float64)
        pts_2d[:, 0] = pts_cam[:, 0] / pts_cam[:, 2] * self.f + px
        pts_2d[:, 1] = pts_cam[:, 1] / pts_cam[:, 2] * self.f + py
        return pts_2d


def make_2d_keypoints(vertex_infos: dict, joint_infos: dict, cam_info: dict) -> np.ndarray:
    """
    Exact port of robopose/third_party/craves/get_2d_gt.py:make_2d_keypoints.
    Returns a (17, 2) array of GT 2D keypoints in native CRAVES pixel coords.
    """
    loc, rot = cam_info["Location"], cam_info["Rotation"]
    f = fov2f(cam_info.get("Fov", 90.0), cam_info["FilmWidth"])
    cam = CRAVESCameraPose(loc["X"], loc["Y"], loc["Z"],
                           rot["Pitch"], rot["Yaw"], rot["Roll"],
                           cam_info["FilmWidth"], cam_info["FilmHeight"], f)

    joints_world = joint_infos[CRAVES_ACTOR_NAME]["WorldJoints"]
    joints3d = np.array([[joints_world[n]["X"], joints_world[n]["Y"], joints_world[n]["Z"]]
                         for n in CRAVES_JOINT_KP_NAMES], dtype=np.float64)
    joints2d = cam.project_to_2d(joints3d)[1:]  # drop Rotation -> (3, 2)

    vertex = vertex_infos[CRAVES_ACTOR_NAME]
    vertex3d = np.array([[v["X"], v["Y"], v["Z"]] for v in vertex], dtype=np.float64)
    vertex2d = cam.project_to_2d(vertex3d)
    vert_avg = np.array([vertex2d[seq].mean(axis=0) for seq in CRAVES_VERTEX_SEQ])

    return np.concatenate([joints2d, vert_avg], axis=0)  # (17, 2)


def craves_camera_opencv(cam_info: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an OpenCV-style K and world->camera T0C from CRAVES caminfo.json.

    Construction mirrors CRAVESCameraPose.project_to_2d exactly so GT 2D
    projection via K·T0C matches the reference pixel-for-pixel.
    Translation is in metres (UE units × 0.001).

    Returns: K (3,3), T0C (4,4) where X_cam = T0C @ X_world.
    """
    W, H = cam_info["FilmWidth"], cam_info["FilmHeight"]
    f = fov2f(cam_info["Fov"], W)
    K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    loc, rot = cam_info["Location"], cam_info["Rotation"]
    cam_T = _make_translation(loc["X"], loc["Y"], loc["Z"])
    cam_R = _make_rotation_ue(rot["Pitch"], rot["Yaw"], rot["Roll"])
    T0C = _make_rearrange() @ cam_R.T @ cam_T
    T0C[:3, 3] *= 0.001  # UE units -> metres
    return K, T0C


# UE world <-> URDF world bridge
# UE is left-handed (Y right); URDF/ROS is right-handed (Y left).
# Verified: p_ue = diag(1,-1,1) @ p_urdf - X and Z identical, Y negated.
# Post-multiplying T0C_ue[:3,:3] by R_URDF_TO_UE yields a proper rotation
# matrix (det=+1) consistent with NeMO's cTr_pred.

R_URDF_TO_UE = np.diag([1.0, -1.0, 1.0])  # Y-flip; det=-1, correct bridge


def T0C_ue_to_urdf_world(T0C_ue: np.ndarray) -> np.ndarray:
    """
    Map T0C_ue (d3-style UE w2c) to T0C_urdf by post-multiplying the rotation
    block: R' = R @ R_URDF_TO_UE = R @ diag(1,-1,1) (Y-flip).
    Translation is unchanged (already in camera space).
    """
    T = np.asarray(T0C_ue, dtype=np.float64).copy()
    T[:3, :3] = T[:3, :3] @ R_URDF_TO_UE
    return T


# Segmentation bbox

def bbox_from_craves_seg(seg_path: str | Path) -> np.ndarray:
    """Derive tight bbox [x1,y1,x2,y2] from a CRAVES seg PNG (robot pixels: R==0)."""
    mask = np.asarray(Image.open(str(seg_path)))
    robot = mask[..., 0] == 0 if mask.ndim == 3 else mask == 0
    ys, xs = np.where(robot)
    if xs.size == 0:
        raise ValueError(f"empty robot mask in {seg_path}")
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


# Joint angle parsing

def parse_craves_joint_angles(angles_json_path: str | Path) -> np.ndarray:
    """
    Load first 4 entries of angles/*.json, convert deg->rad, apply q[0]*=-1
    sign flip to match URDF convention.  Returns (4,) float64.
    """
    arr = np.asarray(json.loads(Path(angles_json_path).read_text()))
    q = arr[:4].astype(np.float64) * np.pi / 180.0
    q[0] *= -1.0
    return q


def load_craves_cam_info(caminfo_path: str | Path) -> Dict:
    return json.loads(Path(caminfo_path).read_text())


def load_craves_vertex_joint(vertex_path: str | Path, joint_path: str | Path) -> Tuple[Dict, Dict]:
    return json.loads(Path(vertex_path).read_text()), json.loads(Path(joint_path).read_text())


# 17-keypoint projection
# Each keypoint is (link_name, offset_in_link_frame).  At eval:
# world_pos = R_link @ offset + t_link, then p_cam = R_pred @ world_pos + t_pred
# -> project through K.  Mirrors robopose/lib3d/urdf_layer.py::get_keypoints.
# Link index mapping (FK output order): 0=Model, 1=Rotation, 2=Base, 3=Elbow, 4=Wrist

_LINK_NAME_TO_IDX = {"Model": 0, "Rotation": 1, "Base": 2, "Elbow": 3, "Wrist": 4}

# The link-local form of CRAVES_VERTEX_SEQ above: RoboPose resolved each vertex
# group to a fixed point in its parent link's frame, so the keypoints can be
# carried through FK instead of through the CAD model.  The first three are the
# Base/Elbow/Wrist link origins; the other 14 are surface landmarks.
# Verbatim from the RoboPose deps archive, owi-description/keypoints.json:
#   https://www.paris.inria.fr/archive_ylabbeprojectsdata/robopose/deps/owi-description/keypoints.json
#   sha256 dddfda183d46834701b5ddbfcc4fbc1ab459b1d3ced4bf083b85b6cba551bb9f
# Order is the file's order, which is the GT keypoint order - do not re-sort.
# These offsets are metres in the OWI-535 URDF link frames; they only make sense
# against that URDF (robot-renderer's owi535 registry entry).
CRAVES_KEYPOINT_OFFSETS: list[tuple[str, tuple[float, float, float]]] = [
    ("Base",  (0.0, 0.0, 0.0)),
    ("Elbow", (0.0, 0.0, 0.0)),
    ("Wrist", (0.0, 0.0, 0.0)),
    ("Model", (-0.044427200317382814, 0.05549999999999998, 0.03913042449951173)),
    ("Model", (0.03557075500488281, 5.541761655926608e-18, -0.021309238433837887)),
    ("Model", (-0.12441761779785157, 0.0005000000000000055, -0.025305164337158195)),
    ("Model", (-0.04462409591674805, 0.061499999999999964, 0.10689374923706053)),
    ("Base",  (-0.02613829168050258, 0.03436878813214973, -0.00040912808190532446)),
    ("Base",  (-0.026138293158161616, 0.05537155448348488, -0.0004653226341373856)),
    ("Base",  (0.02586170453989227, 0.03459580056914274, -8.115660148472204e-05)),
    ("Base",  (0.025861703110828502, 0.05572965118322966, 0.0002343427905677209)),
    ("Elbow", (0.0012836476515605058, 5.050504386699267e-05, 0.030249485404606022)),
    ("Elbow", (0.001283639357362483, 0.01705050183571323, -0.028250515527677267)),
    ("Elbow", (0.001783636187225969, -0.016949498578384996, -0.035250513742780404)),
    ("Elbow", (0.0016809297289447772, -0.0006151571130747699, -0.06255600320862012)),
    ("Wrist", (-0.0017172939368220244, 0.013035988795018827, -0.07269680390359037)),
    ("Wrist", (-0.0013837937138754808, 0.025710973679562765, -0.04182110519139265)),
]


def craves_keypoint_offsets() -> tuple[np.ndarray, list[int]]:
    """
    Return the (17,3) CRAVES keypoint offsets in link-local frames and the FK
    link index each one belongs to.
    """
    offsets = np.array([off for _, off in CRAVES_KEYPOINT_OFFSETS], dtype=np.float64)
    link_idxs = [_LINK_NAME_TO_IDX[name] for name, _ in CRAVES_KEYPOINT_OFFSETS]
    return offsets, link_idxs


def predict_all_17_keypoints_2d(
    kp_offsets: np.ndarray,
    kp_link_idxs: list[int],
    R_list: np.ndarray,
    t_list: np.ndarray,
    R_pred: np.ndarray,
    t_pred: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Project all 17 CRAVES keypoints using the predicted pose.
    Mirrors RoboPose's urdf_layer.get_keypoints() + project_points().

    Returns (17, 2) predicted 2D keypoints in native pixels.
    """
    R_arr = np.asarray(R_list, dtype=np.float64)  # (5,3,3)
    t_arr = np.asarray(t_list, dtype=np.float64)  # (5,3)
    pts_world = np.stack([R_arr[li] @ kp_offsets[i] + t_arr[li]
                          for i, li in enumerate(kp_link_idxs)])   # (17,3)
    pts_cam = (R_pred @ pts_world.T).T + t_pred                    # (17,3)
    pix = (K @ pts_cam.T).T
    return pix[:, :2] / (pix[:, 2:3] + 1e-8)                      # (17,2)


# Dataset

CRAVES_DEFAULT_W = 1280
CRAVES_DEFAULT_H = 720


class CRAVESLabDataset(Dataset):
    """
    Iterates over the CRAVES lab_test_real split (428 frames by default).

    Parameters
    ----------
    data_folder   : path to .../test_20181024/
    scale         : image resize factor; kp2d_gt and bbox stay in native pixels.
    frame_slice   : optional (start, stop[, step]) to restrict frames.
    trans_to_tensor : optional transform; default = ToTensor().
    """

    def __init__(
        self,
        data_folder: str,
        scale: float = 1.0,
        frame_slice: Optional[Sequence[int]] = None,
        trans_to_tensor=None,
    ):
        self.data_folder = Path(os.path.expanduser(data_folder))
        if not self.data_folder.exists():
            raise FileNotFoundError(f"CRAVES data folder not found: {self.data_folder}")
        self.scale = float(scale)

        self.cam_dir    = self.data_folder / "FusionCameraActor3_2"
        self.img_dir    = self.cam_dir / "lit"
        self.seg_dir    = self.cam_dir / "seg"
        self.caminfo_dir = self.cam_dir / "caminfo"
        self.angles_dir = self.data_folder / "angles"
        self.joint_dir  = self.data_folder / "joint"
        self.vertex_dir = self.data_folder / "vertex"

        for p in (self.img_dir, self.seg_dir, self.caminfo_dir,
                  self.angles_dir, self.joint_dir, self.vertex_dir):
            if not p.exists():
                raise FileNotFoundError(f"missing expected subdir: {p}")

        frame_ids = sorted(p.stem for p in self.img_dir.glob("*.jpg"))
        if frame_slice is not None:
            frame_ids = frame_ids[slice(*frame_slice)]
        if not frame_ids:
            raise RuntimeError(f"no frames found under {self.img_dir}")
        self.frame_ids = frame_ids

        if trans_to_tensor is None:
            import torchvision.transforms as T
            trans_to_tensor = T.ToTensor()
        self.trans_to_tensor = trans_to_tensor

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_image(self, frame_id: str) -> torch.Tensor:
        with Image.open(self.img_dir / f"{frame_id}.jpg") as im:
            im = im.convert("RGB")
            if self.scale != 1.0:
                im = im.resize(
                    (int(round(im.width * self.scale)), int(round(im.height * self.scale))),
                    Image.Resampling.BILINEAR,
                )
            return self.trans_to_tensor(im)

    def __getitem__(self, idx: int) -> dict:
        frame_id = self.frame_ids[idx]
        image = self._load_image(frame_id)

        q = parse_craves_joint_angles(self.angles_dir / f"{frame_id}.json")
        joints_rad = torch.from_numpy(q.astype(np.float32))

        cam_info = load_craves_cam_info(self.caminfo_dir / f"{frame_id}.json")
        K_native, T0C_ue = craves_camera_opencv(cam_info)
        T0C = T0C_ue_to_urdf_world(T0C_ue)  # canonical bridge: R_URDF_TO_UE = diag(1,-1,1)

        K = K_native.copy()
        if self.scale != 1.0:
            # K follows the pixels actually produced by _load_image, not the
            # requested scale: rounding makes the x and y factors differ slightly
            native_w, native_h = craves_native_size()
            K = scale_K_to_size(
                K_native, native_w, native_h, *scaled_size(native_w, native_h, self.scale)
            )

        vertex, joint = load_craves_vertex_joint(
            self.vertex_dir / f"{frame_id}.json",
            self.joint_dir  / f"{frame_id}.json",
        )
        kp2d_gt = make_2d_keypoints(vertex, joint, cam_info).astype(np.float32)
        bbox = bbox_from_craves_seg(self.seg_dir / f"{frame_id}.png")

        return {
            "image":     image,
            "joints_rad": joints_rad,
            "K":         K.astype(np.float32),
            "K_native":  K_native.astype(np.float32),
            "T0C":       T0C.astype(np.float32),
            "kp2d_gt":   kp2d_gt,               # (17,2) native 1280×720 pixels
            "bbox":      bbox.astype(np.float32),
            "frame_id":  frame_id,
        }


def craves_native_size() -> tuple:
    """Canonical CRAVES lab image size."""
    return (CRAVES_DEFAULT_W, CRAVES_DEFAULT_H)
