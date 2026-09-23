"""Loader for the CtRNet Baxter real-world dataset.

Images may be scaled, while ground-truth coordinates remain at the native
2048x1536 resolution. The fixed intrinsics below are reported by CtRNet.
"""
import glob
import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from loaders.utils import _pil_loader, scaled_size

# Full-resolution camera intrinsics (Azure Kinect, as reported in CtRNet paper).
BAXTER_FX = 960.41357421875
BAXTER_FY = 960.22314453125
BAXTER_CX = 1021.7171020507812
BAXTER_CY = 776.2381591796875
BAXTER_WIDTH_FULL  = 2048
BAXTER_HEIGHT_FULL = 1536
BAXTER_EVAL_SCALE  = 0.3125   # used by CtRNet: brings image to 640 × 480


def baxter_K_full() -> np.ndarray:
    """Return 3×3 intrinsic matrix at full (2048×1536) resolution."""
    return np.array(
        [[BAXTER_FX, 0., BAXTER_CX],
         [0., BAXTER_FY, BAXTER_CY],
         [0., 0., 1.]], dtype=np.float32
    )


def baxter_K_scaled(scale: float = BAXTER_EVAL_SCALE) -> np.ndarray:
    """Return 3×3 intrinsic matrix scaled to evaluation resolution."""
    K = baxter_K_full()
    K[0, 0] *= scale
    K[0, 2] *= scale
    K[1, 1] *= scale
    K[1, 2] *= scale
    return K


class BaxterDataset(Dataset):
    """Iterate over images, joints and native-resolution endpoint labels."""

    def __init__(
        self,
        data_folder: str,
        scale: float = BAXTER_EVAL_SCALE,
        trans_to_tensor=None,
    ):
        self.data_folder = os.path.expanduser(data_folder)
        self.scale = scale

        if trans_to_tensor is None:
            import torchvision.transforms as T
            trans_to_tensor = T.ToTensor()
        self.trans_to_tensor = trans_to_tensor

        gt_path = os.path.join(self.data_folder, "ground_truth_data")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(
                f"ground_truth_data not found in {self.data_folder}. "
                "Expected the CtRNet Baxter dataset layout."
            )
        with open(gt_path, "rb") as f:
            self._ground_truth = pickle.load(f)

        # Discover pose directories and sort numerically.
        pose_dirs = sorted(
            [d for d in os.listdir(self.data_folder)
             if d.startswith("pose_") and os.path.isdir(os.path.join(self.data_folder, d))],
            key=lambda d: int(d.split("_")[1]),
        )

        # Build flat sample list: (image_path, pose_key)
        self._samples = []
        for pose_dir in pose_dirs:
            pose_key = pose_dir  # e.g. "pose_3"
            if pose_key not in self._ground_truth:
                continue
            pattern = os.path.join(self.data_folder, pose_dir, "*.png")
            img_files = sorted(glob.glob(pattern))
            for img_path in img_files:
                self._samples.append((img_path, pose_key))

        if not self._samples:
            raise FileNotFoundError(
                f"No PNG images found under pose_N/ directories in {self.data_folder}."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        img_path, pose_key = self._samples[idx]
        image_pil = _pil_loader(img_path)
        if self.scale != 1.0:
            w, h = image_pil.size
            # same rounding as the other loaders, so K and pixels agree
            image_pil = image_pil.resize(
                scaled_size(w, h, self.scale), Image.Resampling.BILINEAR,
            )
        image = self.trans_to_tensor(image_pil)

        gt = self._ground_truth[pose_key]
        joint_angles = torch.tensor(np.array(gt["joints"], dtype=np.float32))
        ee_2d_gt = np.array(gt["ee_2d"], dtype=np.float32)
        ee_3d_gt = np.array(gt["ee_3d"], dtype=np.float32)
        return image, joint_angles, ee_2d_gt, ee_3d_gt, pose_key
