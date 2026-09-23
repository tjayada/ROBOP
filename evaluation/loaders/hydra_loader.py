"""Loader for one Hydra ICP benchmark measurement.

Each directory contains a shared camera pose and intrinsics, 15 images and a
joint-state JSON. Image scaling also scales intrinsics, but not the pose.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from loaders.utils import _pil_loader, scale_K_to_size, scaled_size

HYDRA_NATIVE_HEIGHT = 720
HYDRA_NATIVE_WIDTH  = 1280


class HydraDataset(Dataset):
    """Yield scaled images, joint angles, the shared pose and frame index.

    ``K`` matches scaled images; ``K_native`` retains the 1280x720 intrinsics.
    """

    def __init__(
        self,
        measurement_dir: str | Path,
        scale: float = 1.0,
        trans_to_tensor=None,
    ):
        self.meas_dir = Path(measurement_dir)
        self.scale    = float(scale)

        if not self.meas_dir.exists():
            raise FileNotFoundError(f"Measurement directory not found: {self.meas_dir}")

        if trans_to_tensor is None:
            import torchvision.transforms as T
            trans_to_tensor = T.ToTensor()
        self.trans_to_tensor = trans_to_tensor

        self.T_cam_base: np.ndarray = np.load(self.meas_dir / "T_cam_base.npy")
        self.K_native: np.ndarray   = np.load(self.meas_dir / "camera_K.npy").astype(np.float32)

        # K is resolved ONCE here, from the integer output size the resize will
        # actually produce (rounding makes the effective x/y factors differ from
        # `scale`). All frames of a measurement share one camera, and callers read
        # dataset.K before iterating, so it must be correct before the first
        # __getitem__ rather than recomputed per frame.
        self._out_size = None
        self.K = self.K_native.copy()
        if self.scale != 1.0:
            with Image.open(self.meas_dir / "images" / "000.png") as probe:
                w, h = probe.size
            self._out_size = scaled_size(w, h, self.scale)
            self.K = scale_K_to_size(
                self.K_native, w, h, *self._out_size
            ).astype(np.float32)

        with open(self.meas_dir / "ground_truth.json") as fh:
            gt_raw = json.load(fh)

        # JSON keys are strings; convert to int and joints back to ndarray.
        gt = {int(k): {"joints": np.array(v["joints"], dtype=np.float64)}
              for k, v in gt_raw.items()}

        self._frames    = sorted(gt.items())   # [(frame_idx, {"joints": ...}), ...]
        self._image_dir = self.meas_dir / "images"

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        frame_idx, gt = self._frames[idx]
        img_path      = self._image_dir / f"{frame_idx:03d}.png"
        img_pil       = _pil_loader(str(img_path))
        if self._out_size is not None:
            img_pil = img_pil.resize(self._out_size, Image.Resampling.BILINEAR)
        image        = self.trans_to_tensor(img_pil)
        joint_angles = torch.tensor(gt["joints"].astype(np.float32))
        return image, joint_angles, self.T_cam_base, frame_idx
