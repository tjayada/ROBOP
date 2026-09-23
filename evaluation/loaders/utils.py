"""Shared utilities for ROBOP data loaders."""
import numpy as np
from PIL import Image


def _pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def scaled_size(width: int, height: int, scale: float) -> tuple:
    """Output size for a resize by `scale`, rounded to whole pixels."""
    return int(round(width * scale)), int(round(height * scale))


def scale_K_to_size(K, width: int, height: int, out_width: int, out_height: int):
    """Scale intrinsics to an actual output size.

    The per-axis factors come from the integer output dimensions rather than
    the requested scale, because rounding to whole pixels makes the effective
    x and y factors differ slightly for non-integral resizes.
    """
    if min(width, height, out_width, out_height) <= 0:
        raise ValueError(
            f"invalid sizes: {width}x{height} to {out_width}x{out_height}"
        )
    sx, sy = out_width / width, out_height / height
    K_out = np.array(K, dtype=np.float64).copy()
    K_out[0, 0] *= sx
    K_out[0, 2] *= sx
    K_out[1, 1] *= sy
    K_out[1, 2] *= sy
    return K_out
