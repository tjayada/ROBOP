import numpy as np


def binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert a 2-D binary mask to COCO run-length encoding (RLE).

    Encoding is column-major (Fortran order) to match the COCO convention.
    The returned dict contains 'counts' (list of run lengths) and 'size'
    ([height, width]), which is sufficient to reconstruct the mask.
    """
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle["counts"]
    mask = binary_mask.ravel(order="F")
    if len(mask) > 0:
        if mask[0] == 1:
            counts.append(0)
        mask_changes = mask[:-1] != mask[1:]
        changes_indx = np.where(np.concatenate(([True], mask_changes, [True]), 0))[0]
        counts.extend(np.diff(changes_indx).tolist())
    return rle
