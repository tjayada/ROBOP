import json

import numpy as np
import pytest

from evaluation.eval_utils import (
    decode_detection,
    detection_for_frame,
    load_robot_detections,
)


def _mask_to_rle(mask):
    """Column-major RLE matching _rle_to_mask (runs alternate, background first)."""
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    counts, val, run = [], 0, 0
    for v in flat:
        if v == val:
            run += 1
        else:
            counts.append(run)
            val = 1 - val
            run = 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def _blob_mask(H=20, W=30):
    m = np.zeros((H, W), dtype=np.uint8)
    m[5:10, 7:12] = 1
    return m


def _entry(frame_id, mask, bbox=None, found=True):
    e = {"frame_id": frame_id, "found": found}
    if found:
        if bbox is None:
            ys, xs = np.nonzero(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        e["segmentation"] = _mask_to_rle(mask)
        e["bbox_xyxy"] = bbox
    return e


def _write(tmp_path, entries):
    p = tmp_path / "det.json"
    p.write_text(json.dumps({"detections": entries}))
    return p


def test_load_valid_file(tmp_path):
    p = _write(tmp_path, [_entry("f0", _blob_mask()), _entry("f1", None, found=False)])
    found, all_ids, meta = load_robot_detections(p)
    assert set(found) == {"f0"} and all_ids == {"f0", "f1"}
    assert meta["n_entries"] == 2 and meta["n_found"] == 1
    assert len(meta["sha256"]) == 64 and meta["path"] == str(p)


def test_load_none_path():
    assert load_robot_detections(None) == ({}, set(), None)


def test_decode_opening_is_per_estimator():
    m = _blob_mask()
    m[15, 20] = 1  # isolated pixel, removed by the FoundPose-style 3x3 opening
    det = _entry("f0", m)
    raw, _ = decode_detection(det, 20, 30, apply_opening=False)
    assert raw[15, 20] == 1
    opened, _ = decode_detection(det, 20, 30, apply_opening=True)
    assert opened[15, 20] == 0
    assert opened[7, 9] == 1  # blob interior survives


def test_decode_scale_and_aspect_mismatch():
    det = _entry("f0", _blob_mask())              # 20 x 30 detector image
    mask, bbox = decode_detection(det, 40, 60, apply_opening=False)   # eval at 2x
    assert mask.shape == (40, 60)
    np.testing.assert_allclose(bbox, [14, 10, 24, 20])
    with pytest.raises(ValueError, match="aspect"):
        decode_detection(det, 40, 30, apply_opening=False)


def test_detection_for_frame_policy():
    found = {"f0": _entry("f0", _blob_mask())}
    all_ids = {"f0", "f1"}
    mask, bbox, miss = detection_for_frame(found, all_ids, "f0", 20, 30, False)
    assert not miss and mask is not None and bbox is not None
    mask, bbox, miss = detection_for_frame(found, all_ids, "f1", 20, 30, False)
    assert miss and mask is None and bbox is None
    with pytest.raises(KeyError):
        detection_for_frame(found, all_ids, "f2", 20, 30, False)
