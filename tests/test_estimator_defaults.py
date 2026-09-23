"""Pin published estimator defaults that affect reported metrics."""
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "estimator"


def _cfg(name):
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def test_megapose_uses_the_official_inference_path():
    cfg = _cfg("megapose")
    assert cfg["batch_size_images"] == 128     # load_named_model's default


def test_megapose_refiner_batch_size():
    assert _cfg("megapose_refiner")["batch_size_images"] == 128


def test_megapose_coarse_is_coarse_only():
    assert _cfg("megapose_coarse")["coarse_only"] is True


def test_foundpose_template_defaults():
    cfg = _cfg("foundpose")
    assert cfg["sphere_distance_mm"] is None   # per-state distance
    assert cfg["ssaa_factor"] == 2.0           # ours; the released LMO config ships 4


def test_nemo_uses_the_published_regime():
    cfg = _cfg("nemo")
    assert cfg["crop_mode"] == "bbox"                 # external detection, not bootstrap
    assert cfg["include_query_in_templates"] is False
    assert cfg["pnp_min_correspondences"] == 256      # the published PnP floor
    assert cfg["pnp_min_inlier_ratio"] == 0.3
    # fill_frame matches bbox query framing; 3.0 is the shared no-clip bound.
    assert cfg["fill_frame"] is True
    assert cfg["sphere_distance_factor"] == 3.0
