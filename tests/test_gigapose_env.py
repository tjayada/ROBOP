"""ensure_gigapose_path must provide CUDA_VISIBLE_DEVICES when unset.

MegaPose's panda3d_scene_renderer.App asserts the variable exists and names a
single device. Outside a scheduler allocation (plain shell, tmux, ssh) it is
typically unset; the forked render workers then die while the parent waits on
the render queue. The fallback must default to one device and preserve any
scheduler or user setting.
"""
import os

import robop.gigapose_estimator as ge
import robop.megapose_estimator as me
# Via the package, not the bare module: conftest puts src/robop on sys.path too,
# so a bare `import estimator_utils` would be a second, unrelated module object.
from robop import estimator_utils


def _run_fresh(monkeypatch):
    monkeypatch.setattr(estimator_utils, "_GP_ROOT", None)   # force the body to re-run
    estimator_utils.ensure_gigapose_path()


def test_defaults_cuda_visible_devices_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _run_fresh(monkeypatch)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_existing_cuda_visible_devices_is_kept(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    _run_fresh(monkeypatch)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_both_estimators_use_the_shared_helper():
    """Guards the dedup: neither module may reintroduce a private copy."""
    assert ge.ensure_gigapose_path is estimator_utils.ensure_gigapose_path
    assert me.ensure_gigapose_path is estimator_utils.ensure_gigapose_path
