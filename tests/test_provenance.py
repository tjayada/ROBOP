import subprocess

import pytest
from omegaconf import OmegaConf

from evaluation.eval_utils import _git_revision, run_provenance


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git("init", "-q", cwd=tmp_path)
    _git("config", "user.email", "t@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("one")
    _git("add", "a.txt", cwd=tmp_path)
    _git("commit", "-q", "-m", "init", cwd=tmp_path)
    return tmp_path


def test_git_revision_clean(repo):
    rev = _git_revision(repo)
    assert len(rev["commit"]) == 40
    assert rev["modified"] is False


def test_git_revision_untracked_is_not_modified(repo):
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "x.pyc").write_bytes(b"x")
    assert _git_revision(repo)["modified"] is False


def test_git_revision_tracked_change_is_modified(repo):
    (repo / "a.txt").write_text("two")
    assert _git_revision(repo)["modified"] is True


def test_run_provenance_fields():
    prov = run_provenance(OmegaConf.create({"seed": 7}))
    assert prov["seed"] == 7
    assert set(prov) >= {"timestamp_utc", "robop", "externals", "python", "torch", "cuda_device"}
    # ROBOP itself is a git checkout with the pinned externals beside it
    assert prov["robop"] is not None
    assert {"NeMO", "foundpose", "gigapose", "robot-renderer"} <= set(prov["externals"])
