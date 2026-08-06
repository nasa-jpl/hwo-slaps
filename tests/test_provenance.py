"""Tests for run provenance capture."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from hwoslaps.lensing.image_source import load_source_image_asset
from hwoslaps.provenance import capture_provenance, config_hash, write_provenance

from test_image_source import _write_asset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_hash_is_stable_and_key_order_insensitive():
    """Hash a config to 16 chars regardless of key order."""
    config_a = {"psf": {"wavelength": 5.0e-7}, "run_name": "demo"}
    config_b = {"run_name": "demo", "psf": {"wavelength": 5.0e-7}}

    digest = config_hash(config_a)
    assert digest == config_hash(config_b)
    assert len(digest) == 16
    assert digest != config_hash({"run_name": "other"})


def test_capture_provenance_records_expected_fields():
    """Record command, config hash, Python, packages, and git hash."""
    config = {"run_name": "demo"}
    command = ["runner.py", "--config", "demo.yaml"]

    provenance = capture_provenance(config=config, command=command)

    assert provenance["command"] == command
    assert provenance["config_hash"] == config_hash(config)
    assert isinstance(provenance["python"], str)
    assert "hwoslaps" in provenance["package_versions"]
    assert "autolens" in provenance["package_versions"]

    expected_git_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    assert provenance["git_hash"] == expected_git_hash
    assert len(provenance["git_hash"]) == 40
    assert isinstance(provenance["git_dirty"], bool)
    if provenance["git_dirty"]:
        assert isinstance(provenance["git_dirty_paths"], list)
        assert isinstance(provenance["git_diff_sha256"], str)
    else:
        assert provenance["git_dirty_paths"] == []
        assert provenance["git_diff_sha256"] is None


def test_package_versions_prefer_module_version():
    """Prefer module __version__ over stale distribution metadata."""
    provenance = capture_provenance()

    # Module __version__ wins over distribution metadata, which is stale for
    # source installs (e.g. autolens reports 1.0.dev0 from metadata).
    assert provenance["package_versions"]["pyyaml"] == yaml.__version__
    autolens = pytest.importorskip("autolens")
    assert provenance["package_versions"]["autolens"] == autolens.__version__


def test_capture_provenance_outside_repo_records_no_git_hash(tmp_path):
    """Record null git state when run outside the repository."""
    provenance = capture_provenance(repo_dir=tmp_path)
    for key in ("git_hash", "git_dirty", "git_dirty_paths", "git_diff_sha256"):
        assert provenance[key] is None


def _init_git_repo(path):
    """Create a temporary repository with one committed tracked file."""
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=path,
        check=True,
    )
    tracked = path / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "initial"],
        cwd=path,
        check=True,
    )
    return tracked


def test_capture_provenance_records_full_git_state(tmp_path):
    """Capture clean and dirty tracked-tree state without untracked files."""
    repo = tmp_path / "repo"
    tracked = _init_git_repo(repo)

    clean = capture_provenance(repo_dir=repo)
    expected_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    assert clean["git_hash"] == expected_hash
    assert len(clean["git_hash"]) == 40
    assert clean["git_dirty"] is False
    assert clean["git_dirty_paths"] == []
    assert clean["git_diff_sha256"] is None

    tracked.write_text("changed\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("ignored by git state\n", encoding="utf-8")
    dirty = capture_provenance(repo_dir=repo)
    expected_diff = subprocess.check_output(
        ["git", "diff", "HEAD"], cwd=repo
    )
    expected_diff_sha = hashlib.sha256(expected_diff).hexdigest()
    assert dirty["git_dirty"] is True
    assert dirty["git_dirty_paths"] == ["tracked.txt"]
    assert dirty["git_diff_sha256"] == expected_diff_sha
    assert capture_provenance(repo_dir=repo)["git_diff_sha256"] == expected_diff_sha


def test_capture_provenance_records_image_asset_identity(tmp_path):
    """Persist Image-source asset identity and omit it for analytic sources."""
    asset_path = _write_asset(tmp_path / "source.npz")
    asset = load_source_image_asset(asset_path)
    image_config = {
        "lensing": {
            "source_galaxy": {
                "light": {"type": "Image", "asset_path": str(asset_path)}
            }
        }
    }

    provenance = capture_provenance(config=image_config, repo_dir=tmp_path)
    assert provenance["source_image_asset"] == {
        "asset_path": str(asset_path.resolve()),
        "sha256_16": asset.sha256_16,
        "pixel_scale_arcsec": pytest.approx(asset.pixel_scale_arcsec),
        "shape": list(asset.sb.shape),
    }

    analytic_config = {
        "lensing": {
            "source_galaxy": {"light": {"type": "Exponential"}}
        }
    }
    assert "source_image_asset" not in capture_provenance(
        config=analytic_config, repo_dir=tmp_path
    )


def test_write_provenance_round_trips_through_yaml(tmp_path):
    """Round-trip the provenance record through its YAML file."""
    config = {"run_name": "demo", "global_seed": 11}
    path = tmp_path / "provenance.yaml"

    written = write_provenance(path, config=config, command=["runner.py"])

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert loaded == written
    assert loaded["config_hash"] == config_hash(config)
