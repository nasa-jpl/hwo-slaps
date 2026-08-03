"""Tests for run provenance capture."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from hwoslaps.provenance import capture_provenance, config_hash, write_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_hash_is_stable_and_key_order_insensitive():
    config_a = {"psf": {"wavelength": 5.0e-7}, "run_name": "demo"}
    config_b = {"run_name": "demo", "psf": {"wavelength": 5.0e-7}}

    digest = config_hash(config_a)
    assert digest == config_hash(config_b)
    assert len(digest) == 16
    assert digest != config_hash({"run_name": "other"})


def test_capture_provenance_records_expected_fields():
    config = {"run_name": "demo"}
    command = ["runner.py", "--config", "demo.yaml"]

    provenance = capture_provenance(config=config, command=command)

    assert provenance["command"] == command
    assert provenance["config_hash"] == config_hash(config)
    assert isinstance(provenance["python"], str)
    assert "hwoslaps" in provenance["package_versions"]
    assert "autolens" in provenance["package_versions"]

    expected_git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    assert provenance["git_hash"] == expected_git_hash


def test_package_versions_prefer_module_version():
    provenance = capture_provenance()

    # Module __version__ wins over distribution metadata, which is stale for
    # source installs (e.g. autolens reports 1.0.dev0 from metadata).
    assert provenance["package_versions"]["pyyaml"] == yaml.__version__
    autolens = pytest.importorskip("autolens")
    assert provenance["package_versions"]["autolens"] == autolens.__version__


def test_capture_provenance_outside_repo_records_no_git_hash(tmp_path):
    provenance = capture_provenance(repo_dir=tmp_path)
    assert provenance["git_hash"] is None


def test_write_provenance_round_trips_through_yaml(tmp_path):
    config = {"run_name": "demo", "global_seed": 11}
    path = tmp_path / "provenance.yaml"

    written = write_provenance(path, config=config, command=["runner.py"])

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert loaded == written
    assert loaded["config_hash"] == config_hash(config)
