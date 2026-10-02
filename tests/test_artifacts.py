"""Forecast artifact identity contracts without a scientific runtime."""

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwoslaps import artifacts
from hwoslaps.provenance import config_hash


def _stub_writer(monkeypatch):
    written = []
    module = ModuleType("hwoslaps.modeling.utils_fisher")

    def save(grid_map, path):
        written.append((grid_map, path))
        path.write_bytes(b"stub scientific arrays")
        return path

    module.save_fisher_grid_map_npz = save
    monkeypatch.setitem(sys.modules, "hwoslaps.modeling.utils_fisher", module)
    monkeypatch.setattr(artifacts, "revision_provenance", lambda path: {
        "git_hash": "revision", "git_dirty": True, "worktree_diff_sha256": "diff",
    })
    return written


def _config(tmp_path):
    return {"run_name": "unit", "plotting": {"output_dir": str(tmp_path)}}


def test_grid_output_binds_exact_snapshot_and_preserves_runtime_metadata(tmp_path, monkeypatch):
    written = _stub_writer(monkeypatch)
    monkeypatch.setenv("HWOSLAPS_CAMPAIGN_UUID", "campaign")
    config = _config(tmp_path)
    directory = tmp_path / "unit"
    directory.mkdir()
    (directory / "config_used.yaml").write_text(yaml.safe_dump(config))
    grid_map = SimpleNamespace(runtime_provenance={"backend": "numpy"})

    output = artifacts.write_fisher_grid_map(SimpleNamespace(has_grid_map=True, grid_map=grid_map), config)

    assert output == directory / "modeling/fisher_grid_map.npz"
    assert output.is_file()
    assert grid_map.config_hash == config_hash(config)
    assert grid_map.git_hash == "revision"
    assert grid_map.campaign_uuid == "campaign"
    assert grid_map.runtime_provenance["backend"] == "numpy"
    assert grid_map.runtime_provenance["source_worktree_diff_sha256"] == "diff"
    assert written == [(grid_map, output)]


def test_mismatched_snapshot_refuses_scientific_output(tmp_path, monkeypatch):
    written = _stub_writer(monkeypatch)
    directory = tmp_path / "unit"
    directory.mkdir()
    (directory / "config_used.yaml").write_text("run_name: different\n")
    result = SimpleNamespace(has_grid_map=True, grid_map=SimpleNamespace(runtime_provenance=None))
    with pytest.raises(ValueError, match="refusing to bind"):
        artifacts.write_fisher_grid_map(result, _config(tmp_path))
    assert not written
    assert not (directory / "modeling").exists()


def test_standalone_grid_output_has_no_snapshot_or_campaign_binding(tmp_path, monkeypatch):
    _stub_writer(monkeypatch)
    monkeypatch.delenv("HWOSLAPS_CAMPAIGN_UUID", raising=False)
    result = SimpleNamespace(has_grid_map=True, grid_map=SimpleNamespace(runtime_provenance=None))
    artifacts.write_fisher_grid_map(result, _config(tmp_path))
    assert result.grid_map.config_hash is None
    assert result.grid_map.campaign_uuid is None


def test_local_forecast_does_not_create_a_grid_artifact(tmp_path):
    assert artifacts.write_fisher_grid_map(SimpleNamespace(has_grid_map=False), _config(tmp_path)) is None
    assert not (tmp_path / "unit").exists()
