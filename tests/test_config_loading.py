"""Portable configuration composition and path contracts."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hwoslaps.config import load_config, merge_configs, resolve_config_paths, run_directory


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_composition_resolves_paths_in_each_declaring_file(tmp_path, monkeypatch):
    base = _write(tmp_path / "instrument" / "base.yaml", {
        "plotting": {"enabled": False, "output_dir": "outputs"},
        "lensing": {"cosmology": "Planck15"},
        "modeling": {"fisher": {"covariance_path": "noise.npy", "snr_threshold": 3}},
    })
    scene = _write(tmp_path / "scene" / "source.yaml", {
        "lensing": {"source_galaxy": {"light": {"type": "Image", "asset_path": "source.npz"}}},
        "modeling": {"fisher": {"snr_threshold": 5}},
    })
    monkeypatch.chdir(tmp_path.parent)

    config = load_config([base, scene], validate=False)

    assert config["plotting"]["output_dir"] == str(base.parent / "outputs")
    assert config["lensing"]["source_galaxy"]["light"]["asset_path"] == str(scene.parent / "source.npz")
    assert config["modeling"]["fisher"]["covariance_path"] == str(base.parent / "noise.npy")
    assert config["modeling"]["fisher"]["snr_threshold"] == 5
    assert config["lensing"]["cosmology"] == "Planck15"


def test_composition_replaces_sequences_and_does_not_mutate_inputs():
    base = {"map": {"positions": [[1, 2], [3, 4]], "engine": "numpy"}}
    overlay = {"map": {"positions": [[5, 6]]}}
    original_base, original_overlay = deepcopy(base), deepcopy(overlay)
    merged = merge_configs(base, overlay)
    assert merged == {"map": {"positions": [[5, 6]], "engine": "numpy"}}
    merged["map"]["positions"][0][0] = 10
    assert base == original_base
    assert overlay == original_overlay


def test_explicit_base_directory_preserves_archived_path_convention(tmp_path):
    path = _write(tmp_path / "configs" / "archived.yaml", {
        "plotting": {"output_dir": "outputs"},
        "lensing": {"source_galaxy": {"light": {"asset_path": "configs/assets/image.npz"}}},
    })
    config = load_config(path, base_dir=tmp_path, validate=False)
    assert config["plotting"]["output_dir"] == str(tmp_path / "outputs")
    assert config["lensing"]["source_galaxy"]["light"]["asset_path"] == str(
        tmp_path / "configs/assets/image.npz"
    )


def test_python_overrides_use_caller_directory_and_do_not_mutate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = _write(tmp_path / "configs" / "base.yaml", {"plotting": {"output_dir": "outputs"}})
    overrides = {"plotting": {"output_dir": "other"}, "run_name": "new"}
    original = deepcopy(overrides)
    config = load_config(path, overrides=overrides, validate=False)
    assert config["plotting"]["output_dir"] == str(tmp_path / "other")
    assert overrides == original


@pytest.mark.parametrize("document", [None, [], "string", 4, True])
def test_non_mapping_yaml_is_rejected(tmp_path, document):
    path = _write(tmp_path / "bad.yaml", document)
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_config(path, validate=False)


def test_yaml_syntax_error_names_source_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("key: [unterminated", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML.*broken.yaml"):
        load_config(path, validate=False)


def test_empty_config_sequence_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        load_config([], validate=False)


def test_resolver_changes_only_schema_paths_and_preserves_null(tmp_path):
    original = {
        "plotting": {"output_dir": Path("results")},
        "modeling": {"fisher": {"covariance_path": None, "engine": "jax_gpu"}},
        "population": {"name": "my-population", "asset_path": "unchanged"},
    }
    resolved = resolve_config_paths(original, base_dir=tmp_path)
    assert resolved["plotting"]["output_dir"] == str(tmp_path / "results")
    assert resolved["modeling"]["fisher"]["covariance_path"] is None
    assert resolved["population"] == original["population"]
    assert original["plotting"]["output_dir"] == Path("results")


@pytest.mark.parametrize("name", ["", " ", ".", "..", "../escape", "a/b", "a\\b"])
def test_artifact_run_name_is_one_directory_component(name):
    with pytest.raises(ValueError, match="run_name"):
        run_directory({"run_name": name, "plotting": {"output_dir": "/tmp/results"}})


def test_repository_master_config_validates_without_science_imports():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/master_config.yaml")
    assert Path(config["plotting"]["output_dir"]).is_absolute()
