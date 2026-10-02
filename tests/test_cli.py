"""CLI isolation and consistent artifact capture without scientific runtime."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hwoslaps import cli
from hwoslaps.provenance import config_hash


def _master_config():
    return yaml.safe_load((ROOT / "configs/master_config.yaml").read_text())


def test_validation_only_does_not_import_scientific_runtime_or_create_outputs(tmp_path):
    config = _master_config()
    config["plotting"]["output_dir"] = str(tmp_path / "outputs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "from hwoslaps.cli import main; "
        f"assert main(['-c', {str(path)!r}, '--validate-only']) == 0; "
        "assert 'autolens' not in sys.modules; "
        "assert 'hwoslaps.pipeline' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "Configuration valid" in result.stdout
    assert not (tmp_path / "outputs").exists()


def test_cli_overrides_and_composition_use_one_resolved_config(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(_master_config()))
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("modeling:\n  enabled: false\n")
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "run_with_artifacts", lambda config, **kwargs: calls.append((config, kwargs)))

    assert cli.main([
        "-c", str(base), "-c", str(overlay), "--output-dir", "results", "--run-name", "new", "-q",
    ]) == 0

    config, kwargs = calls[0]
    assert config["modeling"]["enabled"] is False
    assert config["run_name"] == "new"
    assert config["plotting"]["output_dir"] == str(tmp_path / "results")
    assert kwargs["verbose"] is False


def test_snapshot_provenance_and_pipeline_share_resolved_configuration(tmp_path, monkeypatch, capsys):
    config = _master_config()
    config["run_name"] = "unit"
    config["plotting"]["output_dir"] = "outputs"
    original = deepcopy(config)
    received = []
    result = object()
    pipeline_module = ModuleType("hwoslaps.pipeline")

    class StubPipeline:
        def __init__(self, verbose):
            assert verbose is False

        def run(self, resolved):
            received.append(resolved)
            print("scientific output")
            return result

    pipeline_module.Pipeline = StubPipeline
    monkeypatch.setitem(sys.modules, "hwoslaps.pipeline", pipeline_module)
    import hwoslaps.provenance as provenance

    def write_provenance(path, config, command):
        path.write_text(yaml.safe_dump({"config_hash": config_hash(config), "command": command}))

    monkeypatch.setattr(provenance, "write_provenance", write_provenance)

    assert cli.run_with_artifacts(config, verbose=False, base_dir=tmp_path, command=["demo"]) is result

    run_dir = tmp_path / "outputs/unit"
    snapshot = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    recorded = yaml.safe_load((run_dir / "provenance.yaml").read_text())
    assert snapshot == received[0]
    assert recorded["config_hash"] == config_hash(snapshot)
    assert "scientific output" in (run_dir / "run.log").read_text()
    assert "scientific output" in capsys.readouterr().out
    assert config == original


def test_invalid_configuration_does_not_create_artifacts(tmp_path):
    config = {"run_name": "bad", "plotting": {"output_dir": str(tmp_path / "outputs")}}
    with pytest.raises(ValueError):
        cli.run_with_artifacts(config)
    assert not (tmp_path / "outputs").exists()


def test_existing_run_directory_preserves_all_previous_artifacts(tmp_path):
    config = _master_config()
    config["run_name"] = "previous"
    config["plotting"]["output_dir"] = str(tmp_path)
    run_dir = tmp_path / "previous"
    run_dir.mkdir()
    originals = {
        "config_used.yaml": b"previous resolved config",
        "run.log": b"previous log",
        "provenance.yaml": b"previous provenance",
        "science.npz": b"previous numerical arrays",
    }
    for name, content in originals.items():
        (run_dir / name).write_bytes(content)

    with pytest.raises(FileExistsError, match="Choose a new run_name"):
        cli.run_with_artifacts(config)

    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == originals


def test_source_checkout_runner_uses_installed_cli_options():
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "runner.py"), "--help"],
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--validate-only" in result.stdout
    assert "--base-dir" in result.stdout
