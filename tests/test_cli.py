"""Public command routing and real command-owned output artifacts."""

from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from hwoslaps import cli
from hwoslaps.modeling.forecast_results import ForecastResult
from hwoslaps.provenance import config_hash

ROOT = Path(__file__).resolve().parents[1]


def _config():
    config = yaml.safe_load((ROOT / "configs/master_config.yaml").read_text())
    config.pop("run_name", None)
    config.pop("plotting", None)
    config["modeling"].pop("enabled", None)
    config["modeling"].pop("detection", None)
    config["modeling"]["fisher"].pop("mode", None)
    return config


def _write_config(path, config=None):
    path.write_text(yaml.safe_dump(_config() if config is None else config))
    return path


def test_validate_is_backend_free_and_creates_no_outputs(tmp_path):
    path = _write_config(tmp_path / "config.yaml")
    code = (
        "import sys; from hwoslaps.cli import main; "
        f"assert main(['validate', '-c', {str(path)!r}]) == 0; "
        "assert not {'autolens', 'hcipy', 'jax', 'hwoslaps.pipeline'} & sys.modules.keys()"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "Configuration valid" in result.stdout
    assert {p.name for p in tmp_path.iterdir()} == {"config.yaml"}


def test_forecast_routes_explicit_arrays_and_persists_real_result(tmp_path, monkeypatch):
    import hwoslaps

    path = _write_config(tmp_path / "config.yaml")
    positions = [[0.1, -0.2], [0.3, 0.4]]
    positions_path = tmp_path / "positions.json"
    positions_path.write_text("[[0.1,-0.2],[0.3,0.4]]")
    effective = _config()
    effective["run_name"] = "forecast"
    prepared = SimpleNamespace(config=effective)
    received = []
    result = ForecastResult(
        masses_msun=np.array([1e7, 1e8]), positions_yx=np.array(positions),
        q_asimov=np.array([[1.0, 3.0], [10.0, 30.0]]),
        fisher_raw=np.ones((2, 2)), fisher_profiled=np.ones((2, 2)),
        sigma_amplitude=np.ones((2, 2)), degradation=np.ones((2, 2)),
        runtime_provenance={"backend": "reference"},
    )

    def prepare(config):
        received.append(config)
        return prepared

    def evaluate(value, *, masses, positions):
        assert value is prepared
        np.testing.assert_array_equal(masses, [1e7, 1e8])
        np.testing.assert_array_equal(positions, result.positions_yx)
        print("forecast evaluated")
        return result

    monkeypatch.setattr(hwoslaps, "prepare_forecast", prepare)
    monkeypatch.setattr(hwoslaps, "forecast", evaluate)
    output = tmp_path / "forecast"
    assert cli.main([
        "forecast", "-c", str(path), "--output-dir", str(output),
        "--masses", "1e7", "1e8", "--positions", str(positions_path),
    ]) == 0
    loaded = ForecastResult.load_npz(output / "forecast.npz")
    np.testing.assert_array_equal(loaded.q_asimov, result.q_asimov)
    snapshot = yaml.safe_load((output / "config_used.yaml").read_text())
    provenance = yaml.safe_load((output / "provenance.yaml").read_text())
    assert snapshot == prepared.config
    assert snapshot["run_name"] == "forecast"
    assert "run_name" not in received[0]
    assert provenance["config_hash"] == config_hash(snapshot)
    assert "forecast evaluated" in (output / "run.log").read_text()


def test_simulate_uses_public_operation_and_writes_observation_arrays(tmp_path, monkeypatch):
    import hwoslaps

    config = _config()
    config.pop("modeling")
    path = _write_config(tmp_path / "config.yaml", config)
    observation = SimpleNamespace(
        data=SimpleNamespace(native=np.array([[3.0, 4.0]])),
        noise_map=SimpleNamespace(native=np.array([[0.5, 0.6]])),
        noiseless_source_eps=np.array([[1.0, 2.0]]), pixel_scale=0.05,
        psf=SimpleNamespace(native=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])),
        metadata={"noise_seed": 17, "truth_kernel": {"kernel_sha256": "abc"}, "shape": np.array([1, 2])},
    )
    monkeypatch.setattr(hwoslaps, "simulate", lambda config: observation)
    output = tmp_path / "observation"
    assert cli.main(["simulate", "-c", str(path), "--output-dir", str(output)]) == 0
    with np.load(output / "observation.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["data_adu"], [[3.0, 4.0]])
        np.testing.assert_array_equal(stored["noise_adu"], [[0.5, 0.6]])
        np.testing.assert_array_equal(stored["noiseless_source_eps"], [[1.0, 2.0]])
        assert stored["pixel_scale_arcsec"] == 0.05
        np.testing.assert_array_equal(stored["imaging_psf_kernel"], observation.psf.native)
        assert json.loads(str(stored["metadata_json"])) == {
            "noise_seed": 17, "truth_kernel": {"kernel_sha256": "abc"}, "shape": [1, 2],
        }


@pytest.mark.parametrize("operation", ["simulate", "forecast"])
def test_invalid_configuration_creates_no_output(tmp_path, operation):
    path = _write_config(tmp_path / "invalid.yaml", {"global_seed": 1})
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        cli.main([operation, "-c", str(path), "--output-dir", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_existing_output_directory_preserves_all_previous_bytes(tmp_path):
    path = _write_config(tmp_path / "config.yaml")
    output = tmp_path / "previous"
    output.mkdir()
    original = {"forecast.npz": b"previous arrays", "config_used.yaml": b"previous config"}
    for name, content in original.items():
        (output / name).write_bytes(content)
    with pytest.raises(SystemExit) as error:
        cli.main(["forecast", "-c", str(path), "--output-dir", str(output)])
    assert error.value.code == 2
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original
