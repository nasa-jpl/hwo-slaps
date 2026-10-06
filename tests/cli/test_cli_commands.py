"""The installed command's real configuration, output and replay boundaries."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def run_cli(*arguments):
    return subprocess.run([sys.executable, "-m", "hwoslaps", *map(str, arguments)], capture_output=True, text=True,
                          timeout=120)


def test_validate_reports_dotted_key_and_reference_is_backend_free(minimal_mapping, write_config):
    minimal_mapping["scene"]["grid"]["over_sample_size"] = 0
    path = write_config(minimal_mapping)
    failed = run_cli("validate", path)
    assert failed.returncode == 2
    assert "scene.grid.over_sample_size" in failed.stderr
    # Invalid configuration must refuse before either operation publishes its directory.
    for operation, flags in (("simulate", ("--expected", "--smooth")), ("forecast", ("--masses", "1e8"))):
        output = path.parent / operation
        invalid = run_cli(operation, path, *flags, "-o", output)
        assert invalid.returncode == 2 and "scene.grid.over_sample_size" in invalid.stderr
        assert not output.exists()
    # The actual command must compose all owning schemas without a scientific backend.
    program = r'''
import importlib.abc, sys
forbidden = {'autolens', 'autogalaxy', 'autoarray', 'autofit', 'autoconf', 'hcipy',
             'jax', 'jaxlib', 'nautilus', 'numba', 'matplotlib'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise ModuleNotFoundError('blocked scientific backend: ' + fullname, name=fullname)
sys.meta_path.insert(0, Block())
from hwoslaps.cli import main
result = main(['reference', *sys.argv[1:]])
assert not any(name.split('.')[0] in forbidden for name in sys.modules)
raise SystemExit(result)
'''
    def reference(*sections):
        return subprocess.run([sys.executable, "-c", program, *sections], capture_output=True,
                              text=True, timeout=30)

    documents = reference()
    assert documents.returncode == 0, documents.stderr
    for heading in ("top level", "scene.grid", "population", "batch", "fit", "sampler", "refine", "classification"):
        assert f"## {heading}\n" in documents.stdout
    sections = {
        "scene.grid": ("pixel_scale_arcsec", "arcsec"),
        "population": ("variables", "max_attempts"),
        "batch.execution": ("devices", "workers_per_device"),
        "fit": ("mode", "mass_support"),
        "sampler": ("n_live_smooth", "jax_n_batch"),
        "refine": ("original_start_count", "repeat_gtol"),
        "classification.acceptance": ("smooth", "subhalo"),
    }
    for section, keys in sections.items():
        selected = reference(section)
        assert selected.returncode == 0, selected.stderr
        assert selected.stdout.startswith(f"## {section}\n")
        assert all(key in selected.stdout for key in keys)
        assert "## top level\n" not in selected.stdout
    for key, default in (("n_live_smooth", "`100`"), ("repeat_gtol", "`1e-12`"),
                         ("stationarity_tolerance", "required")):
        row = next(line for line in documents.stdout.splitlines() if line.startswith(f"| `{key}` |"))
        assert row.split("|")[3].strip() == default
    unknown = reference("absent_section")
    assert unknown.returncode == 2 and "unknown section" in unknown.stderr


def test_output_directory_must_not_exist(minimal_mapping, write_config, tmp_path):
    path = write_config(minimal_mapping)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "previous").write_bytes(b"scientific result")
    completed = run_cli("forecast", path, "--masses", "1e8", "-o", output)
    assert completed.returncode == 2
    assert (output / "previous").read_bytes() == b"scientific result"
    assert sorted(item.name for item in output.iterdir()) == ["previous"]


@pytest.mark.parametrize("arguments", [[], ["--masses", "nan"], ["--masses", "-1"]])
def test_forecast_usage_errors_write_nothing(minimal_mapping, write_config, tmp_path, arguments):
    path = write_config(minimal_mapping)
    output = tmp_path / "result"
    completed = run_cli("forecast", path, *arguments, "-o", output)
    assert completed.returncode == 2 and not output.exists()


@pytest.mark.backend
def test_simulate_and_forecast_write_loadable_artifacts_and_replay_exactly(minimal_mapping, write_config, tmp_path):
    from hwoslaps.artifacts import load_forecast, load_observation
    from hwoslaps.config.schema import load_config
    minimal_mapping["scene"]["injection"] = {"mass_msun": 1e8, "position": {"kind": "direct", "centre": [0.0, 0.8]}}
    path = write_config(minimal_mapping)
    for name, flags in (("expected", ["--expected"]), ("noisy", ["--noise-seed", "11"]),
                        ("control", ["--expected", "--smooth"])):
        output = tmp_path / name
        completed = run_cli("simulate", path, *flags, "-o", output)
        assert completed.returncode == 0, completed.stderr
        observed = load_observation(output / "observation.npz")
        assert observed.noise_seed == (11 if name == "noisy" else None)
        assert (observed.subhalo is None) == (name == "control")
        record = json.loads((output / "provenance.json").read_text())
        assert record["operation"] == "simulate"
        assert record["config_digest"] == observed.config_digest
    output = tmp_path / "forecast"
    completed = run_cli("forecast", path, "--masses", "1e7", "1e8", "1e9", "-o", output)
    assert completed.returncode == 0, completed.stderr
    first = load_forecast(output / "forecast.npz")
    assert json.loads((output / "provenance.json").read_text())["config_digest"] == first.provenance["config_digest"]
    effective = output / "effective_config.yaml"
    assert load_config(effective).digest() == load_config(path).digest()
    replay = tmp_path / "replay"
    completed = run_cli("forecast", effective, "--masses", "1e7", "1e8", "1e9", "-o", replay)
    assert completed.returncode == 0, completed.stderr
    second = load_forecast(replay / "forecast.npz")
    assert first.positions_yx.tobytes() == second.positions_yx.tobytes()
    assert first.fisher_profiled.tobytes() == second.fisher_profiled.tobytes()


@pytest.mark.backend
@pytest.mark.parametrize("operation", ["simulate", "forecast"])
def test_companion_identity_comes_from_the_actual_product_after_valid_publication(
        minimal_mapping, write_config, tmp_path, monkeypatch, operation):
    from hwoslaps.artifacts import load_forecast, load_observation
    from hwoslaps.cli import main
    from hwoslaps.config.schema import load_config
    from hwoslaps.provenance import capture_provenance
    import hwoslaps.provenance as provenance

    path = write_config(minimal_mapping)
    before = load_config(path).digest()
    kernel_path = Path(minimal_mapping["psf"]["truth"]["path"])
    original = np.load(kernel_path)
    changed = original.copy()
    changed[3, 3] *= 1.1
    changed /= changed.sum()
    replacement = tmp_path / "replacement.npy"
    np.save(replacement, changed)
    published = False
    def capture_with_publication(*args, **kwargs):
        nonlocal published
        assert not published
        replacement.replace(kernel_path)
        published = True
        return capture_provenance(*args, **kwargs)
    monkeypatch.setattr(provenance, "capture_provenance", capture_with_publication)
    output = tmp_path / operation
    arguments = (["simulate", path, "--expected", "--smooth", "-o", output] if operation == "simulate" else
                 ["forecast", path, "--masses", "1e8", "-o", output])
    assert main(list(map(str, arguments))) == 0
    assert published
    record = json.loads((output / "provenance.json").read_text())
    identity = (load_observation(output / "observation.npz").config_digest if operation == "simulate" else
                load_forecast(output / "forecast.npz").provenance["config_digest"])
    assert identity != before
    assert record["config_digest"] == identity
