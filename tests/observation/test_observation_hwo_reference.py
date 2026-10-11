"""The shipped HWO reference resolves the pinned observing inputs through real simulation."""

from pathlib import Path
import runpy

import pytest

from hwoslaps.config.schema import load_config
from hwoslaps.simulation import simulate


@pytest.mark.backend
def test_packaged_hwo_reference_rates_units_and_detector_convention():
    root = Path(__file__).resolve().parents[2]
    directory = root / "examples" / "hwo_reference"
    client = runpy.run_path(root / "examples" / "example_support.py")
    client["verify_sei"](directory / "sei_v0.1.9")
    paths = [directory / name for name in (
        "scene_smooth_ring.yaml", "instrument.yaml", "forecast.yaml",
    )]
    config = load_config(paths)
    observation = simulate(config, subhalo=None, noise_seed=None)
    assert observation.kind == "expected"
    photometry = observation.photometry
    assert photometry.collecting_area_m2 == pytest.approx(33.606448937520405, rel=1e-12, abs=0)
    # Intrinsic source total, before lensing; sky is a rate per detector pixel.
    component = photometry.components["source.disk"]
    assert component["rate_e_per_s"] == pytest.approx(8.951505744562876, rel=1e-9, abs=0)
    assert observation.exposure.sky_rate_e_per_s == pytest.approx(0.002510279845963486, rel=1e-9, abs=0)
    assert component["amplitude"] == pytest.approx(0.003174147284617635, rel=2e-7, abs=0)
    # AB photometry resolves the sky independently of the rounded paper literal.
    # Two reads combine into one exposure: sky + dark counts and 2 * (0.2 e-)^2.
    blank_from_resolved_sky = (0.002 * 2000.0 + observation.exposure.sky_rate_e_per_s * 2000.0) + 2 * 0.2**2
    assert observation.exposure.blank_variance_e2 == pytest.approx(blank_from_resolved_sky, rel=1e-15, abs=0)
    # The actual paper overlay retains the literal rate and strict historical target.
    paper_observation = simulate(load_config([*paths, directory / "paper_values.yaml"]),
                                 subhalo=None, noise_seed=None)
    assert paper_observation.exposure.sky_rate_e_per_s == 0.002510279845963486
    assert paper_observation.photometry.components["source.disk"]["amplitude"] == 0.003174147284617635
    assert paper_observation.exposure.blank_variance_e2 == pytest.approx(9.100559691926973, rel=1e-15, abs=0)


def test_example_output_collision_precedes_expensive_preparation(tmp_path, monkeypatch):
    from hwoslaps.fisher import api

    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(root / "examples/example_support.py")
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("prior products")

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("preparation ran despite an existing output directory")

    monkeypatch.setattr(api, "prepare_forecast", unexpected_preparation)
    with pytest.raises(FileExistsError):
        client["run_product"](None, output, masses=[1e8], engine="reference", q_threshold=5,
                              budget_s=0, command=["example"])
    assert (output / "keep.txt").read_text() == "prior products"


@pytest.mark.backend
def test_example_records_slow_execution_without_losing_products(tmp_path):
    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(root / "examples/example_support.py")
    output = tmp_path / "slow_example"
    result, record = client["run_product"](
        load_config(root / "configs/minimal.yaml"), output, masses=[1e8],
        engine="reference", q_threshold=5, budget_s=0, command=["example"], noise_seed=11,
    )
    assert record["elapsed_s"] > 0
    assert record["within_budget"] is False
    assert result.masses_msun.tolist() == [1e8]
    assert {path.name for path in output.iterdir()} >= {"forecast.npz", "expected.npz", "noisy.npz", "run.json"}


@pytest.mark.backend
@pytest.mark.xtx_multi_gpu
def test_example_accepts_multiple_visible_gpus_and_records_actual_default_device():
    import jax

    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(root / "examples/example_support.py")
    devices = jax.devices("gpu")
    with jax.default_device(devices[1]):
        lane = client["require_lane"]("jax")
    assert lane["lane"] == "gpu"
    assert lane["device"] == str(devices[1])
    assert str(devices[0]) in lane["devices"]


@pytest.mark.backend
def test_example_rejects_cpu_fallback_for_jax():
    import jax

    root = Path(__file__).resolve().parents[2]
    client = runpy.run_path(root / "examples/example_support.py")
    with jax.default_device(jax.devices("cpu")[0]), pytest.raises(RuntimeError, match="requires a GPU backend"):
        client["require_lane"]("jax")
