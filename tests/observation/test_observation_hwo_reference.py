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
