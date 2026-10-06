"""Actual current engine products used by the optional Axes consumers."""

import numpy as np
import pytest


@pytest.fixture
def plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot
    yield pyplot
    pyplot.close("all")


@pytest.fixture
def forecast_product(minimal_mapping):
    from hwoslaps.fisher.api import forecast, prepare_forecast

    minimal_mapping["forecast"]["positions"] = {"kind": "grid", "spacing_arcsec": .4, "half_width_arcsec": .4}
    with prepare_forecast(minimal_mapping) as prepared:
        return forecast(prepared, masses_msun=[1e8, 2e8])


@pytest.fixture
def observation_product(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast

    minimal_mapping["scene"]["lens"]["light"] = {"light": {"type": "Exponential", "centre": [0., 0.],
        "ell_comps": [.1, .02], "intensity": .5, "effective_radius": .4}}
    minimal_mapping["instrument"]["detector"].update(gain_e_per_adu=2.5, read_noise_e=2., dark_current_e_per_s=.003)
    minimal_mapping["observation"].update(exposure_time_s=17., exposure_count=3)
    with prepare_forecast(minimal_mapping) as prepared:
        return prepared.observation


@pytest.fixture
def area_product():
    from hwoslaps.analysis.knowledge_error import knowledge_error_areas
    from hwoslaps.fisher.positions import grid_positions
    from hwoslaps.fisher.result import ForecastResult
    from hwoslaps.optics.kernels import DetectorPSF, KernelBinding

    positions = grid_positions((0, 0), spacing_arcsec=.5, half_width_arcsec=.5, annulus=None)
    values = np.array([[4., 0, 0, 0, 0, 0, 0, 0, 0], [4., 4, 4, 4, 0, 0, 0, 0, 0]])
    provenance = {"comparison_digest": "same-science", "mask": {"digest": "same-mask"},
                  "nuisance_names": [], "truth_kernels": KernelBinding.uniform(
                      DetectorPSF.from_array(np.ones((3, 3)), .05), ("source.light.light",)).to_mapping()}
    reference = ForecastResult(np.array([1e8, 2e8]), positions, values + 1, values, None, None, "matched", {},
                               provenance | {"psf_relation": "matched"})
    amplitude = np.zeros((2, 9))
    amplitude[1, [0, 2, 4, 5]] = 1
    spurious = np.zeros((2, 9))
    spurious[1, [1, 5, 8]] = 1
    mismatch = ForecastResult(reference.masses_msun, positions, np.full((2, 9), 5.), np.full((2, 9), 4.),
                              amplitude, spurious, "kernel", {}, provenance | {"psf_relation": "kernel"})
    return knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=2,
                                 selection=np.arange(9) < 6)
