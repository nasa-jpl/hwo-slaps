"""Forecast/profile likelihood agreement on the same pixels and kernel in the weak-halo regime."""

import json

import numpy as np
import pytest

from hwoslaps.inference.api import validate_nonlinear
from hwoslaps.inference.result import ForecastReference
from hwoslaps.inference.settings import FitSpec, RefineSettings, SamplerSettings

pytestmark = [pytest.mark.backend, pytest.mark.xtx_gpu]
MASS = 10.0 ** 8.5
POSITION = (0.4, -0.6)
MODEL_SIGMA_PIXELS = 1.01


@pytest.mark.parametrize("relation", ["matched", "mismatched"])
def test_nonlinear_q_matches_the_forecast_for_a_weak_subhalo(relation, prepared_forecast_factory, tmp_path):
    from hwoslaps.fisher.api import forecast
    from hwoslaps.simulation import simulate

    overrides = {"forecast": {"mask": {"kind": "psf_border"},
                               "nuisances": {"background_offset": False, "priors": {}, "wavefront": None}}}
    if relation == "mismatched":
        y, x = np.mgrid[-3:4, -3:4].astype(float)
        model = np.exp(-(x ** 2 + y ** 2) / (2.0 * MODEL_SIGMA_PIXELS ** 2))
        path = tmp_path / "model_kernel.npy"
        np.save(path, model / model.sum())
        overrides["psf"] = {"model": {"kind": "kernel", "path": str(path), "pixel_scale_arcsec": 0.05}}
    prepared = prepared_forecast_factory(overrides)
    masses = [MASS, MASS / np.sqrt(10.0)] if relation == "matched" else [MASS]
    prediction = forecast(prepared, masses_msun=masses, positions=[POSITION])
    assert 4.0 <= prediction.fisher_profiled[0, 0] <= 9.0
    sampler = SamplerSettings(use_jax=True, n_live_smooth=50, n_live_subhalo_fixed=50, n_eff=200,
                              jax_n_batch=50, retain_search_internal=True)
    measurements = []
    for index, mass in enumerate(masses):
        trial = prepared.hypothesis(mass, POSITION)
        reference = ForecastReference.from_result(prediction, mass_index=index, position_index=0)
        result = validate_nonlinear(prepared, trial, simulate(prepared, subhalo=trial, noise_seed=None),
                                    fit=FitSpec(mode="fixed_template", mask="forecast_mask_minus_psf_border",
                                                h1="truth_anchor" if relation == "matched" else "search"),
                                    sampler=sampler, sampler_seed=20261005, refine=RefineSettings(),
                                    output_dir=tmp_path / "fits", case_id=f"{relation}_{index}", forecast_reference=reference)
        assert result.data["mask"]["digest"] == reference.mask_digest
        assert tuple(reference.nuisance_names) == result.fitted_parameters
        assert reference.model_kernel == prepared.psfs.model_kernels.single.kernel_identity()
        information = float(prediction.fisher_profiled[index, 0])
        amplitude = 1.0 if relation == "matched" else float(prediction.amplitude_hat[index, 0])
        linear_fixed = (2.0 * amplitude - 1.0) * information
        gap = abs(result.q_signed / linear_fixed - 1.0)
        if relation == "mismatched":
            assert 0.05 <= abs(amplitude - 1.0) <= 0.2
            assert reference.q == float(prediction.q_mismatch[index, 0])
            assert reference.amplitude == amplitude
        else:
            assert reference.q == information and reference.amplitude is None
        assert gap < 0.05
        measurements.append({"mass_msun": mass, "q_nonlinear": result.q_signed, "F": information,
                             "amplitude": amplitude, "q_fixed_linear": linear_fixed, "relative_gap": gap,
                             "q_forecast": reference.q, "conservatism_gap": (amplitude - 1.0) ** 2 * information,
                             "amplitude_spurious": None if relation == "matched"
                             else float(prediction.amplitude_spurious[index, 0])})
    if relation == "matched":
        assert measurements[1]["relative_gap"] < measurements[0]["relative_gap"]
    print("LINEAR_REGIME " + json.dumps({"relation": relation, "model_sigma_pixels": MODEL_SIGMA_PIXELS,
                                         "measurements": measurements}))
