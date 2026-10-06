"""Chromatic observations reach the actual one-kernel nonlinear likelihood."""

from copy import deepcopy

import numpy as np
import pytest

pytestmark=pytest.mark.backend


@pytest.mark.parametrize("model",["kernel","monochromatic"])
def test_chromatic_truth_is_fit_with_one_actual_model_kernel(minimal_mapping,model):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.inference.api import prepare_case
    from hwoslaps.inference.settings import FitSpec
    from hwoslaps.simulation import simulate
    mapping=deepcopy(minimal_mapping)
    fixed_kernel=deepcopy(mapping["psf"]["truth"])
    mapping["scene"]["grid"]["shape"]=[40,40]
    mapping["scene"]["source"]["light"]["light"]["sed"]={"kind":"power_law","index":-3.}
    second=deepcopy(mapping["scene"]["source"]["light"]["light"])
    second.update(centre=[.06,-.02],intensity=.2,effective_radius=.08,sed={"kind":"power_law","index":3.})
    mapping["scene"]["source"]["light"]["second"]=second
    mapping["psf"]={"truth":{"kind":"optical","pupil":{"kind":"circular","diameter_m":2.,
        "pixels":64,"supersampling":2},"focal_length_m":20.,"wavelength_samples":5,
        "detector_oversampling":3,"kernel_shape":[9,9]},
        "model":{**fixed_kernel,"kind":"kernel"} if model=="kernel" else {"kind":"monochromatic","wavelength_nm":500.}}
    mapping["instrument"]["bandpass"]={"kind":"top_hat","min_nm":450.,"max_nm":550.,"throughput":.21}
    mapping["forecast"]["nuisances"]={"fixed":["*"],"background_offset":False}
    with prepare_forecast(mapping) as prepared:
        assert len(prepared.psfs.truth_kernels.kernels)==2
        assert len(prepared.psfs.model_kernels.kernels)==1
        trial=prepared.hypothesis(1.e8,(.4,-.6))
        observation=simulate(prepared,subhalo=trial,noise_seed=None)
        case=prepare_case(prepared,trial,observation,fit=FitSpec(mode="fixed_template"),use_jax=False)
        assert case.data.record["truth_kernels"]
        assert case.data.record["model_kernel"]==prepared.psfs.model_kernels.single.kernel_identity().to_mapping()
        assert np.isfinite(case.log_likelihood("smooth",case.truth_vector("smooth")))
        assert np.isfinite(case.log_likelihood("subhalo",case.truth_vector("subhalo")))
        prepared.validate_identity()
