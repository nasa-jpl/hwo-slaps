"""Chromatic observations reach the actual one-kernel nonlinear likelihood."""

from copy import deepcopy

import numpy as np
import pytest

pytestmark=pytest.mark.backend


@pytest.mark.parametrize("model",["kernel","monochromatic"])
@pytest.mark.parametrize("use_jax",[False,True])
def test_chromatic_truth_is_fit_with_one_actual_model_kernel(minimal_mapping,model,image_asset,use_jax):
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
    mapping["scene"]["source"]["light"]["light"]={"type":"Image","asset_path":str(image_asset),
        "centre":[-.03,.08],"total_flux":.7,"flux_scale":.8,"size_scale":.9,"rotation_deg":12.,
        "sed":{"kind":"power_law","index":-3.}}
    mapping["scene"]["source"]["light"]["third"]={"type":"Image","asset_path":str(image_asset),
        "centre":[-.07,.03],"total_flux":.4,"flux_scale":.65,"size_scale":1.2,"rotation_deg":-17.,
        "sed":{"kind":"power_law","index":-3.}}
    mapping["psf"]={"truth":{"kind":"optical","pupil":{"kind":"circular","diameter_m":2.,
        "pixels":64,"supersampling":2},"focal_length_m":20.,"wavelength_samples":5,
        "detector_oversampling":3,"kernel_shape":[9,9]},
        "model":{**fixed_kernel,"kind":"kernel"} if model=="kernel" else {"kind":"monochromatic","wavelength_nm":500.}}
    mapping["instrument"]["bandpass"]={"kind":"top_hat","min_nm":450.,"max_nm":550.,"throughput":.21}
    mapping["forecast"]["nuisances"]={"fixed":["lens.mass.*","source.light.*.centre_*",
        "source.light.*.ell_comp_*","source.light.*.effective_radius","source.light.*.size_scale",
        "source.light.*.rotation_deg"],"background_offset":False}
    with prepare_forecast(mapping) as prepared:
        assert len(prepared.psfs.truth_kernels.kernels)==2
        assert len(prepared.psfs.model_kernels.kernels)==1
        trial=prepared.hypothesis(1.e8,(.4,-.6))
        observation=simulate(prepared,subhalo=trial,noise_seed=None)
        case=prepare_case(prepared,trial,observation,fit=FitSpec(mode="fixed_template"),use_jax=use_jax)
        assert tuple(prepared.scene.light_groups)==("source:light","source:second")
        assert prepared.scene.light_groups["source:light"].components==("light","third")
        model_names=case.model("smooth").parameter_names
        assert model_names==("galaxies.source.light.flux_scale","galaxies.source.second.intensity",
                            "galaxies.source.third.flux_scale")
        np.testing.assert_array_equal(case.truth_vector("smooth"),[.8,.2,.65])
        instance=case.autofit_models["smooth"].instance_from_vector(vector=case.truth_vector("smooth").tolist())
        for name in ("light","third"):
            profile=getattr(instance.galaxies.source,name)
            actual=next(component for component in prepared.scene.spec.source.light if component.name==name)
            assert profile.rotation_deg==actual.values["rotation_deg"]
            assert profile.size_scale==actual.values["size_scale"]
            assert profile.total_flux==actual.values["total_flux"]
            np.testing.assert_array_equal(profile.sb,prepared.renderer.assets[str(image_asset)].sb)
        import autolens as al
        tracer=al.Tracer(galaxies=list(instance.galaxies),cosmology=prepared.scene.cosmology.autogalaxy())
        np.testing.assert_allclose(tracer.image_2d_from(grid=prepared.scene.grid).native,
            sum(prepared.scene.light_images.values()),rtol=1.e-12,atol=1.e-13*max(image.max() for image in prepared.scene.light_images.values()))
        assert case.data.record["truth_kernels"]
        from hwoslaps.identity import json_ready
        assert json_ready(case.data.record["model_kernel"])==prepared.psfs.model_kernels.single.kernel_identity().to_mapping()
        assert np.isfinite(case.log_likelihood("smooth",case.truth_vector("smooth")))
        assert np.isfinite(case.log_likelihood("subhalo",case.truth_vector("subhalo")))
        prepared.validate_identity()
