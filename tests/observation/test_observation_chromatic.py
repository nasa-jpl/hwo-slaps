"""Independent single-colour scenes and detector noise for a grouped chromatic mean."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

pytestmark=pytest.mark.backend


def test_two_colour_expected_image_is_the_sum_of_single_colour_images(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.optics.chromatic import effective_kernel,sed_weights
    from hwoslaps.optics.kernels import KernelBinding
    from hwoslaps.spectra.bandpass import build_bandpass,parse_bandpass
    from hwoslaps.simulation import simulate
    mapping=deepcopy(minimal_mapping)
    mapping["psf"]={"truth":{"kind":"optical","pupil":{"kind":"circular","diameter_m":2.,
        "pixels":64,"supersampling":2},"focal_length_m":20.,"wavelength_samples":5,
        "detector_oversampling":3,"kernel_shape":[9,9]}}
    mapping["instrument"]["bandpass"]={"kind":"top_hat","min_nm":450.,"max_nm":550.,"throughput":.21}
    mapping["instrument"]["detector"]["gain_e_per_adu"]=2.5
    mapping["scene"]["source"]["light"]["light"]["sed"]={"kind":"power_law","index":-3.}
    mapping["scene"]["source"]["light"]["blue"]={"type":"Sersic","centre":[.05,-.04],
        "ell_comps":[.08,.03],"intensity":.5,"effective_radius":.07,"sersic_index":.75,
        "sed":{"kind":"power_law","index":2.}}
    mapping["scene"]["lens"]["light"]={"lens":{"type":"Exponential","centre":[.01,.02],
        "ell_comps":[.05,.04],"intensity":.1,"effective_radius":.2,"sed":{"kind":"flat_flambda"}}}
    band=build_bandpass(parse_bandpass(mapping["instrument"]["bandpass"],"band"))
    with prepare_forecast(mapping) as prepared:
        assert tuple(prepared.scene.light_groups)==("lens:lens","source:light","source:blue")
        kernels=prepared.psfs.truth.kernels();terms=[]
        for key,group in prepared.scene.light_groups.items():
            spec=prepared.scene.spec
            selected=tuple(component for component in getattr(spec,group.plane).light if component.name in group.components)
            solo=replace(spec,lens=replace(spec.lens,light=selected if group.plane=="lens" else ()),
                         source=replace(spec.source,light=selected if group.plane=="source" else ()))
            scene=prepared.renderer.scene(spec=solo)
            sed=prepared.renderer.loaded_seds[f"{group.plane}.{group.components[0]}"]
            kernel=effective_kernel(kernels,sed_weights(band,sed,prepared.psfs.truth.wavelengths_m),.05,source={})
            terms.append(prepared.renderer.light_rate(scene,KernelBinding.uniform(kernel,(key,))))
        expected=sum(terms)
        np.testing.assert_allclose(prepared.observation.light_rate_e_per_s,expected,rtol=1.e-12,atol=0.)
        exposure=prepared.observation.exposure
        noise=np.sqrt(np.maximum(expected*exposure.exposure_time_s,0.)+
            exposure.sky_rate_e_per_s*exposure.exposure_time_s+
            exposure.detector.dark_current_e_per_s*exposure.exposure_time_s+
            exposure.exposure_count*exposure.detector.read_noise_e**2)/exposure.detector.gain_e_per_adu
        np.testing.assert_allclose(prepared.sigma_adu,noise,rtol=1.e-12,atol=0.)
        direct=simulate(mapping,subhalo=None,noise_seed=None)
        np.testing.assert_array_equal(direct.expected_adu,prepared.mean_truth_adu)
        for group in prepared.scene.light_groups:
            assert direct.psfs.for_group(group).kernel_identity()==prepared.psfs.truth_kernels.for_group(group).kernel_identity()


def test_compact_actual_component_conserves_flux_under_a_unit_chromatic_kernel(minimal_mapping):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.optics.kernels import DetectorPSF,convolve_real_space
    mapping=deepcopy(minimal_mapping)
    mapping["scene"]["grid"].update(shape=[101,101],pixel_scale_arcsec=.03,over_sample_size=2)
    mapping["scene"]["lens"]["mass"]["mass"]["einstein_radius"]=.05
    mapping["scene"]["source"]["light"]["light"].update(effective_radius=.025,centre=[.1,.1],sed={"kind":"flat_fnu"})
    mapping["psf"]={"truth":{"kind":"optical","pupil":{"kind":"circular","diameter_m":2.,
        "pixels":64,"supersampling":2},"focal_length_m":20.,"wavelength_samples":5,
        "detector_oversampling":3,"kernel_shape":[9,9]}}
    mapping["instrument"]["bandpass"]={"kind":"top_hat","min_nm":450.,"max_nm":550.,"throughput":.21}
    mapping["forecast"]["nuisances"]={"fixed":["*"],"background_offset":False}
    with prepare_forecast(mapping) as prepared:
        image=prepared.scene.light_images["source:light"]
        # Explicit unit-sum condition; the finite optical support's captured power remains
        # recorded and unrenormalized in production.
        kernel=DetectorPSF.from_array(prepared.psfs.truth_kernels.single.kernel,.03,normalize=True)
        convolved=convolve_real_space(image,kernel.kernel,.03)
        assert convolved.sum()==pytest.approx(image.sum(),rel=1.e-12,abs=0.)
