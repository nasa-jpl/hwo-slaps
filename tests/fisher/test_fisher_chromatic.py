"""Actual grouped chromatic forecasts, model means and wavefront derivatives."""

from copy import deepcopy
import math

import numpy as np
import pytest

pytestmark=pytest.mark.backend


def chromatic_mapping(mapping,*,count=5,relation="matched"):
    mapping=deepcopy(mapping)
    mapping["scene"]["grid"].update(shape=[41,41],pixel_scale_arcsec=.05,over_sample_size=2)
    mapping["psf"]={"truth":{"kind":"optical","pupil":{"kind":"circular","diameter_m":2.,
        "pixels":64,"supersampling":2},"focal_length_m":20.,"wavelength_samples":count,
        "detector_oversampling":3,"kernel_shape":[9,9],"wavefront":{"zernikes":{4:5.,5:3.}}},
        "model":{"kind":relation}}
    if relation=="knowledge_error":
        mapping["psf"]["model"]["draw"]={"prior":{"packaged":"jwst_wss_drift_v1"},
            "amplitude_rms_nm":10.,"seed":20261005,"family":"global"}
    mapping["instrument"]["bandpass"]={"kind":"top_hat","min_nm":450.,"max_nm":550.,"throughput":.21}
    source=mapping["scene"]["source"]["light"]
    source["light"]["sed"]={"kind":"power_law","index":-3.}
    source["blue"]={"type":"Sersic","centre":[.05,-.04],"ell_comps":[.08,.03],
        "intensity":.5,"effective_radius":.07,"sersic_index":.75,"sed":{"kind":"power_law","index":2.}}
    mapping["scene"]["lens"]["light"]={"lens":{"type":"Exponential","centre":[.01,.02],
        "ell_comps":[.05,.04],"intensity":.1,"effective_radius":.2,"sed":{"kind":"flat_flambda"}}}
    mapping["forecast"]["positions"]={"kind":"explicit","positions_yx":[[y,x] for y in(-.4,0.,.4) for x in(-.4,0.,.4)]}
    # Scene columns remain selected; constrain the two source amplitudes to avoid near-zero
    # profiled information in the tiny scene masking the actual template comparison.
    mapping["forecast"]["nuisances"]={"priors":{"source.light.light.intensity":.1,"source.light.blue.intensity":.1}}
    return mapping


@pytest.mark.parametrize("relation",["matched","knowledge_error"])
@pytest.mark.parametrize("sed_state",["response","no_sed","zero_response"])
def test_single_node_chromatic_forecast_equals_monochromatic_forecast(minimal_mapping,relation,sed_state,tmp_path):
    from hwoslaps.fisher.api import forecast,prepare_forecast
    from hwoslaps.spectra.bandpass import build_bandpass,parse_bandpass
    sampled=chromatic_mapping(minimal_mapping,count=1,relation=relation)
    outside=tmp_path/"zero-response-sed.npz";np.savez(outside,wave=[400.,465.,470.,475.,600.],value=[0.,0.,1.,0.,0.])
    for galaxy in ("lens","source"):
        for component in sampled["scene"][galaxy]["light"].values():
            if sed_state=="no_sed":component.pop("sed")
            elif sed_state=="response":component["sed"]={"kind":"flat_fnu"}
            else:component["sed"]={"kind":"table","path":str(outside),"wavelength_key":"wave",
                "value_key":"value","wavelength_unit":"nm","quantity":"fnu"}
    if sed_state=="zero_response":
        response=tmp_path/"response.npz";np.savez(response,wave=[450.,480.,481.,550.],value=[0.,0.,.21,.21])
        sampled["instrument"]["bandpass"]={"kind":"table","path":str(response),"wavelength_key":"wave",
            "value_key":"value","wavelength_unit":"nm","support_nm":[450.,550.]}
    sampled["forecast"]["nuisances"]["wavefront"]={"modes":{"zernikes":{"nolls":[4,5]}},"step_nm":1.,"prior_sigma_nm":5.}
    mono=deepcopy(sampled);truth=mono["psf"]["truth"]
    truth.pop("wavelength_samples")
    truth["wavelength_nm"]=build_bandpass(parse_bandpass(sampled["instrument"]["bandpass"],"band")).nodes(1)[0]*1.e9
    if sed_state=="no_sed":
        from hwoslaps.config.checks import ConfigError
        from hwoslaps.config.schema import parse_config
        # X7 still requires an SED for sampled optics, even at one node. The supported
        # absent-shape metadata boundary is explicit monochromatic optics with a band.
        with pytest.raises(ConfigError,match="requires an SED"):parse_config(sampled)
        sampled=deepcopy(mono)
        mono["instrument"].pop("bandpass")
    with prepare_forecast(mono) as a,prepare_forecast(sampled) as b:
        np.testing.assert_array_equal(a.mean_truth_adu,b.mean_truth_adu)
        np.testing.assert_array_equal(a.mean_model_adu,b.mean_model_adu)
        assert a.nuisances.names==b.nuisances.names
        np.testing.assert_array_equal(a.nuisances.images,b.nuisances.images)
        for prepared in (a,b):
            if prepared.psfs.spectral is None:
                assert sed_state=="no_sed" and prepared is a
                prepared.validate_identity()
                continue
            for record in prepared.psfs.spectral["truth"]["groups"].values():
                assert record["status"]==sed_state
                assert (record["sed_digest"] is None)==(sed_state=="no_sed")
                if sed_state=="response":
                    np.testing.assert_array_equal(record["weights"]["normalized"],[1.])
                    assert record["weights"]["rates"][0]*math.exp(record["weights"]["log_rate_scale"])==pytest.approx(.21*math.log(550./450.),rel=1.e-12,abs=0.)
                    assert record["effective_wavelength_m"]*1.e9==pytest.approx(100./math.log(550./450.),rel=1.e-9,abs=0.)
                else:
                    assert record["weights"] is None and record["effective_wavelength_m"] is None
                assert record["kernel_wavelength_m"]==prepared.psfs.truth.wavelengths_m[0]
            kernel=prepared.psfs.truth_kernels.single
            assert prepared.psfs.spectral["truth"]["captured_power_fractions"]==(kernel.source["captured_power_fraction"],)
            prepared.validate_identity()
        aa,bb=forecast(a,masses_msun=[1.e8]),forecast(b,masses_msun=[1.e8])
        for name in ("q_asimov","fisher_raw","fisher_profiled","amplitude_hat","amplitude_spurious"):
            left,right=getattr(aa,name),getattr(bb,name)
            if left is None:assert right is None
            else:np.testing.assert_array_equal(left,right)


@pytest.mark.parametrize("relation",["matched","knowledge_error"])
@pytest.mark.parametrize("device",["cpu",pytest.param("gpu",marks=pytest.mark.xtx_gpu)])
def test_chromatic_jax_engine_matches_reference(minimal_mapping,relation,device):
    from hwoslaps.fisher.api import Execution,forecast,prepare_forecast
    mapping=chromatic_mapping(minimal_mapping,relation=relation)
    with prepare_forecast(mapping) as reference,prepare_forecast(mapping,execution=Execution(engine="jax")) as jax:
        if device=="gpu":assert jax.engine.describe()["device"].startswith("cuda")
        actual=forecast(jax,masses_msun=[1.e8]);expected=forecast(reference,masses_msun=[1.e8])
        assert len(jax.psfs.truth_kernels.kernels)==3
        assert len(jax.engine._model_slots)==2 # source slots; lens is an independently bound constant.
        for name in ("q_asimov","fisher_raw","fisher_profiled","amplitude_hat","amplitude_spurious"):
            left,right=getattr(actual,name),getattr(expected,name)
            if left is None:assert right is None
            else:np.testing.assert_allclose(left,right,rtol=1.e-6,atol=0.)


@pytest.mark.parametrize("relation",["matched","knowledge_error","monochromatic"])
def test_chromatic_wavefront_nuisance_is_the_derivative_of_the_forward_model(minimal_mapping,relation):
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.optics.chromatic import effective_kernel,sed_weights
    from hwoslaps.optics.kernels import KernelBinding
    from hwoslaps.fisher.psf_pair import PsfPair
    from hwoslaps.optics.wavefront import WavefrontMode
    mapping=chromatic_mapping(minimal_mapping,relation=relation)
    mapping["forecast"]["nuisances"].update(fixed=["*"],background_offset=False,
        wavefront={"modes":{"zernikes":{"nolls":[4,5]}},"step_nm":1.,"prior_sigma_nm":5.})
    with prepare_forecast(mapping) as prepared:
        provider=prepared.psfs.model.provider
        reconstructed=PsfPair(prepared.psfs.truth,prepared.psfs.model,prepared.psfs.truth_kernels,
                              prepared.psfs.model_kernels,prepared.psfs.spectral)
        from hwoslaps.spectra.bandpass import build_bandpass,parse_bandpass
        band=build_bandpass(parse_bandpass(mapping["instrument"]["bandpass"],"band"))
        for noll in (4,5):
            mode=WavefrontMode("zernikes",noll);value=provider.coefficients.value(mode)
            original_derivatives=prepared.psfs.model_kernel_derivatives(mode,1.)
            recreated_derivatives=reconstructed.model_kernel_derivatives(mode,1.)
            assert tuple(recreated_derivatives)==tuple(original_derivatives)
            for group in original_derivatives:
                np.testing.assert_array_equal(recreated_derivatives[group],original_derivatives[group])
            bindings=[]
            for sign in (1.,-1.):
                coefficients=provider.coefficients.replace(mode,value+sign)
                nodes=provider.kernels(coefficients=coefficients)
                grouped={}
                for key,group in prepared.scene.light_groups.items():
                    if relation=="monochromatic":
                        wavelength=prepared.psfs.spectral["model"]["groups"][key]["kernel_wavelength_m"]
                        grouped[key]=provider.kernel(wavelength,coefficients=coefficients)
                    else:
                        sed=prepared.renderer.loaded_seds[f"{group.plane}.{group.components[0]}"]
                        weights=sed_weights(band,sed,provider.wavelengths_m)
                        grouped[key]=effective_kernel(nodes,weights,.05,source={})
                bindings.append(KernelBinding.from_groups(grouped))
            expected=(prepared.renderer.mean_adu(prepared.scene,bindings[0])-prepared.renderer.mean_adu(prepared.scene,bindings[1]))/2
            actual=prepared.nuisances.images[prepared.nuisances.names.index(f"psf.zernikes[{noll}]")]
            assert np.max(np.abs(actual-expected))<=1.e-10*np.max(np.abs(expected))
        with pytest.raises(TypeError):prepared.psfs.spectral["model"]["groups"]["source:light"]["effective_wavelength_m"]=0.
        exported=prepared.psfs.to_mapping();exported["spectral"]["model"]["groups"].clear()
        assert prepared.psfs.spectral["model"]["groups"]


def test_monochromatic_model_requires_sampled_optical_truth(minimal_mapping):
    from hwoslaps.config.schema import parse_config
    from hwoslaps.config.checks import ConfigError
    bad=deepcopy(minimal_mapping);bad["psf"]["model"]={"kind":"monochromatic"}
    with pytest.raises(ConfigError,match="sampled|optical"):parse_config(bad)
    one=chromatic_mapping(minimal_mapping,count=1,relation="monochromatic")
    with pytest.raises(ConfigError,match="2|two"):parse_config(one)
    two=chromatic_mapping(minimal_mapping,count=2,relation="monochromatic")
    assert parse_config(two).psf.model.wavelength_nm is None


def test_monochromatic_model_mode_binds_each_group_at_its_mean_wavelength(minimal_mapping,tmp_path):
    from hwoslaps.fisher.api import forecast,prepare_forecast
    maxima={"mean":[],"fixed":[]}
    for eps in (.08,.16):
        mapping=chromatic_mapping(minimal_mapping,count=12,relation="monochromatic")
        mapping["scene"]["lens"].pop("light")
        mapping["scene"]["source"]["light"]["blue"]["sed"]["index"]=3.
        width=500.*eps;low,high=500.-width/2,500.+width/2
        path=tmp_path/f"ramp-{eps}.npz";np.savez(path,wave=[low,high],value=[.1,.7])
        mapping["instrument"]["bandpass"]={"kind":"table","path":str(path),"wavelength_key":"wave",
            "value_key":"value","wavelength_unit":"nm","support_nm":[low,high]}
        # Exact primitive for integral(lambda^power), including the logarithmic case.
        def integral(power):
            return math.log(high/low) if power==-1 else (high**(power+1)-low**(power+1))/(power+1)
        a,b=1-1.5*500./width,1.5/width
        for mode in ("mean","fixed"):
            current=deepcopy(mapping)
            if mode=="fixed":current["psf"]["model"]["wavelength_nm"]=500.
            with prepare_forecast(current) as prepared:
                if mode=="mean":
                    for key,group in prepared.scene.light_groups.items():
                        alpha=group.sed.index
                        expected_nm=(a*integral(-alpha)+b*integral(1-alpha))/(a*integral(-alpha-1)+b*integral(-alpha))
                        recorded=prepared.psfs.spectral["model"]["groups"][key]["kernel_wavelength_m"]
                        # Mandated 10001-point trapezoid error <=5.09e-10; root-approved
                        # continuous oracle bound1e-9, with all kernel identities still bitwise.
                        assert recorded*1.e9==pytest.approx(expected_nm,rel=1.e-9,abs=0.)
                        np.testing.assert_array_equal(prepared.psfs.model_kernels.for_group(key).kernel,
                                                      prepared.psfs.truth.kernel(recorded).kernel)
                result=forecast(prepared,masses_msun=[1.e8])
                maxima[mode].append(np.max(np.abs(result.amplitude_spurious)))
    assert 3<maxima["mean"][1]/maxima["mean"][0]<5
    assert 1.5<maxima["fixed"][1]/maxima["fixed"][0]<2.5


def test_chromatic_spawn_keeps_captured_shared_table_colours_and_kernel_groups(minimal_mapping,tmp_path):
    from hwoslaps.fisher.api import Execution,prepare_forecast
    path=tmp_path/"shared-sed.npz";np.savez(path,wave=[400.,500.,600.],value=[.5,1.,.8])
    mapping=chromatic_mapping(minimal_mapping,relation="knowledge_error")
    for component,quantity in (("light","fnu"),("blue","flambda")):
        mapping["scene"]["source"]["light"][component]["sed"]={"kind":"table","path":str(path),
            "wavelength_key":"wave","value_key":"value","wavelength_unit":"nm","quantity":quantity}
    with prepare_forecast(mapping) as serial,prepare_forecast(mapping,execution=Execution(reference_workers=2)) as pooled:
        assert len(pooled.psfs.truth_kernels.kernels)==3
        positions=serial.positions.positions_yx
        expected=serial.engine.evaluate(positions,[1.e8])[0]
        first=pooled.engine.evaluate(positions,[1.e8])[0]
        path.unlink()
        # Engine transport itself uses the already captured runtime map and kernels;
        # public forecast still honestly refuses the now-missing configured file.
        removed=pooled.engine.evaluate(positions,[1.e8])[0]
        for name in ("q_asimov","fisher_raw","fisher_profiled","amplitude_hat","amplitude_spurious"):
            np.testing.assert_array_equal(getattr(first,name),getattr(expected,name))
            np.testing.assert_array_equal(getattr(removed,name),getattr(expected,name))
        with pytest.raises(FileNotFoundError):pooled.validate_identity()
