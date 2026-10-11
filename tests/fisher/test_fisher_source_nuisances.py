"""Sersic index detector column against a differentiated brightness formula."""

import copy

import numpy as np
import pytest

pytestmark=pytest.mark.backend


def test_sersic_index_column_matches_analytic_derivative(minimal_mapping):
    import autolens as al
    from hwoslaps.fisher.api import prepare_forecast

    minimal_mapping["scene"]["grid"]["shape"]=[60,60]
    source=minimal_mapping["scene"]["source"]["light"]["light"]
    source.update(type="Sersic",sersic_index=2.5,ell_comps=[.08,.13])
    with prepare_forecast(minimal_mapping) as prepared:
        scene=prepared.scene;profile=scene.light_profiles["source"][0]
        traced=scene.tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(values=np.asarray(scene.grid.over_sampled)))[-1]
        y,x=(np.asarray(traced)-profile.centre).T
        e=np.hypot(*profile.ell_comps);q=(1-e)/(1+e);phi=.5*np.arctan2(*profile.ell_comps)
        xr=x*np.cos(phi)+y*np.sin(phi);yr=-x*np.sin(phi)+y*np.cos(phi)
        r=np.sqrt(q*xr*xr+yr*yr/q);u=r/profile.effective_radius;n=2.5
        b=2*n-1/3+4/(405*n)+46/(25515*n*n)+131/(1148175*n**3)-2194697/(30690717750*n**4)
        db=2-4/(405*n*n)-92/(25515*n**3)-393/(1148175*n**4)+4*2194697/(30690717750*n**5)
        intensity=profile.intensity*np.exp(-b*(u**(1/n)-1))
        derivative=intensity*(-db*(u**(1/n)-1)+b*u**(1/n)*np.log(u)/n**2)
        native=scene.grid.over_sampler.binned_array_2d_from(array=derivative).native
        from scipy.signal import convolve2d
        kernel=prepared.psfs.model_kernels.for_group("source").kernel
        expected=convolve2d(np.asarray(native),kernel,mode="same")
        exposure=prepared.observation.exposure
        expected*=exposure.exposure_time_s/exposure.detector.gain_e_per_adu
        index=prepared.nuisances.names.index("source.light.light.sersic_index")
        actual=prepared.nuisances.images[index]
        assert np.max(np.abs(actual-expected))<=1e-6*np.max(np.abs(expected))
        assert prepared.nuisances.parameters[index].step==.001


def test_multicomponent_source_columns_have_registry_order_and_index_edges_fail(minimal_mapping):
    from hwoslaps.config.checks import ConfigError
    from hwoslaps.fisher.api import prepare_forecast
    from hwoslaps.scene.parameters import scene_parameters

    source=minimal_mapping["scene"]["source"]["light"]
    source["light"].update(type="Sersic",sersic_index=2.5)
    source["second"]=copy.deepcopy(source["light"]);source["second"].update(sersic_index=4.,centre=[.02,-.03])
    source["third"]=copy.deepcopy(source["light"]);source["third"].update(type="Exponential",centre=[.03,.02]);del source["third"]["sersic_index"]
    with prepare_forecast(minimal_mapping) as prepared:
        expected=[p.name for p in scene_parameters(prepared.scene.spec)]
        assert list(prepared.nuisances.names[:-1])==expected
        assert [name for name in expected if name.endswith("sersic_index")]==["source.light.light.sersic_index","source.light.second.sersic_index"]
    source["light"]["sersic_index"]=.36
    with pytest.raises(ConfigError,match="sersic_index.*half-step"):
        prepare_forecast(minimal_mapping)
    minimal_mapping["forecast"]["nuisances"]={"fixed":["source.light.light.sersic_index"]}
    with prepare_forecast(minimal_mapping) as prepared:
        assert "source.light.light.sersic_index" not in prepared.nuisances.names
