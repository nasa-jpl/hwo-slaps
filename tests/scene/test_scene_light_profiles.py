"""Sersic normalization and component/plane linearity against independent integrals."""

import copy
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammainc, gammaincinv

from hwoslaps.scene.profiles import PROFILE_TYPES, sersic_constant, sersic_unit_integral
from hwoslaps.scene.spec import parse_scene

pytestmark = pytest.mark.backend


@pytest.mark.parametrize("n", [.36,.418,.5,1.,2.5,4.,8.])
def test_sersic_unit_integral_and_b_n_follow_the_rendered_profile(n):
    import autolens as al

    b=sersic_constant(n)
    assert b == al.lp.Sersic(sersic_index=n).sersic_constant
    radius=.11
    integrand=lambda u: 2*math.pi*n*radius**2*u**(2*n-1)*math.exp(-b*(u-1))
    integral=quad(integrand,0,1,epsabs=1e-13,epsrel=1e-13)[0]+quad(integrand,1,np.inf,epsabs=1e-13,epsrel=1e-13)[0]
    assert sersic_unit_integral(radius,n) == pytest.approx(integral,rel=1e-12,abs=0.)
    assert .5 <= gammainc(2*n,b) <= .5+1.7e-4
    assert abs(b/gammaincinv(2*n,.5)-1) <= 5.5e-4
    definition=PROFILE_TYPES["Sersic"].parameters({})[-1]
    assert (definition.domain.lower,definition.domain.upper,definition.domain.open_lower,definition.domain.open_upper)==(.36,8.,False,False)


def test_multi_component_light_is_the_sum_of_its_components(minimal_mapping,image_asset):
    from hwoslaps.scene.builder import build_scene
    from hwoslaps.scene.cosmology import Cosmology,parse_cosmology

    values=copy.deepcopy(minimal_mapping["scene"])
    values["source"]["light"]={
        "bulge":{"type":"Sersic","centre":[-.03,.08],"ell_comps":[.03,.05],"effective_radius":.12,"intensity":.8,"sersic_index":4.},
        "disk":{"type":"Exponential","centre":[-.02,.07],"ell_comps":[.1,.04],"effective_radius":.2,"intensity":.6},
        "clumps":{"type":"Image","centre":[-.04,.09],"asset_path":str(image_asset),"rotation_deg":30.,"total_flux":.1}}
    values["lens"]["light"]={"light":{"type":"Sersic","centre":[0.,0.],"ell_comps":[.1,.0],"effective_radius":.3,"intensity":.05,"sersic_index":3.}}
    cosmology=Cosmology(parse_cosmology({"name":"Planck15"}))
    full=build_scene(parse_scene(values),cosmology,subhalo=None)
    parts=[]
    for name,component in values["source"]["light"].items():
        one=copy.deepcopy(values);one["source"]["light"]={name:component}
        parts.append(np.asarray(build_scene(parse_scene(one),cosmology,subhalo=None).light_images["source"]))
    np.testing.assert_allclose(full.light_images["source"],sum(parts),rtol=1e-13,atol=0.)
    import autolens as al
    lens_profile=full.light_profiles["lens"][0]
    np.testing.assert_array_equal(full.light_images["lens"],np.asarray(lens_profile.image_2d_from(grid=full.grid).native))
    exp=copy.deepcopy(minimal_mapping["scene"])
    ser=copy.deepcopy(exp);ser["source"]["light"]["light"].update(type="Sersic",sersic_index=1.)
    np.testing.assert_array_equal(build_scene(parse_scene(exp),cosmology,subhalo=None).light_images["source"],
                                  build_scene(parse_scene(ser),cosmology,subhalo=None).light_images["source"])
