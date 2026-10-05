"""New mass families against quadrature and amplitude-angle physics oracles."""

import copy
import math

import numpy as np
import pytest
from scipy.integrate import quad

from hwoslaps.config.checks import ConfigError
from hwoslaps.scene.parameters import scene_parameters, with_parameter
from hwoslaps.scene.profiles import instantiate
from hwoslaps.scene.spec import parse_scene

pytestmark = pytest.mark.backend
POINTS = np.array([[0.13, 0.21], [-0.27, 0.15], [0.41, -0.31], [-0.11, -0.38]])


def _scene(mapping, mass):
    values = copy.deepcopy(mapping["scene"])
    values["lens"]["mass"] = {"main": mass}
    return parse_scene(values)


def _mass(type="PowerLaw", **values):
    return {"type": type, "centre": [0.03, -0.02], "einstein_radius": 0.8,
            "ell_comps": [0.07, 0.13], **({"slope": 2.08} if type == "PowerLaw" else {}), **values}


@pytest.mark.parametrize("slope", [1.6, 1.8, 2.0, 2.08, 2.3, 2.4])
@pytest.mark.parametrize("q", [1.0, 0.85, 0.6])
def test_power_law_deflections_equal_keeton_quadrature(slope, q, minimal_mapping):
    import autolens as al

    angle = 0.31
    f = (1.0 - q) / (1.0 + q)
    spec = _scene(minimal_mapping, _mass(slope=slope, ell_comps=[f * math.sin(2 * angle), f * math.cos(2 * angle)]))
    profile = instantiate(spec.lens.mass[0])["main"]
    expected = []
    for y, x in POINTS - [0.03, -0.02]:
        xr, yr = x * math.cos(angle) + y * math.sin(angle), -x * math.sin(angle) + y * math.cos(angle)
        def integrand(u, exponent):
            denom = 1 - (1 - q*q) * u
            eta = math.sqrt(u * (xr*xr + yr*yr / denom))
            return (3 - slope) / (1 + q) * (0.8 / eta) ** (slope - 1) * denom ** exponent
        ax = q * xr * quad(integrand, 0, 1, args=(-0.5,), epsabs=1e-12, epsrel=1e-12)[0]
        ay = q * yr * quad(integrand, 0, 1, args=(-1.5,), epsabs=1e-12, epsrel=1e-12)[0]
        expected.append([ax * math.sin(angle) + ay * math.cos(angle), ax * math.cos(angle) - ay * math.sin(angle)])
    actual = np.asarray(profile.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=POINTS)))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
    if q == 1.0:
        shifted = POINTS - [0.03, -0.02]
        radii = np.linalg.norm(shifted, axis=1)
        np.testing.assert_allclose(actual, 0.8 ** (slope - 1) * radii[:, None] ** (1 - slope) * shifted,
                                   rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("m,slope", [(3, 2.0), (4, 2.08), (4, 1.7), (3, 2.6)])
@pytest.mark.parametrize("centre", [(0., 0.), (0.03, -0.02)])
@pytest.mark.parametrize("comps", [(0.013, -0.021), (-0.05, 0.03), (0., 0.)])
def test_cartesian_multipole_agrees_with_amplitude_angle_and_backend(m, slope, centre, comps):
    import autolens as al
    from hwoslaps.scene.multipole_profile import CartesianPowerLawMultipole

    args = dict(m=m, slope=slope, centre=centre, einstein_radius=0.8, multipole_comps=comps)
    grid = al.Grid2DIrregular(values=POINTS)
    actual = np.asarray(CartesianPowerLawMultipole(**args).deflections_yx_2d_from(grid=grid))
    parent = np.asarray(al.mp.PowerLawMultipole(**args).deflections_yx_2d_from(grid=grid))
    y, x = (POINTS - centre).T
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    amplitude = math.hypot(*comps)
    phase = math.atan2(comps[0], comps[1]) / m
    a = 0.8 ** (slope - 1) / ((3 - slope)**2 - m*m)
    ar = (3 - slope) * a * r ** (2 - slope) * amplitude * np.cos(m*(phi - phase))
    ap = -m * a * r ** (2 - slope) * amplitude * np.sin(m*(phi - phase))
    expected = np.column_stack((ar*np.sin(phi) + ap*np.cos(phi), ar*np.cos(phi) - ap*np.sin(phi)))
    atol = 1e-14 * max(float(np.max(np.abs(expected))), 1e-300)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=atol)
    np.testing.assert_allclose(actual, parent, rtol=1e-12, atol=atol)
    if amplitude == 0:
        np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_shear_deflections_and_registry_order(minimal_mapping):
    import autolens as al

    spec = _scene(minimal_mapping, {"type": "ExternalShear", "gamma_1": 0.08, "gamma_2": -0.05})
    actual = np.asarray(instantiate(spec.lens.mass[0])["main"].deflections_yx_2d_from(
        grid=al.Grid2DIrregular(values=POINTS)))
    y, x = POINTS.T
    np.testing.assert_allclose(actual, np.column_stack((0.08*x + 0.05*y, 0.08*y - 0.05*x)), rtol=1e-12, atol=1e-14)
    assert [p.name for p in scene_parameters(spec)][:2] == ["lens.mass.main.gamma_1", "lens.mass.main.gamma_2"]


def test_multipole_parameter_reads_replacement_and_links_use_registry(minimal_mapping):
    spec = _scene(minimal_mapping, _mass(multipoles={"m4": [0.0, -0.03], "m3": [0.02, 0.0]}))
    names = [p.name.rsplit(".", 1)[-1] for p in scene_parameters(spec) if p.name.startswith("lens.mass")]
    assert names == ["centre_y", "centre_x", "einstein_radius", "ell_comp_1", "ell_comp_2", "slope",
                     "multipole_m3_1", "multipole_m3_2", "multipole_m4_1", "multipole_m4_2"]
    updated = with_parameter(spec, "lens.mass.main.multipole_m3_2", -0.012)
    assert updated.lens.mass[0].values["multipoles"]["m3"] == (0.02, -0.012)
    assert spec.lens.mass[0].values["multipoles"]["m3"] == (0.02, 0.0)
    profiles = instantiate(updated.lens.mass[0])
    assert list(profiles) == ["main", "main_multipole_m3", "main_multipole_m4"]
    for profile in list(profiles.values())[1:]:
        assert profile.centre == profiles["main"].centre
        assert profile.einstein_radius == profiles["main"].einstein_radius
        assert profile.slope == profiles["main"].slope


@pytest.mark.parametrize("kind,q,slope,multipoles,accepted", [
    ("PowerLaw", 1., 2., {"m4": [0., .99]}, True),
    ("PowerLaw", 1., 2., {"m4": [0., 1.]}, False),
    ("Isothermal", 1., 2., {"m4": [0., .99]}, True),
    ("Isothermal", 1., 2., {"m4": [0., -.999997]}, False),
    ("Isothermal", .8, 2., {"m3": [.4, 0.], "m4": [0., -.45]}, True),
    ("Isothermal", .8, 2., {"m3": [.4, 0.], "m4": [0., -.5]}, False),
    ("PowerLaw", 1., 2.9, {"m4": [0., .0999]}, True),
    ("PowerLaw", 1., 2.9, {"m4": [0., .1001]}, False),
])
def test_multipole_domain_keeps_convergence_positive(kind, q, slope, multipoles, accepted, minimal_mapping):
    mass = _mass(kind, centre=[0., 0.], ell_comps=[0., (1-q)/(1+q)], multipoles=multipoles)
    if kind == "PowerLaw":
        mass["slope"] = slope
    phi = np.arange(7200) * 2*np.pi/7200
    evaluated_q = min(q, .99999) if kind == "Isothermal" else q
    eta = np.sqrt(np.cos(phi)**2 + np.sin(phi)**2/evaluated_q**2)
    convergence = (3-slope)/(1+evaluated_q) * (0.8/eta)**(slope-1)
    for order, comps in multipoles.items():
        m = int(order[1:])
        convergence += .5 * .8**(slope-1) * math.hypot(*comps) * np.cos(m*phi - math.atan2(*comps))
    if accepted:
        _scene(minimal_mapping, mass)
        assert np.min(convergence) > 0
    else:
        with pytest.raises(ConfigError) as error:
            _scene(minimal_mapping, mass)
        assert error.value.path == "scene.lens.mass.main.multipoles"
        assert np.min(convergence) < 0


@pytest.mark.parametrize("edit,leaf", [({"slope": 1.}, "slope"), ({"slope": 3.}, "slope"),
    ({"slope": np.nan}, "slope"), ({"multipoles": {}}, "multipoles"),
    ({"multipoles": {"m2": [0.,0.]}}, "multipoles.m2"),
    ({"multipoles": {"m3": [0.]}}, "multipoles.m3"), ({"ell_comps": [.999,0.]}, "ell_comps"),
    ({"unknown": 1.}, "unknown")])
def test_new_mass_keys_fail_at_exact_paths(edit, leaf, minimal_mapping):
    with pytest.raises(ConfigError) as error:
        _scene(minimal_mapping, _mass(**edit))
    assert error.value.path == "scene.lens.mass.main." + leaf


@pytest.mark.parametrize("mass,leaf", [({"type":"ExternalShear","gamma_1":1.,"gamma_2":0.}, "main"),
    ({"type":"ExternalShear","gamma_1":.1,"gamma_2":.1,"centre":[0.,0.]}, "main.centre")])
def test_shear_domain_and_absent_centre_are_strict(mass, leaf, minimal_mapping):
    with pytest.raises(ConfigError) as error:
        _scene(minimal_mapping, mass)
    assert error.value.path == "scene.lens.mass." + leaf


@pytest.mark.parametrize("kind,q,slope,shear", [("PowerLaw",1.,1.6,0.),("PowerLaw",1.,2.4,0.),
    ("Isothermal",.818,2.,0.),("Isothermal",.6,2.,0.),("Isothermal",1.,2.,.15)])
def test_effective_radius_has_the_analytic_mass_definition(kind,q,slope,shear,minimal_mapping):
    from hwoslaps.scene.cosmology import Cosmology, parse_cosmology
    from hwoslaps.scene.critical_curve import effective_einstein_radius

    mapping = copy.deepcopy(minimal_mapping["scene"])
    mass = _mass(kind, centre=[0.,0.], ell_comps=[0.,(1-q)/(1+q)])
    if kind == "PowerLaw": mass["slope"] = slope
    mapping["lens"]["mass"] = {"main":mass}
    if shear: mapping["lens"]["mass"]["shear"] = {"type":"ExternalShear","gamma_1":0.,"gamma_2":shear}
    spec=parse_scene(mapping)
    result=effective_einstein_radius(spec,Cosmology(parse_cosmology({"name":"Planck15"})))
    expected = .8 if kind=="PowerLaw" else 2*math.sqrt(q)*.8/(1+q)
    if shear: expected=.8*math.sqrt(1+shear*shear/2)/(1-shear*shear)
    assert result == pytest.approx(expected,rel=2e-5,abs=0.)
