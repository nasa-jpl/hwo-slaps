"""Plane assembly of halos: lens-plane deflections, the two-plane lens equation, summation order."""

import math
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.constants import ARCSEC_PER_RAD, C_M_S, G_SI, MPC_TO_M, MSUN_KG
from hwoslaps.scene.builder import build_scene
from hwoslaps.scene.halos import FixedConcentration, HaloModel, Moline2017, make_halo
from hwoslaps.scene.perturbers import realize_perturbers
from hwoslaps.scene.spec import parse_scene

POINTS = np.array([(-0.31, 0.27), (0.19, 0.42), (0.44, -0.36), (-0.52, -0.11), (0.08, -0.47), (1.3, 0.2), (-0.9, 1.1)])
POINT_MASS = HaloModel("PointMass", None, None)
SIS = HaloModel("SIS", None, None)
NFW = HaloModel("NFW", Moline2017(x_sub=1.0, h=None), None)
OFF_PLANE_NFW = HaloModel("NFW", FixedConcentration(20.), None)
CENTRE = (0.4, -0.8)


def _point_mass_einstein_radius(mass, geometry):
    """theta_E = sqrt(4 G M D_ds / (c^2 D_d D_s)), arcsec."""
    d_d, d_s, d_ds = (value * MPC_TO_M for value in (geometry.d_deflector_mpc, geometry.d_source_mpc,
                                                       geometry.d_deflector_source_mpc))
    return math.sqrt(4 * G_SI * mass * MSUN_KG * d_ds / (C_M_S**2 * d_d * d_s)) * ARCSEC_PER_RAD


def _r200_m(mass, geometry):
    return (3 * mass * MSUN_KG / (4 * math.pi * 200 * geometry.rho_crit_kg_m3)) ** (1.0 / 3.0)


def _sis_einstein_radius(mass, geometry):
    """theta_E = 4 pi sigma^2 / c^2 D_ds / D_s with sigma^2 = G M200 / (2 r200), arcsec."""
    sigma_squared = G_SI * mass * MSUN_KG / (2 * _r200_m(mass, geometry))
    return 4 * math.pi * sigma_squared / C_M_S**2 * geometry.d_deflector_source_mpc / geometry.d_source_mpc * ARCSEC_PER_RAD


def _nfw_scales(mass, geometry, reduced_h, fixed_concentration=None):
    """(kappa_s, theta_s) of an M200c NFW with the Moline et al. (2017) eq. 7 concentration at x_sub = 1."""
    log_mass = math.log10(mass * reduced_h / 1.0e8)
    c = (19.9 * (1 - 0.195 * log_mass + (0.089 * log_mass) ** 2 + (0.089 * log_mass) ** 3)
         if fixed_concentration is None else fixed_concentration)
    r_s = _r200_m(mass, geometry) / c
    rho_s = geometry.rho_crit_kg_m3 * 200.0 / 3.0 * c**3 / (math.log(1 + c) - c / (1 + c))
    return rho_s * r_s / geometry.sigma_crit_kg_m2, r_s / (geometry.d_deflector_mpc * MPC_TO_M) * ARCSEC_PER_RAD


def _nfw_deflection(radius, kappa_s, theta_s):
    """Bartelmann (1996): alpha(r) = 4 kappa_s theta_s h(x) / x, x = r / theta_s."""
    x = radius / theta_s
    if x < 1:
        f = math.acosh(1 / x) / math.sqrt(1 - x**2)
    else:
        f = math.acos(1 / x) / math.sqrt(x**2 - 1)
    return 4 * kappa_s * theta_s * (math.log(x / 2) + f) / x


def _radial_deflections(points, centre, magnitude):
    offsets = points - np.asarray(centre)
    radii = np.hypot(offsets[:, 0], offsets[:, 1])
    return offsets / radii[:, None] * np.array([magnitude(r) for r in radii])[:, None]


def _deflections(scene, points):
    import autolens as al

    return np.asarray(scene.tracer.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=points)))


def _independent_dense_singular_image(scene, halo, lens, source, regular_size, central_size):
    import autolens as al

    centres = np.asarray(scene.grid)
    pixel = scene.pixel_scale_arcsec
    geometry = halo.geometry
    theta_e = (_point_mass_einstein_radius(halo.mass_msun, geometry) if halo.model.type == "PointMass"
               else _sis_einstein_radius(halo.mass_msun, geometry))
    near = np.all(np.abs(centres-np.asarray(halo.position_yx_arcsec)) <= 1.5*pixel, axis=1)
    image = np.empty(len(centres))
    for selected, size in ((np.flatnonzero(~near), regular_size), (np.flatnonzero(near), central_size)):
        axis = ((np.arange(size)+0.5)/size-0.5)*pixel
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        offsets = np.column_stack([yy.ravel(), xx.ravel()])
        # Bound the sample-array memory even for the independently refined core.
        chunk = max(1, 65536//(size*size))
        for start in range(0, len(selected), chunk):
            indices = selected[start:start+chunk]
            points = (centres[indices, None, :] + offsets[None, :, :]).reshape(-1, 2)
            delta = points - np.asarray(halo.position_yx_arcsec)
            squared = np.sum(delta**2, axis=1)
            assert np.all(squared > 0.), "the independent oracle requires strictly positive quadrature radii"
            factor = theta_e**2/squared if halo.model.type == "PointMass" else theta_e/np.sqrt(squared)
            macro = np.asarray(lens.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=points)))
            traced = points - macro - delta*factor[:, None]
            values = np.asarray(source.image_2d_from(grid=al.Grid2DIrregular(values=traced)))
            image[indices] = values.reshape(len(indices), size*size).mean(axis=1)
    return image.reshape(scene.spec.grid.shape)


@pytest.mark.backend
@pytest.mark.parametrize("role, model", [("subhalo", POINT_MASS), ("subhalo", SIS), ("perturber", NFW)],
                         ids=["point-mass-subhalo", "sis-subhalo", "nfw-perturber"])
def test_lens_plane_halos_add_their_analytic_deflections(scene_mapping, planck15, role, model):
    mass = 3.0e9
    spec = parse_scene(scene_mapping)
    smooth = build_scene(spec, planck15, subhalo=None)
    halo = make_halo(model, mass, CENTRE, redshift=0.2, source_redshift=0.6, cosmology=planck15)
    scene = build_scene(spec, planck15, subhalo=halo if role == "subhalo" else None,
                        perturbers=(halo,) if role == "perturber" else ())
    geometry = planck15.geometry(0.2, 0.6)
    point_mass_radius = _point_mass_einstein_radius(mass, geometry)
    sis_radius = _sis_einstein_radius(mass, geometry)
    kappa_s, theta_s = _nfw_scales(mass, geometry, planck15.reduced_h)

    def magnitude(radius):
        if model.type == "PointMass":
            return point_mass_radius**2 / radius
        if model.type == "SIS":
            return sis_radius
        return _nfw_deflection(radius, kappa_s, theta_s)

    np.testing.assert_allclose(_deflections(scene, POINTS) - _deflections(smooth, POINTS),
                               _radial_deflections(POINTS, CENTRE, magnitude), rtol=1.0e-12, atol=0.0)
    assert scene.plane_count == 2


@pytest.mark.backend
@pytest.mark.parametrize("role, redshift", [("perturber", 0.4), ("perturber", 0.1), ("subhalo", 0.4),
                                            ("subhalo", 0.1)],
                         ids=["perturber-behind-the-lens", "perturber-in-front", "subhalo-behind-the-lens",
                              "subhalo-in-front"])
def test_off_plane_halo_follows_the_two_plane_lens_equation(scene_mapping, planck15, role, redshift):
    import autolens as al

    mass = 1.0e10
    if role == "perturber":
        scene_mapping["perturbers"] = {"halos": [{"type": "NFW", "concentration": {"kind": "fixed", "value": 20.},
                                                  "mass_msun": mass, "centre": list(CENTRE), "redshift": redshift}]}
        spec = parse_scene(scene_mapping)
        scene = build_scene(spec, planck15, subhalo=None, perturbers=realize_perturbers(spec, planck15, seed=0))
    else:
        scene_mapping["subhalo"] = {"type": "NFW", "concentration": {"kind": "fixed", "value": 20.},
                                     "redshift": redshift}
        spec = parse_scene(scene_mapping)
        scene = build_scene(spec, planck15, subhalo=make_halo(OFF_PLANE_NFW, mass, CENTRE, redshift=redshift,
                                                              source_redshift=0.6, cosmology=planck15))
    lens = al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.8, ell_comps=(0.05, 0.0))
    kappa_s, theta_s = _nfw_scales(mass, planck15.geometry(redshift, 0.6), planck15.reduced_h,
                                 fixed_concentration=20.)

    def alpha_lens(points):
        return np.asarray(lens.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=points)))

    def alpha_halo(points):
        return _radial_deflections(points, CENTRE, lambda radius: _nfw_deflection(radius, kappa_s, theta_s))

    lens_geometry, halo_geometry = planck15.geometry(0.2, 0.6), planck15.geometry(redshift, 0.6)
    if redshift > 0.2:
        beta = (planck15.geometry(0.2, redshift).d_deflector_source_mpc * lens_geometry.d_source_mpc
                / (halo_geometry.d_deflector_mpc * lens_geometry.d_deflector_source_mpc))
        halo_plane = POINTS - beta * alpha_lens(POINTS)
        expected = POINTS - alpha_lens(POINTS) - alpha_halo(halo_plane)
    else:
        beta = (planck15.geometry(redshift, 0.2).d_deflector_source_mpc * lens_geometry.d_source_mpc
                / (lens_geometry.d_deflector_mpc * halo_geometry.d_deflector_source_mpc))
        lens_plane = POINTS - beta * alpha_halo(POINTS)
        expected = POINTS - alpha_halo(POINTS) - alpha_lens(lens_plane)
    traced = scene.tracer.traced_grid_2d_list_from(grid=al.Grid2DIrregular(values=POINTS))[-1]
    np.testing.assert_allclose(np.asarray(traced), expected, rtol=1.0e-12, atol=0.0)
    assert scene.plane_count == 3


@pytest.mark.backend
@pytest.mark.parametrize("scene_kind, model", [
    ("assembly", POINT_MASS), ("assembly", SIS), ("assembly", NFW),
    ("gold64", POINT_MASS), ("gold64", SIS), ("gold64", NFW),
], ids=["point-mass", "sis", "nfw", "point-mass-gold64", "sis-gold64", "nfw-gold64"])
def test_subhalo_and_perturbers_sum_in_the_assembly_order(scene_mapping, planck15, scene_kind, model,
                                                         record_property):
    import autolens as al

    if scene_kind == "gold64":
        # Exact physical inputs of974 tests/test_lensing_physics_integration.py.
        # grid.pixel_scale -> pixel_scale_arcsec; missing old subsize default is4.
        # lens/source_galaxy become named plane components; enabled/direct subhalo
        # becomes an explicit real Halo at the same(y,x); null h uses Planck15.
        mapping = {
            "grid": {"shape": [64, 64], "pixel_scale_arcsec": 0.00716, "over_sample_size": 4},
            "lens": {"redshift": 0.2, "mass": {"main": {
                "type": "Isothermal", "centre": [0.0, 0.0], "einstein_radius": 1.0, "ell_comps": [0.1, 0.0]}}},
            "source": {"redshift": 0.6, "light": {"disk": {
                "type": "Exponential", "centre": [-0.03, 0.08], "ell_comps": [0.14516129, 0.25142673],
                "intensity": 2.0, "effective_radius": 0.11}}},
            "subhalo": {"type": model.type},
        }
        if model.type == "NFW":
            mapping["subhalo"]["concentration"] = {"kind": "moline2017_eq7", "x_sub": 1.0, "h": None}
        spec = parse_scene(mapping)
        mass = 1.0e9 if model.type == "NFW" else 1.0e8
        injected = make_halo(model, mass, (0.08, -0.05), redshift=0.2, source_redshift=0.6, cosmology=planck15)
        scene = build_scene(spec, planck15, subhalo=injected)
        image = scene.light_images["source"]
        fixture = Path(__file__).resolve().parents[1] / "fixtures/scene/halo_anchors.json"
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == "1167b2cc3916468b9b77f2d4df3f24ca0d5c9d8a70f3ded05d753ef3e15e1026"
        expected = json.loads(fixture.read_text())["integration_image_summary"][model.type.lower()]
        assert list(image.shape) == expected["shape"]
        assert np.all(np.isfinite(image))
        if model.type == "NFW":
            assert float(np.sum(image)) == pytest.approx(expected["total_flux"], rel=1.0e-10)
            assert float(np.max(image)) == pytest.approx(expected["peak"], rel=1.0e-10)
        else:
            # These historical singular-family anchors used four uniform samples
            # per axis. Validate the new pixel integrals rather than replacing gold.
            lens = al.mp.Isothermal(centre=(0., 0.), einstein_radius=1., ell_comps=(0.1, 0.))
            source = al.lp.Exponential(centre=(-0.03, 0.08), ell_comps=(0.14516129, 0.25142673),
                                       intensity=2., effective_radius=0.11)
            coarse = _independent_dense_singular_image(scene, injected, lens, source, 32, 512)
            fine = _independent_dense_singular_image(scene, injected, lens, source, 64, 1024)
            uncertainty = np.abs(fine - coarse)
            reference_refinement = np.sum(uncertainty) / np.sum(np.abs(fine))
            image_error = np.sum(np.abs(image-fine)) / np.sum(np.abs(fine))
            record_property("dense_reference_relative_l1_refinement", float(reference_refinement))
            record_property("adaptive_image_relative_l1_error", float(image_error))
            record_property("historical_sampled_total_flux", expected["total_flux"])
            record_property("integrated_total_flux", float(np.sum(image)))
            assert reference_refinement < 2.e-6
            assert image_error < 4.e-6
            assert np.all(np.abs(image-fine) <= 2.e-6*np.abs(fine) + 1.e-10 + 2.*uncertainty)
        return

    scene_mapping["perturbers"] = {"halos": [{"type": "NFW", "concentration": {"kind": "moline2017_eq7", "x_sub": 1.0},
                                              "mass_msun": 5.0e8, "centre": [-0.5, 0.6]}]}
    spec = parse_scene(scene_mapping)
    (perturber,) = realize_perturbers(spec, planck15, seed=0)
    subhalo = make_halo(model, 1.0e9, CENTRE, redshift=0.2, source_redshift=0.6, cosmology=planck15)
    scene = build_scene(spec, planck15, subhalo=subhalo, perturbers=(perturber,))
    source = al.Galaxy(redshift=0.6, disk=al.lp.Exponential(centre=(0.02, -0.03), ell_comps=(0.1, 0.05), intensity=1.0,
                                                            effective_radius=0.12))
    separate = al.Tracer(galaxies=[
        al.Galaxy(redshift=0.2, main=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.8, ell_comps=(0.05, 0.0))),
        al.Galaxy(redshift=0.2, perturber_0=perturber.autolens_profile()),
        al.Galaxy(redshift=0.2, subhalo=subhalo.autolens_profile()),
        source,
    ], cosmology=al.cosmo.Planck15())
    # Separate same-redshift galaxies preserve the summation order independently
    # of builder assembly. Positive-radius rays remain valid for all three families.
    probe = al.Grid2DIrregular(values=POINTS)
    np.testing.assert_array_equal(_deflections(scene, POINTS),
                                  np.asarray(separate.deflections_yx_2d_from(grid=probe)))
    expected_image = separate.image_2d_from(grid=scene.grid).native.array
    if model.type in ("PointMass", "SIS"):
        from hwoslaps.scene.singular_integration import correct_native_images
        def evaluator(points):
            return np.asarray(separate.image_2d_from(grid=al.Grid2DIrregular(values=points)))
        expected_image = correct_native_images(expected_image.reshape(-1), np.asarray(scene.grid),
                                               scene.grid.pixel_scales, [subhalo.position_yx_arcsec],
                                               evaluator)[:, 0].reshape(scene.spec.grid.shape)
    np.testing.assert_array_equal(scene.light_images["source"], expected_image)


def test_listed_perturbers_are_realized_in_list_order(scene_mapping, planck15):
    scene_mapping["perturbers"] = {"halos": [
        {"type": "SIS", "mass_msun": 2.0e9, "centre": [0.1, 0.2]},
        {"type": "NFW", "concentration": {"kind": "fixed", "value": 12.0}, "mass_msun": 4.0e9, "centre": [-0.3, 0.0],
         "redshift": 0.45},
    ]}
    spec = parse_scene(scene_mapping)
    first, second = realize_perturbers(spec, planck15, seed=7)
    assert (first.model.type, first.mass_msun, first.position_yx_arcsec, first.redshift) == ("SIS", 2.0e9, (0.1, 0.2), 0.2)
    assert (second.model.type, second.mass_msun, second.position_yx_arcsec, second.redshift) == (
        "NFW", 4.0e9, (-0.3, 0.0), 0.45)
    assert first.source_redshift == second.source_redshift == 0.6 and first.cosmology == planck15
    assert realize_perturbers(spec, planck15, seed=8) == (first, second)


@pytest.mark.backend
@pytest.mark.parametrize("model", [POINT_MASS, SIS], ids=["pointmass", "sis"])
def test_singular_halo_values_distinguish_exact_centre_from_positive_radii(planck15, model, record_property):
    import autolens as al

    halo = make_halo(model, 1.e8, CENTRE, redshift=0.2, source_redshift=0.6, cosmology=planck15)
    point = np.asarray(CENTRE)
    adjacent = np.array([point, [point[0], np.nextafter(point[1], np.inf)],
                         [np.nextafter(point[0], np.inf), point[1]]])
    actual = np.asarray(halo.autolens_profile().deflections_yx_2d_from(grid=al.Grid2DIrregular(values=adjacent)))
    assert np.all(np.isnan(actual[0]))
    geometry = planck15.geometry(0.2, 0.6)
    theta_e = (_point_mass_einstein_radius(halo.mass_msun, geometry) if model.type == "PointMass"
               else _sis_einstein_radius(halo.mass_msun, geometry))
    magnitude = lambda radius: theta_e**2/radius if model.type == "PointMass" else theta_e
    expected = _radial_deflections(adjacent[1:], CENTRE, magnitude)
    physical_magnitude = np.linalg.norm(expected, axis=1)
    relative_vector_error = np.linalg.norm(actual[1:]-expected, axis=1)/physical_magnitude
    direction = expected/physical_magnitude[:, None]
    orthogonal = actual[1:] - np.sum(actual[1:]*direction, axis=1)[:, None]*direction
    record_property("adjacent_radii_arcsec", np.hypot(*(adjacent[1:]-point).T).tolist())
    record_property("physical_deflection_magnitudes_arcsec", physical_magnitude.tolist())
    record_property("max_relative_vector_error", float(np.max(relative_vector_error)))
    record_property("max_orthogonal_roundoff_arcsec", float(np.max(np.abs(orthogonal))))
    # The inherited angular transform can leave a roundoff-sized orthogonal
    # component. Compare vectors in units of their physical magnitude; this
    # retains the 1e-14 relative criterion even for enormous adjacent-ray values.
    assert np.all(physical_magnitude > 0.)
    assert np.all(np.isfinite(actual[1:]))
    assert np.all(relative_vector_error < 1.e-14)


@pytest.mark.backend
@pytest.mark.parametrize("model", [POINT_MASS, SIS], ids=["pointmass", "sis"])
@pytest.mark.parametrize("redshift", [0.1, 0.4])
@pytest.mark.parametrize("role", ["subhalo", "perturber"])
def test_off_plane_singular_scene_integration_has_explicit_capability_boundary(scene_mapping, planck15,
                                                                             model, redshift, role):
    from hwoslaps.scene.singular_integration import SingularIntegrationCapabilityError

    halo = make_halo(model, 1.e8, CENTRE, redshift=redshift, source_redshift=0.6, cosmology=planck15)
    with pytest.raises(SingularIntegrationCapabilityError, match="two-plane scene and a lens-plane halo"):
        build_scene(parse_scene(scene_mapping), planck15, subhalo=halo if role == "subhalo" else None,
                    perturbers=(halo,) if role == "perturber" else ())
