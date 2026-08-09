"""Focused regression tests for the Item 7b JAX mass-mapping port."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import multiprocessing
import pickle

import autolens as al
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import PchipInterpolator

from hwoslaps.constants import KM_TO_M, MPC_TO_M
from hwoslaps.lensing import mass_models
from hwoslaps.lensing.image_source import ImageSource
from hwoslaps.modeling.nonlinear import autolens_model_builder as model_builder
from hwoslaps.modeling.nonlinear.clumpy_profiles import (
    ClumpyTemplateContext,
    ClumpyTransformedSource,
)
from hwoslaps.modeling.nonlinear import mass_mapping
from hwoslaps.modeling.nonlinear.model_specs import uniform as real_uniform
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial


MASS_GRID = np.logspace(6.0, 8.5, 512)


def _assert_true_relative(actual, expected, tolerance):
    """Assert the literal relative-error inequality required by the spec."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert np.all(np.abs(actual - expected) <= tolerance * np.abs(expected))


def _assert_finite_difference(actual, expected):
    """Apply the required relative gradient gate with a near-zero fallback."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert np.all(np.isfinite(actual))
    assert np.all(np.isfinite(expected))
    difference = np.abs(actual - expected)
    tolerance = 1.0e-6 * np.abs(expected) + 1.0e-8
    assert np.all(difference <= tolerance), (actual, expected, difference)


def _central_difference(function, parameters, step=1.0e-5):
    """Return independent central differences for every scalar parameter."""
    parameters = np.asarray(parameters, dtype=float)
    derivatives = []
    for index in range(parameters.size):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        derivatives.append((function(upper) - function(lower)) / (2.0 * step))
    return np.asarray(derivatives)


def _geometry():
    """Build the canonical bench geometry through the public eager API."""
    return mass_models.lensing_geometry_scalars(0.2, 0.6, al.cosmo.Planck15())


def _legacy_nfw(masses, relation):
    """Evaluate the untouched legacy NFW path over the fixed mass grid."""
    cosmology = al.cosmo.Planck15()
    concentrations = []
    parameters = []
    for mass in masses:
        if relation == "moline2017_eq7":
            concentration = mass_models.concentration_moline2017_eq7(
                mass,
                x_sub=1.0,
                h=0.6774,
            )
        else:
            concentration = mass_models.concentration_power_law(mass, z=0.2)
        concentrations.append(concentration)
        parameters.append(
            mass_models.nfw_lensing_parameters(
                mass,
                concentration,
                0.2,
                0.6,
                cosmology,
            )
        )
    return np.asarray(concentrations), np.asarray(parameters)


def _legacy_einstein_radii(masses, model):
    """Evaluate one untouched legacy Einstein-radius path."""
    cosmology = al.cosmo.Planck15()
    function = {
        "SIS": mass_models.einstein_radius_sis_m200,
        "PointMass": mass_models.einstein_radius_point_mass,
    }[model]
    return np.asarray(
        [function(mass, 0.2, 0.6, cosmology) for mass in masses]
    )


@pytest.mark.parametrize("relation", ["moline2017_eq7", "power_law"])
def test_t1_numpy_xp_nfw_kernels_match_legacy_dense_grid(relation):
    """Catch changed concentration or NFW arithmetic in the NumPy kernels."""
    expected_c200, expected_parameters = _legacy_nfw(MASS_GRID, relation)
    if relation == "moline2017_eq7":
        actual_c200 = mass_models.concentration_moline2017_eq7_xp(
            MASS_GRID,
            1.0,
            0.6774,
            np,
        )
    else:
        actual_c200 = mass_models.concentration_power_law_xp(
            MASS_GRID,
            0.2,
            np,
        )
    actual_parameters = mass_models.nfw_lensing_parameters_xp(
        MASS_GRID,
        actual_c200,
        _geometry(),
        np,
    )
    _assert_true_relative(actual_c200, expected_c200, 1.0e-15)
    _assert_true_relative(actual_parameters[0], expected_parameters[:, 0], 1.0e-15)
    _assert_true_relative(actual_parameters[1], expected_parameters[:, 1], 1.0e-15)


@pytest.mark.parametrize("model", ["SIS", "PointMass"])
def test_t1_numpy_xp_einstein_radius_kernels_match_legacy_dense_grid(model):
    """Catch changed SIS or point-mass arithmetic in the NumPy kernels."""
    expected = _legacy_einstein_radii(MASS_GRID, model)
    function = {
        "SIS": mass_models.einstein_radius_sis_m200_xp,
        "PointMass": mass_models.einstein_radius_point_mass_xp,
    }[model]
    actual = function(MASS_GRID, _geometry(), np)
    _assert_true_relative(actual, expected, 1.0e-15)


def test_t2_jnp_kernels_match_numpy_dense_grid():
    """Catch backend-specific constants, precision, or operation ordering."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    geometry = _geometry()

    numpy_outputs = (
        mass_models.concentration_moline2017_eq7_xp(
            MASS_GRID, 1.0, 0.6774, np
        ),
        mass_models.concentration_power_law_xp(MASS_GRID, 0.2, np),
        *mass_models.nfw_lensing_parameters_xp(
            MASS_GRID,
            mass_models.concentration_moline2017_eq7_xp(
                MASS_GRID, 1.0, 0.6774, np
            ),
            geometry,
            np,
        ),
        mass_models.einstein_radius_sis_m200_xp(MASS_GRID, geometry, np),
        mass_models.einstein_radius_point_mass_xp(MASS_GRID, geometry, np),
    )
    jax_masses = jnp.asarray(MASS_GRID)
    jax_concentration = mass_models.concentration_moline2017_eq7_xp(
        jax_masses, 1.0, 0.6774, jnp
    )
    jax_outputs = (
        jax_concentration,
        mass_models.concentration_power_law_xp(jax_masses, 0.2, jnp),
        *mass_models.nfw_lensing_parameters_xp(
            jax_masses,
            jax_concentration,
            geometry,
            jnp,
        ),
        mass_models.einstein_radius_sis_m200_xp(jax_masses, geometry, jnp),
        mass_models.einstein_radius_point_mass_xp(jax_masses, geometry, jnp),
    )
    for actual, expected in zip(jax_outputs, numpy_outputs):
        _assert_true_relative(np.asarray(actual), expected, 1.0e-12)


def test_t3_geometry_scalars_match_direct_legacy_computation_and_are_frozen():
    """Catch omitted geometry inputs, retained astropy state, or mutability."""
    cosmology = al.cosmo.Planck15()
    geometry = mass_models.lensing_geometry_scalars(0.2, 0.6, cosmology)
    d_l_m = mass_models.angular_diameter_distance_mpc(cosmology, 0.2) * MPC_TO_M
    d_s_m = mass_models.angular_diameter_distance_mpc(cosmology, 0.6) * MPC_TO_M
    d_ls_m = (
        mass_models.angular_diameter_distance_z1z2_mpc(cosmology, 0.2, 0.6)
        * MPC_TO_M
    )
    g_si = float(const.G.value)
    c_si = float(const.c.value)
    h_z_si = (
        mass_models.hubble_parameter_km_s_mpc(cosmology, 0.2)
        * KM_TO_M
        / MPC_TO_M
    )
    expected = {
        "z_lens": 0.2,
        "z_source": 0.6,
        "d_l_m": d_l_m,
        "d_s_m": d_s_m,
        "d_ls_m": d_ls_m,
        "rho_crit_z_lens_kg_m3": 3 * h_z_si**2 / (8 * np.pi * g_si),
        "sigma_crit_kg_m2": (
            c_si**2 / (4 * np.pi * g_si)
        ) * (d_s_m / (d_l_m * d_ls_m)),
        "msun_kg": float((1 * u.Msun).to(u.kg).value),
        "g_si": g_si,
        "c_si": c_si,
    }
    assert [field.name for field in fields(geometry)] == list(expected)
    for name, value in expected.items():
        _assert_true_relative(getattr(geometry, name), value, 1.0e-15)
    assert pickle.loads(pickle.dumps(geometry)) == geometry
    assert hash(pickle.loads(pickle.dumps(geometry))) == hash(geometry)
    with pytest.raises(FrozenInstanceError):
        geometry.d_l_m = 1.0


def _context(model="NFW", relation="moline2017_eq7", **overrides):
    """Build one public v2 context with valid defaults."""
    arguments = {
        "subhalo_model": model,
        "concentration_model": relation if model == "NFW" else None,
        "x_sub": 1.0 if model == "NFW" and relation == "moline2017_eq7" else None,
        "h": 0.6774 if model == "NFW" and relation == "moline2017_eq7" else None,
        "z_lens": 0.2,
        "z_source": 0.6,
        "cosmology_name": "Planck15",
        "log10_m200_range": (6.0, 8.5),
    }
    arguments.update(overrides)
    return mass_mapping.build_mass_mapping_context_explicit(**arguments)


def _spawn_context_hash(queue):
    """Build the canonical context in a fresh interpreter."""
    queue.put(_context().context_hash)


def test_t4_context_v2_fields_hash_pickle_and_spawn_identity():
    """Catch retained tables, omitted hash inputs, or unstable hashes."""
    context = _context()
    assert [field.name for field in fields(context)] == [
        "subhalo_model",
        "concentration_model",
        "x_sub",
        "h",
        "z_lens",
        "z_source",
        "cosmology_name",
        "log10_m200_lower",
        "log10_m200_upper",
        "geometry",
        "context_hash",
    ]
    assert pickle.loads(pickle.dumps(context)) == context
    variants = [
        _context(x_sub=0.9),
        _context(h=0.7),
        _context(z_lens=0.21),
        _context(z_source=0.61),
        _context(log10_m200_range=(6.0, 8.6)),
        _context(relation="power_law"),
        _context(model="SIS"),
    ]
    assert all(item.context_hash != context.context_hash for item in variants)

    spawn = multiprocessing.get_context("spawn")
    queue = spawn.Queue()
    process = spawn.Process(target=_spawn_context_hash, args=(queue,))
    process.start()
    spawned_hash = queue.get(timeout=30)
    process.join(timeout=30)
    assert process.exitcode == 0
    assert spawned_hash == context.context_hash


@pytest.mark.parametrize("model", ["NFW", "SIS", "PointMass"])
@pytest.mark.parametrize("relation", ["moline2017_eq7", "power_law"])
def test_t4_closed_form_mapping_stays_within_old_pchip_numerics(model, relation):
    """Catch discontinuity larger than the retired table's 1e-11 contract."""
    if model != "NFW" and relation == "power_law":
        pytest.skip("SIS and PointMass have no concentration relation")
    context = _context(model=model, relation=relation)
    node_count = 2049
    while True:
        nodes = np.linspace(6.0, 8.5, node_count)
        masses = 10.0**nodes
        if model == "NFW":
            _, legacy = _legacy_nfw(masses, relation)
            tables = [
                PchipInterpolator(nodes, legacy[:, 0]),
                PchipInterpolator(nodes, legacy[:, 1]),
            ]
            keys = ["kappa_s", "scale_radius_arcsec"]
        else:
            legacy = _legacy_einstein_radii(masses, model)
            tables = [PchipInterpolator(nodes, legacy)]
            keys = ["einstein_radius_arcsec"]
        validation_probes = np.linspace(6.0, 8.5, 4 * node_count)
        validation_masses = 10.0**validation_probes
        if model == "NFW":
            _, direct = _legacy_nfw(validation_masses, relation)
            direct_outputs = [direct[:, 0], direct[:, 1]]
        else:
            direct_outputs = [
                _legacy_einstein_radii(validation_masses, model)
            ]
        relative_errors = [
            np.max(np.abs(table(validation_probes) - direct) / np.abs(direct))
            for table, direct in zip(tables, direct_outputs)
        ]
        if max(relative_errors) <= 1.0e-11:
            break
        if node_count == 32769:
            pytest.fail("retired adaptive PCHIP algorithm did not converge")
        node_count = min(2 * node_count, 32769)

    probes = np.linspace(6.0, 8.5, 512)
    closed_form = [mass_mapping.evaluate_mass_mapping(context, value) for value in probes]
    for key, table in zip(keys, tables):
        _assert_true_relative(
            [item[key] for item in closed_form],
            table(probes),
            1.0e-11,
        )


@pytest.mark.parametrize("log_mass", [5.999, 8.501, np.nan])
def test_t4_eager_mapping_rejects_out_of_range_and_nan(log_mass):
    """Catch accidental eager extrapolation or NaN acceptance."""
    with pytest.raises(ValueError, match="outside"):
        mass_mapping.evaluate_mass_mapping(_context(), log_mass)


@pytest.mark.parametrize(
    "overrides",
    [
        {"z_lens": np.nan},
        {"z_lens": np.inf},
        {"z_lens": 0.0},
        {"z_lens": -0.1},
        {"z_source": np.nan},
        {"z_source": np.inf},
        {"z_source": 0.2},
        {"z_source": 0.1},
        {"log10_m200_range": (np.nan, 8.5)},
        {"log10_m200_range": (6.0, np.inf)},
        {"log10_m200_range": (8.5, 6.0)},
        {"log10_m200_range": (6.0, 6.0)},
        {"log10_m200_range": (5.99, 8.5)},
        {"log10_m200_range": (6.0, 12.001)},
    ],
)
def test_t4_context_rejects_invalid_scalar_domains(overrides):
    """Catch deletion of explicit pre-trace scalar and Moline-mass checks."""
    with pytest.raises(ValueError):
        _context(**overrides)


@pytest.mark.parametrize("x_sub", [None, 0.0, -0.1, np.nan, np.inf, 1.5001])
def test_t4_context_rejects_invalid_moline_x_sub(x_sub):
    """Catch missing or out-of-domain Moline radial-position validation."""
    with pytest.raises(ValueError, match="x_sub"):
        _context(x_sub=x_sub)


@pytest.mark.parametrize("h", [0.0, -0.1, np.nan, np.inf])
def test_t4_context_rejects_invalid_moline_h(h):
    """Catch nonpositive or nonfinite resolved-H validation."""
    with pytest.raises(ValueError, match="h"):
        _context(h=h)


def test_t4_context_rejects_invalid_model_and_relation():
    """Catch acceptance of unsupported model/relation combinations."""
    with pytest.raises(ValueError, match="subhalo_model"):
        _context(model="Gaussian")
    with pytest.raises(ValueError, match="concentration_model"):
        _context(relation="duffy2008")


@pytest.mark.parametrize("field_name,bad_value", [("d_l_m", 0.0), ("g_si", np.nan)])
def test_t4_context_rejects_invalid_derived_geometry(monkeypatch, field_name, bad_value):
    """Catch missing derived-geometry validation before tracing."""
    valid_geometry = _geometry()
    monkeypatch.setattr(
        mass_mapping,
        "lensing_geometry_scalars",
        lambda *args: replace(valid_geometry, **{field_name: bad_value}),
        raising=False,
    )
    mass_mapping._build_context.cache_clear()
    try:
        with pytest.raises(ValueError, match="geometry"):
            _context(log10_m200_range=(6.0, 8.499))
    finally:
        mass_mapping._build_context.cache_clear()


def _freed_config_and_trial():
    """Return minimal complete inputs for the real freed-model builder."""
    config = {
        "lensing": {
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "einstein_radius": 1.0,
                    "ell_comps": [0.1, 0.0],
                },
            },
            "source_galaxy": {
                "redshift": 0.6,
                "light": {
                    "type": "Exponential",
                    "centre": [-0.03, 0.08],
                    "ell_comps": [0.1, 0.2],
                    "intensity": 2.0,
                    "effective_radius": 0.11,
                },
            },
            "subhalo": {
                "enabled": True,
                "model": "NFW",
                "mass": 1.0e7,
                "concentration": {
                    "model": "moline2017_eq7",
                    "x_sub": 1.0,
                    "h": None,
                },
            },
            "cosmology": "Planck15",
        },
        "observation": {"throughput": 1.0},
    }
    trial = SubhaloTrial(
        case_id="prior-support",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model="NFW",
        profile_class="NFWSph",
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01,
        scale_radius_arcsec=0.2,
    )
    return config, trial


def test_t5_builder_rejects_mass_prior_support_outside_context(monkeypatch):
    """Catch removal of the final builder-boundary prior-support gate."""
    config, trial = _freed_config_and_trial()
    context = _context()

    def expanded_mass_uniform(lower, upper):
        if lower == context.log10_m200_lower and upper == context.log10_m200_upper:
            return real_uniform(lower - 0.1, upper)
        return real_uniform(lower, upper)

    monkeypatch.setattr(model_builder, "uniform", expanded_mass_uniform)
    with pytest.raises(ValueError, match="log10_m200 prior support"):
        model_builder.subhalo_model_spec_from_trial(
            config,
            trial,
            fit_mode="freed",
            mass_context=context,
        )


def test_t5_builder_accepts_mass_prior_at_exact_context_bounds():
    """Catch an exclusive or rounded prior-support boundary check."""
    config, trial = _freed_config_and_trial()
    context = _context()
    spec = model_builder.subhalo_model_spec_from_trial(
        config,
        trial,
        fit_mode="freed",
        mass_context=context,
    )
    prior = spec.galaxies["lens"].components["subhalo"].parameters["log10_m200"]
    assert prior.lower == context.log10_m200_lower
    assert prior.upper == context.log10_m200_upper


def test_t7_xp_selector_finds_nested_jax_arrays_and_tracers():
    """Catch shallow traversal or omission of either supported JAX type."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    concrete = jnp.asarray(0.37)
    assert mass_mapping._xp_for({"outer": [(concrete,)]}) is jnp

    @jax.jit
    def traced(value):
        xp = mass_mapping._xp_for({"outer": [(value,)]})
        return xp.sin(value)

    actual = traced(concrete)
    np.testing.assert_allclose(np.asarray(actual), np.sin(0.37), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "model,relation,adapter",
    [
        ("NFW", "moline2017_eq7", "NFWMCRSubhaloSph"),
        ("NFW", "power_law", "NFWMCRSubhaloSph"),
        ("SIS", None, "SISMCRSubhalo"),
        ("PointMass", None, "PointMassMCRSubhalo"),
    ],
)
def test_t7_mass_adapters_construct_and_evaluate_under_persistent_jit(
    model,
    relation,
    adapter,
):
    """Catch tracer casts, NumPy derivations, and frozen traced mass values."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    context = _context(
        model=model,
        relation=relation or "moline2017_eq7",
    )
    adapter_class = getattr(mass_mapping, adapter)
    grid = al.Grid2D.uniform(
        shape_native=(3, 3),
        pixel_scales=0.14,
        origin=(0.04, -0.02),
    )

    def traced_image(log_mass):
        profile = adapter_class(
            centre=(0.02, -0.03),
            log10_m200=log_mass,
            mapping_context=context,
        )
        return profile.deflections_yx_2d_from(grid=grid, xp=jnp).array

    def eager_objective(log_mass):
        profile = adapter_class(
            centre=(0.02, -0.03),
            log10_m200=float(log_mass),
            mapping_context=context,
        )
        values = profile.deflections_yx_2d_from(grid=grid, xp=np)
        return float(np.asarray(values).sum())

    persistent = jax.jit(traced_image)
    first = persistent(jnp.asarray(7.0))
    changed = persistent(jnp.asarray(7.2))
    eager = adapter_class(
        centre=(0.02, -0.03),
        log10_m200=7.0,
        mapping_context=context,
    ).deflections_yx_2d_from(grid=grid, xp=np)
    np.testing.assert_allclose(
        np.asarray(first),
        np.asarray(eager),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert not np.array_equal(np.asarray(first), np.asarray(changed))

    gradient = jax.grad(lambda value: jnp.sum(traced_image(value)))(
        jnp.asarray(7.0)
    )
    finite_difference = (
        eager_objective(7.0 + 1.0e-5)
        - eager_objective(7.0 - 1.0e-5)
    ) / (2.0e-5)
    _assert_finite_difference(np.asarray(gradient), finite_difference)


def _clumpy_context():
    """Return a nondegenerate fixed host-and-clump template."""
    return ClumpyTemplateContext(
        host=(0.11, -0.07, 1.8, 0.16, 1.4),
        host_centre=(0.03, -0.04),
        clumps=(
            (0.09, -0.06, 0.03, -0.02, 0.7, 0.035, 1.1),
            (-0.08, 0.05, -0.04, 0.01, 0.5, 0.028, 0.9),
        ),
        context_hash="item7b-t8",
    )


def _clumpy_profile(parameters, mode):
    """Construct a rigid or host-free profile from flat parameters."""
    arguments = {
        "centre": (parameters[0], parameters[1]),
        "flux_scale": parameters[2],
        "size_scale": parameters[3],
        "host_ell_comps": (0.11, -0.07),
        "host_intensity": 1.8,
        "host_effective_radius": 0.16,
        "host_sersic_index": 1.4,
        "template_context": _clumpy_context(),
    }
    if mode == "host_free":
        arguments.update(
            host_ell_comps=(parameters[4], parameters[5]),
            host_intensity=parameters[6],
            host_effective_radius=parameters[7],
            host_sersic_index=parameters[8],
        )
    return ClumpyTransformedSource(**arguments)


@pytest.mark.parametrize(
    "mode,parameters",
    [
        ("rigid", [0.03, -0.04, 1.15, 0.92]),
        (
            "host_free",
            [0.03, -0.04, 1.15, 0.92, 0.11, -0.07, 1.8, 0.16, 1.4],
        ),
    ],
)
def test_t8_clumpy_profiles_trace_every_sampled_scalar(mode, parameters):
    """Catch scalar casts, lost xp threading, or detached clumpy parameters."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    parameters = np.asarray(parameters, dtype=float)
    grid = al.Grid2D.uniform(
        shape_native=(3, 3),
        pixel_scales=0.11,
        origin=(0.04, -0.02),
    )

    def traced_image(values):
        image = _clumpy_profile(values, mode).image_2d_from(
            grid=grid,
            xp=jnp,
        )
        return image.array

    def eager_image(values):
        return np.asarray(
            _clumpy_profile(values, mode).image_2d_from(
                grid=grid,
                xp=np,
            )
        )

    def eager_objective(values):
        return float(eager_image(values).sum())

    persistent = jax.jit(traced_image)
    first = persistent(jnp.asarray(parameters))
    changed_parameters = parameters.copy()
    changed_parameters[:4] += np.asarray([0.006, -0.004, 0.05, 0.03])
    if mode == "host_free":
        changed_parameters[4:] += np.asarray(
            [0.006, -0.005, 0.08, 0.008, 0.04]
        )
    changed = persistent(jnp.asarray(changed_parameters))
    np.testing.assert_allclose(
        np.asarray(first),
        eager_image(parameters),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert not np.array_equal(np.asarray(first), np.asarray(changed))

    gradient = jax.grad(lambda values: jnp.sum(traced_image(values)))(
        jnp.asarray(parameters)
    )
    finite_difference = _central_difference(
        eager_objective,
        parameters,
        step=3.0e-6,
    )
    _assert_finite_difference(np.asarray(gradient), finite_difference)


def _image_samples():
    """Return a deterministic normalized asymmetric image asset."""
    rows, cols = np.indices((8, 10), dtype=float)
    values = np.exp(
        -0.5 * (((rows - 2.7) / 1.1) ** 2 + ((cols - 5.6) / 1.4) ** 2)
    )
    return values / (0.2**2 * values.sum())


def _sky_points_from_pixels(rows, cols, centre, rotation_deg, size_scale):
    """Map literal pixel coordinates to sky coordinates independently."""
    theta = np.deg2rad(rotation_deg)
    cosine = np.cos(theta)
    sine = np.sin(theta)
    u = (np.asarray(cols) - 4.5) * 0.2 * size_scale
    v = (np.asarray(rows) - 3.5) * 0.2 * size_scale
    dx = u * cosine - v * sine
    dy = u * sine + v * cosine
    return np.column_stack((centre[0] + dy, centre[1] + dx))


@pytest.mark.parametrize("size_scale", [0.85, 1.2])
def test_t9_image_source_jax_matches_complete_zero_pad_matrix(size_scale):
    """Catch wrong bilinear corners, transforms, masks, or external leakage."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    centre = (0.13, -0.21)
    rows = np.asarray(
        [2.25, 0.0, 7.0, -0.5, 7.5, -1.0, 8.0, -1.01, 8.01, 3.3]
    )
    cols = np.asarray(
        [4.60, 0.0, 9.0, 4.2, 5.1, 2.0, 7.0, 3.0, 6.0, 10.01]
    )
    points = _sky_points_from_pixels(
        rows,
        cols,
        centre,
        rotation_deg=37.3,
        size_scale=size_scale,
    )
    profile = ImageSource(
        centre=centre,
        rotation_deg=37.3,
        pixel_scale_arcsec=0.2,
        sb=_image_samples(),
        total_flux=1.7,
        flux_scale=1.15,
        size_scale=size_scale,
    )
    grid = al.Grid2DIrregular(values=points)
    expected = np.asarray(profile.image_2d_from(grid=grid, xp=np))
    actual = profile.image_2d_from(grid=grid, xp=jnp)
    np.testing.assert_allclose(
        np.asarray(actual),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.array_equal(np.asarray(actual)[7:], np.zeros(3))


def _image_profile(parameters):
    """Construct a traced ImageSource while its sample asset stays fixed."""
    return ImageSource(
        centre=(parameters[0], parameters[1]),
        rotation_deg=37.3,
        pixel_scale_arcsec=0.2,
        sb=_image_samples(),
        total_flux=1.7,
        flux_scale=parameters[2],
        size_scale=parameters[3],
    )


def test_t9_image_source_persistent_jit_gradients_and_warm_transfer_guard():
    """Catch traced casts, stale values, bad gradients, or warm transfers."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    jax.config.update("jax_enable_x64", True)
    parameters = np.asarray([0.13, -0.21, 1.15, 0.92], dtype=float)
    grid = al.Grid2D.uniform(
        shape_native=(3, 3),
        pixel_scales=0.11,
        origin=tuple(parameters[:2]),
    )

    def traced_image(values):
        return _image_profile(values).image_2d_from(
            grid=grid,
            xp=jnp,
        ).array

    def eager_objective(values):
        profile = _image_profile(values)
        return float(
            np.asarray(profile.image_2d_from(grid=grid, xp=np).array).sum()
        )

    persistent = jax.jit(traced_image)
    parameters_device = jax.device_put(parameters)
    first = persistent(parameters_device)
    first.block_until_ready()
    changed_parameters = parameters + np.asarray([0.006, -0.004, 0.05, 0.03])
    changed = persistent(jax.device_put(changed_parameters))
    changed.block_until_ready()
    assert not np.array_equal(np.asarray(first), np.asarray(changed))

    with jax.transfer_guard("disallow"):
        persistent(parameters_device).block_until_ready()

    gradient = jax.grad(lambda values: jnp.sum(traced_image(values)))(
        parameters_device
    )
    finite_difference = _central_difference(eager_objective, parameters)
    _assert_finite_difference(np.asarray(gradient), finite_difference)


def _spawn_profile_evaluation(serialized_profiles, queue):
    """Unpickle every Item 7 custom profile and evaluate it eagerly."""
    try:
        grid = al.Grid2DIrregular(
            values=np.asarray(
                [[0.071, -0.013], [-0.043, 0.097]],
                dtype=float,
            )
        )
        results = []
        for serialized in serialized_profiles:
            profile = pickle.loads(serialized)
            if hasattr(profile, "mapping_context"):
                values = profile.deflections_yx_2d_from(grid=grid, xp=np)
            else:
                values = profile.image_2d_from(grid=grid, xp=np)
            results.append(
                (
                    profile.__class__.__module__,
                    profile.__class__.__qualname__,
                    float(np.asarray(values).sum()),
                )
            )
        queue.put(("ok", results))
    except Exception as error:
        queue.put(("error", repr(error)))


def test_t11_all_custom_profiles_pickle_spawn_and_evaluate_on_cpu():
    """Catch lost module identity or non-picklable traced-profile state."""
    profiles = [
        mass_mapping.NFWMCRSubhaloSph(
            log10_m200=7.0,
            mapping_context=_context(),
        ),
        mass_mapping.SISMCRSubhalo(
            log10_m200=7.0,
            mapping_context=_context(model="SIS"),
        ),
        mass_mapping.PointMassMCRSubhalo(
            log10_m200=7.0,
            mapping_context=_context(model="PointMass"),
        ),
        _clumpy_profile(np.asarray([0.03, -0.04, 1.15, 0.92]), "rigid"),
        _image_profile(np.asarray([0.13, -0.21, 1.15, 0.92])),
    ]
    serialized = [pickle.dumps(profile) for profile in profiles]
    restored = [pickle.loads(payload) for payload in serialized]
    assert [profile.__class__ for profile in restored] == [
        profile.__class__ for profile in profiles
    ]
    assert all(profile.__class__.__qualname__ == profile.__class__.__name__ for profile in restored)

    spawn = multiprocessing.get_context("spawn")
    queue = spawn.Queue()
    process = spawn.Process(
        target=_spawn_profile_evaluation,
        args=(serialized, queue),
    )
    process.start()
    status, payload = queue.get(timeout=60)
    process.join(timeout=60)
    assert process.exitcode == 0
    assert status == "ok", payload
    results = payload
    assert all(np.isfinite(value) for _, _, value in results)
    assert [module for module, _, _ in results] == [
        profile.__class__.__module__ for profile in profiles
    ]
