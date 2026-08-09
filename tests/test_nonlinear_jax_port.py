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
from hwoslaps.modeling.nonlinear import autolens_model_builder as model_builder
from hwoslaps.modeling.nonlinear import mass_mapping
from hwoslaps.modeling.nonlinear.model_specs import uniform as real_uniform
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial


MASS_GRID = np.logspace(6.0, 8.5, 512)


def _assert_true_relative(actual, expected, tolerance):
    """Assert the literal relative-error inequality required by the spec."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert np.all(np.abs(actual - expected) <= tolerance * np.abs(expected))


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
