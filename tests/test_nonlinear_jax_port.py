"""Focused regression tests for the Item 7b JAX mass-mapping port."""

from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields, replace
import importlib.util
import multiprocessing
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import autolens as al
import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import PchipInterpolator

from hwoslaps.constants import KM_TO_M, MPC_TO_M
from hwoslaps.lensing import mass_models
from hwoslaps.lensing.image_source import ImageSource
from hwoslaps.modeling.nonlinear import autolens_runner
from hwoslaps.modeling.nonlinear import autolens_model_builder as model_builder
from hwoslaps.modeling.nonlinear.autolens_runner import (
    AutoLensFitRunner,
    NonlinearSearchSettings,
)
from hwoslaps.modeling.nonlinear.clumpy_profiles import (
    ClumpyTemplateContext,
    ClumpyTransformedSource,
)
from hwoslaps.modeling.nonlinear.dataset_builder import (
    NonlinearDatasetMetadata,
)
from hwoslaps.modeling.nonlinear import mass_mapping
from hwoslaps.modeling.nonlinear.model_specs import uniform as real_uniform
from hwoslaps.modeling.nonlinear.output_schema import (
    NONLINEAR_CASE_CSV_COLUMNS,
    NonlinearCaseResult,
    NonlinearFitSummary,
)
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial
from hwoslaps.modeling.nonlinear.validator import NonlinearMetricValidator
from hwoslaps.psf.utils import make_pyauto_convolver, make_pyauto_kernel


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

    spec = model_builder.subhalo_model_spec_from_trial(
        config,
        trial,
        fit_mode="freed",
        mass_context=context,
    )
    prior = spec.galaxies["lens"].components["subhalo"].parameters["log10_m200"]
    assert prior.lower == context.log10_m200_lower
    assert prior.upper == context.log10_m200_upper

    def expanded_mass_prior(_mass_context):
        return real_uniform(
            _mass_context.log10_m200_lower - 0.1,
            _mass_context.log10_m200_upper,
        )

    monkeypatch.setattr(model_builder, "_freed_mass_prior", expanded_mass_prior)
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


JAX_CORES_ERROR = (
    "JAX likelihood requires number_of_cores=1; AutoFit vectorizes parameter "
    "batches and ignores the process count; parallelize cases via "
    "CUDA_VISIBLE_DEVICES"
)


def _patch_supported_jax_analysis(monkeypatch, events=None):
    """Install a minimal effective-JAX analysis without tracing."""
    events = [] if events is None else events

    def ensure_x64():
        events.append("x64")

    def ensure_backend():
        events.append("backend")

    class AnalysisImaging:
        """Record explicit JAX construction and expose its effective state."""

        def __init__(self, dataset, use_jax):
            events.append(("analysis", use_jax))
            self.dataset = dataset
            self._use_jax = use_jax

    monkeypatch.setattr(autolens_runner, "ensure_jax_x64", ensure_x64)
    monkeypatch.setattr(
        autolens_runner,
        "ensure_target_jax_backend",
        ensure_backend,
    )
    monkeypatch.setattr(al, "AnalysisImaging", AnalysisImaging)
    monkeypatch.setattr(
        autolens_runner,
        "_patch_analysis_imaging_adapt_images_compat",
        lambda module: None,
    )
    return events


def test_t6_jax_rejects_nonunit_process_count_with_exact_message(tmp_path):
    """Catch silent acceptance of a process count AutoFit ignores for JAX."""
    with pytest.raises(ValueError) as error:
        AutoLensFitRunner(
            NonlinearSearchSettings(use_jax=True, number_of_cores=2),
            output_dir=tmp_path,
        )
    assert str(error.value) == JAX_CORES_ERROR


@pytest.mark.parametrize("jax_n_batch", [0, -1])
def test_t6_jax_batch_size_must_be_positive(jax_n_batch):
    """Catch zero or negative vectorized batch settings before search setup."""
    with pytest.raises(ValueError, match="jax_n_batch must be a positive integer"):
        NonlinearSearchSettings(jax_n_batch=jax_n_batch)


def test_t6_missing_jax_fails_explicitly(monkeypatch):
    """Catch import errors that do not identify the requested engine."""
    real_import = builtins.__import__

    def import_without_jax(name, *args, **kwargs):
        if name == "jax":
            raise ModuleNotFoundError("synthetic missing jax")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_jax)
    with pytest.raises(RuntimeError, match="JAX likelihood was requested.*import"):
        autolens_runner.ensure_jax_x64()


def test_t6_x64_enablement_failure_is_fatal(monkeypatch):
    """Catch a JAX config update that leaves 64-bit likelihoods disabled."""

    class DisabledConfig:
        """Ignore x64 updates to emulate an ineffective installed backend."""

        jax_enable_x64 = False

        def update(self, name, value):
            assert name == "jax_enable_x64"
            assert value is True

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(config=DisabledConfig()))
    with pytest.raises(RuntimeError, match="64-bit mode"):
        autolens_runner.ensure_jax_x64()


def test_t6_backend_feature_check_names_every_missing_fitness_model_seam(
    monkeypatch,
):
    """Catch version-only acceptance of missing traced-vector APIs."""
    import autofit as af
    from autofit.non_linear import fitness as fitness_module

    class UnsupportedFitness:
        """Omit the target vectorization and batch constructor keywords."""

        def __init__(self, model, analysis):
            self.model = model
            self.analysis = analysis

    def unsupported_instance_from_vector(self, vector):
        return vector

    monkeypatch.setattr(fitness_module, "Fitness", UnsupportedFitness)
    monkeypatch.setattr(
        af.Model,
        "instance_from_vector",
        unsupported_instance_from_vector,
    )
    with pytest.raises(RuntimeError) as error:
        autolens_runner.ensure_target_jax_backend()
    message = str(error.value)
    assert "installed" in message
    assert "Fitness.use_jax_vmap" in message
    assert "Fitness.batch_size" in message
    assert "Model.instance_from_vector(xp=...)" in message


def test_t6_x64_is_enabled_before_autolens_analysis_construction(
    monkeypatch,
    tmp_path,
):
    """Catch importing or constructing AutoLens before x64 is established."""
    events = _patch_supported_jax_analysis(monkeypatch)
    analysis = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True),
        output_dir=tmp_path,
    ).make_analysis(dataset=object(), model_metadata={})
    assert analysis._use_jax is True
    assert events == ["x64", "backend", ("analysis", True)]


@pytest.mark.xtx_gpu
def test_t6_x64_precedes_first_autolens_import_in_fresh_interpreter():
    """Catch transitive AutoLens imports before the runner enables x64."""
    source = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys

        import jax

        jax.config.update("jax_enable_x64", False)
        events = []
        tracked = {"autolens", "autogalaxy", "autoarray", "autofit"}

        # Enforced contract: x64 is enabled before the runner imports
        # AutoLens. The wider PyAuto stack cannot be gated the same way:
        # hwoslaps.modeling.nonlinear eagerly imports ImageSource, which
        # subclasses ag.LightProfile and applies aa decorators at class
        # definition, so autogalaxy/autoarray (and autofit through
        # autogalaxy) load before x64 on every real path. Those imports
        # are recorded as observations only; the runner-level guarantee
        # is x64-before-AutoLens-import and before any traced evaluation.


        class AutoLensImportGuard(importlib.abc.MetaPathFinder,
                                  importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                top_level = fullname.split(".")[0]
                if top_level not in tracked:
                    return None

                x64_enabled = bool(jax.config.jax_enable_x64)
                if top_level not in sys.modules:
                    events.append((top_level, x64_enabled))
                if top_level != "autolens":
                    return None

                if not x64_enabled:
                    raise RuntimeError(
                        "AutoLens imported before JAX x64 was enabled"
                    )
                is_package = "." not in fullname
                return importlib.util.spec_from_loader(
                    fullname,
                    self,
                    is_package=is_package,
                )

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                class AnalysisImaging:
                    def __init__(self, dataset, use_jax):
                        self.dataset = dataset
                        self._use_jax = use_jax

                module.AnalysisImaging = AnalysisImaging


        sys.meta_path.insert(0, AutoLensImportGuard())

        from hwoslaps.modeling.nonlinear import autolens_runner

        assert "autolens" not in sys.modules
        autolens_runner.ensure_target_jax_backend = lambda: None
        autolens_runner._patch_analysis_imaging_adapt_images_compat = (
            lambda module: None
        )
        analysis = autolens_runner.AutoLensFitRunner(
            autolens_runner.NonlinearSearchSettings(use_jax=True),
            output_dir=".",
        ).make_analysis(dataset=object(), model_metadata={})
        assert analysis._use_jax is True
        autolens_events = [
            enabled for name, enabled in events if name == "autolens"
        ]
        assert autolens_events == [True]
        assert {name for name, _ in events} <= tracked
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_t6_jax_analysis_typeerror_never_silently_downgrades(
    monkeypatch,
    tmp_path,
):
    """Catch the TypeError fallback disabling an explicit JAX request."""
    calls = []

    class RejectingAnalysis:
        """Reject the target keyword and record fallback attempts."""

        def __init__(self, dataset, **kwargs):
            calls.append(kwargs)
            if "use_jax" in kwargs:
                raise TypeError("synthetic unsupported use_jax")

    monkeypatch.setattr(autolens_runner, "ensure_jax_x64", lambda: None)
    monkeypatch.setattr(
        autolens_runner,
        "ensure_target_jax_backend",
        lambda: None,
    )
    monkeypatch.setattr(al, "AnalysisImaging", RejectingAnalysis)
    monkeypatch.setattr(
        autolens_runner,
        "_patch_analysis_imaging_adapt_images_compat",
        lambda module: None,
    )
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True),
        output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="could not construct.*use_jax=True"):
        runner.make_analysis(dataset=object(), model_metadata={})
    assert calls == [{"use_jax": True}]


def test_t6_cpu_analysis_retains_legacy_typeerror_fallback(monkeypatch, tmp_path):
    """Catch removal of the old-backend CPU construction fallback."""
    calls = []

    class LegacyAnalysis:
        """Accept only the legacy constructor without a use_jax keyword."""

        def __init__(self, dataset, **kwargs):
            calls.append(kwargs)
            if kwargs:
                raise TypeError("legacy constructor")
            self.dataset = dataset

    monkeypatch.setattr(al, "AnalysisImaging", LegacyAnalysis)
    monkeypatch.setattr(
        autolens_runner,
        "_patch_analysis_imaging_adapt_images_compat",
        lambda module: None,
    )
    analysis = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=False),
        output_dir=tmp_path,
    ).make_analysis(dataset=object(), model_metadata={})
    assert analysis.dataset is not None
    assert calls == [{"use_jax": False}, {}]


def test_t6_requires_cpu_guard_remains_general_and_precedes_backend_import(
    monkeypatch,
    tmp_path,
):
    """Catch deletion of the dormant guard or coupling it to Item 7 wording."""
    monkeypatch.setattr(
        autolens_runner,
        "ensure_jax_x64",
        lambda: pytest.fail("requires_cpu must fail before importing JAX"),
        raising=False,
    )
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True),
        output_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="model marked requires_cpu=True"):
        runner.make_analysis(
            dataset=object(),
            model_metadata={"requires_cpu": True},
        )


def _t10_source_config(source_family):
    """Return one complete config for a stock, clumpy, or image source."""
    config, _ = _freed_config_and_trial()
    if source_family == "stock":
        return config
    if source_family == "clumpy":
        config["lensing"]["source_galaxy"]["light"] = {
            "type": "Clumpy",
            "host": {
                "centre": [-0.03, 0.08],
                "ell_comps": [0.12, -0.06],
                "intensity": 1.8,
                "effective_radius": 0.11,
                "sersic_index": 1.2,
            },
            "clumps": [
                {
                    "centre": [0.03, 0.12],
                    "ell_comps": [0.0, 0.0],
                    "intensity": 0.8,
                    "effective_radius": 0.025,
                    "sersic_index": 1.0,
                },
                {
                    "centre": [-0.08, 0.01],
                    "ell_comps": [0.02, -0.01],
                    "intensity": 0.6,
                    "effective_radius": 0.03,
                    "sersic_index": 0.9,
                },
            ],
            "flux_scale": 1.0,
            "size_scale": 1.0,
        }
        return config
    if source_family == "image":
        asset_path = (
            Path(__file__).resolve().parents[1]
            / "configs/source_assets/cosmos_48849_hlr011.npz"
        )
        config["lensing"]["source_galaxy"]["light"] = {
            "type": "Image",
            "asset_path": str(asset_path),
            "centre": [-0.03, 0.08],
            "rotation_deg": 17.0,
            "total_flux": 0.5,
            "flux_scale": 1.0,
            "size_scale": 1.0,
        }
        return config
    raise AssertionError(f"unsupported T10 source family: {source_family}")


def _t10_freed_trial(model):
    """Return a physical freed trial for one supported mass family."""
    return SubhaloTrial(
        case_id=f"item7b-t10-{model.lower()}",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.2, -0.1),
        model=model,
        profile_class={
            "NFW": "NFWSph",
            "SIS": "IsothermalSph",
            "PointMass": "PointMass",
        }[model],
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01 if model == "NFW" else None,
        scale_radius_arcsec=0.2 if model == "NFW" else None,
        einstein_radius_arcsec=0.01 if model != "NFW" else None,
    )


def _t10_spec(family):
    """Build the real repository ModelSpec for one T10 family."""
    if family == "smooth":
        return model_builder.smooth_model_spec_from_config(
            _t10_source_config("stock")
        )
    if family.startswith("freed_"):
        model = {
            "freed_nfw": "NFW",
            "freed_sis": "SIS",
            "freed_point_mass": "PointMass",
        }[family]
        config = deepcopy(_t10_source_config("stock"))
        config["lensing"]["subhalo"]["model"] = model
        context = _context(model=model)
        return model_builder.subhalo_model_spec_from_trial(
            config,
            _t10_freed_trial(model),
            fit_mode="freed",
            mass_context=context,
        )
    if family in {"clumpy_rigid", "clumpy_host_free"}:
        mode = family.removeprefix("clumpy_")
        return model_builder.smooth_model_spec_from_config(
            _t10_source_config("clumpy"),
            clumpy_fit_parameterization=mode,
        )
    if family == "image_source":
        return model_builder.smooth_model_spec_from_config(
            _t10_source_config("image")
        )
    raise AssertionError(f"unsupported T10 family: {family}")


T10_FAMILIES = (
    "smooth",
    "freed_nfw",
    "freed_sis",
    "freed_point_mass",
    "clumpy_rigid",
    "clumpy_host_free",
    "image_source",
)


def test_t6_all_intended_item7b_specs_have_no_requires_cpu_stamp():
    """Catch retaining any CPU stamp after its complete T10 gate passes."""
    for family in T10_FAMILIES[1:]:
        spec = _t10_spec(family)
        assert "requires_cpu" not in spec.metadata, (
            f"{family} still carries requires_cpu="
            f"{spec.metadata['requires_cpu']!r}"
        )


def test_t6_search_name_and_vectorized_kwargs_change_only_for_jax(
    monkeypatch,
    tmp_path,
):
    """Catch CPU resume drift or omission of the JAX execution token/kwargs."""
    import autofit as af

    captured = []

    class CapturingNautilus:
        """Expose effective execution attributes passed by runner."""

        def __init__(self, **kwargs):
            captured.append(dict(kwargs))
            for name, value in kwargs.items():
                setattr(self, name, value)

    monkeypatch.setattr(af, "Nautilus", CapturingNautilus)
    cpu_runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=False),
        output_dir=tmp_path,
    )
    cpu_runner._make_search("case", "smooth", 5, "analysis")
    jax_runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True, jax_n_batch=32),
        output_dir=tmp_path,
    )
    jax_runner._make_search("case", "smooth", 5, "analysis")

    cpu_kwargs, jax_kwargs = captured
    assert cpu_kwargs["name"] == "case_smooth_analysis"
    assert "n_batch" not in cpu_kwargs
    assert "use_jax_vmap" not in cpu_kwargs
    assert jax_kwargs["name"] == "case_smooth_analysis_jax_vmap_b32"
    assert jax_kwargs["n_batch"] == 32
    assert jax_kwargs["use_jax_vmap"] is True
    without_jax = {
        key: value
        for key, value in jax_kwargs.items()
        if key not in {"name", "n_batch", "use_jax_vmap"}
    }
    without_cpu_name = {
        key: value for key, value in cpu_kwargs.items() if key != "name"
    }
    assert without_jax == without_cpu_name


def test_t6_constructed_nautilus_must_expose_effective_vectorized_seam(
    monkeypatch,
    tmp_path,
):
    """Catch a kwargs-tolerant backend silently ignoring JAX batch controls."""
    import autofit as af

    class IgnoringNautilus:
        """Accept but discard every keyword like an unsupported loose API."""

        def __init__(self, **kwargs):
            self.name = kwargs["name"]

    monkeypatch.setattr(af, "Nautilus", IgnoringNautilus)
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True, jax_n_batch=8),
        output_dir=tmp_path,
    )
    with pytest.raises(RuntimeError) as error:
        runner._make_search("case", "subhalo", 5, "analysis")
    message = str(error.value)
    assert "Nautilus" in message
    assert "n_batch" in message
    assert "use_jax_vmap" in message
    assert "autofit=" in message
    assert "autolens=" in message


def _successful_result(log_likelihood=-5.0):
    """Return the smallest result accepted by runner summary extraction."""
    sample = SimpleNamespace(log_likelihood=log_likelihood)
    return SimpleNamespace(
        samples=SimpleNamespace(max_log_likelihood=lambda: sample),
    )


def test_t12_successful_jax_summary_uses_effective_analysis_and_search_state(
    monkeypatch,
    tmp_path,
):
    """Catch request-copy provenance that ignores constructed runtime state."""
    import autofit as af

    class EffectiveNautilus:
        """Expose target attributes and return one successful search result."""

        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

        def fit(self, model, analysis):
            assert analysis._use_jax is True
            return _successful_result()

    monkeypatch.setattr(af, "Nautilus", EffectiveNautilus)
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True, jax_n_batch=32),
        output_dir=tmp_path,
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=2),
        analysis=SimpleNamespace(_use_jax=True),
        role="subhalo",
        fit_mode="freed",
        case_id="case",
        n_live=5,
        analysis_key="analysis",
    )
    assert summary.status == "success"
    assert summary.use_jax_requested is True
    assert summary.use_jax_effective is True
    assert summary.jax_n_batch_effective == 32


def test_t12_requested_effective_disagreement_fails_before_search_fit(
    monkeypatch,
    tmp_path,
):
    """Catch a successful summary after runtime JAX state was downgraded."""
    import autofit as af

    fit_called = []

    class DowngradedNautilus:
        """Expose a backend that ignored the vectorization request."""

        def __init__(self, **kwargs):
            self.n_batch = kwargs["n_batch"]
            self.use_jax_vmap = False

        def fit(self, model, analysis):
            fit_called.append(True)
            return _successful_result()

    monkeypatch.setattr(af, "Nautilus", DowngradedNautilus)
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True, jax_n_batch=8),
        output_dir=tmp_path,
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=2),
        analysis=SimpleNamespace(_use_jax=True),
        role="subhalo",
        fit_mode="freed",
        case_id="case",
        n_live=5,
        analysis_key="analysis",
    )
    assert summary.status == "failed"
    assert "effective JAX execution" in summary.error
    assert summary.use_jax_effective is None
    assert summary.jax_n_batch_effective is None
    assert fit_called == []


def test_t12_failure_after_effective_verification_retains_provenance(
    monkeypatch,
    tmp_path,
):
    """Catch loss of verified JAX provenance after a later fit failure."""
    import autofit as af

    class FailingEffectiveNautilus:
        """Expose effective JAX state, then fail inside search execution."""

        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

        def fit(self, model, analysis):
            raise RuntimeError("synthetic search failure after verification")

    monkeypatch.setattr(af, "Nautilus", FailingEffectiveNautilus)
    runner = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True, jax_n_batch=32),
        output_dir=tmp_path,
    )
    summary = runner.run_model(
        model=SimpleNamespace(total_free_parameters=2),
        analysis=SimpleNamespace(_use_jax=True),
        role="subhalo",
        fit_mode="freed",
        case_id="case",
        n_live=5,
        analysis_key="analysis",
    )
    assert summary.status == "failed"
    assert "synthetic search failure" in summary.error
    assert summary.use_jax_effective is True
    assert summary.jax_n_batch_effective == 32


def test_t12_x64_helper_is_idempotent():
    """Catch stateful x64 setup that fails or disables itself on repetition."""
    jax = pytest.importorskip("jax")
    autolens_runner.ensure_jax_x64()
    autolens_runner.ensure_jax_x64()
    assert jax.config.jax_enable_x64 is True


@pytest.mark.xtx_gpu
def test_t12_real_target_fitness_output_is_float64():
    """Catch float32 contamination at the installed vectorized Fitness seam."""
    try:
        autolens_runner.ensure_target_jax_backend()
    except RuntimeError as error:
        pytest.skip(str(error))
    autolens_runner.ensure_jax_x64()

    import autofit as af
    from autofit.non_linear.fitness import Fitness
    import jax.numpy as jnp

    class ScalarModel:
        """Trace-safe scalar instance for the real Fitness boundary."""

        def __init__(self, value=0.0):
            self.value = value

    class QuadraticAnalysis:
        """Return a scalar likelihood without imposing a NumPy dtype."""

        _xp = jnp

        def log_likelihood_function(self, instance):
            return -jnp.square(instance.value)

    model = af.Model(ScalarModel)
    model.value = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
    fitness = Fitness(
        model=model,
        analysis=QuadraticAnalysis(),
        paths=None,
        use_jax_vmap=True,
        batch_size=2,
    )
    result = fitness.call_wrap(np.asarray([[-0.25], [0.5]], dtype=np.float64))
    assert np.asarray(result).dtype == np.dtype(np.float64)


def _validator_dataset_and_metadata():
    """Return deterministic identity inputs for validator contract tests."""
    dataset = SimpleNamespace(
        data=np.zeros((2, 2), dtype=float),
        noise_map=np.ones((2, 2), dtype=float),
        psf=np.eye(2, dtype=float),
    )
    metadata = NonlinearDatasetMetadata(
        dataset_kind="asimov",
        data_units="adu",
        background_treatment="subtract_known",
        sky_dark_background_adu=0.0,
        mask_name="all_pixels",
        n_unmasked_pixels=4,
        psf_truth_label="truth",
        psf_fit_label="fit",
    )
    return dataset, metadata


class _IdentityAnalysis:
    """Record the object receiving the C1 fixed-point evaluation."""

    def __init__(self):
        self.fixed_point_calls = 0

    def log_likelihood_function(self, instance):
        self.fixed_point_calls += 1
        return -5.0


class _IdentityRunner:
    """Record analysis identity while returning deterministic summaries."""

    def __init__(self, use_jax=False):
        self.settings = NonlinearSearchSettings(use_jax=use_jax)
        self.analysis = _IdentityAnalysis()
        self.calls = []

    def make_analysis(self, dataset, model_metadata=None):
        return self.analysis

    def run_model(self, **kwargs):
        self.calls.append(kwargs)
        return NonlinearFitSummary(
            model_role=kwargs["role"],
            fit_mode=kwargs["fit_mode"],
            status="success",
            log_likelihood_max=-10.0 if kwargs["role"] == "smooth" else -4.0,
            analysis_key=kwargs["analysis_key"],
            use_jax_requested=self.settings.use_jax,
        )


def test_t6_c1_fixed_point_uses_exact_freed_analysis_object():
    """Catch reconstruction of C1 on an engine-distinct analysis object."""
    config, trial = _freed_config_and_trial()
    dataset, metadata = _validator_dataset_and_metadata()
    runner = _IdentityRunner()
    NonlinearMetricValidator(runner).validate_case(
        dataset=dataset,
        dataset_metadata=metadata,
        full_config=config,
        trial=trial,
        fit_mode="freed",
        mass_context=_context(),
    )
    assert runner.analysis.fixed_point_calls == 1
    assert len(runner.calls) == 2
    assert all(call["analysis"] is runner.analysis for call in runner.calls)


def test_t6_smooth_engine_mismatch_is_informational_and_exact():
    """Catch missing/misfiring engine provenance flags on smooth reuse."""
    config, trial = _freed_config_and_trial()
    dataset, metadata = _validator_dataset_and_metadata()
    cpu_runner = _IdentityRunner(use_jax=False)
    first = NonlinearMetricValidator(cpu_runner).validate_case(
        dataset=dataset,
        dataset_metadata=metadata,
        full_config=config,
        trial=trial,
    )

    same = NonlinearMetricValidator(_IdentityRunner(use_jax=False)).validate_case(
        dataset=dataset,
        dataset_metadata=metadata,
        full_config=config,
        trial=trial,
        smooth_result=first.smooth_fit,
    )
    mixed = NonlinearMetricValidator(_IdentityRunner(use_jax=True)).validate_case(
        dataset=dataset,
        dataset_metadata=metadata,
        full_config=config,
        trial=trial,
        smooth_result=first.smooth_fit,
    )
    assert "smooth_engine_mismatch" not in same.quality_flags
    assert "smooth_engine_mismatch" in mixed.quality_flags
    assert mixed.subhalo_fit.status == "success"


def test_t6_legacy_unknown_engine_reuse_does_not_flag_cpu_mismatch():
    """Catch treating absent legacy provenance as a known CPU disagreement."""
    config, trial = _freed_config_and_trial()
    dataset, metadata = _validator_dataset_and_metadata()
    initial = NonlinearMetricValidator(
        _IdentityRunner(use_jax=False)
    ).validate_case(
        dataset=dataset,
        dataset_metadata=metadata,
        full_config=config,
        trial=trial,
    )
    legacy_smooth = replace(
        initial.smooth_fit,
        use_jax_requested=None,
    )
    reused = NonlinearMetricValidator(
        _IdentityRunner(use_jax=False)
    ).validate_case(
        dataset=dataset,
        dataset_metadata=metadata,
        full_config=config,
        trial=trial,
        smooth_result=legacy_smooth,
    )
    assert "smooth_reused" in reused.quality_flags
    assert "smooth_engine_mismatch" not in reused.quality_flags


def test_t12_effective_provenance_serializes_and_old_payloads_default_null():
    """Catch omitted JSON/CSV fields or incompatible legacy construction."""
    legacy = NonlinearFitSummary("smooth", "fixed_template", "success")
    assert legacy.to_dict()["use_jax_effective"] is None
    assert legacy.to_dict()["jax_n_batch_effective"] is None

    effective = NonlinearFitSummary(
        "subhalo",
        "freed",
        "success",
        use_jax_requested=True,
        use_jax_effective=True,
        jax_n_batch_effective=100,
    )
    config, trial = _freed_config_and_trial()
    del config
    dataset, metadata = _validator_dataset_and_metadata()
    del dataset
    case = NonlinearCaseResult(
        case_id=trial.case_id,
        trial=trial,
        dataset_metadata=metadata,
        fit_mode="freed",
        psf_case="nominal",
        smooth_fit=legacy,
        subhalo_fit=effective,
        metric=None,
        quality_flags=["smooth_engine_mismatch"],
    )
    nested = case.to_dict()["subhalo_fit"]
    row = case.to_csv_row()
    assert nested["use_jax_effective"] is True
    assert nested["jax_n_batch_effective"] == 100
    assert row["use_jax_effective"] is True
    assert row["jax_n_batch_effective"] == 100
    assert row["smooth_engine_mismatch"] is True
    assert set(row) == set(NONLINEAR_CASE_CSV_COLUMNS)


def _load_two_gpu_launcher():
    """Load the ignored-path V7 launcher as an importable test module."""
    path = (
        Path(__file__).resolve().parents[1]
        / "scratch/item7b_probes/run_two_gpu_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("item7b_v7_launcher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v7_launch_failure_terminates_reaps_and_closes_partial_children(
    monkeypatch,
    tmp_path,
):
    """Catch an orphaned first GPU fit when the second child cannot start."""
    launcher = _load_two_gpu_launcher()
    arguments = SimpleNamespace(
        config=["first.yaml", "second.yaml"],
        gpu=["0", "1"],
        python="python",
        runner="runner.py",
        log_dir=str(tmp_path),
    )
    monkeypatch.setattr(launcher, "_arguments", lambda: arguments)

    class PartialChild:
        """Record cleanup calls for the one successfully started process."""

        pid = 1234

        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return -15

    child = PartialChild()
    handles = []

    def popen(command, **kwargs):
        handles.append(kwargs["stdout"])
        if len(handles) == 1:
            return child
        raise OSError("synthetic second-child launch failure")

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    with pytest.raises(OSError, match="second-child launch failure"):
        launcher.main()
    assert child.terminated is True
    assert child.waited is True
    assert len(handles) == 2
    assert all(handle.closed for handle in handles)


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


def _require_t10_target_gpu():
    """Skip T10 only when the pinned target API or a GPU is absent."""
    try:
        autolens_runner.ensure_target_jax_backend()
    except RuntimeError as error:
        pytest.skip(str(error))
    autolens_runner.ensure_jax_x64()
    import jax

    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("T10 requires an available target GPU")
    return jax


def _t10_real_imaging(model):
    """Generate one actual 61x61 Imaging dataset from a real model instance."""
    instance = model.instance_from_prior_medians()
    grid = al.Grid2D.uniform(
        shape_native=(61, 61),
        pixel_scales=0.04,
    )
    tracer = al.Tracer(
        galaxies=[instance.galaxies.lens, instance.galaxies.source]
    )
    image = tracer.image_2d_from(grid=grid)
    data = al.Array2D.no_mask(
        values=np.asarray(image.native),
        pixel_scales=0.04,
    )
    noise = al.Array2D.full(
        fill_value=0.2,
        shape_native=(61, 61),
        pixel_scales=0.04,
    )
    unit_kernel = make_pyauto_kernel(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        pixel_scales=0.04,
    )
    return al.Imaging(
        data=data,
        noise_map=noise,
        psf=make_pyauto_convolver(unit_kernel),
        over_sample_size_lp=1,
    )


def _t10_changed_physical_batches(model, family):
    """Build two same-shape batches varying every real free parameter."""
    batch_size = 3
    parameter_count = model.prior_count
    column_offsets = np.linspace(-0.03, 0.03, parameter_count)
    row_offsets = np.linspace(-0.04, 0.04, batch_size)[:, None]
    unit_a = 0.40 + row_offsets + column_offsets[None, :]
    unit_b = 0.60 - row_offsets - column_offsets[None, :]
    batch_a = np.asarray(
        [model.vector_from_unit_vector(row) for row in unit_a],
        dtype=np.float64,
    )
    batch_b = np.asarray(
        [model.vector_from_unit_vector(row) for row in unit_b],
        dtype=np.float64,
    )
    if family == "freed_nfw":
        names = list(model.model_component_and_parameter_names)
        mass_name = "galaxies.lens.subhalo.log10_m200"
        mass_index = names.index(mass_name)
        batch_a[:, mass_index] = np.asarray([6.0, 7.0, 8.5])
        batch_b[:, mass_index] = np.asarray([6.2, 7.2, 8.3])
        assert np.array_equal(batch_a[:, mass_index], [6.0, 7.0, 8.5])
    assert batch_a.shape == batch_b.shape == (batch_size, parameter_count)
    assert np.all(np.any(batch_a != batch_b, axis=0))
    assert np.all(np.ptp(batch_a, axis=0) > 0.0)
    assert np.all(np.ptp(batch_b, axis=0) > 0.0)
    return batch_a, batch_b


def _t10_fitness_pair(model, dataset, tmp_path, batch_size):
    """Construct persistent real target and NumPy Fitness objects."""
    from autofit.non_linear.fitness import Fitness

    jax_analysis = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=True, number_of_cores=1),
        output_dir=tmp_path / "jax",
    ).make_analysis(dataset=dataset, model_metadata={})
    numpy_analysis = AutoLensFitRunner(
        NonlinearSearchSettings(use_jax=False, number_of_cores=1),
        output_dir=tmp_path / "numpy",
    ).make_analysis(dataset=dataset, model_metadata={})
    resample_merit = -1.23456789e99
    jax_fitness = Fitness(
        model=model,
        analysis=jax_analysis,
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=resample_merit,
        use_jax_vmap=True,
        batch_size=batch_size,
    )
    numpy_fitness = Fitness(
        model=model,
        analysis=numpy_analysis,
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=resample_merit,
        use_jax_vmap=False,
    )
    return jax_fitness, numpy_fitness, resample_merit


def _t10_numpy_values(fitness, physical_batch):
    """Evaluate one physical batch serially through real NumPy Fitness."""
    return np.asarray(
        [fitness.call_wrap(vector) for vector in physical_batch],
        dtype=np.float64,
    )


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("family", T10_FAMILIES)
def test_t10_each_real_model_family_matches_numpy_through_persistent_fitness(
    family,
    tmp_path,
):
    """Catch traced model, decorator, precision, or persistent-cache breaks."""
    jax = _require_t10_target_gpu()
    spec = _t10_spec(family)
    model = model_builder.autofit_model_from_spec(spec)
    dataset = _t10_real_imaging(model)
    batch_a, batch_b = _t10_changed_physical_batches(model, family)
    jax_fitness, numpy_fitness, _ = _t10_fitness_pair(
        model,
        dataset,
        tmp_path,
        batch_size=batch_a.shape[0],
    )

    first = jax_fitness.call_wrap(batch_a)
    jax.block_until_ready(first)
    first = np.asarray(first)
    changed = jax_fitness.call_wrap(batch_b)
    jax.block_until_ready(changed)
    changed = np.asarray(changed)

    assert first.shape == changed.shape == (batch_a.shape[0],)
    assert first.dtype == changed.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(changed))
    assert np.any(first != changed)
    numpy_first = _t10_numpy_values(numpy_fitness, batch_a)
    numpy_changed = _t10_numpy_values(numpy_fitness, batch_b)
    assert np.all(np.isfinite(numpy_first))
    assert np.all(np.isfinite(numpy_changed))
    assert np.all(np.abs(first - numpy_first) <= 1.0e-5)
    assert np.all(np.abs(changed - numpy_changed) <= 1.0e-5)


@pytest.mark.xtx_gpu
def test_t10_nonfinite_dynamic_vector_resamples_without_poisoning_fitness(
    tmp_path,
):
    """Catch NaN leakage or persistent-state corruption after resampling."""
    jax = _require_t10_target_gpu()
    model = model_builder.autofit_model_from_spec(_t10_spec("smooth"))
    dataset = _t10_real_imaging(model)
    valid_a, valid_b = _t10_changed_physical_batches(model, "smooth")
    jax_fitness, numpy_fitness, resample_merit = _t10_fitness_pair(
        model,
        dataset,
        tmp_path,
        batch_size=valid_a.shape[0],
    )

    baseline = jax_fitness.call_wrap(valid_a)
    jax.block_until_ready(baseline)
    invalid = valid_a.copy()
    invalid[1, 0] = np.nan
    invalid_values = jax_fitness.call_wrap(invalid)
    jax.block_until_ready(invalid_values)
    invalid_values = np.asarray(invalid_values)
    assert invalid_values[1] == resample_merit

    recovered = jax_fitness.call_wrap(valid_b)
    jax.block_until_ready(recovered)
    recovered = np.asarray(recovered)
    expected = _t10_numpy_values(numpy_fitness, valid_b)
    assert recovered.shape == (valid_b.shape[0],)
    assert recovered.dtype == np.dtype(np.float64)
    assert np.all(np.isfinite(recovered))
    assert np.all(np.abs(recovered - expected) <= 1.0e-5)


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
