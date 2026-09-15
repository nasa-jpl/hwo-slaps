"""Independent correctness tests for Plan B's radial-table lookup.

The tests deliberately build abscissae with the same ``log(logspace())``
operation used by the engine.  In particular, an exactly uniform linspace is
not a sufficient test of the proposed shortcut.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("jax")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hwoslaps.modeling.fisher_grid_jax import (
    JaxGridTemplateEngine,
    _affine_log_grid_interval_index,
    _affine_log_grid_parameters,
    _interp_on_affine_log_grid,
)


def _actual_log_radii(n: int, r_min: float, r_max: float) -> np.ndarray:
    """Reproduce the engine's stored abscissae, including round-off."""
    radii = np.logspace(np.log10(r_min), np.log10(r_max), n)
    return np.log(radii)


def _queries_at_knots(log_radii: np.ndarray) -> np.ndarray:
    """Return knots, adjacent representable values, and interval midpoints."""
    knots = np.asarray(log_radii, dtype=np.float64)
    midpoints = 0.5 * (knots[:-1] + knots[1:])
    return np.concatenate(
        (
            knots,
            np.nextafter(knots, -np.inf),
            np.nextafter(knots, np.inf),
            midpoints,
            np.asarray([knots[0] - 1.0, knots[-1] + 1.0]),
        )
    )


@pytest.mark.parametrize("n", [8192, 32768, 131072])
@pytest.mark.parametrize(
    "r_min,r_max",
    [(1.0e-6, 2.0), (1.0e-5, 40.0), (3.0e-8, 300.0)],
)
def test_actual_logspace_grids_are_accepted(n, r_min, r_max):
    """Ordinary, mismatch, and extended production-like grids validate."""
    log_radii = _actual_log_radii(n, r_min, r_max)
    parameters = _affine_log_grid_parameters(log_radii)
    assert parameters is not None
    origin, inverse_step = parameters
    assert origin == float(log_radii[0])
    assert np.isfinite(inverse_step) and inverse_step > 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: np.array([x[0], x[0], *x[2:]]),
        lambda x: np.array([*x[:-1], x[-2], x[-1]]),
        lambda x: x.at[17].set(np.nan) if hasattr(x, "at") else _nan_numpy(x),
        lambda x: x[:1],
        lambda x: np.asarray(x, dtype=float).reshape(-1, 1),
        lambda x: _perturb_one_knot(x),
    ],
)
def test_nonuniform_or_invalid_grids_fall_back(mutation):
    """The affine path must never silently accept an invalid table."""
    base = _actual_log_radii(257, 1.0e-6, 8.0)
    mutated = mutation(base)
    assert _affine_log_grid_parameters(mutated) is None


def _nan_numpy(x):
    result = np.array(x, copy=True)
    result[17] = np.nan
    return result


def _perturb_one_knot(x):
    result = np.array(x, copy=True)
    result[len(result) // 2] += 0.75 * np.median(np.diff(result))
    return result


@pytest.mark.parametrize("n", [8192, 32768, 131072])
@pytest.mark.parametrize(
    "r_min,r_max",
    [(1.0e-6, 2.0), (1.0e-5, 40.0), (3.0e-8, 300.0)],
)
def test_every_knot_and_midpoint_matches_jax_right_interp(n, r_min, r_max):
    """Compare exact bins and values against JAX's right-sided reference."""
    log_radii = _actual_log_radii(n, r_min, r_max)
    values = np.sin(np.linspace(-2.0, 3.0, n)) + 0.17 * np.cos(
        np.linspace(0.0, 19.0, n) ** 1.3
    )
    origin, inverse_step = _affine_log_grid_parameters(log_radii)
    assert origin is not None
    queries = _queries_at_knots(log_radii)
    expected_bins = np.clip(
        np.searchsorted(log_radii, queries, side="right") - 1,
        0,
        n - 2,
    )
    got_bins = np.asarray(
        _affine_log_grid_interval_index(
            jnp.asarray(queries),
            jnp.asarray(log_radii),
            origin,
            inverse_step,
        )
    )
    np.testing.assert_array_equal(got_bins, expected_bins)
    got = np.asarray(
        _interp_on_affine_log_grid(
            jnp.asarray(queries),
            jnp.asarray(log_radii),
            jnp.asarray(values),
            origin,
            inverse_step,
        )
    )
    expected = np.asarray(
        jnp.interp(jnp.asarray(queries), jnp.asarray(log_radii), jnp.asarray(values))
    )
    np.testing.assert_allclose(got, expected, rtol=2.0e-14, atol=2.0e-15)


def test_right_side_semantics_are_checked_at_each_knot():
    """Adjacent ``nextafter`` queries must select the exact right interval."""
    knots = _actual_log_radii(513, 1.0e-6, 12.0)
    origin, inverse_step = _affine_log_grid_parameters(knots)
    assert origin is not None
    interior = knots[1:-1]
    queries = np.stack(
        (np.nextafter(interior, -np.inf), interior, np.nextafter(interior, np.inf)),
        axis=1,
    ).reshape(-1)
    expected = np.clip(np.searchsorted(knots, queries, side="right") - 1, 0, len(knots) - 2)
    got = np.asarray(
        _affine_log_grid_interval_index(jnp.asarray(queries), jnp.asarray(knots), origin, inverse_step)
    )
    np.testing.assert_array_equal(got, expected)


def test_float32_abscissae_are_rejected_from_shortcut():
    """The production table is float64; lower precision geometry falls back."""
    assert _affine_log_grid_parameters(
        _actual_log_radii(129, 1.0e-6, 5.0).astype(np.float32)
    ) is None


@pytest.mark.parametrize("query_dtype", [np.float32, np.float64])
def test_scalar_batch_jit_vmap_dtype_zero_and_nonfinite(query_dtype):
    """Scalar, batched, vmapped, dtype, zero, and non-finite contracts."""
    knots = _actual_log_radii(129, 1.0e-6, 5.0)
    values = np.array([0.0, *np.geomspace(1.0e-300, 1.0e-12, len(knots) - 1)])
    origin, inverse_step = _affine_log_grid_parameters(knots)
    assert origin is not None
    x = np.asarray(
        [knots[0], 0.5 * (knots[3] + knots[4]), knots[-1], -np.inf, np.inf, np.nan],
        dtype=query_dtype,
    )
    x_jax = jnp.asarray(x)
    k_jax = jnp.asarray(knots, dtype=np.float64)
    v_jax = jnp.asarray(values, dtype=np.float64)

    def direct(q):
        return _interp_on_affine_log_grid(q, k_jax, v_jax, origin, inverse_step)

    scalar = direct(x_jax[1])
    assert np.asarray(scalar).shape == ()
    assert scalar.dtype == jnp.asarray(values, dtype=np.float64).dtype
    expected = jnp.interp(x_jax, k_jax, v_jax)
    np.testing.assert_allclose(
        np.asarray(direct(x_jax)), np.asarray(expected), equal_nan=True, rtol=2e-14, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(direct)(x_jax)), np.asarray(expected), equal_nan=True, rtol=2e-14, atol=0.0
    )
    vmapped = jax.jit(jax.vmap(direct))(x_jax)
    np.testing.assert_allclose(
        np.asarray(vmapped), np.asarray(expected), equal_nan=True, rtol=2e-14, atol=0.0
    )


@pytest.mark.parametrize("n", [8192, 32768])
@pytest.mark.parametrize("mass", [1.0e7, 1.0e8, 1.0e9])
def test_actual_nfw_tables_match_reference(n, mass):
    """Use the production NFW constructor and compare several masses."""
    pytest.importorskip("autolens")
    import autolens as al
    from hwoslaps.lensing.generator import _create_subhalo

    config = {
        "enabled": True,
        "model": "NFW",
        "mass": mass,
        "position": {"type": "direct", "centre": [0.0, 0.0]},
        "concentration": {"model": "moline2017_eq7", "x_sub": 1.0, "h": None},
    }
    lens = al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=0.5))
    profile, info = _create_subhalo(config, 0.5, 2.0, lens, 0.00716, al.cosmo.Planck15())
    radii = np.logspace(-6, 2, n)
    alpha = JaxGridTemplateEngine._radial_deflection_on(profile, radii)
    log_radii = np.log(radii)
    origin, inverse_step = _affine_log_grid_parameters(log_radii)
    assert origin is not None
    rng = np.random.default_rng(int(mass) + n)
    queries = np.concatenate(
        (
            rng.uniform(log_radii[0], log_radii[-1], 1001),
            np.log([info["scale_radius_arcsec"], 1.0e-6, 1.0e2]),
        )
    )
    expected = jnp.interp(queries, log_radii, alpha)
    got = _interp_on_affine_log_grid(queries, log_radii, alpha, origin, inverse_step)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=2.0e-14, atol=1.0e-18)
    assert np.any(np.abs(alpha) > 0.0)


@pytest.fixture(scope="module")
def tiny_engine(tmp_path_factory):
    """Build the existing tiny real scene for signal/reduction integration."""
    pytest.importorskip("autolens")
    pytest.importorskip("hcipy")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(tmp_path_factory.mktemp("numba")))
    import test_fisher_grid_map as grid_tests
    from hwoslaps.lensing import generate_lensing_system

    tmp = tmp_path_factory.mktemp("radial-engine")
    config = grid_tests._build_grid_config(tmp)
    baseline_config = copy.deepcopy(config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    baseline = generate_lensing_system(baseline_config["lensing"], full_config=baseline_config)
    # A tiny identity PSF keeps this focused integration independent of the
    # optional HCIPy PSF generator while exercising the real render/FFT path.
    kernel = np.zeros((3, 3), dtype=float)
    kernel[1, 1] = 1.0
    truth_kernel = np.zeros_like(kernel)
    truth_kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
    positions = [(-0.2, -0.2), (0.0, 0.0), (0.2, 0.2)]
    npix = int(baseline.image.size)
    nuisance = np.column_stack((np.ones(npix), np.linspace(0.0, 1.0, npix)))

    def build(template, mismatch=False):
        engine = JaxGridTemplateEngine(
            lensing_baseline=baseline,
            map_config_template=template,
            psf_kernel_native=kernel,
            truth_psf_kernel_native=truth_kernel if mismatch else None,
            mu0_adu_2d=np.zeros_like(np.asarray(baseline.image.native, dtype=float)),
            mask_2d=np.ones_like(baseline.image, dtype=bool),
            candidate_positions=positions,
            batch_size=2,
            sigma_masked=np.ones(npix),
            nuisance_whitened=nuisance,
        )
        return engine

    engine = build(config)
    reference = build(config)
    reference._log_grid_origin = None
    reference._log_grid_inverse_step = None
    mismatch_engine = build(config, mismatch=True)
    mismatch_reference = build(config, mismatch=True)
    mismatch_reference._log_grid_origin = None
    mismatch_reference._log_grid_inverse_step = None
    return {"engine": engine, "reference": reference, "mismatch": mismatch_engine,
            "mismatch_reference": mismatch_reference, "config": config, "build": build,
            "positions": positions}


def test_engine_signals_and_reductions_use_lookup_and_retarget(tiny_engine):
    """Matched per-position signals/reductions stay stable through A-B-A."""
    setup = tiny_engine
    engine = setup["engine"]
    reference = setup["reference"]
    mismatch_engine = setup["mismatch"]
    mismatch_reference = setup["mismatch_reference"]
    config = setup["config"]
    positions = setup["positions"]
    nuisance = np.column_stack(
        (np.ones(engine._mask_flat_idx.size), np.linspace(0.0, 1.0, engine._mask_flat_idx.size))
    )
    nuisance_pinv = np.linalg.inv(nuisance.T @ nuisance)

    def stack_reductions(chunks, field):
        return np.concatenate([getattr(chunk, field) for chunk in chunks], axis=0)

    def q_from_reduction(raw, cross):
        return raw - np.einsum("ij,jk,ik->i", cross, nuisance_pinv, cross)

    def cache_size(function):
        size = getattr(function, "_cache_size", None)
        return None if size is None else int(size())

    def assert_reductions_equal(actual, expected):
        for field in (
            "raw",
            "cross",
            "finite",
            "signal_data_inner",
            "data_cross",
            "data_finite",
            "signal_bias_inner",
        ):
            if getattr(actual[0], field) is None:
                assert all(getattr(item, field) is None for item in expected)
                continue
            actual_value = stack_reductions(actual, field)
            expected_value = stack_reductions(expected, field)
            if field in {"finite", "data_finite"}:
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                np.testing.assert_allclose(
                    actual_value, expected_value, rtol=2e-13, atol=2e-13
                )

    signal_a = np.asarray(list(engine.signal_iterator(positions)))
    reference_a = np.asarray(list(reference.signal_iterator(positions)))
    reduction_a = list(engine.reduction_iterator(positions))
    reference_reduction_a = list(reference.reduction_iterator(positions))
    signal_cache_a = cache_size(engine._batch_signals)
    reduction_cache_a = cache_size(engine._batch_reductions)
    np.testing.assert_allclose(signal_a, reference_a, rtol=2e-13, atol=2e-13)
    config_b = copy.deepcopy(config)
    config_b["lensing"]["subhalo"]["mass"] = 3.0e8
    engine.retarget_subhalo(config_b)
    signal_b = np.asarray(list(engine.signal_iterator(positions)))
    reduction_b = list(engine.reduction_iterator(positions))
    assert cache_size(engine._batch_signals) == signal_cache_a
    assert cache_size(engine._batch_reductions) == reduction_cache_a
    config_a = copy.deepcopy(config)
    engine.retarget_subhalo(config_a)
    signal_a_again = np.asarray(list(engine.signal_iterator(positions)))
    reduction_a_again = list(engine.reduction_iterator(positions))
    assert cache_size(engine._batch_signals) == signal_cache_a
    assert cache_size(engine._batch_reductions) == reduction_cache_a
    np.testing.assert_array_equal(signal_a_again, signal_a)
    raw_a = stack_reductions(reduction_a, "raw")
    cross_a = stack_reductions(reduction_a, "cross")
    raw_a_again = stack_reductions(reduction_a_again, "raw")
    cross_a_again = stack_reductions(reduction_a_again, "cross")
    np.testing.assert_array_equal(raw_a_again, raw_a)
    np.testing.assert_array_equal(cross_a_again, cross_a)
    assert np.any(np.abs(signal_b - signal_a) > 0.0)
    raw_b = stack_reductions(reduction_b, "raw")
    cross_b = stack_reductions(reduction_b, "cross")
    assert np.any(np.abs(raw_b - raw_a) > 0.0)
    raw_reference_a = stack_reductions(reference_reduction_a, "raw")
    cross_reference_a = stack_reductions(reference_reduction_a, "cross")
    np.testing.assert_allclose(raw_a, raw_reference_a, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(cross_a, cross_reference_a, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        q_from_reduction(raw_a, cross_a),
        q_from_reduction(raw_reference_a, cross_reference_a),
        rtol=2e-13,
        atol=2e-13,
    )

    fresh_b = setup["build"](config_b)
    np.testing.assert_allclose(
        signal_b,
        np.asarray(list(fresh_b.signal_iterator(positions))),
        rtol=2e-13,
        atol=2e-13,
    )

    mismatch_signals = np.asarray(list(mismatch_engine.signal_iterator(positions)))
    mismatch_reference_signals = np.asarray(list(mismatch_reference.signal_iterator(positions)))
    mismatch_reductions = list(mismatch_engine.reduction_iterator(positions))
    mismatch_reference_reductions = list(mismatch_reference.reduction_iterator(positions))
    mismatch_signals_a = mismatch_signals.copy()
    mismatch_reductions_a = mismatch_reductions
    assert mismatch_signals.shape == (len(positions), 2, engine._mask_flat_idx.size)
    np.testing.assert_allclose(mismatch_signals, mismatch_reference_signals, rtol=2e-13, atol=2e-13)
    mismatch_raw = stack_reductions(mismatch_reductions, "raw")
    mismatch_cross = stack_reductions(mismatch_reductions, "cross")
    mismatch_inner = stack_reductions(mismatch_reductions, "signal_data_inner")
    mismatch_data_cross = stack_reductions(mismatch_reductions, "data_cross")
    mismatch_reference_raw = stack_reductions(mismatch_reference_reductions, "raw")
    mismatch_reference_cross = stack_reductions(mismatch_reference_reductions, "cross")
    mismatch_reference_inner = stack_reductions(
        mismatch_reference_reductions, "signal_data_inner"
    )
    mismatch_reference_data_cross = stack_reductions(
        mismatch_reference_reductions, "data_cross"
    )
    np.testing.assert_allclose(mismatch_raw, mismatch_reference_raw, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(mismatch_cross, mismatch_reference_cross, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(mismatch_inner, mismatch_reference_inner, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        mismatch_data_cross,
        mismatch_reference_data_cross,
        rtol=2e-13,
        atol=2e-13,
    )
    mismatch_q = q_from_reduction(mismatch_raw, mismatch_cross)
    mismatch_reference_q = q_from_reduction(mismatch_reference_raw, mismatch_reference_cross)
    np.testing.assert_allclose(mismatch_q, mismatch_reference_q, rtol=2e-13, atol=2e-13)
    mismatch_numerator = mismatch_inner - np.einsum(
        "ij,jk,ik->i", mismatch_cross, nuisance_pinv, mismatch_data_cross
    )
    mismatch_q = mismatch_numerator**2 / mismatch_q
    mismatch_reference_numerator = mismatch_reference_inner - np.einsum(
        "ij,jk,ik->i",
        mismatch_reference_cross,
        nuisance_pinv,
        mismatch_reference_data_cross,
    )
    mismatch_reference_q = mismatch_reference_numerator**2 / mismatch_reference_q
    np.testing.assert_allclose(
        mismatch_q, mismatch_reference_q, rtol=2e-13, atol=2e-13
    )
    for index, signal in enumerate(mismatch_signals):
        model, data = signal
        np.testing.assert_allclose(mismatch_raw[index], np.sum(model * model), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mismatch_inner[index], np.sum(model * data), rtol=1e-12, atol=1e-12)
        nuisance_reference = np.linspace(0.0, 1.0, model.size)
        np.testing.assert_allclose(
            mismatch_cross[index],
            np.array([np.sum(model), np.dot(model, nuisance_reference)]),
            rtol=5e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            mismatch_data_cross[index],
            np.array([np.sum(data), np.dot(data, nuisance_reference)]),
            rtol=5e-12,
            atol=1e-12,
        )

    # Retargeting must update the mismatch fit/truth pair as well.  Compare
    # both the signal arrays and every reduction field with a fresh B engine,
    # then return to A and require exact reproduction of the original state.
    mismatch_engine.retarget_subhalo(config_b)
    mismatch_signals_b = np.asarray(list(mismatch_engine.signal_iterator(positions)))
    mismatch_reductions_b = list(mismatch_engine.reduction_iterator(positions))
    fresh_mismatch_b = setup["build"](config_b, mismatch=True)
    np.testing.assert_allclose(
        mismatch_signals_b,
        np.asarray(list(fresh_mismatch_b.signal_iterator(positions))),
        rtol=2e-13,
        atol=2e-13,
    )
    assert_reductions_equal(
        mismatch_reductions_b,
        list(fresh_mismatch_b.reduction_iterator(positions)),
    )
    mismatch_engine.retarget_subhalo(config_a)
    np.testing.assert_array_equal(
        mismatch_signals_a,
        np.asarray(list(mismatch_engine.signal_iterator(positions))),
    )
    assert_reductions_equal(
        list(mismatch_engine.reduction_iterator(positions)), mismatch_reductions_a
    )
