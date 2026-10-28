"""Residual-space reductions on CPU/CUDA against an independent augmented QR."""

import numpy as np
import pytest

from hwoslaps.fisher.statistics import BankReductions, ProfileLikelihoodWorkspace, residual_blocks


def check_jax_reductions():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(18271)
    design = rng.normal(size=(31, 5))
    signals, data, bias = rng.normal(size=(7, 31)), rng.normal(size=(7, 31)), rng.normal(size=31)
    for precision in (np.zeros(5), np.array([0.0, 1.5, 0.0, 0.3, 2.0])):
        augmented = np.vstack((design, np.diag(np.sqrt(precision))))
        q, _ = np.linalg.qr(augmented, mode="reduced")
        def oracle_residual(values):
            values = np.pad(values, ((0, 0), (0, 5)))
            return values - (values @ q) @ q.T
        sr, dr, br = oracle_residual(signals), oracle_residual(data), oracle_residual(bias[None, :])[0]
        expected_f = np.sum(sr * sr, axis=1)
        expected_a = np.sum(sr * dr, axis=1) / expected_f
        expected_b = sr @ br / expected_f
        for scales in (np.ones(5), 10.0 ** np.array([-12, -6, 0, 6, 12])):
            space = ProfileLikelihoodWorkspace(design / scales, precision / scales**2, tuple(f"p{i}" for i in range(5)))
            qd, qp, b = jnp.asarray(space.basis_data), jnp.asarray(space.basis_prior), jnp.asarray(bias)
            @jax.jit
            def reduce(s, d):
                rs, ps = residual_blocks(s, qd, qp)
                rd, pd = residual_blocks(d, qd, qp)
                rb, pb = residual_blocks(b, qd, qp)
                return dict(raw=jnp.sum(s*s, axis=1), profiled=jnp.sum(rs*rs, axis=1)+jnp.sum(ps*ps, axis=1),
                            finite=jnp.all(jnp.isfinite(s), axis=1), data_finite=jnp.all(jnp.isfinite(d), axis=1),
                            signal_data_residual_inner=jnp.sum(rs*rd, axis=1)+jnp.sum(ps*pd, axis=1),
                            signal_bias_residual_inner=rs@rb+ps@pb)
            reduced = BankReductions(**{name: np.asarray(value) for name, value in reduce(
                jnp.asarray(signals), jnp.asarray(data)).items()})
            actual = space.evaluate_reductions(reduced, bias_whitened=bias)
            for name, oracle in (("q_asimov", expected_f), ("amplitude_hat", expected_a),
                                 ("amplitude_spurious", expected_b)):
                np.testing.assert_allclose(getattr(actual, name), oracle, rtol=5e-12, atol=1e-14)
            np.testing.assert_allclose(actual.z_mismatch, expected_a*np.sqrt(expected_f), rtol=5e-12, atol=1e-14)
            np.testing.assert_allclose(actual.q_spurious, expected_b**2*expected_f, rtol=5e-12, atol=1e-14)


@pytest.mark.backend
def test_jax_residual_reductions_against_augmented_qr():
    check_jax_reductions()


@pytest.mark.backend
@pytest.mark.xtx_gpu
def test_cuda_residual_reductions_against_augmented_qr():
    check_jax_reductions()
