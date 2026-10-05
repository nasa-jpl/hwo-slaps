"""Profiled linear-Gaussian algebra of fisher.statistics.

Oracles: closed forms of the Schur complement written out here, planted
amplitudes recovered exactly, explicit inverse-covariance algebra, hand values
of the pseudo-inverse cutoff, and literals printed by the 974cee9
``modeling.fisher_core`` workspace on a hand bank (run-directory script
``probes/sta_literals.py``, W1-STA ledger).
"""

import pickle
from dataclasses import fields

import numpy as np
import pytest

from hwoslaps.fisher.statistics import BankReductions, ProfileLikelihoodWorkspace, SignalBankResult, Whitener

nan, inf = np.nan, np.inf
BANK_FIELDS = tuple(field.name for field in fields(SignalBankResult))


def workspace(design, precision=None):
    design = np.asarray(design, dtype=float)
    p = design.shape[1]
    return ProfileLikelihoodWorkspace(design, np.zeros(p) if precision is None else precision,
                                      [f"n{i}" for i in range(p)])


def reductions_of(signals, design, data=None, bias=None):
    """The reductions an accelerator returns for whitened vectors it never sends to the host."""
    return BankReductions(
        raw=np.einsum("ij,ij->i", signals, signals), cross=signals @ design,
        finite=np.all(np.isfinite(signals), axis=1),
        signal_data_inner=None if data is None else np.einsum("ij,ij->i", signals, data),
        data_cross=None if data is None else data @ design,
        data_finite=None if data is None else np.all(np.isfinite(data), axis=1),
        signal_bias_inner=None if bias is None else signals @ bias)


def assert_banks_equal(actual, expected):
    for name in BANK_FIELDS:
        left, right = getattr(actual, name), getattr(expected, name)
        if right is None:
            assert left is None, name
        else:
            np.testing.assert_array_equal(left, right, err_msg=name)


ORTHOGONAL = ([[1.0], [0.0], [0.0], [0.0]], None,
              [[0.0, 1.0, 2.0, 0.0], [0.0, 0.0, -3.0, 0.5]], [5.0, 9.25])
IN_SPAN = ([[1.0, 2.0], [0.5, -1.0], [0.0, 1.0]], None,
           [[2.0, 1.0, 0.0], [3.0, -0.5, 1.0]], [0.0, 0.0])
PRIOR = ([[1.0], [2.0], [-1.0], [0.5]], [0.75],
         [[1.0, 0.0, 0.0, 0.0], [0.5, -1.0, 2.0, 1.0], [0.0, 2.0, 0.0, 0.0]],
         [1.0 - 1.0 / 7.0, 6.25 - 9.0 / 7.0, 4.0 - 16.0 / 7.0])
NONE = (np.empty((3, 0)), None, [[1.0, -2.0, 0.5], [0.0, 3.0, 4.0]], [5.25, 25.0])


@pytest.mark.parametrize("design, precision, signals, expected", [ORTHOGONAL, IN_SPAN, PRIOR, NONE],
                         ids=["orthogonal-nuisance", "signal-in-nuisance-span", "one-nuisance-with-prior",
                              "no-nuisance"])
def test_profiled_information_matches_closed_forms(design, precision, signals, expected):
    signals = np.asarray(signals)
    bank = workspace(design, None if precision is None else np.asarray(precision)).evaluate_bank(signals)
    raw = np.sum(signals * signals, axis=1)
    np.testing.assert_allclose(bank.fisher_raw, raw, rtol=1e-15)
    np.testing.assert_allclose(bank.fisher_profiled, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(bank.q_asimov, bank.fisher_profiled)
    positive = np.asarray(expected) > 1e-9
    np.testing.assert_allclose(bank.sigma_amplitude[positive], 1.0 / np.sqrt(np.asarray(expected)[positive]),
                               rtol=1e-12)
    np.testing.assert_allclose(bank.degradation, np.asarray(expected) / raw, rtol=1e-12, atol=1e-12)
    for name in ("amplitude_hat", "q_mismatch", "z_mismatch", "amplitude_spurious", "q_spurious", "z_spurious"):
        assert getattr(bank, name) is None


def planted_bank(n_nuisance):
    """Design, two signals whose profiled residuals are 2 q_a and 3 q_b, orthonormal to each other and to J."""
    rng = np.random.default_rng(5)
    basis, _ = np.linalg.qr(rng.normal(size=(9, n_nuisance + 2)))
    design = basis[:, :n_nuisance] @ np.array([[1.5, 0.3, 0.0], [0.0, 0.8, -0.2], [0.0, 0.0, 2.0]])[
        :n_nuisance, :n_nuisance]
    in_span = design @ rng.normal(size=(n_nuisance, 2))
    signals = np.stack((2.0 * basis[:, n_nuisance] + in_span[:, 0], 3.0 * basis[:, n_nuisance + 1] + in_span[:, 1]))
    return design, signals, design @ rng.normal(size=(n_nuisance, 2))


@pytest.mark.parametrize("n_nuisance", [3, 0])
def test_mismatch_and_spurious_amplitudes_recover_planted_values(n_nuisance):
    design, signals, nuisance_shift = planted_bank(n_nuisance)
    amplitudes, betas = np.array([0.7, -1.3]), np.array([0.05, -0.02])
    data = amplitudes[:, None] * signals + nuisance_shift.T
    bias = betas[0] * signals[0] + betas[1] * signals[1] + nuisance_shift[:, 0]
    bank = workspace(design).evaluate_bank(signals, data_whitened=data, bias_whitened=bias)
    np.testing.assert_allclose(bank.fisher_profiled, [4.0, 9.0], rtol=1e-12)
    np.testing.assert_allclose(bank.amplitude_hat, amplitudes, rtol=1e-12)
    np.testing.assert_allclose(bank.z_mismatch, amplitudes * [2.0, 3.0], rtol=1e-12)
    np.testing.assert_allclose(bank.q_mismatch, (amplitudes * [2.0, 3.0]) ** 2, rtol=1e-12)
    np.testing.assert_allclose(bank.amplitude_spurious, betas, rtol=1e-10)
    np.testing.assert_allclose(bank.z_spurious, np.abs(betas) * [2.0, 3.0], rtol=1e-10)
    np.testing.assert_allclose(bank.q_spurious, (betas * [2.0, 3.0]) ** 2, rtol=1e-10)


def test_bank_finisher_reproduces_974cee9_bits():
    design = np.array([[1.0, 0.5], [0.25, -1.0], [2.0, 0.75], [0.75, 0.0]])
    signals = np.array([[1.0, 2.0, -1.0, 0.5], [0.0, 0.0, 0.0, 0.0], [0.3, -0.7, 1.1, 2.0],
                        [-0.4, 0.9, 0.6, -1.2]])
    data = np.array([[0.9, 1.5, -0.6, 0.2], [0.1, -0.1, 0.2, 0.0], [0.2, -0.5, 1.4, 1.7], [0.1, 0.8, 0.9, -1.0]])
    bias = np.array([0.05, -0.02, 0.1, 0.03])
    space = workspace(design, np.array([0.0, 0.3]))
    bank = space.evaluate_bank(signals, data_whitened=data, bias_whitened=bias)
    expected = {
        "fisher_raw": [6.25, 0.0, 5.790000000000001, 2.77],
        "fisher_profiled": [3.129340124003542, 0.0, 3.03898937112489, 2.464574844995571],
        "sigma_amplitude": [0.5652930104699595, inf, 0.5736346932248904, 0.6369846901680947],
        "q_asimov": [3.129340124003542, 0.0, 3.03898937112489, 2.464574844995571],
        "degradation": [0.5006944198405667, 0.0, 0.524868630591518, 0.8897382111897368],
        "amplitude_hat": [0.7641058299898813, nan, 0.7953287346995158, 0.8815042560570421],
        "q_mismatch": [1.8270893881043693, nan, 1.922306029497982, 1.9150972756546962],
        "z_mismatch": [1.3516987046322007, nan, 1.3864725130697622, 1.3838703969861832],
        "amplitude_spurious": [-0.005494866297294817, nan, 0.0018837260883719862, -0.0015365650735578204],
        "q_spurious": [9.448590510410367e-05, nan, 1.0783622747348909e-05, 5.8189406306434576e-06],
        "z_spurious": [0.009720386057359228, nan, 0.0032838426800547113, 0.0024122480450076973],
    }
    for name, values in expected.items():
        assert getattr(bank, name).tobytes() == np.array(values).tobytes(), name
    assert space.nuisance_rank == 2
    assert space.condition_number == 4.56873856363062


def test_reductions_finish_equal_to_dense_bank():
    rng = np.random.default_rng(11)
    design = rng.normal(size=(30, 4))
    signals, data = rng.normal(size=(6, 30)), rng.normal(size=(6, 30))
    bias = 0.1 * rng.normal(size=30)
    space = workspace(design, np.array([0.0, 2.0, 0.0, 0.5]))
    dense = space.evaluate_bank(signals, data_whitened=data, bias_whitened=bias)
    reduced = space.evaluate_reductions(reductions_of(signals, design, data, bias), bias_whitened=bias)
    assert_banks_equal(reduced, dense)
    halves = BankReductions.concatenate([reductions_of(signals[:2], design, data[:2], bias),
                                         reductions_of(signals[2:], design, data[2:], bias)])
    joined = space.evaluate_reductions(halves, bias_whitened=bias)
    for name in BANK_FIELDS:
        np.testing.assert_allclose(getattr(joined, name), getattr(dense, name), rtol=1e-12, err_msg=name)
    broken = signals.copy()
    broken[3, 7] = np.nan
    with pytest.raises(ValueError, match="signals_whitened contains non-finite"):
        space.evaluate_reductions(reductions_of(broken, design))
    broken_data = data.copy()
    broken_data[0, 0] = np.inf
    with pytest.raises(ValueError, match="data_whitened contains non-finite"):
        space.evaluate_reductions(reductions_of(signals, design, broken_data))
    with pytest.raises(ValueError, match="together"):
        space.evaluate_reductions(reductions_of(signals, design), bias_whitened=bias)


@pytest.mark.parametrize("design", [np.empty((3, 0)), np.array([[1.0], [1.0], [0.0]])], ids=["no-nuisance",
                                                                                            "absorbing-nuisance"])
def test_zero_information_statistics_are_undefined(design):
    signals = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    signals = signals[:1] if design.shape[1] == 0 else signals
    data = np.ones_like(signals)
    bank = workspace(design).evaluate_bank(signals, data_whitened=data, bias_whitened=np.ones(3))
    np.testing.assert_array_equal(bank.fisher_profiled, 0.0)
    np.testing.assert_array_equal(bank.sigma_amplitude, inf)
    np.testing.assert_array_equal(bank.degradation, 0.0)
    for name in ("amplitude_hat", "q_mismatch", "z_mismatch", "amplitude_spurious", "q_spurious", "z_spurious"):
        assert np.all(np.isnan(getattr(bank, name))), name


def test_pseudoinverse_is_invariant_to_duplicate_columns_and_uniform_rescaling():
    for amplitude in (1.0e-6, 1.1e-6):
        bank = workspace([[amplitude], [0.0]]).evaluate_bank(np.array([[1.0, 0.0]]))
        assert bank.fisher_profiled[0] == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(123)
    signals = rng.normal(size=(3, 8))
    design = rng.normal(size=(8, 3))
    precision = np.array([0.2, 1.0, 4.0])
    reference = workspace(design, precision).evaluate_bank(signals).fisher_profiled
    for exponent in range(-12, 13, 3):
        scale = 10.0**exponent
        scaled = workspace(design / scale, precision / scale**2).evaluate_bank(signals).fisher_profiled
        np.testing.assert_allclose(scaled, reference, rtol=1e-10, err_msg=f"scale 1e{exponent}")
    flat = workspace(design).evaluate_bank(signals).fisher_profiled
    duplicated = workspace(np.column_stack((design, design[:, 1])))
    np.testing.assert_allclose(duplicated.evaluate_bank(signals).fisher_profiled, flat, rtol=1e-10)
    assert duplicated.nuisance_rank == 3


def test_gram_cutoff_depends_on_individual_column_units():
    signal = np.array([[0.0, 1.0]])
    small_column = workspace(np.diag([1.0, 1.0e-7]))
    assert small_column.evaluate_bank(signal).fisher_profiled[0] == 1.0
    assert small_column.nuisance_rank == 1
    unit_columns = workspace(np.eye(2))
    assert unit_columns.evaluate_bank(signal).fisher_profiled[0] == 0.0
    assert unit_columns.nuisance_rank == 2


def test_profiled_information_depends_only_on_the_nuisance_span():
    rng = np.random.default_rng(42)
    signals = rng.normal(size=(2, 7))
    design = rng.normal(size=(7, 3))
    transform = np.array([[2.0, -1.0, 0.3], [0.0, 1.5, 0.2], [0.0, 0.0, 0.7]])
    np.testing.assert_allclose(workspace(design @ transform).evaluate_bank(signals).fisher_profiled,
                               workspace(design).evaluate_bank(signals).fisher_profiled, rtol=1e-12)


def test_dense_whitening_reproduces_the_inverse_covariance_metric():
    sigma = np.array([0.5, 2.0, 1.25])
    values = np.array([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]])
    np.testing.assert_allclose(Whitener.from_covariance(np.diag(sigma**2)).apply(values), values / sigma[:, None],
                               rtol=1e-15)
    covariance = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]])
    whitener = Whitener.from_covariance(covariance)
    signal = np.array([0.5, -1.2, 2.0])
    design = np.array([[1.0, 0.2], [0.0, 1.0], [0.3, -0.4]])
    whitened = whitener.apply(values)
    np.testing.assert_allclose(whitened.T @ whitened, values.T @ np.linalg.solve(covariance, values), rtol=1e-12)
    precision_signal = np.linalg.solve(covariance, signal)
    raw = signal @ precision_signal
    cross = design.T @ precision_signal
    profiled = raw - cross @ np.linalg.solve(design.T @ np.linalg.solve(covariance, design), cross)
    bank = workspace(whitener.apply(design)).evaluate_bank(whitener.apply(signal)[None, :])
    np.testing.assert_allclose(bank.fisher_raw, [raw], rtol=1e-12)
    np.testing.assert_allclose(bank.fisher_profiled, [profiled], rtol=1e-12)
    with pytest.raises(ValueError, match="positive definite"):
        Whitener.from_covariance(np.array([[1.0, 2.0], [2.0, 1.0]]))


@pytest.mark.parametrize("design", [np.empty((5, 0)), np.ones((5, 2))], ids=["no-nuisance", "two-nuisances"])
def test_workspace_data_size_is_fixed_at_construction(design):
    space = workspace(design)
    assert space.n_data == 5
    assert space.evaluate_bank(np.ones((1, 5))).fisher_raw[0] == 5.0
    with pytest.raises(ValueError, match=r"\(n_signals >= 1, 5\)"):
        space.evaluate_bank(np.ones((2, 4)))
    with pytest.raises(ValueError, match=r"bias_whitened must have shape \(5,\)"):
        space.evaluate_bank(np.ones((1, 5)), bias_whitened=np.ones(4))


def test_significantly_negative_profiled_information_raises():
    space = workspace(np.eye(2))
    with pytest.raises(ValueError, match="significantly negative"):
        space.evaluate_reductions(BankReductions(raw=np.array([1.0]), cross=np.array([[2.0, 0.0]]),
                                                 finite=np.array([True])))
    rounding = space.evaluate_reductions(BankReductions(raw=np.array([1.0]), cross=np.array([[1.0 + 1e-12, 0.0]]),
                                                        finite=np.array([True])))
    assert rounding.fisher_profiled[0] == 0.0


def mismatched_bank():
    space = workspace([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]], np.array([0.0, 0.5]))
    return space.evaluate_bank(np.array([[1.0, 2.0, 3.0]]), data_whitened=np.array([[0.5, 1.0, 1.5]]),
                               bias_whitened=np.ones(3))


@pytest.mark.parametrize("build, names", [
    (lambda: workspace([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]], np.array([0.0, 0.5])),
     ("nuisance_whitened", "normal_pinv")),
    (mismatched_bank, BANK_FIELDS),
    (lambda: Whitener.from_sigma([0.5, 2.0]), ("sigma",)),
    (lambda: Whitener.from_covariance([[4.0, 1.0], [1.0, 3.0]]), ("cholesky_factor",)),
], ids=["workspace", "bank", "diagonal-whitener", "dense-whitener"])
def test_stored_arrays_stay_read_only_after_pickling(build, names):
    stored = build()
    restored = pickle.loads(pickle.dumps(stored))
    for name in names:
        np.testing.assert_array_equal(getattr(restored, name), getattr(stored, name), err_msg=name)
        for copy in (stored, restored):
            with pytest.raises(ValueError, match="read-only"):
                getattr(copy, name).flat[0] = 1.0


@pytest.mark.parametrize("field", ["signals", "data", "bias", "design"])
def test_non_finite_whitened_inputs_raise(field):
    arrays = {"signals": np.ones((2, 3)), "data": np.ones((2, 3)), "bias": np.ones(3), "design": np.ones((3, 1))}
    arrays[field] = arrays[field].copy()
    arrays[field].flat[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        workspace(arrays["design"]).evaluate_bank(arrays["signals"], data_whitened=arrays["data"],
                                                  bias_whitened=arrays["bias"])
