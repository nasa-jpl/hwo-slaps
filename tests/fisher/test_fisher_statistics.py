"""Profiled linear-Gaussian residuals, priors, units and identifiability.

Independent oracles use analytic controls and augmented QR. Protocol, covariance,
threading and immutability checks retain their separate numerical protections.
"""

import pickle
from dataclasses import fields

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from hwoslaps.fisher.statistics import BankReductions, ProfileLikelihoodWorkspace, SignalBankResult, Whitener, residual_blocks

nan, inf = np.nan, np.inf
BANK_FIELDS = tuple(field.name for field in fields(SignalBankResult))


def workspace(design, precision=None):
    design = np.asarray(design, dtype=float)
    p = design.shape[1]
    return ProfileLikelihoodWorkspace(design, np.zeros(p) if precision is None else precision,
                                      [f"n{i}" for i in range(p)])


def reductions_of(signals, space, data=None, bias=None):
    """Device residual-space protocol; independent numerical oracles are below."""
    sr, sp = residual_blocks(signals, space.basis_data, space.basis_prior)
    dr, dp = (None, None) if data is None else residual_blocks(data, space.basis_data, space.basis_prior)
    br, bp = (None, None) if bias is None else residual_blocks(bias, space.basis_data, space.basis_prior)
    return BankReductions(
        raw=np.einsum("ij,ij->i", signals, signals),
        profiled=np.einsum("ij,ij->i", sr, sr) + np.einsum("ij,ij->i", sp, sp),
        finite=np.all(np.isfinite(signals), axis=1),
        signal_data_residual_inner=None if data is None else np.einsum("ij,ij->i", sr, dr) + np.einsum("ij,ij->i", sp, dp),
        data_raw=None if data is None else np.einsum("ij,ij->i", data, data),
        data_profiled=None if data is None else np.einsum("ij,ij->i", dr, dr) + np.einsum("ij,ij->i", dp, dp),
        data_finite=None if data is None else np.all(np.isfinite(data), axis=1),
        signal_bias_residual_inner=None if bias is None else sr @ br + sp @ bp)


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
BACKGROUND_KNOWN = ([[1.0], [-1.0], [0.0]], None, [[1.0, 1.0, 1.0]], [3.0])
BACKGROUND_FREE = ([[1.0, 1.0], [-1.0, 1.0], [0.0, 1.0]], None, [[1.0, 1.0, 1.0]], [0.0])


@pytest.mark.parametrize("design, precision, signals, expected, max_zero_q", [
    (*ORTHOGONAL, None), (*IN_SPAN, None), (*PRIOR, None), (*NONE, None),
    (*BACKGROUND_KNOWN, None), (*BACKGROUND_FREE, 1e-20)],
                         ids=["orthogonal-nuisance", "signal-in-nuisance-span", "one-nuisance-with-prior",
                              "no-nuisance", "constant-background-known", "constant-background-free"])
def test_profiled_information_matches_closed_forms(design, precision, signals, expected, max_zero_q):
    signals = np.asarray(signals)
    bank = workspace(design, None if precision is None else np.asarray(precision)).evaluate_bank(signals)
    if max_zero_q is not None:
        assert np.all(bank.q_asimov < max_zero_q), "free constant background must absorb the signal"
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


def test_bank_finisher_preserves_well_conditioned_closed_form_values():
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
        np.testing.assert_allclose(getattr(bank, name), np.nan_to_num(values) if name.startswith(("q_", "z_")) else values,
                                   rtol=2e-12, atol=1e-14, equal_nan=True, err_msg=name)
    assert space.nuisance_rank == 2
    assert 1.0 <= space.condition_number < 3.0


def test_reductions_finish_equal_to_dense_bank():
    rng = np.random.default_rng(11)
    design = rng.normal(size=(30, 4))
    signals, data = rng.normal(size=(6, 30)), rng.normal(size=(6, 30))
    bias = 0.1 * rng.normal(size=30)
    space = workspace(design, np.array([0.0, 2.0, 0.0, 0.5]))
    dense = space.evaluate_bank(signals, data_whitened=data, bias_whitened=bias)
    reduced = space.evaluate_reductions(reductions_of(signals, space, data, bias), bias_whitened=bias)
    assert_banks_equal(reduced, dense)
    halves = BankReductions.concatenate([reductions_of(signals[:2], space, data[:2], bias),
                                         reductions_of(signals[2:], space, data[2:], bias)])
    joined = space.evaluate_reductions(halves, bias_whitened=bias)
    for name in BANK_FIELDS:
        np.testing.assert_allclose(getattr(joined, name), getattr(dense, name), rtol=1e-12, err_msg=name)
    broken = signals.copy()
    broken[3, 7] = np.nan
    with pytest.raises(ValueError, match="signals_whitened contains non-finite"):
        space.evaluate_reductions(reductions_of(broken, space))
    broken_data = data.copy()
    broken_data[0, 0] = np.inf
    with pytest.raises(ValueError, match="data_whitened contains non-finite"):
        space.evaluate_reductions(reductions_of(signals, space, broken_data))
    with pytest.raises(ValueError, match="together"):
        space.evaluate_reductions(reductions_of(signals, space), bias_whitened=bias)


@pytest.mark.parametrize("design", [np.empty((3, 0)), np.array([[1.0], [1.0], [0.0]])], ids=["no-nuisance",
                                                                                            "absorbing-nuisance"])
def test_zero_information_amplitudes_are_unidentified_and_q_is_zero(design):
    signals = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]])
    signals = signals[:1] if design.shape[1] == 0 else signals
    data = np.ones_like(signals)
    bank = workspace(design).evaluate_bank(signals, data_whitened=data, bias_whitened=np.ones(3))
    np.testing.assert_array_equal(bank.fisher_profiled, 0.0)
    np.testing.assert_array_equal(bank.sigma_amplitude, inf)
    np.testing.assert_array_equal(bank.degradation, 0.0)
    for name in ("amplitude_hat", "amplitude_spurious"):
        assert np.all(np.isnan(getattr(bank, name))), name
    for name in ("q_mismatch", "z_mismatch", "q_spurious", "z_spurious"):
        np.testing.assert_array_equal(getattr(bank, name), 0.0)
    np.testing.assert_array_equal(bank.template_status, "zero_information")


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


def test_normalized_svd_cutoff_is_invariant_to_individual_column_units():
    signal = np.array([[0.0, 1.0]])
    small_column = workspace(np.diag([1.0, 1.0e-7]))
    assert small_column.evaluate_bank(signal).fisher_profiled[0] == 0.0
    assert small_column.nuisance_rank == 2
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


@pytest.mark.parametrize("n_nuisance", [12, 128])
def test_host_algebra_does_not_depend_on_the_caller_thread_count(n_nuisance):
    rng = np.random.default_rng(7)
    root = rng.normal(size=(64, 64)) / 8.0
    covariance = root @ root.T + np.eye(64)
    block = rng.normal(size=(64, 64))
    design = rng.normal(size=(2000, n_nuisance))
    signals, data = rng.normal(size=(256, 2000)), rng.normal(size=(256, 2000))
    bias = rng.normal(size=2000)
    runs = []
    for threads in (1, 4):
        with threadpool_limits(limits=threads):
            whitener = Whitener.from_covariance(covariance)
            space = workspace(design, np.linspace(0.0, 1.0, n_nuisance))
            bank = space.evaluate_bank(signals, data_whitened=data, bias_whitened=bias)
            with threadpool_limits(limits=1):
                reductions = reductions_of(signals, space, data, bias)
            reduced = space.evaluate_reductions(reductions, bias_whitened=bias)
            runs.append({"cholesky_factor": whitener.cholesky_factor, "whitened": whitener.apply(block),
                         "basis_data": space.basis_data,
                         **{f"bank.{name}": getattr(bank, name) for name in BANK_FIELDS},
                         **{f"reduced.{name}": getattr(reduced, name) for name in BANK_FIELDS}})
    single, threaded = runs
    for name, values in single.items():
        assert values.tobytes() == threaded[name].tobytes(), name


@pytest.mark.parametrize("design", [np.empty((5, 0)), np.ones((5, 2))], ids=["no-nuisance", "two-nuisances"])
def test_workspace_data_size_is_fixed_at_construction(design):
    space = workspace(design)
    assert space.n_data == 5
    assert space.evaluate_bank(np.ones((1, 5))).fisher_raw[0] == 5.0
    with pytest.raises(ValueError, match=r"\(n_signals >= 1, 5\)"):
        space.evaluate_bank(np.ones((2, 4)))
    with pytest.raises(ValueError, match=r"bias_whitened must have shape \(5,\)"):
        space.evaluate_bank(np.ones((1, 5)), bias_whitened=np.ones(4))


@pytest.mark.parametrize("names, precision, match", [
    (["shear_1", "shear_1"], [0.0, 0.0], "must be unique"),
    (["shear_1"], [0.0, 0.0], "must be 2 non-empty strings"),
    (["shear_1", "shear_2"], [0.0, -1.0], "must be non-negative"),
    (["shear_1", "shear_2"], [0.0], r"prior_precision must have shape \(2,\)"),
], ids=["duplicate-names", "name-count", "negative-precision", "precision-shape"])
def test_workspace_refuses_inconsistent_nuisance_inputs(names, precision, match):
    with pytest.raises(ValueError, match=match):
        ProfileLikelihoodWorkspace(np.eye(3)[:, :2], np.array(precision), names)


@pytest.mark.parametrize("profiled, message", [(-1.0, "non-negative"), (2.0, "exceeds raw"), (np.inf, "non-finite")])
def test_invalid_residual_reductions_raise(profiled, message):
    space = workspace(np.eye(2))
    with pytest.raises(ValueError, match=message):
        space.evaluate_reductions(BankReductions(raw=np.array([1.0]), profiled=np.array([profiled]),
                                                finite=np.array([True])))


def mismatched_bank():
    space = workspace([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]], np.array([0.0, 0.5]))
    return space.evaluate_bank(np.array([[1.0, 2.0, 3.0]]), data_whitened=np.array([[0.5, 1.0, 1.5]]),
                               bias_whitened=np.ones(3))


@pytest.mark.parametrize("build, names", [
    (lambda: workspace([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]], np.array([0.0, 0.5])),
     ("nuisance_whitened", "prior_precision", "basis_data", "basis_prior", "normalized_singular_values")),
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


def augmented_qr_oracle(design, precision, signals, data, bias):
    """Independent augmented QR calculation with explicit stacked prior rows."""
    p = design.shape[1]
    augmented = np.vstack((design, np.diag(np.sqrt(precision))))
    q, _ = np.linalg.qr(augmented, mode="reduced")
    def residual(values):
        padded = np.pad(values, ((0, 0), (0, p)))
        return padded - (padded @ q) @ q.T
    sr, dr, br = residual(signals), residual(data), residual(bias[None, :])[0]
    information = np.einsum("ij,ij->i", sr, sr)
    return information, np.einsum("ij,ij->i", sr, dr) / information, sr @ br / information


@pytest.mark.parametrize("gaussian", [False, True])
def test_independent_augmented_qr_and_individual_unit_changes(gaussian):
    rng = np.random.default_rng(17829)
    design = rng.normal(size=(40, 6))
    precision = np.exp(rng.normal(size=6)) if gaussian else np.zeros(6)
    signals, data, bias = rng.normal(size=(5, 40)), rng.normal(size=(5, 40)), rng.normal(size=40)
    expected = augmented_qr_oracle(design, precision, signals, data, bias)
    for scales in (np.ones(6), 10.0 ** np.array([-12, -5, 0, 3, 7, 12])):
        space = workspace(design / scales, precision / scales**2)
        actual = space.evaluate_bank(signals, data_whitened=data, bias_whitened=bias)
        for name, oracle in zip(("q_asimov", "amplitude_hat", "amplitude_spurious"), expected):
            np.testing.assert_allclose(getattr(actual, name), oracle, rtol=3e-12, atol=2e-14, err_msg=name)
        assert space.nuisance_rank == 6
        assert not space.rank_tolerance_sensitive


def test_prior_residual_is_included_and_unit_transformed():
    # s=J is absorbed with flat prior; a unit Gaussian constraint leaves F=1/2.
    for scale in (1e-12, 1.0, 1e12):
        space = workspace([[scale], [0.0]], [scale**2])
        bank = space.evaluate_bank([[1.0, 0.0]], data_whitened=[[-2.0, 0.0]], bias_whitened=[3.0, 0.0])
        np.testing.assert_allclose(bank.q_asimov, [0.5], rtol=1e-14)
        np.testing.assert_allclose(bank.amplitude_hat, [-2.0], rtol=1e-14)
        np.testing.assert_allclose(bank.amplitude_spurious, [3.0], rtol=1e-14)
        np.testing.assert_array_equal(bank.template_status, "valid")


def test_exact_and_near_span_keep_small_physical_information():
    # Subtractive normal-equation information loses this positive 1e-16 signal.
    design = np.array([[1.0, 0.0], [0.0, 1e-12], [0.0, 0.0], [0.0, 0.0]])
    signals = np.array([[1e4, -2e4, 0.0, 0.0], [1e4, -2e4, 1e-8, 0.0], [0.0, 0.0, 0.0, 0.0]])
    bank = workspace(design).evaluate_bank(signals, data_whitened=-0.7 * signals,
                                            bias_whitened=[0.0, 0.0, 2e-9, 0.0])
    assert bank.q_asimov[0] == 0.0
    assert bank.q_asimov[1] == pytest.approx(1e-16, rel=1e-14)
    assert bank.amplitude_hat[1] == pytest.approx(-0.7, rel=1e-14)
    assert bank.amplitude_spurious[1] == pytest.approx(0.2, rel=1e-14)
    np.testing.assert_array_equal(bank.template_status, ["zero_information", "valid", "zero_information"])


def test_true_degeneracy_zero_columns_and_rank_tolerance_refinement():
    space = workspace([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    assert space.nuisance_rank == 1
    np.testing.assert_allclose(space.evaluate_bank([[0.0, 3.0]]).q_asimov, [9.0])
    design = np.array([[1.0, 1.0], [0.0, 1e-12], [0.0, 0.0]])
    coarse = ProfileLikelihoodWorkspace(design, [0.0, 0.0], ["a", "b"], rcond=1e-12)
    refined = ProfileLikelihoodWorkspace(design, [0.0, 0.0], ["a", "b"], rcond=1e-13)
    assert coarse.rank_tolerance_sensitive
    assert coarse.nuisance_rank == 1 and coarse.rank_at_refined_tolerance == refined.nuisance_rank == 2


@pytest.mark.parametrize("precision,signal", [([0.0], [1e16, 10.0]), ([1e-30], [1e16, 0.0])],
                         ids=["untouched_data_residual", "weak_gaussian_prior_residual"])
def test_resolved_information_is_not_clipped_by_raw_signal_scale(precision, signal):
    space = workspace([[1.0], [0.0]], precision)
    bank = space.evaluate_bank([signal], data_whitened=[signal], bias_whitened=signal)
    np.testing.assert_allclose(bank.q_asimov, [100.0], rtol=2e-15)
    np.testing.assert_array_equal(bank.template_status, "valid")
    np.testing.assert_allclose(bank.amplitude_hat, [1.0], rtol=2e-15)
    np.testing.assert_allclose(bank.amplitude_spurious, [1.0], rtol=2e-15)
    assert bank.projection_error_bound[0] < 1e-12


def test_near_dependent_false_detection_refines_original_float_inputs_exactly():
    # Independent 100-digit exact-float oracle supplied by the adversarial review.
    a = np.array([1., 2., 3., 4., 5.])
    v = np.array([2., -1., 3., -2., 1.])
    u = np.array([3., 2., -1., 4., -2.])
    design = np.column_stack((a, a + 1e-11 * v))
    signal = 1e6 * (v + 1e-8*u)
    expected = 0.002663131800320136
    for scales in (np.ones(2), np.array([2.**-32, 2.**32])):
        space = workspace(design / scales)
        bank = space.evaluate_bank(signal[None, :], data_whitened=(0.7 * signal)[None, :], bias_whitened=-0.2 * signal)
        assert space.nuisance_rank == 2
        assert bank.q_asimov[0] == pytest.approx(expected, rel=2e-15)
        assert bank.template_status[0] == "valid"
        assert bank.q_asimov[0] < 25.0
        # Data/bias are their original rounded float inputs: the exact amplitude
        # oracle is not simply 0.7/-0.2 in such a cancellation-sensitive example.
        assert np.isfinite(bank.amplitude_hat[0]) and np.isfinite(bank.amplitude_spurious[0])
        assert bank.projection_error_bound[0] < expected * 1e-12


def test_unresolved_large_problem_reports_failed_precision_without_a_false_nondetection():
    a = np.linspace(1.0, 5.0, 10000)
    v = np.sin(np.arange(10000.0))
    design = np.column_stack((a, a + 1e-11*v))
    signal = 1e6 * (v + 1e-8*np.cos(np.arange(10000.0)))
    space = workspace(design)
    assert not space.exact_refinement_available
    bank = space.evaluate_bank(signal[None, :])
    assert bank.template_status[0] == "failed_numeric"
    assert np.isnan(bank.q_asimov[0])
    assert np.isfinite(bank.projection_error_bound[0]) and bank.projection_error_bound[0] > 0.0


def test_unresolved_streamed_reductions_cannot_publish_the_near_span_false_detection():
    a = np.array([1., 2., 3., 4., 5.])
    v = np.array([2., -1., 3., -2., 1.])
    u = np.array([3., 2., -1., 4., -2.])
    space = workspace(np.column_stack((a, a + 1e-11*v)))
    signal = (1e6*(v + 1e-8*u))[None, :]
    reduced = reductions_of(signal, space)
    assert reduced.profiled[0] > 25.0  # Uncertified floating basis gives a false detection.
    bank = space.evaluate_reductions(reduced)
    assert bank.template_status[0] == "failed_numeric"
    assert np.isnan(bank.q_asimov[0])
    assert bank.projection_error_bound[0] > reduced.profiled[0]


def test_nonzero_signal_norm_underflow_is_an_explicit_failure():
    space = workspace(np.empty((2, 0)))
    with pytest.raises(ValueError, match="signal information underflow"):
        space.evaluate_bank([[1e-200, 0.0]])


def test_positive_unrepresentable_refined_information_does_not_become_zero_information():
    space = workspace([[1.0], [0.0]], [1e-320])
    bank = space.evaluate_bank([[1e-100, 0.0]])
    assert bank.template_status[0] == "failed_numeric"
    assert np.isnan(bank.q_asimov[0])
    assert bank.projection_error_bound[0] > 0.0
