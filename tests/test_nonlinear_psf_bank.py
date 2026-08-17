"""Tests for prior-sampled nonlinear PSF nuisance banks."""

from __future__ import annotations

import contextlib
import copy
from dataclasses import fields, replace
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("autolens")
pytest.importorskip("hcipy")

from hwoslaps.config.validation import validate_or_raise, validate_psf_config
from hwoslaps.modeling.nonlinear.autolens_runner import (
    _array_hash,
    analysis_key_from,
)
from hwoslaps.modeling.nonlinear.dataset_builder import (
    imaging_from_observation,
)
from hwoslaps.modeling.nonlinear.likelihood_metrics import (
    profile_likelihood_ratio,
)
from hwoslaps.modeling.nonlinear.output_schema import (
    NonlinearCaseResult,
    NonlinearFitSummary,
)
from hwoslaps.modeling.nonlinear.psf_bank import (
    PsfBankCandidateFit,
    _anchor_diagnostic,
    _kernel_sha256,
    _resolve_prior_table_path,
    build_psf_bank,
    combine_psf_bank_fits,
    load_psf_bank_npz,
    run_psf_bank_case,
    save_psf_bank_npz,
)
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial
from hwoslaps.provenance import config_hash
from hwoslaps.psf.generator import generate_psf_system


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fit(
    label,
    log_l_smooth,
    log_l_subhalo,
    logz_smooth,
    logz_subhalo,
    success=True,
    amplitude=None,
):
    """Build one compact candidate-fit input."""
    return PsfBankCandidateFit(
        label=label,
        amplitude_rms_nm=amplitude,
        log_l_smooth=log_l_smooth,
        log_l_subhalo=log_l_subhalo,
        log_evidence_smooth=logz_smooth,
        log_evidence_subhalo=logz_subhalo,
        success=success,
    )


def test_combination_matches_manual_algebra_and_independent_profiles():
    """Match hand algebra while profiling the two hypotheses separately."""
    candidates = [
        _fit("a", -10.0, -12.0, -15.0, -14.0),
        _fit("b", -13.0, -7.0, -16.0, -10.0),
        _fit("c", -11.0, -9.0, -18.0, -13.0),
    ]

    summary = combine_psf_bank_fits(candidates)
    log_prior = -math.log(3.0)
    expected_smooth = math.log(sum(math.exp(value + log_prior) for value in (-15, -16, -18)))
    expected_subhalo = math.log(sum(math.exp(value + log_prior) for value in (-14, -10, -13)))

    assert summary.best_smooth_label == "a"
    assert summary.best_subhalo_label == "b"
    assert summary.log_l_smooth_profile == -10.0
    assert summary.log_l_subhalo_profile == -7.0
    assert summary.q_fit_psf_profile == 6.0
    assert summary.log_evidence_smooth_psf_marg == pytest.approx(expected_smooth)
    assert summary.log_evidence_subhalo_psf_marg == pytest.approx(expected_subhalo)
    assert summary.delta_log_evidence_psf_marg == pytest.approx(
        expected_subhalo - expected_smooth
    )
    assert summary.n_success == 3
    assert summary.n_evidence == 3
    assert summary.signed_q_fit_psf_profile == 6.0
    assert summary.censored is False
    assert summary.lost_evidence_prior_mass_fraction == 0.0
    assert combine_psf_bank_fits(candidates, allow_censored=True) == summary


@pytest.mark.parametrize(
    "subhalo_log_l,subhalo_logz,q_expected,q_flag,evidence_flag",
    [
        (-6.0, -5.0, 8.0, False, False),
        (-5.0, -4.0, 10.0, True, False),
        (-4.0, -3.9, 12.0, True, True),
        (-12.0, -20.0, 0.0, False, False),
    ],
)
def test_combination_clamp_and_detection_thresholds(
    subhalo_log_l,
    subhalo_logz,
    q_expected,
    q_flag,
    evidence_flag,
):
    """Clamp q and apply inclusive-q and strict-evidence thresholds."""
    summary = combine_psf_bank_fits([
        _fit("draw000", -10.0, subhalo_log_l, -9.0, subhalo_logz)
    ])

    assert summary.q_fit_psf_profile == q_expected
    assert summary.detected_fit_scdd_psf_profile is q_flag
    assert summary.detected_evidence_psf_marg is evidence_flag


def test_signed_profile_statistic_preserves_negative_contrast():
    """Keep the exact negative signed statistic while clipping q at zero."""
    summary = combine_psf_bank_fits([
        _fit("a", -5.0, -8.0, -6.0, -7.0),
        _fit("b", -5.5, -9.0, -6.5, -7.5),
    ])

    assert summary.q_fit_psf_profile == 0.0
    assert summary.signed_q_fit_psf_profile == -6.0


def test_combination_is_order_invariant_with_lexical_exact_ties():
    """Use lexical labels to break exact profile ties under shuffling."""
    candidates = [
        _fit("zeta", -5.0, -3.0, -8.0, -6.0),
        _fit("alpha", -5.0, -3.0, -8.0, -6.0),
        _fit("middle", -7.0, -4.0, -9.0, -7.0),
    ]

    forward = combine_psf_bank_fits(candidates)
    shuffled = combine_psf_bank_fits([candidates[2], candidates[0], candidates[1]])

    assert forward == shuffled
    assert forward.best_smooth_label == "alpha"
    assert forward.best_subhalo_label == "alpha"


def test_profile_q_is_bounded_by_paired_candidate_q_and_differs():
    """Restore the SPIE profile-q upper-bound regression."""
    candidates = [
        _fit("large-pair", 0.0, 10.0, 0.0, 0.0),
        _fit("profile", 9.0, 11.0, 0.0, 0.0),
    ]

    summary = combine_psf_bank_fits(candidates)
    paired_q = max(
        max(0.0, 2.0*(item.log_l_subhalo - item.log_l_smooth))
        for item in candidates
    )

    assert summary.q_fit_psf_profile == 4.0
    assert paired_q == 20.0
    assert summary.q_fit_psf_profile <= paired_q
    assert summary.log_l_smooth_profile >= max(item.log_l_smooth for item in candidates)
    assert summary.log_l_subhalo_profile >= max(item.log_l_subhalo for item in candidates)


def test_combination_rejects_empty_bank():
    """Reject an empty marginalization set."""
    with pytest.raises(ValueError, match="at least one candidate"):
        combine_psf_bank_fits([])


def test_paired_evidence_excludes_asymmetric_missingness_counterexample():
    """Use only the shared finite evidence pair in both log-sum-exps."""
    candidates = [
        _fit("shared", -2.0, -1.0, 0.0, 0.0),
        _fit("smooth-only", -2.0, -1.0, 100.0, None),
        _fit("subhalo-only", -2.0, -1.0, None, 1000.0),
    ]

    with pytest.raises(ValueError, match="allow_censored"):
        combine_psf_bank_fits(candidates)
    summary = combine_psf_bank_fits(candidates, allow_censored=True)

    assert summary.n_success == 3
    assert summary.n_evidence == 1
    assert summary.log_evidence_smooth_psf_marg == pytest.approx(0.0)
    assert summary.log_evidence_subhalo_psf_marg == pytest.approx(0.0)
    assert summary.delta_log_evidence_psf_marg == pytest.approx(0.0)
    assert summary.censored is True
    assert summary.lost_evidence_prior_mass_fraction == pytest.approx(
        2.0 / 3.0
    )


def test_nonfinite_and_missing_values_follow_paired_set_semantics():
    """Treat NaN, infinities, and missing log likelihoods as unavailable."""
    candidates = [
        _fit("usable", -4.0, -3.0, -8.0, -7.0),
        _fit("nan-evidence", -5.0, -2.0, float("nan"), -6.0),
        _fit("inf-evidence", -5.0, -2.0, -6.0, float("inf")),
        _fit("missing-logl", None, -1.0, -5.0, -4.0),
        _fit("inf-logl", -5.0, -float("inf"), -5.0, -4.0),
    ]

    with pytest.raises(ValueError, match="allow_censored"):
        combine_psf_bank_fits(candidates)
    summary = combine_psf_bank_fits(candidates, allow_censored=True)

    assert summary.n_success == 3
    assert summary.n_evidence == 1
    assert summary.delta_log_evidence_psf_marg == pytest.approx(1.0)
    assert summary.censored is True
    assert summary.lost_evidence_prior_mass_fraction == pytest.approx(0.8)


def test_all_missing_and_all_failed_banks_return_none_statistics():
    """Return null censored summaries with exact paired-set counts."""
    missing_candidates = [
        _fit("a", None, -1.0, 0.0, 1.0),
        _fit("b", -2.0, None, 0.0, 1.0),
    ]
    failed_candidates = [
        _fit("a", -2.0, -1.0, 0.0, 1.0, success=False),
    ]

    for candidates in (missing_candidates, failed_candidates):
        with pytest.raises(ValueError, match="allow_censored"):
            combine_psf_bank_fits(candidates)
    missing = combine_psf_bank_fits(missing_candidates, allow_censored=True)
    failed = combine_psf_bank_fits(failed_candidates, allow_censored=True)

    for summary, total in ((missing, 2), (failed, 1)):
        assert summary.n_candidates == total
        assert summary.n_success == 0
        assert summary.n_evidence == 0
        assert summary.log_l_smooth_profile is None
        assert summary.log_l_subhalo_profile is None
        assert summary.q_fit_psf_profile is None
        assert summary.signed_q_fit_psf_profile is None
        assert summary.log_evidence_smooth_psf_marg is None
        assert summary.log_evidence_subhalo_psf_marg is None
        assert summary.delta_log_evidence_psf_marg is None
        assert summary.ess_evidence_smooth is None
        assert summary.ess_evidence_subhalo is None
        assert summary.detected_fit_scdd_psf_profile is None
        assert summary.detected_evidence_psf_marg is None
        assert summary.censored is True
        assert summary.lost_evidence_prior_mass_fraction == 1.0


def test_censored_log_prior_renormalizes_over_usable_set():
    """Normalize the censored evidence prior over usable candidates only."""
    candidates = [
        _fit("usable", -2.0, -1.0, 7.0, 9.0),
        _fit("missing", -3.0, -2.0, None, None),
        _fit("failed", -3.0, -2.0, 30.0, 40.0, success=False),
    ]

    with pytest.raises(ValueError, match="allow_censored"):
        combine_psf_bank_fits(candidates)
    summary = combine_psf_bank_fits(candidates, allow_censored=True)

    assert summary.log_evidence_smooth_psf_marg == pytest.approx(7.0)
    assert summary.log_evidence_subhalo_psf_marg == pytest.approx(9.0)
    assert summary.censored is True
    assert summary.lost_evidence_prior_mass_fraction == pytest.approx(
        2.0 / 3.0
    )


def test_censored_two_usable_candidates_match_restricted_prior_algebra():
    """Match hand algebra for a censored bank with two usable candidates."""
    candidates = [
        _fit("a", -10.0, -12.0, -15.0, -14.0),
        _fit("b", -13.0, -7.0, -16.0, -10.0),
        _fit("c", -11.0, -9.0, None, None),
    ]

    summary = combine_psf_bank_fits(candidates, allow_censored=True)
    log_prior = -math.log(2.0)
    expected_smooth = math.log(
        sum(math.exp(value + log_prior) for value in (-15, -16))
    )
    expected_subhalo = math.log(
        sum(math.exp(value + log_prior) for value in (-14, -10))
    )

    assert summary.n_success == 3
    assert summary.n_evidence == 2
    assert summary.log_l_smooth_profile == -10.0
    assert summary.log_l_subhalo_profile == -7.0
    assert summary.q_fit_psf_profile == 6.0
    assert summary.signed_q_fit_psf_profile == 6.0
    assert summary.log_evidence_smooth_psf_marg == pytest.approx(
        expected_smooth
    )
    assert summary.log_evidence_subhalo_psf_marg == pytest.approx(
        expected_subhalo
    )
    assert summary.censored is True
    assert summary.lost_evidence_prior_mass_fraction == pytest.approx(
        1.0 / 3.0
    )


def test_fail_closed_names_counts_and_offending_labels():
    """Name every count and offending label in the fail-closed error."""
    nonfinite = [
        _fit("good", -2.0, -1.0, -4.0, -3.0),
        _fit("bad", float("nan"), -1.0, -4.0, -3.0),
    ]
    with pytest.raises(ValueError) as nonfinite_error:
        combine_psf_bank_fits(nonfinite)
    message = str(nonfinite_error.value)
    assert "2 declared candidates" in message
    assert "1 likelihood-usable" in message
    assert "1 evidence-usable" in message
    assert "['bad']" in message
    assert "allow_censored=True" in message

    missing_evidence = [
        _fit("good", -2.0, -1.0, -4.0, -3.0),
        _fit("noz", -2.0, -1.0, None, -3.0),
    ]
    with pytest.raises(ValueError) as missing_error:
        combine_psf_bank_fits(missing_evidence)
    message = str(missing_error.value)
    assert "2 declared candidates" in message
    assert "2 likelihood-usable" in message
    assert "1 evidence-usable" in message
    assert "['noz']" in message
    assert "allow_censored=True" in message


def test_combination_rejects_non_boolean_allow_censored():
    """Reject non-boolean censoring flags loudly."""
    candidates = [_fit("a", -2.0, -1.0, -4.0, -3.0)]

    for value in (1, "true", None):
        with pytest.raises(ValueError, match="allow_censored must be"):
            combine_psf_bank_fits(candidates, allow_censored=value)


def test_evidence_effective_sample_size_contract():
    """Match equal, singleton, dominant, and hand-computed ESS values."""
    equal = combine_psf_bank_fits([
        _fit(str(index), 0.0, 0.0, 5.0, 5.0) for index in range(3)
    ])
    singleton = combine_psf_bank_fits(
        [
            _fit("one", 0.0, 0.0, 5.0, 5.0),
            _fit("missing", 0.0, 0.0, None, None),
        ],
        allow_censored=True,
    )
    dominant = combine_psf_bank_fits([
        _fit("high", 0.0, 0.0, 100.0, 100.0),
        _fit("low", 0.0, 0.0, 0.0, 0.0),
    ])
    hand = combine_psf_bank_fits([
        _fit("a", 0.0, 0.0, math.log(1.0), math.log(2.0)),
        _fit("b", 0.0, 0.0, math.log(2.0), math.log(3.0)),
        _fit("c", 0.0, 0.0, math.log(3.0), math.log(5.0)),
    ])

    assert equal.ess_evidence_smooth == pytest.approx(3.0)
    assert equal.ess_evidence_subhalo == pytest.approx(3.0)
    assert singleton.ess_evidence_smooth == pytest.approx(1.0)
    assert dominant.ess_evidence_smooth == pytest.approx(1.0)
    assert hand.ess_evidence_smooth == pytest.approx(36.0 / 14.0)
    assert hand.ess_evidence_subhalo == pytest.approx(100.0 / 38.0)


def test_random_paired_evidence_obeys_difference_bound():
    """Keep marginalized evidence differences inside paired differences."""
    rng = np.random.default_rng(20260811)
    for iteration in range(50):
        smooth = rng.normal(size=7)
        subhalo = rng.normal(size=7)
        candidates = [
            _fit(str(index), 0.0, 0.0, smooth[index], subhalo[index])
            for index in range(7)
        ]
        summary = combine_psf_bank_fits(candidates)
        differences = subhalo - smooth
        assert np.min(differences) <= summary.delta_log_evidence_psf_marg
        assert summary.delta_log_evidence_psf_marg <= np.max(differences)


def test_anchor_diagnostic_reports_signed_and_clipped_statistics():
    """Carry the exact negative signed anchor statistic beside clipped q."""
    case = SimpleNamespace(
        smooth_fit=SimpleNamespace(
            log_likelihood_max=-5.0,
            log_evidence=-6.0,
        ),
        subhalo_fit=SimpleNamespace(
            log_likelihood_max=-8.0,
            log_evidence=-7.0,
        ),
    )
    missing = SimpleNamespace(
        smooth_fit=SimpleNamespace(
            log_likelihood_max=None,
            log_evidence=-6.0,
        ),
        subhalo_fit=SimpleNamespace(
            log_likelihood_max=-8.0,
            log_evidence=-7.0,
        ),
    )

    assert _anchor_diagnostic(case) == {
        "q_fit": 0.0,
        "signed_q_fit": -6.0,
        "delta_log_evidence": -1.0,
    }
    assert _anchor_diagnostic(missing) == {
        "q_fit": None,
        "signed_q_fit": None,
        "delta_log_evidence": -1.0,
    }


def test_freed_mode_nulls_only_scdd_detection_flag():
    """Keep statistics while nulling fixed-calibration SCDD in freed mode."""
    candidates = [_fit("draw000", -10.0, -4.0, -12.0, -4.0)]

    freed = combine_psf_bank_fits(candidates, fit_mode="freed")
    fixed = combine_psf_bank_fits(candidates, fit_mode="fixed_template")

    assert freed.q_fit_psf_profile == 12.0
    assert freed.detected_fit_scdd_psf_profile is None
    assert freed.detected_evidence_psf_marg is True
    assert fixed.detected_fit_scdd_psf_profile is True


@pytest.fixture()
def prior_table(tmp_path) -> Path:
    """Write a tiny two-mode prior table."""
    path = tmp_path / "prior.yaml"
    path.write_text(
        yaml.safe_dump({
            "name": "tiny",
            "segment_variance_fraction": 0.0,
            "global_weights": {4: 1.0, 5: 0.5},
            "metadata": {"basis_convention": "test"},
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def compact_config(prior_table) -> dict:
    """Load a small optical configuration with one prior-draw bank."""
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    config["plotting"]["enabled"] = False
    config["lensing"]["grid"] = {"shape": [25, 25], "pixel_scale": 0.1}
    config["psf"]["telescope"]["num_rings"] = 1
    config["psf"]["telescope"]["supersampling_factor"] = 1
    config["psf"]["hres_psf"].update({
        "num_pix": 128,
        "num_airy": 6,
        "sampling": 5,
        "save_highres_psf_npy": False,
    })
    config["psf"]["kernel"]["shape_native"] = [11, 11]
    config["psf"]["aberrations"] = {
        "enable_segment_pistons": False,
        "enable_segment_tiptilts": False,
        "enable_segment_hexikes": False,
        "enable_global_zernikes": True,
        "segment_pistons": {},
        "segment_tiptilts": {},
        "segment_hexikes": {},
        "global_zernikes": {4: 3.0, 5: -1.0},
    }
    config["modeling"]["fit_psf"] = {
        "mode": "bank",
        "bank": {
            "kind": "prior_draws",
            "prior_table": str(prior_table),
            "amplitude_rms_nm": 20.0,
            "n_draws": 2,
            "seed": 20260811,
        },
    }
    validate_or_raise(config)
    return config


def _quiet_build(config):
    """Build a bank without emitting HCIPy progress output."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return build_psf_bank(config)


def _quiet_psf(config, aberrations=None):
    """Generate one compact PSF without progress output."""
    psf_config = copy.deepcopy(config["psf"])
    if aberrations is not None:
        psf_config["aberrations"] = copy.deepcopy(aberrations)
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return generate_psf_system(psf_config, full_config=config)


def test_prior_draw_generation_is_deterministic_and_prefix_stable(compact_config):
    """Reproduce draws and retain the old prefix when growing the bank."""
    config_four = copy.deepcopy(compact_config)
    config_four["modeling"]["fit_psf"]["bank"]["n_draws"] = 4
    config_eight = copy.deepcopy(config_four)
    config_eight["modeling"]["fit_psf"]["bank"]["n_draws"] = 8

    first = _quiet_build(config_four)
    second = _quiet_build(config_four)
    grown = _quiet_build(config_eight)

    assert first.bank_id == second.bank_id
    for left, right in zip(first.candidates, second.candidates):
        assert left.orthonormal_segment == right.orthonormal_segment
        assert left.orthonormal_global == right.orthonormal_global
        assert left.kernel_sha256 == right.kernel_sha256
    for left, right in zip(first.candidates, grown.candidates[:4]):
        assert left.orthonormal_segment == right.orthonormal_segment
        assert left.orthonormal_global == right.orthonormal_global
        assert left.kernel_sha256 == right.kernel_sha256


def test_prior_draw_seed_and_amplitude_cycle_contract(compact_config):
    """Change coefficients by seed and cycle balanced amplitudes in order."""
    first = copy.deepcopy(compact_config)
    first["modeling"]["fit_psf"]["bank"].update({
        "amplitude_rms_nm": [10.0, 20.0],
        "n_draws": 4,
    })
    second = copy.deepcopy(first)
    second["modeling"]["fit_psf"]["bank"]["seed"] += 1

    bank_first = _quiet_build(first)
    bank_second = _quiet_build(second)

    assert [candidate.amplitude_rms_nm for candidate in bank_first.candidates] == [
        10.0,
        20.0,
        10.0,
        20.0,
    ]
    assert any(
        left.orthonormal_global != right.orthonormal_global
        for left, right in zip(bank_first.candidates, bank_second.candidates)
    )


def test_generated_draws_and_anchors_obey_structure(compact_config):
    """Validate draw structure, exact RMS, and separated anchor controls."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["bank"].update({
        "include_perfect": True,
        "include_truth": True,
    })

    bank = _quiet_build(config)

    assert [candidate.label for candidate in bank.candidates] == ["draw000", "draw001"]
    assert [anchor.label for anchor in bank.anchors] == ["perfect", "truth"]
    for candidate in bank.candidates:
        candidate_psf = copy.deepcopy(config["psf"])
        candidate_psf["aberrations"] = candidate.aberrations
        validate_psf_config(candidate_psf)
        aberrations = candidate.aberrations
        assert aberrations["enable_segment_pistons"] is False
        assert aberrations["enable_segment_tiptilts"] is False
        assert aberrations["segment_pistons"] == {}
        assert aberrations["segment_tiptilts"] == {}
        assert aberrations["enable_segment_hexikes"] is bool(
            aberrations["segment_hexikes"]
        )
        assert aberrations["enable_global_zernikes"] is bool(
            aberrations["global_zernikes"]
        )
        assert candidate.measured_total_rms_nm == pytest.approx(
            candidate.amplitude_rms_nm,
            rel=1.0e-5,
        )
    perfect, truth = bank.anchors
    assert perfect.kind == "perfect"
    assert perfect.amplitude_rms_nm == 0.0
    assert perfect.measured_total_rms_nm == pytest.approx(0.0, abs=1.0e-12)
    assert all(
        perfect.aberrations[key] is False
        for key in (
            "enable_segment_pistons",
            "enable_segment_tiptilts",
            "enable_segment_hexikes",
            "enable_global_zernikes",
        )
    )
    assert truth.kind == "truth"
    assert truth.amplitude_rms_nm is None
    assert truth.aberrations == config["psf"]["aberrations"]


def test_bank_id_covers_every_generation_input(compact_config, tmp_path):
    """Change bank identity for every identity-bearing generation input."""
    base_config = copy.deepcopy(compact_config)
    base_config["modeling"]["fit_psf"]["bank"].update({
        "amplitude_rms_nm": [10.0, 20.0],
        "n_draws": 2,
    })
    base = _quiet_build(base_config)
    assert _quiet_build(copy.deepcopy(base_config)).bank_id == base.bank_id

    variants = []
    seed = copy.deepcopy(base_config)
    seed["modeling"]["fit_psf"]["bank"]["seed"] += 1
    variants.append(seed)
    draws = copy.deepcopy(base_config)
    draws["modeling"]["fit_psf"]["bank"]["n_draws"] = 4
    variants.append(draws)
    amplitudes = copy.deepcopy(base_config)
    amplitudes["modeling"]["fit_psf"]["bank"]["amplitude_rms_nm"] = [10.0, 30.0]
    variants.append(amplitudes)
    amplitude_order = copy.deepcopy(base_config)
    amplitude_order["modeling"]["fit_psf"]["bank"]["amplitude_rms_nm"] = [20.0, 10.0]
    variants.append(amplitude_order)
    perfect = copy.deepcopy(base_config)
    perfect["modeling"]["fit_psf"]["bank"]["include_perfect"] = True
    variants.append(perfect)
    truth = copy.deepcopy(base_config)
    truth["modeling"]["fit_psf"]["bank"]["include_truth"] = True
    variants.append(truth)
    optics = copy.deepcopy(base_config)
    optics["psf"]["telescope"]["focal_length"] += 1.0
    variants.append(optics)
    pixel_scale = copy.deepcopy(base_config)
    pixel_scale["lensing"]["grid"]["pixel_scale"] = 0.11
    variants.append(pixel_scale)
    changed_table = tmp_path / "changed-prior.yaml"
    changed_table.write_text(
        yaml.safe_dump({
            "name": "tiny",
            "segment_variance_fraction": 0.0,
            "global_weights": {4: 1.0, 5: 0.75},
        }),
        encoding="utf-8",
    )
    table = copy.deepcopy(base_config)
    table["modeling"]["fit_psf"]["bank"]["prior_table"] = str(changed_table)
    variants.append(table)

    variant_ids = {_quiet_build(config).bank_id for config in variants}
    assert base.bank_id not in variant_ids
    assert len(variant_ids) == len(variants)


def test_bank_id_uses_prior_content_instead_of_path(compact_config, tmp_path):
    """Keep identity stable when identical prior bytes move to a new path."""
    original = Path(
        compact_config["modeling"]["fit_psf"]["bank"]["prior_table"]
    )
    copied = tmp_path / "same-content.yaml"
    copied.write_bytes(original.read_bytes())
    moved = copy.deepcopy(compact_config)
    moved["modeling"]["fit_psf"]["bank"]["prior_table"] = str(copied)

    assert _quiet_build(moved).bank_id == _quiet_build(compact_config).bank_id


def test_build_psf_bank_rejects_nonbank_fit_mode(compact_config):
    """Reject direct construction outside fit-PSF bank mode."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"] = {"mode": "matched"}

    with pytest.raises(ValueError, match="modeling.fit_psf.mode"):
        _quiet_build(config)


def _assert_bank_roundtrip_equal(saved, loaded):
    """Compare bank metadata exactly and kernels bitwise."""
    for field in fields(saved):
        if field.name in {"candidates", "anchors"}:
            continue
        assert getattr(loaded, field.name) == getattr(saved, field.name)
    for left_group, right_group in (
        (saved.candidates, loaded.candidates),
        (saved.anchors, loaded.anchors),
    ):
        assert len(left_group) == len(right_group)
        for left, right in zip(left_group, right_group):
            for field in fields(left):
                if field.name == "kernel":
                    continue
                assert getattr(right, field.name) == getattr(left, field.name)
            np.testing.assert_array_equal(right.kernel, left.kernel)


def test_kernel_digest_includes_array_shape():
    """Distinguish equal float64 bytes stored with different shapes."""
    kernel = np.arange(9, dtype=np.float64).reshape(3, 3)

    assert _kernel_sha256(kernel) != _kernel_sha256(kernel.reshape(1, 9))


def test_npz_roundtrip_preserves_typed_metadata_and_kernel_hashes(
    compact_config,
    tmp_path,
):
    """Round-trip every field while restoring integer coefficient keys."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["bank"].update({
        "include_perfect": True,
        "include_truth": True,
    })
    bank = _quiet_build(config)
    path = tmp_path / "bank.npz"

    save_psf_bank_npz(bank, path)
    loaded = load_psf_bank_npz(path)

    _assert_bank_roundtrip_equal(bank, loaded)
    assert all(
        isinstance(key, int)
        for key in loaded.candidates[0].orthonormal_global
    )
    assert all(
        isinstance(key, int)
        for key in loaded.candidates[0].aberrations["global_zernikes"]
    )
    candidate_psf = copy.deepcopy(config["psf"])
    candidate_psf["aberrations"] = loaded.candidates[0].aberrations
    validate_psf_config(candidate_psf)


def test_npz_loader_rejects_tampered_kernel_and_metadata(compact_config, tmp_path):
    """Reject content corruption in either NPZ kernels or metadata."""
    bank = _quiet_build(compact_config)
    original = tmp_path / "bank.npz"
    save_psf_bank_npz(bank, original)
    with np.load(original, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}

    kernel_arrays = dict(arrays)
    kernel_arrays["kernel_draw000"][0, 0] += 1.0e-8
    kernel_path = tmp_path / "tampered-kernel.npz"
    np.savez(kernel_path, **kernel_arrays)
    with pytest.raises(ValueError, match="kernel.*sha256"):
        load_psf_bank_npz(kernel_path)

    metadata_arrays = dict(arrays)
    metadata = json.loads(str(metadata_arrays["metadata_json"].item()))
    metadata["bank_id"] = "0"*16
    metadata_arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    metadata_path = tmp_path / "tampered-metadata.npz"
    np.savez(metadata_path, **metadata_arrays)
    with pytest.raises(ValueError, match="bank_id"):
        load_psf_bank_npz(metadata_path)


def test_npz_loader_rejects_shape_preserving_kernel_byte_tamper(
    compact_config,
    tmp_path,
):
    """Reject a saved kernel reshaped without changing its float64 bytes."""
    bank = _quiet_build(compact_config)
    original = tmp_path / "bank.npz"
    save_psf_bank_npz(bank, original)
    with np.load(original, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    kernel = arrays["kernel_draw000"]
    arrays["kernel_draw000"] = kernel.reshape(1, kernel.size)
    tampered = tmp_path / "reshaped-kernel.npz"
    np.savez(tampered, **arrays)

    with pytest.raises(ValueError, match="kernel.*sha256"):
        load_psf_bank_npz(tampered)


def test_npz_loader_binds_manifest_structure_to_bank_config(
    compact_config,
    tmp_path,
):
    """Reject deletion, reclassification, reordering, and count edits."""
    bank = _quiet_build(compact_config)
    original = tmp_path / "bank.npz"
    save_psf_bank_npz(bank, original)
    with np.load(original, allow_pickle=False) as stored:
        original_arrays = {
            name: stored[name].copy() for name in stored.files
        }

    deleted_arrays = dict(original_arrays)
    deleted_metadata = json.loads(
        str(deleted_arrays["metadata_json"].item())
    )
    removed = deleted_metadata["candidates"].pop()
    deleted_arrays.pop(f"kernel_{removed['label']}")
    deleted_arrays["metadata_json"] = np.asarray(
        json.dumps(deleted_metadata)
    )
    deleted_path = tmp_path / "deleted-candidate.npz"
    np.savez(deleted_path, **deleted_arrays)
    with pytest.raises(ValueError, match="structure check.*candidate labels"):
        load_psf_bank_npz(deleted_path)

    moved_arrays = dict(original_arrays)
    moved_metadata = json.loads(str(moved_arrays["metadata_json"].item()))
    moved_metadata["anchors"].append(moved_metadata["candidates"].pop())
    moved_arrays["metadata_json"] = np.asarray(json.dumps(moved_metadata))
    moved_path = tmp_path / "moved-candidate.npz"
    np.savez(moved_path, **moved_arrays)
    with pytest.raises(ValueError, match="structure check.*candidate labels"):
        load_psf_bank_npz(moved_path)

    reordered_arrays = dict(original_arrays)
    reordered_metadata = json.loads(
        str(reordered_arrays["metadata_json"].item())
    )
    reordered_metadata["candidates"].reverse()
    reordered_arrays["metadata_json"] = np.asarray(
        json.dumps(reordered_metadata)
    )
    reordered_path = tmp_path / "reordered-candidates.npz"
    np.savez(reordered_path, **reordered_arrays)
    with pytest.raises(ValueError, match="structure check.*candidate labels"):
        load_psf_bank_npz(reordered_path)

    count_arrays = dict(original_arrays)
    count_metadata = json.loads(str(count_arrays["metadata_json"].item()))
    count_metadata["n_draws"] += 1
    count_arrays["metadata_json"] = np.asarray(json.dumps(count_metadata))
    count_path = tmp_path / "edited-count.npz"
    np.savez(count_path, **count_arrays)
    with pytest.raises(ValueError, match="structure check.*n_draws"):
        load_psf_bank_npz(count_path)

    seed_arrays = dict(original_arrays)
    seed_metadata = json.loads(str(seed_arrays["metadata_json"].item()))
    seed_metadata["bank_config"]["seed"] += 1
    seed_arrays["metadata_json"] = np.asarray(json.dumps(seed_metadata))
    seed_path = tmp_path / "edited-seed.npz"
    np.savez(seed_path, **seed_arrays)
    with pytest.raises(ValueError, match="bank_id"):
        load_psf_bank_npz(seed_path)


class _Observation:
    """Small observation seam accepted by `imaging_from_observation`."""

    def __init__(self, psf, pixel_scale):
        shape = (25, 25)
        self.psf = psf
        self.pixel_scale = float(pixel_scale)
        self.noiseless_source_eps = np.ones(shape, dtype=float)
        self.data = SimpleNamespace(native=np.ones(shape, dtype=float))
        self.noise_map = SimpleNamespace(native=np.ones(shape, dtype=float))
        self.gain = 1.0
        self.exposure_time = 1.0
        self.sky_electrons_per_pixel = 0.0
        self.dark_electrons_per_pixel = 0.0


def _trial() -> SubhaloTrial:
    """Return one fixed PointMass trial for stubbed executor tests."""
    return SubhaloTrial(
        case_id="bank-case",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.1, -0.2),
        model="PointMass",
        profile_class="PointMass",
        lens_redshift=0.2,
        source_redshift=0.6,
        einstein_radius_arcsec=0.01,
    )


class FakeValidator:
    """Return canned case summaries without starting a nonlinear search."""

    def __init__(self, fail_labels=(), missing_evidence_labels=()):
        self.calls = []
        self.psf_hashes = {}
        self.fail_labels = set(fail_labels)
        self.missing_evidence_labels = set(missing_evidence_labels)

    def validate_case(
        self,
        dataset,
        dataset_metadata,
        full_config,
        trial,
        fit_mode="fixed_template",
        psf_case="nominal",
        priors_config=None,
        mass_context=None,
        clumpy_fit_parameterization="host_free",
        smooth_result=None,
        expected_psf_fit_sha256=None,
    ):
        """Record one executor call and return deterministic fit summaries."""
        del full_config, priors_config, mass_context, clumpy_fit_parameterization
        del expected_psf_fit_sha256
        label = psf_case.rsplit(":", 1)[-1]
        index = len(self.calls)
        smooth_key = analysis_key_from(
            dataset,
            dataset_metadata,
            {"fit_mode": "smooth", "resolved_prior_widths": {}},
        )
        subhalo_key = analysis_key_from(
            dataset,
            dataset_metadata,
            {"fit_mode": fit_mode, "resolved_prior_widths": {}},
        )
        failed = label in self.fail_labels
        smooth_log_l = None if failed else -10.0 - index
        subhalo_log_l = -7.0 - index
        smooth_logz = None if label in self.missing_evidence_labels else -12.0 - index
        subhalo_logz = -9.0 - index
        smooth = NonlinearFitSummary(
            model_role="smooth",
            fit_mode=fit_mode,
            status="failed" if failed else "success",
            log_likelihood_max=smooth_log_l,
            log_evidence=smooth_logz,
            analysis_key=smooth_key,
        )
        subhalo = NonlinearFitSummary(
            model_role="subhalo",
            fit_mode=fit_mode,
            status="success",
            log_likelihood_max=subhalo_log_l,
            log_evidence=subhalo_logz,
            analysis_key=subhalo_key,
        )
        metric = (
            profile_likelihood_ratio(smooth_log_l, subhalo_log_l)
            if not failed
            else None
        )
        case = NonlinearCaseResult(
            case_id=trial.case_id,
            trial=trial,
            dataset_metadata=dataset_metadata,
            fit_mode=fit_mode,
            psf_case=psf_case,
            smooth_fit=smooth,
            subhalo_fit=subhalo,
            metric=metric,
            quality_flags=["fit_failed"] if failed else [],
        )
        self.calls.append({
            "label": label,
            "psf_fit_label": dataset_metadata.psf_fit_label,
            "smooth_result": smooth_result,
            "case": case,
        })
        self.psf_hashes[label] = _array_hash(dataset.psf)
        return case


def _observation_from_config(config):
    """Build an observation seam carrying the configured truth kernel."""
    truth_psf = _quiet_psf(config)
    return _Observation(
        truth_psf.kernel,
        config["lensing"]["grid"]["pixel_scale"],
    )


def test_truth_and_explicit_candidates_preserve_matched_kernel_identity(compact_config):
    """Preserve the matched PSF byte hash through executor kernel wrapping."""
    observation = _observation_from_config(compact_config)
    matched_dataset, _ = imaging_from_observation(observation, psf_for_fit=None)
    matched_hash = _array_hash(matched_dataset.psf)

    truth_config = copy.deepcopy(compact_config)
    truth_config["modeling"]["fit_psf"]["bank"]["include_truth"] = True
    truth_bank = _quiet_build(truth_config)
    truth_validator = FakeValidator()
    run_psf_bank_case(
        truth_validator,
        observation,
        truth_config,
        _trial(),
        truth_bank,
        fit_mode="fixed_template",
    )
    assert truth_validator.psf_hashes["truth"] == matched_hash

    explicit_config = copy.deepcopy(compact_config)
    explicit_config["modeling"]["fit_psf"]["bank"] = {
        "kind": "explicit",
        "candidates": [copy.deepcopy(explicit_config["psf"]["aberrations"])],
    }
    validate_or_raise(explicit_config)
    explicit_bank = _quiet_build(explicit_config)
    explicit_validator = FakeValidator()
    result = run_psf_bank_case(
        explicit_validator,
        observation,
        explicit_config,
        _trial(),
        explicit_bank,
        fit_mode="fixed_template",
    )

    assert explicit_validator.psf_hashes["explicit000"] == matched_hash
    assert result.summary.q_fit_psf_profile == 6.0
    assert result.summary.delta_log_evidence_psf_marg == pytest.approx(3.0)


def test_executor_wires_candidates_anchors_callbacks_and_slim_json(
    compact_config,
    tmp_path,
):
    """Run each bank member independently and keep anchors out of summary."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["bank"].update({
        "n_draws": 3,
        "include_perfect": True,
    })
    bank = _quiet_build(config)
    observation = _observation_from_config(config)
    validator = FakeValidator()
    callback_labels = []

    def callback(case):
        """Record each completed case callback."""
        callback_labels.append(case.psf_case.rsplit(":", 1)[-1])

    result = run_psf_bank_case(
        validator,
        observation,
        config,
        _trial(),
        bank,
        fit_mode="fixed_template",
        on_candidate=callback,
    )

    expected_labels = ["draw000", "draw001", "draw002", "perfect"]
    assert [call["label"] for call in validator.calls] == expected_labels
    assert callback_labels == expected_labels
    assert all(call["smooth_result"] is None for call in validator.calls)
    fit_labels = [call["psf_fit_label"] for call in validator.calls]
    assert fit_labels == [
        f"bank:{bank.bank_id}:{label}" for label in expected_labels
    ]
    assert len({call["case"].subhalo_fit.analysis_key for call in validator.calls}) == 4
    assert result.summary.n_candidates == 3
    assert len(result.candidate_results) == 3
    assert len(result.anchor_results) == 1
    assert result.anchor_diagnostics["perfect"] == {
        "q_fit": 6.0,
        "signed_q_fit": 6.0,
        "delta_log_evidence": 3.0,
    }
    assert result.quality_flags == []
    output = tmp_path / "case.json"
    result.write_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "kernel" not in payload["candidate_results"][0]
    assert payload["candidate_results"][0]["label"] == "draw000"
    assert payload["bank_id"] == bank.bank_id


def test_executor_flags_failures_missing_evidence_and_callback_errors(compact_config):
    """Exclude bad candidates and record every bank-level quality condition."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["bank"].update({
        "n_draws": 3,
        "include_perfect": True,
    })
    bank = _quiet_build(config)
    validator = FakeValidator(
        fail_labels={"draw000", "perfect"},
        missing_evidence_labels={"draw001"},
    )
    callback_count = 0

    def failing_callback(case):
        """Raise after proving the callback was attempted."""
        nonlocal callback_count
        callback_count += 1
        if case.psf_case.endswith("draw002"):
            raise RuntimeError("callback failed")

    result = run_psf_bank_case(
        validator,
        _observation_from_config(config),
        config,
        _trial(),
        bank,
        fit_mode="fixed_template",
        on_candidate=failing_callback,
        allow_censored=True,
    )

    assert result.summary.n_success == 2
    assert result.summary.n_evidence == 1
    assert result.summary.censored is True
    assert result.summary.lost_evidence_prior_mass_fraction == pytest.approx(
        2.0 / 3.0
    )
    assert callback_count == 4
    assert set(result.quality_flags) == {
        "bank_candidate_failed",
        "bank_missing_evidence",
        "bank_anchor_failed",
        "bank_on_candidate_callback_failed",
    }


def test_executor_fails_closed_after_candidate_fits_complete(compact_config):
    """Refuse the censored combination only after every fit has run."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["bank"]["n_draws"] = 3
    bank = _quiet_build(config)
    validator = FakeValidator(fail_labels={"draw001"})

    with pytest.raises(ValueError, match="allow_censored"):
        run_psf_bank_case(
            validator,
            _observation_from_config(config),
            config,
            _trial(),
            bank,
            fit_mode="fixed_template",
        )
    assert [call["label"] for call in validator.calls] == [
        "draw000",
        "draw001",
        "draw002",
    ]


def test_version_drift_is_a_soft_execution_diagnostic(
    compact_config,
    tmp_path,
):
    """Load stale version metadata and flag it without blocking execution."""
    bank = _quiet_build(compact_config)
    edited_versions = dict(bank.versions)
    edited_versions["numpy"] = "tampered-version"
    edited_bank = replace(bank, versions=edited_versions)
    path = tmp_path / "version-mismatch.npz"
    save_psf_bank_npz(edited_bank, path)

    loaded = load_psf_bank_npz(path)
    result = run_psf_bank_case(
        FakeValidator(),
        _observation_from_config(compact_config),
        compact_config,
        _trial(),
        loaded,
        fit_mode="fixed_template",
    )

    assert loaded.versions["numpy"] == "tampered-version"
    assert "bank_version_mismatch" in result.quality_flags
    assert result.bank_provenance["version_mismatches"]["numpy"] == {
        "bank": "tampered-version",
        "current": bank.versions["numpy"],
    }


def test_executor_checks_freed_context_before_any_fit(compact_config):
    """Reject a missing freed mass context before validator invocation."""
    bank = _quiet_build(compact_config)
    validator = FakeValidator()

    with pytest.raises(ValueError, match="freed mode requires mass_context"):
        run_psf_bank_case(
            validator,
            _observation_from_config(compact_config),
            compact_config,
            _trial(),
            bank,
        )
    assert validator.calls == []


def test_executor_rejects_incompatible_bank_inputs(compact_config):
    """Reject PSF config, lensing scale, and observation scale mismatches."""
    bank = _quiet_build(compact_config)
    observation = _observation_from_config(compact_config)
    validator = FakeValidator()

    bad_hash = replace(bank, psf_config_hash="0"*16)
    with pytest.raises(ValueError, match=bank.bank_id):
        run_psf_bank_case(
            validator,
            observation,
            compact_config,
            _trial(),
            bad_hash,
            fit_mode="fixed_template",
        )

    bad_lensing_scale = replace(
        bank,
        lensing_pixel_scale=bank.lensing_pixel_scale + 0.01,
    )
    with pytest.raises(ValueError, match=bank.bank_id):
        run_psf_bank_case(
            validator,
            observation,
            compact_config,
            _trial(),
            bad_lensing_scale,
            fit_mode="fixed_template",
        )

    bad_candidate = replace(
        bank.candidates[0],
        kernel_pixel_scale=bank.candidates[0].kernel_pixel_scale + 0.01,
    )
    bad_kernel_scale = replace(
        bank,
        candidates=(bad_candidate,) + bank.candidates[1:],
    )
    with pytest.raises(ValueError, match=bank.bank_id):
        run_psf_bank_case(
            validator,
            observation,
            compact_config,
            _trial(),
            bad_kernel_scale,
            fit_mode="fixed_template",
        )
    assert validator.calls == []


def test_executor_rejects_corrupted_in_memory_kernel(compact_config):
    """Check every stored kernel hash before starting candidate fits."""
    bank = _quiet_build(compact_config)
    kernel = bank.candidates[0].kernel.copy()
    kernel[0, 0] += 1.0e-8
    corrupted_candidate = replace(bank.candidates[0], kernel=kernel)
    corrupted = replace(
        bank,
        candidates=(corrupted_candidate,) + bank.candidates[1:],
    )
    validator = FakeValidator()

    with pytest.raises(ValueError, match="kernel.*sha256"):
        run_psf_bank_case(
            validator,
            _observation_from_config(compact_config),
            compact_config,
            _trial(),
            corrupted,
            fit_mode="fixed_template",
        )
    assert validator.calls == []


def test_explicit_bank_build_roundtrip_and_stubbed_execution(
    compact_config,
    tmp_path,
):
    """Keep explicit coefficients unchanged through save, load, and run."""
    config = copy.deepcopy(compact_config)
    aberrations = copy.deepcopy(config["psf"]["aberrations"])
    aberrations["global_zernikes"] = {4: 7, 5: -2}
    config["modeling"]["fit_psf"]["bank"] = {
        "kind": "explicit",
        "candidates": [aberrations],
    }
    validate_or_raise(config)

    bank = _quiet_build(config)
    expected = _quiet_psf(config, aberrations=aberrations)

    assert bank.seed is None
    assert bank.n_draws == 0
    assert bank.prior_table_path is None
    assert bank.candidates[0].label == "explicit000"
    assert bank.candidates[0].kind == "explicit"
    assert bank.candidates[0].amplitude_rms_nm is None
    assert bank.candidates[0].aberrations == aberrations
    assert bank.candidates[0].measured_total_rms_nm == pytest.approx(
        expected.total_rms_nm
    )

    path = tmp_path / "explicit.npz"
    save_psf_bank_npz(bank, path)
    loaded = load_psf_bank_npz(path)
    _assert_bank_roundtrip_equal(bank, loaded)
    result = run_psf_bank_case(
        FakeValidator(),
        _observation_from_config(config),
        config,
        _trial(),
        loaded,
        fit_mode="fixed_template",
    )
    assert result.summary.n_candidates == 1
    assert result.summary.q_fit_psf_profile == 6.0


def test_explicit_tiptilt_pairs_are_canonical_and_identity_stable(
    compact_config,
    tmp_path,
):
    """Normalize tuple pairs before identity, persistence, and execution."""
    tuple_config = copy.deepcopy(compact_config)
    aberrations = copy.deepcopy(tuple_config["psf"]["aberrations"])
    aberrations["enable_segment_tiptilts"] = True
    aberrations["segment_tiptilts"] = {0: (1.0, 2.0)}
    tuple_config["modeling"]["fit_psf"]["bank"] = {
        "kind": "explicit",
        "candidates": [aberrations],
    }
    validate_or_raise(tuple_config)

    list_config = copy.deepcopy(tuple_config)
    list_config["modeling"]["fit_psf"]["bank"]["candidates"][0][
        "segment_tiptilts"
    ] = {0: [1.0, 2.0]}
    tuple_bank = _quiet_build(tuple_config)
    list_bank = _quiet_build(list_config)
    path = tmp_path / "tuple-tiptilt.npz"
    save_psf_bank_npz(tuple_bank, path)
    loaded = load_psf_bank_npz(path)

    assert tuple_bank.bank_id == list_bank.bank_id
    assert tuple_bank.bank_config["candidates"][0]["segment_tiptilts"] == {
        0: [1.0, 2.0]
    }
    assert tuple_bank.candidates[0].aberrations["segment_tiptilts"] == {
        0: [1.0, 2.0]
    }
    _assert_bank_roundtrip_equal(tuple_bank, loaded)


def test_prior_table_path_resolution_absolute_cwd_repo_and_missing(
    prior_table,
    tmp_path,
    monkeypatch,
):
    """Resolve absolute, current-directory, and repository-relative tables."""
    assert _resolve_prior_table_path(prior_table) == prior_table.resolve()

    monkeypatch.chdir(tmp_path)
    assert _resolve_prior_table_path(prior_table.name) == prior_table.resolve()

    repository_relative = Path("configs/psf_priors/jwst_wss_drift_v1.yaml")
    assert _resolve_prior_table_path(repository_relative) == (
        PROJECT_ROOT / repository_relative
    ).resolve()
    with pytest.raises(FileNotFoundError, match="missing-prior.yaml"):
        _resolve_prior_table_path("missing-prior.yaml")


def test_bank_id_uses_config_hash_and_lensing_scale(compact_config):
    """Record the exact PSF config hash and consumed detector pixel scale."""
    bank = _quiet_build(compact_config)

    assert bank.psf_config_hash == config_hash(compact_config["psf"])
    assert bank.lensing_pixel_scale == compact_config["lensing"]["grid"][
        "pixel_scale"
    ]


def test_nonlinear_package_exposes_psf_bank_symbols_lazily():
    """Expose all Item 8 public symbols through the package namespace."""
    import hwoslaps.modeling.nonlinear as nonlinear

    expected = {
        "PsfBank",
        "PsfBankCandidate",
        "PsfBankCandidateFit",
        "PsfBankSummary",
        "PsfBankCaseResult",
        "build_psf_bank",
        "save_psf_bank_npz",
        "load_psf_bank_npz",
        "combine_psf_bank_fits",
        "run_psf_bank_case",
    }

    assert expected <= set(dir(nonlinear))
    assert nonlinear.PsfBankCandidateFit is PsfBankCandidateFit
