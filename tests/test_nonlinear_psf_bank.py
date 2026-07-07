"""Tests for the discrete PSF-bank statistic combination rules."""

import math

import pytest

from hwoslaps.modeling.nonlinear.psf_bank import (
    DEFAULT_PSF_BANK_SCALES,
    PsfBankCandidateFit,
    combine_psf_bank_fits,
    scale_psf_aberrations,
)


def _candidate(scale, log_l_smooth, log_l_subhalo, logz_smooth, logz_subhalo, success=True):
    return PsfBankCandidateFit(
        scale=scale,
        log_l_smooth=log_l_smooth,
        log_l_subhalo=log_l_subhalo,
        log_evidence_smooth=logz_smooth,
        log_evidence_subhalo=logz_subhalo,
        success=success,
    )


def test_scale_psf_aberrations_scales_and_flags():
    aberrations = {
        "enable_segment_hexikes": True,
        "enable_global_zernikes": True,
        "enable_segment_pistons": False,
        "enable_segment_tiptilts": False,
        "segment_hexikes": {0: {2: 4.0}},
        "global_zernikes": {4: 3.0},
        "segment_pistons": {},
        "segment_tiptilts": {},
    }
    scaled = scale_psf_aberrations(aberrations, 0.5)
    assert scaled["segment_hexikes"] == {0: {2: 2.0}}
    assert scaled["global_zernikes"] == {4: 1.5}
    assert scaled["enable_segment_hexikes"] is True
    assert scaled["enable_segment_pistons"] is False
    # The input is not mutated.
    assert aberrations["segment_hexikes"] == {0: {2: 4.0}}


def test_scale_zero_produces_perfect_psf_candidate():
    aberrations = {
        "enable_segment_hexikes": True,
        "segment_hexikes": {0: {2: 4.0}},
        "global_zernikes": {},
        "segment_pistons": {},
        "segment_tiptilts": {},
    }
    scaled = scale_psf_aberrations(aberrations, 0.0)
    for family in ("segment_pistons", "segment_tiptilts", "segment_hexikes", "global_zernikes"):
        assert scaled[family] == {}
        assert scaled[f"enable_{family}"] is False


def test_profiling_maximizes_each_hypothesis_independently():
    candidates = [
        _candidate(0.0, log_l_smooth=-100.0, log_l_subhalo=-120.0, logz_smooth=-110.0, logz_subhalo=-130.0),
        _candidate(1.0, log_l_smooth=-105.0, log_l_subhalo=-90.0, logz_smooth=-115.0, logz_subhalo=-100.0),
    ]
    summary = combine_psf_bank_fits(candidates)
    assert summary.best_smooth_scale == 0.0
    assert summary.best_subhalo_scale == 1.0
    assert summary.log_l_smooth_profile == -100.0
    assert summary.log_l_subhalo_profile == -90.0
    assert summary.q_fit_psf_profile == pytest.approx(20.0)
    assert summary.detected_fit_scdd_psf_profile is True


def test_q_fit_is_clamped_at_zero():
    candidates = [
        _candidate(0.0, log_l_smooth=-90.0, log_l_subhalo=-95.0, logz_smooth=-100.0, logz_subhalo=-105.0),
    ]
    summary = combine_psf_bank_fits(candidates)
    assert summary.q_fit_psf_profile == 0.0
    assert summary.detected_fit_scdd_psf_profile is False


def test_equal_prior_weights_cancel_in_delta_log_evidence():
    candidates = [
        _candidate(scale, -100.0 - scale, -95.0 - scale, -110.0 - scale, -102.0 - scale)
        for scale in DEFAULT_PSF_BANK_SCALES
    ]
    summary = combine_psf_bank_fits(candidates)
    # With identical per-candidate offsets, the -log(n) normalization is
    # common to both hypotheses and must cancel in the difference.
    expected = 8.0  # -102 - (-110) at every scale
    assert summary.delta_log_evidence_psf_marg == pytest.approx(expected)
    assert summary.detected_evidence_psf_marg is True


def test_marginalized_evidence_matches_manual_logsumexp():
    candidates = [
        _candidate(0.0, -100.0, -95.0, -110.0, -104.0),
        _candidate(1.0, -101.0, -93.0, -112.0, -101.0),
    ]
    summary = combine_psf_bank_fits(candidates)
    log_prior = -math.log(2)
    expected_smooth = math.log(
        math.exp(-110.0 + log_prior) + math.exp(-112.0 + log_prior)
    )
    expected_subhalo = math.log(
        math.exp(-104.0 + log_prior) + math.exp(-101.0 + log_prior)
    )
    assert summary.log_evidence_smooth_psf_marg == pytest.approx(expected_smooth)
    assert summary.log_evidence_subhalo_psf_marg == pytest.approx(expected_subhalo)
    assert summary.delta_log_evidence_psf_marg == pytest.approx(expected_subhalo - expected_smooth)


def test_profile_q_is_never_larger_than_best_paired_q():
    candidates = [
        _candidate(0.0, -100.0, -96.0, -110.0, -106.0),
        _candidate(0.5, -98.0, -94.0, -108.0, -104.0),
        _candidate(1.0, -103.0, -91.0, -113.0, -101.0),
    ]
    summary = combine_psf_bank_fits(candidates)
    paired_q = max(
        2.0*(cand.log_l_subhalo - cand.log_l_smooth) for cand in candidates
    )
    assert summary.q_fit_psf_profile <= paired_q


def test_failed_candidates_are_excluded():
    candidates = [
        _candidate(0.0, -100.0, -95.0, -110.0, -104.0),
        _candidate(1.0, -50.0, -10.0, -60.0, -20.0, success=False),
    ]
    summary = combine_psf_bank_fits(candidates)
    assert summary.n_candidates == 2
    assert summary.n_success == 1
    assert summary.log_l_subhalo_profile == -95.0
    # The -log(2) prior over the full bank remains in each hypothesis but
    # cancels in the difference.
    assert summary.delta_log_evidence_psf_marg == pytest.approx(6.0)


def test_empty_bank_raises():
    with pytest.raises(ValueError):
        combine_psf_bank_fits([])
