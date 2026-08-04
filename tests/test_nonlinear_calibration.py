"""Tests for Fisher-versus-nonlinear calibration summaries."""

from __future__ import annotations

from hwoslaps.modeling.nonlinear.calibration import (
    fit_q_calibration,
    pair_fisher_and_nonlinear,
)
from hwoslaps.modeling.nonlinear.dataset_builder import NonlinearDatasetMetadata
from hwoslaps.modeling.nonlinear.likelihood_metrics import profile_likelihood_ratio
from hwoslaps.modeling.nonlinear.output_schema import (
    NonlinearCaseResult,
    NonlinearDetectionData,
    NonlinearFitSummary,
)
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial


def _case(case_id: str, q_fisher: float, q_fit: float) -> NonlinearCaseResult:
    trial = SubhaloTrial(
        case_id=case_id,
        mass_msun=1.0e7,
        position_yx_arcsec=(0.0, 1.0),
        model="PointMass",
        profile_class="PointMass",
        lens_redshift=0.2,
        source_redshift=0.6,
        einstein_radius_arcsec=0.001,
        fisher_q=q_fisher,
        fisher_z=q_fisher**0.5,
        fisher_delta_log_l_equiv=0.5*q_fisher,
    )
    metric = profile_likelihood_ratio(
        log_l_smooth=-100.0,
        log_l_subhalo=-100.0 + 0.5*q_fit,
    )
    return NonlinearCaseResult(
        case_id=case_id,
        trial=trial,
        dataset_metadata=NonlinearDatasetMetadata(
            dataset_kind="asimov",
            data_units="adu",
            background_treatment="subtract_known",
            sky_dark_background_adu=0.0,
            mask_name="all",
            n_unmasked_pixels=10,
            psf_truth_label="truth",
            psf_fit_label="fit",
        ),
        fit_mode="fixed_template",
        psf_case="nominal",
        smooth_fit=NonlinearFitSummary("smooth", "fixed_template", "success"),
        subhalo_fit=NonlinearFitSummary("subhalo", "fixed_template", "success"),
        metric=metric,
        fisher_q=q_fisher,
        fisher_z=q_fisher**0.5,
        fisher_delta_log_l_equiv=0.5*q_fisher,
    )


def test_pair_fisher_and_nonlinear_skips_cases_without_metrics():
    """Pair only the cases that carry both Fisher and fit metrics."""
    data = NonlinearDetectionData(
        run_name="run",
        backend="pyautolens",
        cases=[_case("a", 10.0, 12.0)],
        thresholds={"q_threshold": 10.0},
        config={},
    )

    pairs = pair_fisher_and_nonlinear(data)

    assert len(pairs) == 1
    assert pairs[0].q_fisher == 10.0
    assert pairs[0].q_fit == 12.0
    assert pairs[0].delta_log_l_fit == 6.0


def test_fit_q_calibration_reports_ratios_and_threshold_confusion():
    """Report q ratios, a linear slope, and the confusion counts."""
    data = NonlinearDetectionData(
        run_name="run",
        backend="pyautolens",
        cases=[
            _case("a", 5.0, 4.0),
            _case("b", 10.0, 12.0),
            _case("c", 20.0, 18.0),
        ],
        thresholds={"q_threshold": 10.0},
        config={},
    )
    calibration = fit_q_calibration(pair_fisher_and_nonlinear(data))

    assert calibration.n_pairs == 3
    assert calibration.median_q_ratio is not None
    assert calibration.alpha_linear_q is not None
    assert calibration.threshold_confusion["fisher_detected_fit_detected"] == 2
    assert calibration.threshold_confusion["fisher_not_detected_fit_not_detected"] == 1
