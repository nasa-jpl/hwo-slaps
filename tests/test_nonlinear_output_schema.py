"""Serialization tests for nonlinear validation outputs."""

from __future__ import annotations

import json

from hwoslaps.modeling.nonlinear.dataset_builder import NonlinearDatasetMetadata
from hwoslaps.modeling.nonlinear.likelihood_metrics import profile_likelihood_ratio
from hwoslaps.modeling.nonlinear.output_schema import (
    NONLINEAR_CASE_CSV_COLUMNS,
    NonlinearCaseResult,
    NonlinearDetectionData,
    NonlinearFitSummary,
)
from hwoslaps.modeling.nonlinear.trial import SubhaloTrial


def _trial() -> SubhaloTrial:
    return SubhaloTrial(
        case_id="case",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.1, -0.2),
        model="NFW",
        profile_class="NFWSph",
        lens_redshift=0.2,
        source_redshift=0.6,
        kappa_s=0.01,
        scale_radius_arcsec=0.2,
        fisher_q=10.0,
        fisher_z=10.0**0.5,
        fisher_delta_log_l_equiv=5.0,
    )


def _metadata() -> NonlinearDatasetMetadata:
    return NonlinearDatasetMetadata(
        dataset_kind="asimov",
        data_units="adu",
        background_treatment="subtract_known",
        sky_dark_background_adu=1.0,
        mask_name="all_pixels",
        n_unmasked_pixels=25,
        psf_truth_label="truth",
        psf_fit_label="fit",
    )


def test_case_result_serializes_to_json_safe_dictionary():
    case = NonlinearCaseResult(
        case_id="case",
        trial=_trial(),
        dataset_metadata=_metadata(),
        fit_mode="fixed_template",
        psf_case="known_degraded",
        smooth_fit=NonlinearFitSummary(
            model_role="smooth",
            fit_mode="fixed_template",
            status="success",
            log_likelihood_max=-105.0,
        ),
        subhalo_fit=NonlinearFitSummary(
            model_role="subhalo",
            fit_mode="fixed_template",
            status="success",
            log_likelihood_max=-100.0,
        ),
        metric=profile_likelihood_ratio(-105.0, -100.0),
        fisher_q=10.0,
        fisher_z=10.0**0.5,
        fisher_delta_log_l_equiv=5.0,
    )

    payload = case.to_dict()
    json.dumps(payload)
    row = case.to_csv_row(run_name="run")

    assert set(row) == set(NONLINEAR_CASE_CSV_COLUMNS)
    assert row["q_fit"] == 10.0
    assert row["detected_fit_scdd"] is True
    assert row["q_fit_over_q_fisher"] == 1.0


def test_failed_case_serializes_with_null_metric():
    case = NonlinearCaseResult(
        case_id="failed",
        trial=_trial(),
        dataset_metadata=_metadata(),
        fit_mode="fixed_template",
        psf_case="nominal",
        smooth_fit=NonlinearFitSummary(
            model_role="smooth",
            fit_mode="fixed_template",
            status="failed",
            error="boom",
        ),
        subhalo_fit=NonlinearFitSummary(
            model_role="subhalo",
            fit_mode="fixed_template",
            status="skipped",
        ),
        metric=None,
    )

    data = NonlinearDetectionData(
        run_name="run",
        backend="pyautolens",
        cases=[case],
        thresholds={"q_threshold": 10.0},
        config={},
    )

    payload = data.to_dict()
    json.dumps(payload)
    assert payload["cases"][0]["metric"] is None
    assert payload["summary"]["n_failed"] == 1
