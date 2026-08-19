"""Serialization tests for nonlinear validation outputs."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from hwoslaps.modeling.nonlinear.dataset_builder import NonlinearDatasetMetadata
from hwoslaps.modeling.nonlinear.likelihood_metrics import profile_likelihood_ratio
from hwoslaps.modeling.nonlinear.output_schema import (
    NONLINEAR_CASE_CSV_COLUMNS,
    NonlinearCaseResult,
    NonlinearDetectionData,
    NonlinearFitSummary,
    SubhaloRecovery,
    _weighted_quantiles,
    extract_subhalo_recovery,
)
from hwoslaps.modeling.nonlinear.mass_mapping import (
    build_mass_mapping_context_explicit,
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
    """Serialize a successful case to JSON and to the CSV columns."""
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
    """Serialize a failed case with null metric fields."""
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


def test_subhalo_recovery_json_and_csv_round_trip(tmp_path):
    """Serialize recovery values while legacy-mode columns stay null-safe."""
    recovery = SubhaloRecovery(
        log10_m200_ml=7.1,
        centre_ml_y=0.1,
        centre_ml_x=-0.2,
        concentration_ml=20.0,
        kappa_s_ml=0.01,
        scale_radius_arcsec_ml=0.2,
        log10_m200_p16=6.9,
        log10_m200_p50=7.1,
        log10_m200_p84=7.3,
        mass_at_lower_bound=False,
        mass_at_upper_bound=False,
        pdf_converged=True,
        extraction_method="synthetic",
        n_samples=10,
    )
    case = NonlinearCaseResult(
        case_id="case",
        trial=_trial(),
        dataset_metadata=_metadata(),
        fit_mode="freed",
        psf_case="nominal",
        smooth_fit=NonlinearFitSummary(
            model_role="smooth",
            fit_mode="freed",
            status="success",
            log_likelihood_max=-10.0,
            analysis_key="key",
        ),
        subhalo_fit=NonlinearFitSummary(
            model_role="subhalo",
            fit_mode="freed",
            status="success",
            log_likelihood_max=-8.0,
            analysis_key="key",
            n_like_max_reached=False,
        ),
        metric=profile_likelihood_ratio(-10.0, -8.0),
        subhalo_recovery=recovery,
    )
    data = NonlinearDetectionData(
        run_name="run",
        backend="pyautolens",
        cases=[case],
        thresholds={},
        config={},
    )
    json_path = tmp_path / "recovery.json"
    csv_path = tmp_path / "recovery.csv"
    data.write_json(json_path)
    data.write_cases_csv(csv_path)
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert loaded["cases"][0]["subhalo_recovery"]["log10_m200_ml"] == 7.1
    assert row["recovered_log10_m200_ml"] == "7.1"
    assert case.to_csv_row()["q_fit"] == pytest.approx(4.0)
    assert case.to_csv_row()["z_fit_local"] is None
    assert case.to_csv_row()["detected_fit_scdd"] is None

    legacy = NonlinearCaseResult(
        case_id="legacy",
        trial=_trial(),
        dataset_metadata=_metadata(),
        fit_mode="fixed_template",
        psf_case="nominal",
        smooth_fit=case.smooth_fit,
        subhalo_fit=case.subhalo_fit,
        metric=None,
    ).to_csv_row()
    assert legacy["recovered_log10_m200_ml"] is None
    assert legacy["pdf_converged"] is None


class _Samples:
    """Synthetic posterior samples keyed by AutoFit parameter paths."""

    def __init__(
        self,
        mass,
        centre_y,
        centre_x,
        converged,
        weights=None,
    ):
        prefix = ("galaxies", "lens", "subhalo")
        self.values = {
            prefix + ("log10_m200",): mass,
            prefix + ("centre", "centre_0"): centre_y,
            prefix + ("centre", "centre_1"): centre_x,
        }
        self.weight_list = (
            np.ones(len(mass), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        self.pdf_converged = converged

    def values_for_path(self, path):
        return self.values[path]


def _recovery_result(
    log_mass,
    masses,
    converged=True,
    weights=None,
    centre_y=None,
    centre_x=None,
):
    """Return a synthetic AutoFit-like result for recovery extraction."""
    subhalo = SimpleNamespace(
        log10_m200=log_mass,
        centre=(0.1, -0.2),
    )
    instance = SimpleNamespace(
        galaxies=SimpleNamespace(
            lens=SimpleNamespace(subhalo=subhalo),
        )
    )
    samples = _Samples(
        np.asarray(masses),
        (
            np.linspace(0.08, 0.12, len(masses))
            if centre_y is None
            else np.asarray(centre_y)
        ),
        (
            np.linspace(-0.22, -0.18, len(masses))
            if centre_x is None
            else np.asarray(centre_x)
        ),
        converged,
        weights=weights,
    )
    return SimpleNamespace(
        max_log_likelihood_instance=instance,
        samples=samples,
    )


def _mapping_context():
    """Return a canonical NFW recovery context."""
    return build_mass_mapping_context_explicit(
        subhalo_model="NFW",
        concentration_model="moline2017_eq7",
        x_sub=1.0,
        h=0.6774,
        z_lens=0.2,
        z_source=0.6,
        cosmology_name="Planck15",
    )


def test_recovery_boundary_flags_and_posterior_fractions():
    """Flag a bound-hugging ML value and posterior sample pile-up."""
    result = _recovery_result(
        6.005,
        [6.0, 6.01, 6.02, 6.2, 7.0],
    )
    recovery = extract_subhalo_recovery(result, _mapping_context())
    assert recovery.mass_at_lower_bound is True
    assert recovery.mass_at_upper_bound is False
    assert recovery.posterior_mass_frac_lower == pytest.approx(0.6)
    assert recovery.posterior_mass_frac_upper == pytest.approx(0.0)
    assert recovery.n_samples == 5


def test_recovery_upper_boundary_flag_and_posterior_pile_up():
    """Flag an upper-bound ML value and an upper posterior pile-up."""
    result = _recovery_result(
        8.495,
        [8.46, 8.47, 8.48, 8.49, 8.5],
    )
    recovery = extract_subhalo_recovery(result, _mapping_context())
    assert recovery.mass_at_lower_bound is False
    assert recovery.mass_at_upper_bound is True
    assert recovery.posterior_mass_frac_upper == pytest.approx(1.0)


def test_midpoint_ecdf_weighted_quantiles_and_converged_recovery():
    """Match midpoint-ECDF weighted quantiles for recovered coordinates.

    With values ``[1, 2, 3, 4]`` and weights ``[0.01, 0.01, 0.01, 0.97]``
    the midpoint ECDF is ``[0.005, 0.015, 0.025, 0.515]``, so the 16th
    and 50th percentiles interpolate on the ``[0.025, 0.515]`` segment
    and the 84th percentile clamps to the largest sample.
    """
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    weights = np.asarray([0.01, 0.01, 0.01, 0.97])
    expected = (
        3.0 + (0.16 - 0.025) / 0.49,
        3.0 + (0.5 - 0.025) / 0.49,
        4.0,
    )
    assert _weighted_quantiles(values, weights) == pytest.approx(
        expected,
        rel=1.0e-12,
    )

    result = _recovery_result(
        7.0,
        values,
        weights=weights,
        centre_y=values,
        centre_x=values,
    )
    recovery = extract_subhalo_recovery(result, _mapping_context())
    assert (
        recovery.log10_m200_p16,
        recovery.log10_m200_p50,
        recovery.log10_m200_p84,
    ) == pytest.approx(expected, rel=1.0e-12)
    assert (
        recovery.centre_y_p16,
        recovery.centre_y_p50,
        recovery.centre_y_p84,
    ) == pytest.approx(expected, rel=1.0e-12)
    assert (
        recovery.centre_x_p16,
        recovery.centre_x_p50,
        recovery.centre_x_p84,
    ) == pytest.approx(expected, rel=1.0e-12)


def test_weighted_quantiles_clamp_below_heavy_first_weight():
    """Clamp the 16th percentile to a heavily weighted smallest sample.

    With weights ``[0.97, 0.01, 0.01, 0.01]`` the midpoint ECDF is
    ``[0.485, 0.975, 0.985, 0.995]``, so the 16th percentile falls below
    the first CDF value and clamps to the smallest sample.
    """
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    weights = np.asarray([0.97, 0.01, 0.01, 0.01])
    expected = (
        1.0,
        1.0 + (0.5 - 0.485) / 0.49,
        1.0 + (0.84 - 0.485) / 0.49,
    )
    assert _weighted_quantiles(values, weights) == pytest.approx(
        expected,
        rel=1.0e-12,
    )


def test_weighted_quantiles_asymmetric_two_point_case():
    """Interpolate an asymmetric two-point posterior and clamp above.

    With values ``[0, 10]`` and weights ``[0.25, 0.75]`` the midpoint
    ECDF is ``[0.125, 0.625]``, so the 16th and 50th percentiles
    interpolate linearly and the 84th percentile clamps to 10.
    """
    values = np.asarray([0.0, 10.0])
    weights = np.asarray([0.25, 0.75])
    expected = (
        10.0 * (0.16 - 0.125) / 0.5,
        10.0 * (0.5 - 0.125) / 0.5,
        10.0,
    )
    assert _weighted_quantiles(values, weights) == pytest.approx(
        expected,
        rel=1.0e-12,
    )


def test_weighted_quantiles_without_weights_match_numpy_quantile():
    """Fall back to plain np.quantile when no weights are provided."""
    values = np.asarray([3.0, 1.0, 4.0, 1.5, 9.0, 2.6])
    expected = tuple(
        float(value) for value in np.quantile(values, [0.16, 0.5, 0.84])
    )
    assert _weighted_quantiles(values, None) == expected


def test_unconverged_recovery_suppresses_quantiles():
    """Record false PDF convergence without reporting fallback quantiles."""
    result = _recovery_result(7.0, [6.8, 7.0, 7.2], converged=False)
    recovery = extract_subhalo_recovery(result, _mapping_context())
    assert recovery.pdf_converged is False
    assert recovery.log10_m200_p16 is None
    assert recovery.log10_m200_p50 is None
    assert recovery.log10_m200_p84 is None
    assert recovery.posterior_mass_frac_lower == pytest.approx(0.0)


def test_fit_summary_serializes_sampler_provenance_fields():
    """Serialize sampler provenance and default omitted fields to None."""
    summary = NonlinearFitSummary(
        model_role="subhalo",
        fit_mode="freed",
        status="success",
        n_eff_requested=2000.0,
        n_eff_effective=2000.0,
        n_shell_requested=3,
        n_shell_effective=3,
        discard_exploration_requested=True,
        discard_exploration_effective=True,
        search_internal_retention_requested=True,
        search_internal_retained=True,
    )
    payload = summary.to_dict()
    json.dumps(payload)
    assert payload["n_eff_requested"] == 2000.0
    assert payload["n_eff_effective"] == 2000.0
    assert payload["n_shell_requested"] == 3
    assert payload["n_shell_effective"] == 3
    assert payload["discard_exploration_requested"] is True
    assert payload["discard_exploration_effective"] is True
    assert payload["search_internal_retention_requested"] is True
    assert payload["search_internal_retained"] is True

    runtime = NonlinearFitSummary(
        model_role="smooth",
        fit_mode="freed",
        status="success",
        training_workers_requested=4,
        training_workers_effective=4,
        training_start_method="spawn",
        runtime_provenance={
            "training_workers_requested": 4,
            "training_workers_effective": 4,
            "training_start_method": "spawn",
        },
    ).to_dict()
    assert runtime["training_workers_requested"] == 4
    assert runtime["training_workers_effective"] == 4
    assert runtime["training_start_method"] == "spawn"
    assert runtime["runtime_provenance"]["training_start_method"] == "spawn"

    omitted = NonlinearFitSummary(
        model_role="smooth",
        fit_mode="freed",
        status="success",
    ).to_dict()
    assert omitted["n_eff_requested"] is None
    assert omitted["n_eff_effective"] is None
    assert omitted["n_shell_requested"] is None
    assert omitted["n_shell_effective"] is None
    assert omitted["discard_exploration_requested"] is None
    assert omitted["discard_exploration_effective"] is None
    assert omitted["search_internal_retention_requested"] is None
    assert omitted["search_internal_retained"] is None
    assert omitted["training_workers_requested"] is None
    assert omitted["training_workers_effective"] is None
    assert omitted["training_start_method"] is None
    assert omitted["runtime_provenance"] is None


def test_v2_schema_preserves_v1_column_prefix():
    """Bump the schema while retaining every existing CSV column order."""
    v1_columns = (
        "run_name",
        "case_id",
        "fit_mode",
        "psf_case",
        "dataset_kind",
        "mass_msun",
        "y_arcsec",
        "x_arcsec",
        "subhalo_model",
        "profile_class",
        "kappa_s",
        "scale_radius_arcsec",
        "fisher_q",
        "fisher_z",
        "fisher_delta_log_l_equiv",
        "log_l_smooth",
        "log_l_subhalo",
        "signed_delta_log_l_fit",
        "q_fit",
        "z_fit_local",
        "detected_fisher_scdd",
        "detected_fit_scdd",
        "q_fit_over_q_fisher",
        "fit_status_smooth",
        "fit_status_subhalo",
        "n_unmasked_pixels",
        "background_treatment",
        "use_jax_requested",
        "search_engine",
        "n_live_smooth",
        "n_live_subhalo",
        "runtime_s_smooth",
        "runtime_s_subhalo",
        "result_path_smooth",
        "result_path_subhalo",
        "error",
    )
    assert NONLINEAR_CASE_CSV_COLUMNS[: len(v1_columns)] == v1_columns
    payload = NonlinearDetectionData(
        run_name="run",
        backend="pyautolens",
        cases=[],
        thresholds={},
        config={},
    ).to_dict()
    assert payload["schema_version"] == "nonlinear_detection.v2"
