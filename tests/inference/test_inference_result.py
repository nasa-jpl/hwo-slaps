"""Case records: JSON round trips, role acceptance status, the forecast detection rule."""

from __future__ import annotations

import json
import math

import pytest

from hwoslaps.identity import KernelIdentity
from hwoslaps.inference.recovery import SubhaloEstimate, SubhaloRecovery
from hwoslaps.inference.result import (
    ForecastReference, ObservationRecord, RefineOutcome, RetentionInventory, RoleFit, RoleStatus, SamplerRecord,
)
from hwoslaps.inference.settings import MassSupport

KERNEL = KernelIdentity("ab" * 32, (15, 15), 0.03)
NAMES = ("galaxies.lens.mass.einstein_radius", "galaxies.lens.subhalo.log10_m200")


def _sampler_record(*, failed=False):
    retention = RetentionInventory(route="directory", files={"search_internal.dill": {
        "bytes": 13, "sha256": "cd" * 32, "location": "files/search_internal/search_internal.dill"}},
        missing_required=(), bound_to_result_path=True)
    return SamplerRecord(
        name="subhalo_0123456789abcdef", output_path="subhalo_0123456789abcdef/identifier",
        identity="ef" * 32, n_live=50, requested={"n_live_smooth": 50, "n_eff": 200.0, "n_like_max": None},
        effective={"n_live": 50, "n_eff": 200.0, "n_like_max": math.inf, "use_jax_vmap": True}, seed=20261005,
        training_workers=2, log_likelihood_max=None if failed else 16007.682222346557,
        log_evidence=None if failed else 15990.1, likelihood_calls=None if failed else 12345,
        retention=None if failed else retention, runtime_s=12.5)


def _refinement(status=RoleStatus.ACCEPTED):
    return RefineOutcome(acceptance_status=status, best_log_likelihood=16008.188527701757,
                         best_vector=(1.0, 8.9), gates={"support": True, "repeat": status is RoleStatus.ACCEPTED},
                         record={"candidate_best_half_chi2": 3.5, "runs": [{"solver_endpoint_half_chi2": math.nan}]},
                         projected_gradient_linf=0.055, repeat_converged=False)


def _role(role="subhalo", **changes):
    fields = dict(role=role, strategy="search", status="success", log_likelihood=16008.188527701757,
                  truth_log_likelihood=16008.188, parameter_names=NAMES, n_free_parameters=2,
                  sampler=_sampler_record(), refinement=_refinement(),
                  box_edge_margins={NAMES[0]: 0.5, NAMES[1]: 0.31}, anchor_chi2=None, error=None, runtime_s=20.0)
    return RoleFit(**{**fields, **changes})


def _estimate(log10_mass):
    return SubhaloEstimate(log10_mass=log10_mass, centre_yx=(0.39, -0.82),
                           profile_scales={"concentration": 12.5, "kappa_s": 0.003},
                           margin_to_lower_dex=log10_mass - 6.0, margin_to_upper_dex=9.7 - log10_mass)


RECORDS = {
    "search-role-refined": _role(),
    "failed-role": _role(status="failed", log_likelihood=None, sampler=_sampler_record(failed=True),
                         refinement=None, box_edge_margins=None, error="RuntimeError: boom"),
    "anchor-role": _role(strategy="truth_anchor", sampler=None, refinement=None, anchor_chi2=3.0e-12,
                         box_edge_margins=None),
    "sampler-only-role": _role(role="smooth", refinement=None),
    "unresolved-refinement": _refinement(RoleStatus.UNRESOLVED),
    "recovery-sampler-and-refined": SubhaloRecovery(
        sampler=_estimate(8.93), refined=_estimate(8.95), log10_mass_quantiles=(8.89, 8.94, 8.99),
        centre_y_quantiles=(0.38, 0.39, 0.40), centre_x_quantiles=(-0.83, -0.82, -0.81),
        support=MassSupport(6.0, 9.7), pdf_converged=True, sample_count=4000),
    "recovery-unconverged": SubhaloRecovery(
        sampler=_estimate(8.93), refined=None, log10_mass_quantiles=None, centre_y_quantiles=None,
        centre_x_quantiles=None, support=MassSupport(6.0, 9.7), pdf_converged=False, sample_count=12),
    "forecast-reference-mismatch": ForecastReference(
        q=16.0, metric="q_mismatch", mass_msun=1.0e9, position_yx_arcsec=(0.4, -0.8), config_digest="01" * 32,
        mask_digest="02" * 32, nuisance_names=("lens.mass.einstein_radius", "background"), model_kernel=KERNEL,
        amplitude=-2.0, comparison_digest="03" * 32, noise_model="covariance:" + "04" * 32),
    "observation-noisy": ObservationRecord(kind="noisy", noise_seed=11, config_digest="05" * 32,
                                           data_digest="06" * 32, subhalo=None),
}


@pytest.mark.parametrize("name", RECORDS)
def test_case_records_round_trip_through_json(name):
    """Every INFA record type stores as plain JSON (no NaN or inf) and reads back to an equal value."""
    value = RECORDS[name]
    text = json.dumps(value.to_mapping(), allow_nan=False)
    restored = type(value).from_mapping(json.loads(text))
    assert restored == value
    assert restored.to_mapping() == value.to_mapping()


def test_record_storage_writes_non_finite_numbers_as_null_and_refuses_unknown_keys():
    mapping = _role().to_mapping()
    assert mapping["sampler"]["effective"]["n_like_max"] is None
    assert mapping["refinement"]["record"]["runs"][0]["solver_endpoint_half_chi2"] is None
    assert mapping["refinement"]["acceptance_status"] == "accepted_repeatable_profile"
    with pytest.raises(ValueError, match="RoleFit record keys"):
        RoleFit.from_mapping({**mapping, "profile": None})
    with pytest.raises(ValueError, match="SamplerRecord record keys"):
        SamplerRecord.from_mapping({key: value for key, value in mapping["sampler"].items() if key != "seed"})


@pytest.mark.parametrize(("role", "status"), [
    (_role(status="failed", log_likelihood=None, refinement=None), RoleStatus.FAILED),
    (_role(strategy="truth_anchor", sampler=None, refinement=None), RoleStatus.ZERO_RESIDUAL_ANCHOR),
    (_role(refinement=None), RoleStatus.SAMPLER_ONLY),
    (_role(refinement=_refinement(RoleStatus.UNRESOLVED)), RoleStatus.UNRESOLVED),
    (_role(), RoleStatus.ACCEPTED),
], ids=["failed", "anchor", "sampler-only", "unresolved", "accepted"])
def test_role_acceptance_status_follows_the_role_outcome(role, status):
    assert role.acceptance_status is status


@pytest.mark.parametrize(("metric", "q", "amplitude", "detected"), [
    ("q_asimov", 12.0, None, True),
    ("q_asimov", 9.0, None, False),
    ("q_mismatch", 16.0, 2.0, True),
    ("q_mismatch", 16.0, -2.0, False),
], ids=["asimov-above", "asimov-below", "mismatch-positive-amplitude", "mismatch-negative-amplitude"])
def test_forecast_reference_detection_requires_a_positive_amplitude(metric, q, amplitude, detected):
    """SCI-02: q_mismatch = a_hat^2 F with a_hat = -2, F = 4 is 16 but a template of the wrong sign."""
    reference = ForecastReference(q=q, metric=metric, mass_msun=1.0e9, position_yx_arcsec=(0.4, -0.8),
                                  config_digest="01" * 32, mask_digest="02" * 32, nuisance_names=(),
                                  model_kernel=KERNEL, amplitude=amplitude, comparison_digest="03" * 32,
                                  noise_model="diagonal")
    assert reference.detected(q_threshold=10.0) is detected
