"""Classify nonlinear cases and compare detections using the caller's recorded rule.

Status follows role outcomes and retained state; the optional stationarity requirement is
separate from the historical refinement gates. Agreement counts accepted injected cases,
reports comparison-input differences and excludes controls from like-for-like detections.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ..config.checks import Boolean, ConfigError, Key, ListOf, Nullable, Real, Table, Text
from ..identity import mapping_digest
from ..inference.result import CaseResult, RoleStatus
from ..inference.sampler import REQUIRED_SEARCH_INTERNAL_FILES

__all__ = ["AgreementTable", "AttemptSelection", "CaseClassification", "CaseStatus", "ClassificationRule",
           "RoleAcceptance", "StatusResult", "case_status", "classify_case", "detection_agreement", "select_attempt"]

CaseStatus = Literal["accepted", "unresolved", "failed", "incomplete"]
_ROLES = ("smooth", "subhalo")
_POSITIVE = Real(min=0.0, min_open=True)
_NONNEGATIVE = Real(min=0.0)
_STATIONARITY = Nullable(_POSITIVE)
_ACCEPTANCE = Table(tuple(Key(role, ListOf(Text(choices=tuple(status.value for status in RoleStatus)), unique=True),
                              f"accepted {role} role statuses") for role in _ROLES))


@dataclass(frozen=True)
class RoleAcceptance:
    smooth: frozenset[RoleStatus]
    subhalo: frozenset[RoleStatus]

    def __post_init__(self) -> None:
        for role in _ROLES:
            try:
                statuses = frozenset(RoleStatus(value) for value in getattr(self, role))
            except (TypeError, ValueError) as error:
                raise ConfigError(f"classification.acceptance.{role}", "must contain valid RoleStatus values") from error
            object.__setattr__(self, role, statuses)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str]], *, path: str = "classification.acceptance") -> RoleAcceptance:
        return cls(**{role: frozenset(values) for role, values in _ACCEPTANCE.read(mapping, path).items()})

    def to_mapping(self) -> dict[str, list[str]]:
        return {role: sorted(status.value for status in getattr(self, role)) for role in _ROLES}


@dataclass(frozen=True)
class ClassificationRule:
    q_threshold: float
    marginal_half_width: float
    acceptance: RoleAcceptance
    require_retained_state: bool
    retry_log_likelihood_tolerance: float
    stationarity_tolerance: float | None

    def __post_init__(self) -> None:
        for name, check in (("q_threshold", _POSITIVE), ("marginal_half_width", _NONNEGATIVE),
                            ("require_retained_state", Boolean()), ("retry_log_likelihood_tolerance", _NONNEGATIVE),
                            ("stationarity_tolerance", _STATIONARITY)):
            object.__setattr__(self, name, check(getattr(self, name), f"classification.{name}"))
        if not isinstance(self.acceptance, RoleAcceptance):
            raise ConfigError("classification.acceptance", "must be a RoleAcceptance")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], *, path: str = "classification") -> ClassificationRule:
        values = _CLASSIFICATION.read(mapping, path)
        values["acceptance"] = RoleAcceptance.from_mapping(values["acceptance"], path=f"{path}.acceptance")
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        return {"q_threshold": self.q_threshold, "marginal_half_width": self.marginal_half_width,
                "acceptance": self.acceptance.to_mapping(), "require_retained_state": self.require_retained_state,
                "retry_log_likelihood_tolerance": self.retry_log_likelihood_tolerance,
                "stationarity_tolerance": self.stationarity_tolerance}


_CLASSIFICATION = Table((
    Key("q_threshold", _POSITIVE, "required detection threshold"),
    Key("marginal_half_width", _NONNEGATIVE, "open half width about the threshold"),
    Key("acceptance", _ACCEPTANCE, "role statuses accepted by this rule"),
    Key("require_retained_state", Boolean(), "require the raw state of searched roles"),
    Key("retry_log_likelihood_tolerance", _NONNEGATIVE, "allowed decrease in each role on retry"),
    Key("stationarity_tolerance", _STATIONARITY, "None preserves the paper rule; positive bounds the projected gradient"),
))


@dataclass(frozen=True)
class StatusResult:
    status: CaseStatus
    role_statuses: Mapping[str, RoleStatus]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CaseClassification:
    status: CaseStatus
    q_signed: float | None
    q_clipped: float | None
    rule: ClassificationRule
    marginal: bool | None
    detected: bool | None
    role_statuses: Mapping[str, RoleStatus]
    reasons: tuple[str, ...]
    truth_shortfall: Mapping[str, float | None]


@dataclass(frozen=True)
class AttemptSelection:
    selected: Literal["first", "retry"]
    classification: CaseClassification
    detection_changed_by_retry: str | None
    q_classification_changed_by_retry: bool | None


@dataclass(frozen=True)
class AgreementTable:
    rule: ClassificationRule
    both: int
    forecast_only: int
    nonlinear_only: int
    neither: int
    excluded: Mapping[str, int]
    unfitted_forecast_nuisances: Mapping[str, int]
    mask_mismatches: int
    configuration_mismatches: int


def case_status(result: CaseResult, *, acceptance: RoleAcceptance, require_retained_state: bool,
                stationarity_tolerance: float | None) -> StatusResult:
    """Failure, retention, role acceptance and optional stationarity, independent of q."""
    require_retained_state = Boolean()(require_retained_state, "require_retained_state")
    tolerance = _STATIONARITY(stationarity_tolerance, "stationarity_tolerance")
    statuses = {role: result.role(role).acceptance_status for role in _ROLES}
    reasons = tuple(f"{role}_failed: {result.role(role).error}" for role in _ROLES if result.role(role).status == "failed")
    if reasons:
        return StatusResult("failed", statuses, reasons)
    missing = []
    if require_retained_state:
        for role in _ROLES:
            fitted = result.role(role)
            if fitted.strategy != "search":
                continue
            retention = None if fitted.sampler is None else fitted.sampler.retention
            if retention is None:
                missing.append(f"missing_retained_state: {role}: {', '.join(REQUIRED_SEARCH_INTERNAL_FILES)}")
            elif not retention.retained:
                detail = ", ".join(retention.missing_required) or "not bound to result path"
                missing.append(f"missing_retained_state: {role}: {detail}")
    if missing:
        return StatusResult("incomplete", statuses, tuple(missing))
    reasons = [f"{role}: {status.value}" for role, status in statuses.items() if status not in getattr(acceptance, role)]
    if tolerance is not None:
        for role, status in statuses.items():
            if status is RoleStatus.ACCEPTED:
                gradient = result.role(role).refinement.projected_gradient_linf
                if gradient is None or not math.isfinite(gradient) or gradient < 0.0 or gradient > tolerance:
                    reasons.append(f"not_stationary: {role}")
    return StatusResult("unresolved" if reasons else "accepted", statuses, tuple(reasons))


def classify_case(result: CaseResult, rule: ClassificationRule) -> CaseClassification:
    """Check signed q integrity, then apply status, marginal-band and detection rules."""
    q = None if result.q_signed is None else Real()(result.q_signed, "q_signed")
    likelihoods = [result.role(role).log_likelihood for role in _ROLES]
    if all(value is not None and math.isfinite(value) for value in likelihoods):
        expected = 2.0 * (likelihoods[1] - likelihoods[0])
        if q is None or not math.isclose(q, expected, rel_tol=1e-12, abs_tol=1e-7):
            raise ValueError(f"q_signed {q!r} differs from 2 (L_subhalo - L_smooth) = {expected!r}")
    outcome = case_status(result, acceptance=rule.acceptance, require_retained_state=rule.require_retained_state,
                          stationarity_tolerance=rule.stationarity_tolerance)
    if outcome.status == "accepted" and q is None:
        raise ValueError("an accepted case must have a finite q_signed")
    return CaseClassification(outcome.status, q, result.q_clipped, rule,
                              None if q is None else abs(q - rule.q_threshold) < rule.marginal_half_width,
                              None if outcome.status != "accepted" else q >= rule.q_threshold,
                              outcome.role_statuses, outcome.reasons,
                              {role: None if result.role(role).log_likelihood is None
                               else result.role(role).truth_log_likelihood - result.role(role).log_likelihood
                               for role in _ROLES})


def _same_attempt_inputs(first: CaseResult, retry: CaseResult) -> bool:
    return (first.hypothesis == retry.hypothesis and first.observation == retry.observation
            and mapping_digest(first.data) == mapping_digest(retry.data) and first.fit == retry.fit
            and first.models == retry.models and first.comparison_digest == retry.comparison_digest)


def select_attempt(first: CaseResult, retry: CaseResult | None, rule: ClassificationRule) -> AttemptSelection:
    """Promote only an accepted retry of the same case whose role likelihoods stay within tolerance."""
    initial = classify_case(first, rule)
    if retry is not None and not _same_attempt_inputs(first, retry):
        raise ValueError("retry must have the same hypothesis, observation, data, fit and model identities as first")
    if initial.status == "accepted" or retry is None:
        return AttemptSelection("first", initial, None, None)
    repeated = classify_case(retry, rule)
    if repeated.status != "accepted" or any(
            first.role(role).log_likelihood is not None
            and (retry.role(role).log_likelihood is None
                 or retry.role(role).log_likelihood < first.role(role).log_likelihood - rule.retry_log_likelihood_tolerance)
            for role in _ROLES):
        return AttemptSelection("first", initial, None, None)
    changed = None if initial.q_signed is None or repeated.q_signed is None else (
        initial.q_signed >= rule.q_threshold) != (repeated.q_signed >= rule.q_threshold)
    label = f"{initial.status}_to_{'detection' if repeated.detected else 'non_detection'}"
    return AttemptSelection("retry", repeated, label, changed)


def detection_agreement(cases: Iterable[tuple[CaseResult, CaseClassification]]) -> AgreementTable:
    """The 2x2 detection table, exclusions and forecast-input diagnostics under one rule."""
    rule = None
    counts = Counter()
    excluded = Counter()
    unfitted = Counter()
    mask_mismatches = configuration_mismatches = 0
    for result, classification in cases:
        if rule is None:
            rule = classification.rule
        elif classification.rule != rule:
            raise ValueError("detection agreement requires one ClassificationRule; mixed rules were supplied")
        if classification.status == "accepted" and (classification.detected is None or classification.q_signed is None
                                                       or not math.isfinite(classification.q_signed)):
            raise ValueError("an accepted classification must have a finite nonlinear statistic and detection")
        if classification != classify_case(result, classification.rule):
            raise ValueError("classification does not match the supplied CaseResult under its rule")
        reference = result.forecast_reference
        if reference is not None:
            if reference.mass_msun != result.hypothesis.mass_msun or reference.position_yx_arcsec != result.hypothesis.position_yx_arcsec:
                raise ValueError("forecast reference node differs from the case hypothesis")
            unfitted.update({name for name in reference.nuisance_names
                             if name not in result.fitted_parameters or name == "observation.background_offset_adu"
                             or name.startswith("psf.")})
            mask_mismatches += reference.mask_digest != result.data["mask"]["digest"]
            configuration_mismatches += reference.comparison_digest != result.comparison_digest
        if classification.status != "accepted":
            excluded[classification.status] += 1
        elif reference is None:
            excluded["no_forecast_reference"] += 1
        elif result.observation.subhalo is None:
            excluded["control_observation"] += 1
        elif result.observation.subhalo != result.hypothesis:
            excluded["observation_injection_mismatch"] += 1
        else:
            forecast = reference.detected(q_threshold=rule.q_threshold)
            key = "both" if forecast and classification.detected else "forecast_only" if forecast else (
                "nonlinear_only" if classification.detected else "neither")
            counts[key] += 1
    if rule is None:
        raise ValueError("detection agreement needs at least one case to define its ClassificationRule")
    return AgreementTable(rule, counts["both"], counts["forecast_only"], counts["nonlinear_only"], counts["neither"],
                          dict(excluded), dict(unfitted), mask_mismatches, configuration_mismatches)
