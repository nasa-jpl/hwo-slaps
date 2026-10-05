"""Records of a nonlinear case: role statuses, sampler and refinement records, the case result.

The fit API computes and records no thresholded quantity: detection, the marginal band and
retry decisions are reductions in ``analysis.nonlinear`` with the threshold as an argument.
Every record has ``to_mapping`` (plain JSON types, non-finite numbers written as null) and
``from_mapping``, which refuses missing and unknown keys.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from ..identity import KernelIdentity, json_ready
from .settings import FitSpec, RefineSettings, SamplerSettings

if TYPE_CHECKING:
    from ..scene.halos import Halo
    from .recovery import SubhaloRecovery

__all__ = [
    "CaseResult", "ForecastReference", "ObservationRecord", "RefineOutcome", "RetentionInventory", "RoleFit",
    "RoleStatus", "SamplerRecord", "check_record_keys", "finite_json",
]


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def finite_json(value: Any) -> Any:
    """``value`` in plain JSON types (``identity.json_ready``) with every non-finite float as None."""
    return json_ready(_finite_or_none(value))


def check_record_keys(mapping: Mapping[str, Any], cls: type) -> None:
    """Refuse a record of a dataclass whose keys are not exactly its field names."""
    expected = [item.name for item in dataclasses.fields(cls)]
    if set(mapping) != set(expected):
        raise ValueError(f"{cls.__name__} record keys must be {expected}, got {sorted(map(str, mapping))}")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


class RoleStatus(StrEnum):
    """Acceptance status of one role fit; the strings are the values 8fa6209 and the paper recorded."""

    ACCEPTED = "accepted_repeatable_profile"
    UNRESOLVED = "unresolved_optimization"
    SAMPLER_ONLY = "sampler_only"
    ZERO_RESIDUAL_ANCHOR = "verified_zero_residual_anchor"
    FAILED = "failed"


@dataclass(frozen=True)
class RetentionInventory:
    """The raw sampler state a search left on disk, with byte counts and SHA-256 digests.

    ``files`` maps a file name to ``{"bytes", "sha256", "location"}``; ``location`` is the path
    below the search output directory (route ``directory``) or the member name inside
    ``<output_path>.zip`` (route ``zip``).
    """

    route: Literal["directory", "zip"] | None
    files: Mapping[str, Mapping[str, Any]]
    missing_required: tuple[str, ...]
    bound_to_result_path: bool

    @property
    def retained(self) -> bool:
        return not self.missing_required and self.bound_to_result_path

    def to_mapping(self) -> dict[str, Any]:
        return {"route": self.route, "files": {name: dict(entry) for name, entry in self.files.items()},
                "missing_required": list(self.missing_required),
                "bound_to_result_path": self.bound_to_result_path}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> RetentionInventory:
        check_record_keys(mapping, cls)
        return cls(route=mapping["route"], files={name: dict(entry) for name, entry in mapping["files"].items()},
                   missing_required=tuple(mapping["missing_required"]),
                   bound_to_result_path=bool(mapping["bound_to_result_path"]))


@dataclass(frozen=True)
class SamplerRecord:
    """One role search: where it ran, its identity, settings as requested and as constructed, outputs.

    ``output_path`` is relative to the case directory. ``log_likelihood_max``,
    ``log_evidence`` and ``likelihood_calls`` are None for a failed search.
    """

    name: str
    output_path: str
    identity: str
    n_live: int
    requested: Mapping[str, Any]
    effective: Mapping[str, Any]
    seed: int
    training_workers: int
    log_likelihood_max: float | None
    log_evidence: float | None
    likelihood_calls: int | None
    retention: RetentionInventory | None
    runtime_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested", finite_json(self.requested))
        object.__setattr__(self, "effective", finite_json(self.effective))

    def to_mapping(self) -> dict[str, Any]:
        record = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        record["retention"] = None if self.retention is None else self.retention.to_mapping()
        return finite_json(record)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SamplerRecord:
        check_record_keys(mapping, cls)
        retention = mapping["retention"]
        calls = mapping["likelihood_calls"]
        return cls(name=mapping["name"], output_path=mapping["output_path"], identity=mapping["identity"],
                   n_live=int(mapping["n_live"]), requested=dict(mapping["requested"]),
                   effective=dict(mapping["effective"]), seed=int(mapping["seed"]),
                   training_workers=int(mapping["training_workers"]),
                   log_likelihood_max=_optional_float(mapping["log_likelihood_max"]),
                   log_evidence=_optional_float(mapping["log_evidence"]),
                   likelihood_calls=None if calls is None else int(calls),
                   retention=None if retention is None else RetentionInventory.from_mapping(retention),
                   runtime_s=float(mapping["runtime_s"]))


@dataclass(frozen=True)
class RefineOutcome:
    """Result of refining one role maximum.

    ``best_vector`` is physical, in the model's parameter order; ``best_log_likelihood`` is the
    direct likelihood there. ``gates`` holds the six acceptance gates. ``projected_gradient_linf``
    is the L-infinity norm of the unit-box gradient of the half chi-square at the retained best,
    with outward components at an active bound (within 1e-10) zeroed, None when that gradient is
    not finite; ``repeat_converged`` is the scipy success flag of the tighter repeat, None when
    the repeat raised. Neither enters the gates, so the recorded statuses keep their paper
    meaning; a classification rule can require stationarity.
    """

    acceptance_status: RoleStatus
    best_log_likelihood: float | None
    best_vector: tuple[float, ...] | None
    gates: Mapping[str, bool]
    record: Mapping[str, Any]
    projected_gradient_linf: float | None
    repeat_converged: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceptance_status", RoleStatus(self.acceptance_status))
        object.__setattr__(self, "record", finite_json(self.record))
        if self.best_vector is not None:
            object.__setattr__(self, "best_vector", tuple(float(value) for value in self.best_vector))

    def to_mapping(self) -> dict[str, Any]:
        record = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        record["acceptance_status"] = self.acceptance_status.value
        return finite_json(record)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> RefineOutcome:
        check_record_keys(mapping, cls)
        vector = mapping["best_vector"]
        converged = mapping["repeat_converged"]
        return cls(acceptance_status=RoleStatus(mapping["acceptance_status"]),
                   best_log_likelihood=_optional_float(mapping["best_log_likelihood"]),
                   best_vector=None if vector is None else tuple(float(item) for item in vector),
                   gates={name: bool(value) for name, value in mapping["gates"].items()},
                   record=dict(mapping["record"]),
                   projected_gradient_linf=_optional_float(mapping["projected_gradient_linf"]),
                   repeat_converged=None if converged is None else bool(converged))


@dataclass(frozen=True)
class RoleFit:
    """One role (H0 smooth or H1 subhalo) of a case.

    ``log_likelihood`` is the refined direct likelihood when refinement ran, the sampler maximum
    otherwise, the truth value for the anchor, and None for a failed role.
    ``box_edge_margins`` is the retained best's distance to the nearer box edge per free
    parameter, in units of the box width.
    """

    role: Literal["smooth", "subhalo"]
    strategy: Literal["search", "truth_anchor"]
    status: Literal["success", "failed"]
    log_likelihood: float | None
    truth_log_likelihood: float
    parameter_names: tuple[str, ...]
    n_free_parameters: int
    sampler: SamplerRecord | None
    refinement: RefineOutcome | None
    box_edge_margins: Mapping[str, float] | None
    anchor_chi2: float | None
    error: str | None
    runtime_s: float

    def __post_init__(self) -> None:
        if self.role not in ("smooth", "subhalo"):
            raise ValueError(f"role must be smooth or subhalo, got {self.role!r}")
        if self.strategy not in ("search", "truth_anchor"):
            raise ValueError(f"strategy must be search or truth_anchor, got {self.strategy!r}")
        if self.status not in ("success", "failed"):
            raise ValueError(f"status must be success or failed, got {self.status!r}")
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))

    @property
    def acceptance_status(self) -> RoleStatus:
        if self.status == "failed":
            return RoleStatus.FAILED
        if self.strategy == "truth_anchor":
            return RoleStatus.ZERO_RESIDUAL_ANCHOR
        if self.refinement is None:
            return RoleStatus.SAMPLER_ONLY
        return self.refinement.acceptance_status

    def to_mapping(self) -> dict[str, Any]:
        record = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        record["sampler"] = None if self.sampler is None else self.sampler.to_mapping()
        record["refinement"] = None if self.refinement is None else self.refinement.to_mapping()
        return finite_json(record)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> RoleFit:
        check_record_keys(mapping, cls)
        sampler, refinement, margins = mapping["sampler"], mapping["refinement"], mapping["box_edge_margins"]
        return cls(role=mapping["role"], strategy=mapping["strategy"], status=mapping["status"],
                   log_likelihood=_optional_float(mapping["log_likelihood"]),
                   truth_log_likelihood=float(mapping["truth_log_likelihood"]),
                   parameter_names=tuple(mapping["parameter_names"]),
                   n_free_parameters=int(mapping["n_free_parameters"]),
                   sampler=None if sampler is None else SamplerRecord.from_mapping(sampler),
                   refinement=None if refinement is None else RefineOutcome.from_mapping(refinement),
                   box_edge_margins=None if margins is None else {name: float(v) for name, v in margins.items()},
                   anchor_chi2=_optional_float(mapping["anchor_chi2"]), error=mapping["error"],
                   runtime_s=float(mapping["runtime_s"]))


@dataclass(frozen=True)
class ForecastReference:
    """Where a compared forecast q came from: its metric, node, configuration, mask and kernel.

    ``amplitude`` is the free-amplitude estimate at the node for ``q_mismatch`` and None for
    ``q_asimov``; ``noise_model`` is ``"diagonal"`` or ``"covariance:<file sha256>"``.
    """

    q: float
    metric: Literal["q_asimov", "q_mismatch"]
    mass_msun: float
    position_yx_arcsec: tuple[float, float]
    config_digest: str
    mask_digest: str
    nuisance_names: tuple[str, ...]
    model_kernel: KernelIdentity
    amplitude: float | None
    comparison_digest: str
    noise_model: str

    def __post_init__(self) -> None:
        if self.metric not in ("q_asimov", "q_mismatch"):
            raise ValueError(f"metric must be q_asimov or q_mismatch, got {self.metric!r}")
        if (self.amplitude is None) != (self.metric == "q_asimov"):
            raise ValueError("amplitude is recorded for q_mismatch and only for it")
        object.__setattr__(self, "position_yx_arcsec", tuple(float(v) for v in self.position_yx_arcsec))
        object.__setattr__(self, "nuisance_names", tuple(self.nuisance_names))

    def detected(self, *, q_threshold: float) -> bool:
        """``q >= q_threshold`` with a positive amplitude where one is recorded (the forecast rule)."""
        return self.q >= q_threshold and (self.amplitude is None or self.amplitude > 0.0)

    def to_mapping(self) -> dict[str, Any]:
        record = {item.name: getattr(self, item.name) for item in dataclasses.fields(self)}
        record["model_kernel"] = self.model_kernel.to_mapping()
        return finite_json(record)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ForecastReference:
        check_record_keys(mapping, cls)
        return cls(q=float(mapping["q"]), metric=mapping["metric"], mass_msun=float(mapping["mass_msun"]),
                   position_yx_arcsec=tuple(mapping["position_yx_arcsec"]),
                   config_digest=mapping["config_digest"], mask_digest=mapping["mask_digest"],
                   nuisance_names=tuple(mapping["nuisance_names"]),
                   model_kernel=KernelIdentity.from_mapping(mapping["model_kernel"]),
                   amplitude=_optional_float(mapping["amplitude"]),
                   comparison_digest=mapping["comparison_digest"], noise_model=mapping["noise_model"])


def _halo_from_mapping(mapping: Mapping[str, Any] | None) -> Halo | None:
    if mapping is None:
        return None
    from ..scene.halos import Halo

    return Halo.from_mapping(mapping)


@dataclass(frozen=True)
class ObservationRecord:
    """The fitted observation: its kind, noise seed, configuration and data digests, injected halo."""

    kind: Literal["expected", "noisy"]
    noise_seed: int | None
    config_digest: str
    data_digest: str
    subhalo: Halo | None

    def __post_init__(self) -> None:
        if self.kind not in ("expected", "noisy"):
            raise ValueError(f"kind must be expected or noisy, got {self.kind!r}")
        if (self.noise_seed is None) != (self.kind == "expected"):
            raise ValueError("a noisy observation records its noise seed and an expected one has none")

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": self.kind, "noise_seed": self.noise_seed, "config_digest": self.config_digest,
                "data_digest": self.data_digest,
                "subhalo": None if self.subhalo is None else self.subhalo.to_mapping()}

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ObservationRecord:
        check_record_keys(mapping, cls)
        seed = mapping["noise_seed"]
        return cls(kind=mapping["kind"], noise_seed=None if seed is None else int(seed),
                   config_digest=mapping["config_digest"], data_digest=mapping["data_digest"],
                   subhalo=_halo_from_mapping(mapping["subhalo"]))


@dataclass(frozen=True)
class CaseResult:
    """One nonlinear case: hypothesis, fitted observation, settings, both roles, q and recovery.

    ``q_signed`` and ``q_clipped`` come from the two role log-likelihoods when both exist;
    ``delta_log_evidence`` when both roles were searched successfully with finite evidence.
    ``fitted_parameters`` are the scene parameter names H0 frees, in the forecast nuisance
    vocabulary.
    """

    case_id: str
    hypothesis: Halo
    observation: ObservationRecord
    data: Mapping[str, Any]
    fit: FitSpec
    sampler: SamplerSettings
    sampler_seed: int
    refine: RefineSettings | None
    models: Mapping[Literal["smooth", "subhalo"], str]
    smooth: RoleFit
    subhalo: RoleFit
    q_signed: float | None
    q_clipped: float | None
    delta_log_evidence: float | None
    recovery: SubhaloRecovery | None
    forecast_reference: ForecastReference | None
    fitted_parameters: tuple[str, ...]
    comparison_digest: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.smooth.role != "smooth" or self.subhalo.role != "subhalo":
            raise ValueError(f"roles must be (smooth, subhalo), got ({self.smooth.role}, {self.subhalo.role})")
        object.__setattr__(self, "fitted_parameters", tuple(self.fitted_parameters))

    def role(self, name: Literal["smooth", "subhalo"]) -> RoleFit:
        if name == "smooth":
            return self.smooth
        if name == "subhalo":
            return self.subhalo
        raise ValueError(f"role must be smooth or subhalo, got {name!r}")

    def to_mapping(self) -> dict[str, Any]:
        return finite_json({
            "case_id": self.case_id, "hypothesis": self.hypothesis.to_mapping(),
            "observation": self.observation.to_mapping(), "data": self.data, "fit": self.fit.to_mapping(),
            "sampler": self.sampler.to_mapping(), "sampler_seed": self.sampler_seed,
            "refine": None if self.refine is None else self.refine.to_mapping(), "models": self.models,
            "smooth": self.smooth.to_mapping(), "subhalo": self.subhalo.to_mapping(),
            "q_signed": self.q_signed, "q_clipped": self.q_clipped,
            "delta_log_evidence": self.delta_log_evidence,
            "recovery": None if self.recovery is None else self.recovery.to_mapping(),
            "forecast_reference": None if self.forecast_reference is None else self.forecast_reference.to_mapping(),
            "fitted_parameters": self.fitted_parameters, "comparison_digest": self.comparison_digest,
            "provenance": self.provenance,
        })

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> CaseResult:
        from .recovery import SubhaloRecovery

        check_record_keys(mapping, cls)
        if not isinstance(mapping["fit"]["mask"], str):
            raise ValueError("this case was fitted with a custom PixelMask, which is recorded by digest only "
                             "and cannot be rebuilt from the record")
        refine, recovery, reference = mapping["refine"], mapping["recovery"], mapping["forecast_reference"]
        return cls(case_id=mapping["case_id"], hypothesis=_halo_from_mapping(mapping["hypothesis"]),
                   observation=ObservationRecord.from_mapping(mapping["observation"]), data=dict(mapping["data"]),
                   fit=FitSpec.from_mapping(mapping["fit"]), sampler=SamplerSettings.from_mapping(mapping["sampler"]),
                   sampler_seed=int(mapping["sampler_seed"]),
                   refine=None if refine is None else RefineSettings.from_mapping(refine),
                   models=dict(mapping["models"]), smooth=RoleFit.from_mapping(mapping["smooth"]),
                   subhalo=RoleFit.from_mapping(mapping["subhalo"]),
                   q_signed=_optional_float(mapping["q_signed"]), q_clipped=_optional_float(mapping["q_clipped"]),
                   delta_log_evidence=_optional_float(mapping["delta_log_evidence"]),
                   recovery=None if recovery is None else SubhaloRecovery.from_mapping(recovery),
                   forecast_reference=None if reference is None else ForecastReference.from_mapping(reference),
                   fitted_parameters=tuple(mapping["fitted_parameters"]),
                   comparison_digest=mapping["comparison_digest"], provenance=dict(mapping["provenance"]))
