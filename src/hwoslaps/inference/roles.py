"""One case role: a checked search and optional refinement, or the expected-data truth anchor."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .refine import refine as refine_maximum
from .low_dimension import exact_likelihood, scalar_likelihood
from .result import RefineOutcome, RoleFit, RoleStatus
from .sampler import SearchOutcome, run_search
from .settings import PROCEDURE_VERSION, RefineSettings, SamplerSettings
from .starts import InsufficientStartsError, select_starts

if TYPE_CHECKING:
    from .api import PreparedCase
    from .backend import BackendSession

__all__ = ["anchor_role", "run_search_role"]


def _edge_margins(case: PreparedCase, role: str, vector: np.ndarray | None) -> dict[str, float] | None:
    if vector is None:
        return None
    model = case.model(role)
    normalized = (np.asarray(vector, dtype=float) - model.lower) / (model.upper - model.lower)
    return dict(zip(model.parameter_names, map(float, np.minimum(normalized, 1.0 - normalized)), strict=True))


def run_search_role(case: PreparedCase, role: Literal["smooth", "subhalo"], *, sampler: SamplerSettings,
                    sampler_seed: int, refine: RefineSettings | None, session: BackendSession,
                    case_dir: Path, case_id: str) -> tuple[RoleFit, SearchOutcome | None]:
    start = time.perf_counter()
    model = case.model(role)
    truth_log_likelihood = case.log_likelihood(role, case.truth_vector(role))
    dimension = len(model.parameter_names)
    if dimension <= 1:
        return _deterministic_role(case, role, truth_log_likelihood=truth_log_likelihood,
                                   sampler=sampler, start=start), None
    outcome = run_search(fit_model=model, model=case.autofit_models[role], analysis=case.analysis, role=role,
                         n_live=sampler.n_live(role, case.fit.mode), settings=sampler, seed=sampler_seed,
                         case_dir=case_dir, case_id=case_id, data_identity=case.data.record, session=session)
    refinement = None
    log_likelihood = outcome.record.log_likelihood_max
    error = outcome.error
    vector = None
    if outcome.result is not None:
        if refine is None:
            vector = np.asarray(outcome.result.samples.max_log_likelihood(as_instance=False), dtype=float)
        else:
            try:
                starts = select_starts(case_dir / outcome.record.output_path / "files", model.parameter_names,
                                       model.lower, model.upper, refine)
            except InsufficientStartsError as selection_error:
                vector = selection_error.starts[0].physical
                log_likelihood = case.log_likelihood(role, vector)
                if not np.isfinite(log_likelihood):
                    log_likelihood = None
                refinement = RefineOutcome(
                    acceptance_status=RoleStatus.UNRESOLVED, best_log_likelihood=log_likelihood,
                    best_vector=vector, gates=dict.fromkeys(("support", "repeat", "finite_gradient", "scalar_residual",
                                                           "direct_log_likelihood", "incumbent"), False),
                    record={"start_selection_error": str(selection_error),
                            "start_provenance": [start.to_mapping() for start in selection_error.starts],
                            "sampler_incumbent_retained": log_likelihood is not None,
                            "procedure": refine.to_mapping(), "procedure_version": PROCEDURE_VERSION},
                    projected_gradient_linf=None, repeat_converged=None)
            else:
                refinement = refine_maximum(starts, case.objective(role), refine)
            log_likelihood = refinement.best_log_likelihood
            vector = refinement.best_vector
            if log_likelihood is None:
                error = "refinement produced no finite evaluation"
    status = "success" if error is None and log_likelihood is not None else "failed"
    result = RoleFit(role=role, strategy="search", status=status, log_likelihood=log_likelihood,
                     truth_log_likelihood=truth_log_likelihood, parameter_names=model.parameter_names,
                     n_free_parameters=len(model.parameter_names), sampler=outcome.record, refinement=refinement,
                     box_edge_margins=_edge_margins(case, role, vector), anchor_chi2=None, error=error,
                     runtime_s=time.perf_counter() - start)
    return result, outcome


def _deterministic_role(case: PreparedCase, role: Literal["smooth", "subhalo"], *,
                        truth_log_likelihood: float, sampler: SamplerSettings, start: float) -> RoleFit:
    model = case.model(role)
    dimension = len(model.parameter_names)
    strategy = "exact" if dimension == 0 else "scalar"
    vector, likelihood, record, error = None, None, None, None
    try:
        outcome = (exact_likelihood(truth_log_likelihood) if dimension == 0 else
                   scalar_likelihood(lambda values: case.log_likelihood(role, values),
                                     model.lower[0], model.upper[0], parameter_name=model.parameter_names[0],
                                     max_likelihood_calls=sampler.n_like_max,
                                     initial_evaluation=(case.truth_vector(role)[0], truth_log_likelihood)))
        vector, likelihood, record = outcome.best_vector, outcome.best_log_likelihood, outcome.record
        record["requested_likelihood_calls"] = sampler.n_like_max
        record["call_count_convention"] = {
            "role": "truth plus all new scalar search, quadrature and polishing evaluations",
            "external_preflight": "validate_nonlinear evaluates one truth likelihood per role before dispatch",
            "validate_nonlinear_preflight_calls_per_role": 1,
            "external_preflight_in_budget": False, "separate_refinement_calls": 0,
            "cache_hits_counted": False}
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    return RoleFit(role=role, strategy=strategy, status="success" if error is None else "failed",
                   log_likelihood=likelihood, truth_log_likelihood=truth_log_likelihood,
                   parameter_names=model.parameter_names, n_free_parameters=dimension,
                   sampler=None, refinement=None, box_edge_margins=_edge_margins(case, role, vector),
                   anchor_chi2=None, error=error, runtime_s=time.perf_counter() - start, strategy_record=record)


def anchor_role(case: PreparedCase) -> RoleFit:
    start = time.perf_counter()
    role, vector = "subhalo", case.truth_vector("subhalo")
    likelihood = case.log_likelihood(role, vector)
    model = case.model(role)
    return RoleFit(role=role, strategy="truth_anchor", status="success", log_likelihood=likelihood,
                   truth_log_likelihood=likelihood, parameter_names=model.parameter_names,
                   n_free_parameters=len(model.parameter_names), sampler=None, refinement=None,
                   box_edge_margins=_edge_margins(case, role, vector), anchor_chi2=case.chi_squared(role, vector),
                   error=None, runtime_s=time.perf_counter() - start)
