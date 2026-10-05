"""Run B4a/B4b fit-engine anchors or B4L likelihood timing against the captured baseline.

Run on the supported GPU backend with BLAS threads 1 and JAX x64 enabled. --output records
results and timing ratios; every case uses a fresh temporary namespace below --work-dir.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hwoslaps.identity import array_digest, json_ready
from hwoslaps.inference.api import prepare_case, validate_nonlinear
from hwoslaps.inference.result import CaseResult
from hwoslaps.inference.settings import FitSpec, MassSupport, RefineSettings, SamplerSettings

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/paper_parity/engine"


def _inputs(name: str, config_path: Path | None = None):
    from hwoslaps.fisher.api import Execution, prepare_forecast
    scene = "p1_optical_matched" if name == "B4a" else "p2_delta_knowledge_error"
    prepared = prepare_forecast(config_path or FIXTURES / f"{scene}.yaml", execution=Execution(engine="reference"))
    trial = prepared.hypothesis(1.0e9, (0.4, -0.8))
    fit = FitSpec(mode="fixed_template") if name == "B4a" else FitSpec(mode="freed", mass_support=MassSupport(6.0, 9.7))
    return prepared, trial, fit


def _median(call, samples: int) -> float:
    for _ in range(3):
        call()
    elapsed = []
    for _ in range(samples):
        start = perf_counter()
        call()
        elapsed.append(perf_counter() - start)
    return float(statistics.median(elapsed))


def _unit_batch(model, size: int) -> np.ndarray:
    unit = np.random.default_rng(0).uniform(0.4, 0.6, size=(size, model.prior_count))
    return np.array([model.vector_from_unit_vector(unit_vector=list(row)) for row in unit])


def fit_engine_results(result: CaseResult) -> dict[str, Any]:
    roles = {}
    for role in ("smooth", "subhalo"):
        fitted = result.role(role)
        refinement = fitted.refinement
        roles[role] = {"model_role": role, "status": fitted.status,
                       "sampler_log_likelihood_max": fitted.sampler.log_likelihood_max,
                       "refined_log_likelihood_max": fitted.log_likelihood,
                       "candidate_best_log_likelihood": None if refinement is None else refinement.best_log_likelihood,
                       "candidate_best_vector": None if refinement is None else refinement.best_vector,
                       "candidate_best_z": None if refinement is None else refinement.record["candidate_best_z"],
                       "candidate_acceptance_status": fitted.acceptance_status.value,
                       "candidate_best_projected_gradient": None if refinement is None
                       else refinement.record["candidate_best_projected_gradient"]}
    recovery = None if result.recovery is None else result.recovery.to_mapping()
    return json_ready({"roles": roles, "q": result.q_signed,
                       "signed_delta_log_l": None if result.q_signed is None else 0.5 * result.q_signed,
                       "recovery": recovery})


def run_fit_case(name: str, work_dir: Path, *, config_path: Path | None = None) -> tuple[CaseResult, dict[str, Any]]:
    from autofit.non_linear.fitness import Fitness

    if name not in ("B4a", "B4b"):
        raise ValueError(f"fit case must be B4a or B4b, got {name!r}")
    prepared, trial, fit = _inputs(name, config_path)
    try:
        sampler = SamplerSettings(n_live_smooth=50, n_live_subhalo_fixed=50, n_live_subhalo_search=80,
                                  n_eff=200, n_shell=1, f_live=0.01, discard_exploration=False,
                                  use_jax=True, jax_n_batch=50, retain_search_internal=True)
        from hwoslaps.simulation import simulate

        start = perf_counter()
        observation = simulate(prepared, subhalo=trial, noise_seed=None if name == "B4a" else 11)
        simulation_seconds = perf_counter() - start
        fit_start = perf_counter()
        result = validate_nonlinear(prepared, trial, observation, fit=fit, sampler=sampler, sampler_seed=20261005,
                                    refine=RefineSettings(), output_dir=work_dir, case_id=name)
        fit_only_seconds = perf_counter() - fit_start
        wall = perf_counter() - start
        case = prepare_case(prepared, trial, observation, fit=fit, use_jax=True)
        timings = {}
        for role in ("smooth", "subhalo"):
            fitness = Fitness(model=case.autofit_models[role], analysis=case.analysis,
                              fom_is_log_likelihood=True, resample_figure_of_merit=-1.0e99,
                              use_jax_vmap=True, batch_size=50)
            vectors = _unit_batch(case.autofit_models[role], 50)
            timings[f"{role}_fitness_batch_seconds_median"] = _median(lambda: np.asarray(fitness(vectors)), 20)
            fitted = result.role(role)
            if fitted.refinement is None or fitted.refinement.best_vector is None:
                raise RuntimeError(f"{name} {role} did not produce a refined finite maximum: {fitted.error}")
            point = np.asarray(fitted.refinement.record["candidate_best_z"], dtype=float)
            objective = case.objective(role)
            timings[f"{role}_objective_value_and_gradient_seconds_median"] = _median(
                lambda: objective.value_and_gradient(point), 50)
        return result, {"case_wall_seconds": wall, "simulation_seconds": simulation_seconds,
                        "fit_only_seconds": fit_only_seconds, "inclusive_case_wall_seconds": wall, **timings, "results": fit_engine_results(result)}
    finally:
        prepared.close()


def run_likelihood(*, config_path: Path | None = None) -> dict[str, Any]:
    import jax

    prepared, trial, fit = _inputs("B4L", config_path)
    from hwoslaps.simulation import simulate

    try:
        observation = simulate(prepared, subhalo=trial, noise_seed=11)
        case = prepare_case(prepared, trial, observation, fit=fit, use_jax=True)
        model = case.autofit_models["subhalo"]
        def call(parameters):
            instance = model.instance_from_vector(vector=parameters, xp=jax.numpy)
            return case.analysis.log_likelihood_function(instance=instance)
        batched = jax.vmap(jax.jit(call))
        vectors = _unit_batch(model, 32)
        start = perf_counter()
        values = np.asarray(batched(vectors).block_until_ready())
        first = perf_counter() - start
        start = perf_counter()
        for _ in range(200):
            last = batched(vectors)
        last.block_until_ready()
        elapsed = perf_counter() - start
        np.testing.assert_array_equal(np.asarray(last), values)
        digest = array_digest(values)
        return {"first_batch_seconds": first, "ms_per_batch": elapsed * 1000.0 / 200,
                "values": values.tolist(), "values_digest": digest}
    finally:
        prepared.close()


def _recovery_matches(actual: dict[str, Any] | None, baseline: dict[str, Any] | None) -> bool:
    if actual is None or baseline is None:
        return actual is baseline
    sampler = actual["sampler"]
    return (sampler["log10_mass"] == baseline["log10_m200_ml"]
            and sampler["centre_yx"] == [baseline["centre_ml_y"], baseline["centre_ml_x"]]
            and sampler["profile_scales"] == {"kappa_s": baseline["kappa_s_ml"],
                                              "scale_radius": baseline["scale_radius_arcsec_ml"]}
            and actual["log10_mass_quantiles"] == [baseline[f"log10_m200_p{p}"] for p in (16, 50, 84)]
            and actual["centre_y_quantiles"] == [baseline[f"centre_y_p{p}"] for p in (16, 50, 84)]
            and actual["centre_x_quantiles"] == [baseline[f"centre_x_p{p}"] for p in (16, 50, 84)]
            and actual["pdf_converged"] == baseline["pdf_converged"]
            and actual["sample_count"] == baseline["n_samples"])


def check_results(actual: dict[str, Any], baseline: dict[str, Any]) -> bool:
    # Stationarity is recorded beside historical acceptance, as required by SCI-14.
    roles_match = all({key: value for key, value in actual["roles"][role].items()
                       if key != "candidate_best_projected_gradient"}
                      == {key: value for key, value in baseline["roles"][role].items()
                          if key != "candidate_best_projected_gradient"} for role in ("smooth", "subhalo"))
    return (roles_match and actual["q"] == baseline["q"]
            and actual["signed_delta_log_l"] == baseline["signed_delta_log_l"]
            and _recovery_matches(actual["recovery"], baseline["recovery"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("B4a", "B4b", "B4L"), required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    import importlib.util
    from autoconf import conf
    conf.instance.push(str(Path(importlib.util.find_spec("autoarray").origin).parent / "config"), keep_first=True)
    reference = json.loads(args.baseline.read_text())["workloads"][args.case]["summary"]
    with tempfile.TemporaryDirectory(prefix="nonlinear-benchmark-", dir=args.work_dir) as space:
        if args.case == "B4L":
            report = run_likelihood(config_path=args.config)
            ratios = {"likelihood": report["ms_per_batch"] / reference["ms_per_batch_min"]}
            same = report["values_digest"] == reference["values_digest"]
        else:
            _, report = run_fit_case(args.case, Path(space), config_path=args.config)
            ratios = {"wall": report["case_wall_seconds"] / reference["case_wall_seconds_min"]}
            for role in ("smooth", "subhalo"):
                for kind in ("fitness_batch", "objective_value_and_gradient"):
                    field = f"{role}_{kind}_seconds_median"
                    ratios[f"{role}_{kind}"] = report[field] / reference[field + "_min"]
            same = check_results(report["results"], reference["results"])
        passed = same and all(ratio <= (1.10 if name == "wall" else 1.05) for name, ratio in ratios.items())
        report.update(case=args.case, ratios=ratios, results_equal=same, passed=passed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"case": args.case, "ratios": ratios, "results_equal": same, "passed": passed}))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
