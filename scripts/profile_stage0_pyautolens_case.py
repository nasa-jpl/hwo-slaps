#!/usr/bin/env python
"""Profile one Stage 0 PyAutoLens validation case with stage timings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = REPO_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from run_stage0_nonlinear_validation import (  # noqa: E402
    _fisher_mask_from_observation,
    _json_safe,
    _read_stage0_rows,
)

from hwoslaps.modeling.nonlinear.autolens_runner import (  # noqa: E402
    AutoLensFitRunner,
    NonlinearSearchSettings,
    _extract_log_evidence,
    _extract_result_path,
    _model_parameter_count,
    extract_max_log_likelihood_with_method,
)
from hwoslaps.modeling.nonlinear.autolens_model_builder import (  # noqa: E402
    autofit_model_from_spec,
    smooth_model_spec_from_config,
    subhalo_model_spec_from_trial,
)
from hwoslaps.modeling.nonlinear.likelihood_metrics import (  # noqa: E402
    profile_likelihood_ratio,
)
from hwoslaps.modeling.nonlinear.output_schema import NonlinearFitSummary  # noqa: E402


def _set_runtime_env() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("PYAUTO_SKIP_WORKSPACE_VERSION_CHECK", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_MAX_THREADS",
    ):
        os.environ.setdefault(name, "1")


class StageTimer:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    @contextmanager
    def measure(self, name: str, **metadata: Any) -> Iterator[None]:
        start = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            self.events.append(
                {
                    "stage": name,
                    "runtime_s": time.perf_counter() - start,
                    "status": status,
                    **metadata,
                }
            )

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for event in self.events for key in event})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.events)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["plotting"]["enabled"] = False
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _disable_expensive_autofit_output() -> None:
    from autoconf import conf

    output = conf.instance["output"]
    output["latent_during_fit"] = False
    output["latent_after_fit"] = False
    output["latent_csv"] = False
    output["latent_results"] = False
    try:
        conf.instance["visualize"]["plots_search"]["nest"]["corner_anesthetic"] = False
    except KeyError:
        pass


def _disable_analysis_visualization(analysis: Any) -> None:
    def _never_visualize(*_args: Any, **_kwargs: Any) -> bool:
        return False

    analysis.should_visualize = _never_visualize


def _run_one_model(
    *,
    timer: StageTimer,
    runner: AutoLensFitRunner,
    model: Any,
    analysis: Any,
    role: str,
    fit_mode: str,
    case_id: str,
    n_live: int,
) -> NonlinearFitSummary:
    start = time.perf_counter()
    try:
        with timer.measure(f"{role}.make_search", role=role, n_live=n_live):
            search = runner._make_search(case_id=case_id, role=role, n_live=n_live)
        with timer.measure(f"{role}.search_fit", role=role, n_live=n_live):
            result = search.fit(model=model, analysis=analysis)
        with timer.measure(f"{role}.extract_result", role=role):
            log_likelihood, method = extract_max_log_likelihood_with_method(result)
            log_evidence = _extract_log_evidence(result)
            result_path = _extract_result_path(result)
        return NonlinearFitSummary(
            model_role=role,
            fit_mode=fit_mode,
            status="success",
            log_likelihood_max=log_likelihood,
            figure_of_merit_max=log_likelihood,
            log_evidence=log_evidence,
            n_free_parameters=_model_parameter_count(model),
            result_path=result_path,
            runtime_s=time.perf_counter() - start,
            log_likelihood_extraction_method=method,
            use_jax_requested=runner.settings.use_jax,
            search_engine=runner.settings.engine,
            n_live=n_live,
        )
    except Exception as exc:
        return NonlinearFitSummary(
            model_role=role,
            fit_mode=fit_mode,
            status="failed",
            n_free_parameters=_model_parameter_count(model),
            runtime_s=time.perf_counter() - start,
            error=str(exc),
            use_jax_requested=runner.settings.use_jax,
            search_engine=runner.settings.engine,
            n_live=n_live,
        )


def profile_case(args: argparse.Namespace) -> Dict[str, Any]:
    _set_runtime_env()
    if args.fast_output:
        _disable_expensive_autofit_output()

    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.modeling.nonlinear.dataset_builder import imaging_from_observation
    from hwoslaps.modeling.nonlinear.trial import trial_from_lensing_truth
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf import generate_psf_system

    output_dir = Path(args.output_dir).resolve()
    case_dir = output_dir / args.case_label
    timer = StageTimer()
    total_start = time.perf_counter()

    with timer.measure("read_stage0_row"):
        stage0_rows = _read_stage0_rows(Path(args.stage0_results).resolve())
        stage0_row = stage0_rows[args.case]

    config_path = Path(args.config_dir).resolve() / f"{args.case}.yaml"
    with timer.measure("load_validate_config"):
        config = _load_config(config_path)
        config["run_name"] = f"stage0_pyautolens_profile_{args.case_label}"
        validate_or_raise(config)

    with timer.measure("generate_psf"):
        psf_data = generate_psf_system(config["psf"], full_config=config)

    with timer.measure("generate_baseline_lensing"):
        baseline_config = deepcopy(config)
        baseline_config["lensing"]["subhalo"]["enabled"] = False
        validate_or_raise(baseline_config)
        lensing_baseline = generate_lensing_system(
            baseline_config["lensing"],
            full_config=baseline_config,
        )

    with timer.measure("generate_baseline_observation"):
        obs_baseline = generate_observation(
            lensing_data=lensing_baseline,
            psf_data=psf_data,
            observation_config=baseline_config["observation"],
            full_config=baseline_config,
        )

    with timer.measure("generate_test_lensing"):
        test_config = deepcopy(config)
        test_config["lensing"]["subhalo"]["enabled"] = True
        validate_or_raise(test_config)
        lensing_test = generate_lensing_system(
            test_config["lensing"],
            full_config=test_config,
        )

    with timer.measure("generate_test_observation"):
        obs_test = generate_observation(
            lensing_data=lensing_test,
            psf_data=psf_data,
            observation_config=test_config["observation"],
            full_config=test_config,
        )

    with timer.measure("build_mask"):
        mask_bool_use = _fisher_mask_from_observation(obs_baseline, config)

    with timer.measure("build_autolens_dataset"):
        dataset, dataset_metadata = imaging_from_observation(
            obs_test,
            psf_for_fit=None,
            dataset_kind=args.dataset_kind,
            background_treatment="subtract_known",
            mask_bool_use=mask_bool_use,
            psf_truth_label=str(stage0_row["psf_case"]),
            psf_fit_label=str(stage0_row["psf_case"]),
        )

    with timer.measure("build_trial"):
        trial = trial_from_lensing_truth(
            lensing_test,
            case_id=f"{args.case_label}_fixed_template",
        )
        trial = replace(
            trial,
            fisher_q=float(stage0_row["q_f"]),
            fisher_z=float(stage0_row["z_f"]),
            fisher_delta_log_l_equiv=float(stage0_row["delta_log_l_f_equiv"]),
            metadata={
                **trial.metadata,
                "stage0_run_name": args.case,
                "validation_label": args.case_label,
                "gpu": args.gpu,
            },
        )

    settings = NonlinearSearchSettings(
        n_live_smooth=args.n_live_smooth,
        n_live_subhalo_fixed=args.n_live_subhalo,
        number_of_cores=1,
        maxcall=args.maxcall,
        path_prefix="profile_searches",
        unique_tag=args.case_label,
        resume=args.resume,
        use_jax=args.use_jax,
    )
    runner = AutoLensFitRunner(settings=settings, output_dir=str(output_dir))

    with timer.measure("make_analysis"):
        analysis = runner.make_analysis(dataset)
        if args.fast_output:
            _disable_analysis_visualization(analysis)

    with timer.measure("build_smooth_spec"):
        smooth_spec = smooth_model_spec_from_config(config)
    with timer.measure("build_subhalo_spec"):
        subhalo_spec = subhalo_model_spec_from_trial(
            config,
            trial=trial,
            fit_mode="fixed_template",
        )
    with timer.measure("build_smooth_autofit_model"):
        smooth_model = autofit_model_from_spec(smooth_spec)
    with timer.measure("build_subhalo_autofit_model"):
        subhalo_model = autofit_model_from_spec(subhalo_spec)

    smooth_fit = _run_one_model(
        timer=timer,
        runner=runner,
        model=smooth_model,
        analysis=analysis,
        role="smooth",
        fit_mode="fixed_template",
        case_id=trial.case_id,
        n_live=args.n_live_smooth,
    )
    subhalo_fit = _run_one_model(
        timer=timer,
        runner=runner,
        model=subhalo_model,
        analysis=analysis,
        role="subhalo",
        fit_mode="fixed_template",
        case_id=trial.case_id,
        n_live=args.n_live_subhalo,
    )

    with timer.measure("compute_metric"):
        metric = None
        q_fit = None
        if (
            smooth_fit.status == "success"
            and subhalo_fit.status == "success"
            and smooth_fit.log_likelihood_max is not None
            and subhalo_fit.log_likelihood_max is not None
        ):
            metric = profile_likelihood_ratio(
                log_l_smooth=smooth_fit.log_likelihood_max,
                log_l_subhalo=subhalo_fit.log_likelihood_max,
            )
            q_fit = float(metric.q)

    result = {
        "case": args.case,
        "case_label": args.case_label,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "gpu": args.gpu,
        "use_jax": args.use_jax,
        "dataset_kind": args.dataset_kind,
        "n_live_smooth": args.n_live_smooth,
        "n_live_subhalo": args.n_live_subhalo,
        "maxcall": args.maxcall,
        "fast_output": bool(args.fast_output),
        "fisher_q": float(stage0_row["q_f"]),
        "q_fit": q_fit,
        "q_fit_over_q_fisher": None if q_fit is None else q_fit / float(stage0_row["q_f"]),
        "metric": None if metric is None else metric.to_dict(),
        "smooth_fit": smooth_fit.to_dict(),
        "subhalo_fit": subhalo_fit.to_dict(),
        "total_runtime_s": time.perf_counter() - total_start,
        "timings": timer.events,
    }
    _write_json(case_dir / "profile_result.json", result)
    timer.write_csv(case_dir / "timings.csv")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--case-label", required=True)
    parser.add_argument("--stage0-results", required=True)
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--n-live-smooth", type=int, default=100)
    parser.add_argument("--n-live-subhalo", type=int, default=100)
    parser.add_argument("--maxcall", type=int, default=1000)
    parser.add_argument("--dataset-kind", choices=("asimov", "noisy"), default="asimov")
    parser.add_argument("--use-jax", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--fast-output",
        action="store_true",
        default=False,
        help="Disable PyAutoFit visualization and latent-output work for timing validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = profile_case(args)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
