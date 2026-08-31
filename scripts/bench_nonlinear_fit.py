"""Fixed-seed nonlinear (PyAutoLens + Nautilus) performance bench.

Runs the determinism-anchor freed smooth/subhalo pair from the Item 9
science ladder: the matched (delta = 0) case on the ``science_hwo35``
truth config under sampler seed 11, JAX batch 32, and maxcall 500k.

The pair reproduces the standing anchors ``q_fit = 14.570080472738``
and ``Delta logZ = 3.420182478498`` bit-exactly, so the bench doubles
as the identity check for any performance change: a run whose anchors
move is not a valid optimization.

Phase timings and the anchor deltas are written to a JSON report so
runs can be compared without re-reading logs.

Engineering bench, disposable. Not paper data.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import yaml

ANCHOR_Q = 14.570080472738
ANCHOR_DLOGZ = 3.420182478498
ANCHOR_TOL = 1.0e-9

DELTA_BLOCK = {
    "prior_table": "configs/psf_priors/jwst_wss_drift_v1.yaml",
    "seed": 20260814,
    "family": "combined",
    "amplitude_rms_nm": 0.0,
}


def _build_parser() -> argparse.ArgumentParser:
    """Build the bench command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Truth-state configuration YAML")
    parser.add_argument("output_dir", help="Directory for fit outputs")
    parser.add_argument(
        "--case-id",
        default="bench",
        help="Case identifier; change it to force a fresh search path",
    )
    parser.add_argument(
        "--n-live-smooth",
        type=int,
        default=100,
        help="Live points for the smooth fit",
    )
    parser.add_argument(
        "--n-live-subhalo",
        type=int,
        default=200,
        help="Live points for the freed subhalo fit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Nautilus sampler seed",
    )
    parser.add_argument(
        "--jax-n-batch",
        type=int,
        default=32,
        help="Vectorized AutoFit likelihood batch size",
    )
    parser.add_argument(
        "--maxcall",
        type=int,
        default=500_000,
        help="Maximum likelihood calls",
    )
    parser.add_argument(
        "--check-anchor",
        action="store_true",
        help="Assert bit-exact reproduction of the standing anchors",
    )
    return parser


def main(argv=None) -> None:
    """Run the fixed-seed nonlinear bench pair."""
    args = _build_parser().parse_args(argv)

    from hwoslaps.lensing.generator import generate_lensing_system
    from hwoslaps.modeling.nonlinear.autolens_runner import (
        AutoLensFitRunner,
        NonlinearSearchSettings,
    )
    from hwoslaps.modeling.nonlinear.mass_mapping import (
        build_mass_mapping_context,
    )
    from hwoslaps.modeling.nonlinear.psf_mismatch import run_psf_mismatch_case
    from hwoslaps.modeling.nonlinear.trial import trial_from_lensing_truth
    from hwoslaps.modeling.nonlinear.validator import NonlinearMetricValidator
    from hwoslaps.observation.generator import generate_observation
    from hwoslaps.psf.generator import generate_psf_system

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as stream:
        base_config = yaml.safe_load(stream)
    base_config.pop("provenance_note", None)
    base_config["plotting"] = {"enabled": False}

    timings = {}

    start = time.time()
    lensing_data = generate_lensing_system(
        base_config["lensing"], full_config=base_config
    )
    psf_data = generate_psf_system(base_config["psf"], full_config=base_config)
    observation = generate_observation(
        lensing_data=lensing_data,
        psf_data=psf_data,
        observation_config=base_config["observation"],
        full_config=base_config,
    )
    timings["scene_psf_observation_s"] = time.time() - start
    print(
        f"scene+psf+observation: {timings['scene_psf_observation_s']:.1f} s",
        flush=True,
    )

    trial = trial_from_lensing_truth(lensing_data, case_id=args.case_id)
    mass_context = build_mass_mapping_context(base_config)

    config = copy.deepcopy(base_config)
    config["modeling"]["fit_psf"] = {
        "mode": "delta",
        "delta": dict(DELTA_BLOCK),
    }

    runner = AutoLensFitRunner(
        NonlinearSearchSettings(
            n_live_smooth=args.n_live_smooth,
            n_live_subhalo_search=args.n_live_subhalo,
            number_of_cores=1,
            maxcall=args.maxcall,
            seed=args.seed,
            path_prefix="bench",
            use_jax=True,
            jax_n_batch=args.jax_n_batch,
        ),
        output_dir=str(output_dir),
    )
    validator = NonlinearMetricValidator(runner)

    start = time.time()
    result = run_psf_mismatch_case(
        validator,
        observation,
        config,
        trial,
        fit_mode="freed",
        mass_context=mass_context,
    )
    timings["fit_pair_s"] = time.time() - start
    print(
        f"fit pair: {timings['fit_pair_s']:.1f} s "
        f"q_fit={result.q_fit!r} dlogZ={result.delta_log_evidence!r}",
        flush=True,
    )

    case = result.case
    report = {
        "case_id": args.case_id,
        "seed": args.seed,
        "n_live_smooth": args.n_live_smooth,
        "n_live_subhalo": args.n_live_subhalo,
        "jax_n_batch": args.jax_n_batch,
        "maxcall": args.maxcall,
        "timings": timings,
        "q_fit": result.q_fit,
        "delta_log_evidence": result.delta_log_evidence,
        "smooth_status": result.smooth_status,
        "subhalo_status": result.subhalo_status,
        "smooth_log_likelihood_max": case.smooth_fit.log_likelihood_max,
        "subhalo_log_likelihood_max": case.subhalo_fit.log_likelihood_max,
        "smooth_log_evidence": case.smooth_fit.log_evidence,
        "subhalo_log_evidence": case.subhalo_fit.log_evidence,
        "smooth_runtime_s": case.smooth_fit.runtime_s,
        "subhalo_runtime_s": case.subhalo_fit.runtime_s,
        "smooth_analysis_key": case.smooth_fit.analysis_key,
        "subhalo_analysis_key": case.subhalo_fit.analysis_key,
        "quality_flags": list(case.quality_flags),
        "measured_truth_total_rms_nm": psf_data.total_rms_nm,
        "kernel_sha256": result.kernel_sha256,
        "truth_kernel_sha256": result.truth_kernel_sha256,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if args.check_anchor:
        report["abs_q_delta"] = abs(result.q_fit - ANCHOR_Q)
        report["abs_dlogz_delta"] = abs(
            result.delta_log_evidence - ANCHOR_DLOGZ
        )

    result.write_json(output_dir / "bench_case.json")
    (output_dir / "bench_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)

    if args.check_anchor:
        assert result.smooth_status == "success", result.smooth_status
        assert result.subhalo_status == "success", result.subhalo_status
        assert report["abs_q_delta"] < ANCHOR_TOL, report["abs_q_delta"]
        assert report["abs_dlogz_delta"] < ANCHOR_TOL, (
            report["abs_dlogz_delta"]
        )
        print("ANCHOR BIT-EXACT", flush=True)


if __name__ == "__main__":
    main()
