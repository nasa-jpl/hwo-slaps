#!/usr/bin/env python
"""Run one nonlinear-validation arm of one production ladder member.

One invocation is one smooth/subhalo Nautilus fit pair under the
DesignFreeze v3 ``nonlinear_validation`` protocol: the member's staged
ladder configuration is rendered at the production kernel with its trial
subhalo injected (or withheld, for the control arm) at the mass and
position the injection-position artifact declares, and the freed fit
pair runs with the matched delta-zero fit PSF, the declared search
settings and the declared sampler seed stream.

The three arms are declared in the freeze:

- ``asimov_injected``: subhalo in the truth, noiseless Asimov dataset.
- ``noisy_injected``: subhalo in the truth, the system's own declared
  primary-noise realization.
- ``noisy_control``: no subhalo in the truth, the same noisy dataset
  kind; the subhalo search runs at the same trial mass and position.

The sampler seed is derived here, fail-closed, from the freeze entropy
and the declared spawn key; it is not an input. The job artifact is
``nonlinear_validation_<arm>.json`` under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))
if str(REPO_ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"scripts"))

SEED_ENTROPY = 20260823
"""Freeze seed entropy the sampler stream derives from (`int`)."""

SAMPLER_SPAWN_KEY = 5
"""Leading spawn key of the declared sampler stream (`int`)."""

ARMS = {
    "asimov_injected": {"arm_index": 0, "dataset_kind": "asimov", "subhalo": True},
    "noisy_injected": {"arm_index": 1, "dataset_kind": "noisy", "subhalo": True},
    "noisy_control": {"arm_index": 2, "dataset_kind": "noisy", "subhalo": False},
}
"""Declared validation arms (`dict`)."""

KERNEL_SHAPE_NATIVE = [999, 999]
"""Production detector kernel the fits render at (`list`)."""

DELTA_BLOCK = {
    "prior_table": "configs/psf_priors/jwst_wss_drift_v1.yaml",
    "seed": 20260814,
    "family": "combined",
    "amplitude_rms_nm": 0.0,
}
"""Matched delta-zero fit-PSF declaration of the protocol (`dict`)."""

LOG10_M200_RANGE = (6.0, 9.7)
"""Declared freed mass-mapping range of the protocol (`tuple`)."""


def system_index(system_id: str) -> int:
    """Parse the integer system index out of a ``sysNNNN`` identifier.

    Parameters
    ----------
    system_id : `str`
        Member identifier, e.g. ``ladder_parent_sys0625`` or ``sys0625``.

    Returns
    -------
    index : `int`
        The integer system index.

    Raises
    ------
    ValueError
        Raised when the identifier holds no ``sys`` block.
    """
    marker = "sys"
    position = system_id.rfind(marker)
    if position < 0:
        raise ValueError(f"No 'sys' block in system identifier {system_id!r}")
    digits = system_id[position + len(marker):]
    if not digits.isdigit():
        raise ValueError(
            f"System identifier {system_id!r} does not end in digits"
        )
    return int(digits)


def derive_sampler_seed(index: int, arm_index: int) -> int:
    """Derive one arm's declared Nautilus sampler seed.

    Parameters
    ----------
    index : `int`
        System index ``i``.
    arm_index : `int`
        Declared arm index.

    Returns
    -------
    seed : `int`
        The 32-bit sampler seed of the freeze's sampler stream.
    """
    sequence = np.random.SeedSequence(
        entropy=SEED_ENTROPY,
        spawn_key=(SAMPLER_SPAWN_KEY, int(index), int(arm_index)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def build_arm_config(
    staged_config: dict,
    arm: str,
    injection: dict,
) -> dict:
    """Build the rendering configuration of one validation arm.

    Parameters
    ----------
    staged_config : `dict`
        The member's restamped staged ladder configuration.
    arm : `str`
        Declared arm name.
    injection : `dict`
        The member's injection-position artifact payload.

    Returns
    -------
    config : `dict`
        Full configuration the arm's scene and fits are built from.
    """
    declaration = ARMS[arm]
    config = copy.deepcopy(staged_config)
    config.pop("provenance_note", None)
    config["plotting"] = {"enabled": False}
    config["psf"]["kernel"]["shape_native"] = list(KERNEL_SHAPE_NATIVE)
    config["modeling"]["fit_psf"] = {
        "mode": "delta",
        "delta": dict(DELTA_BLOCK),
    }
    subhalo = config["lensing"]["subhalo"]
    subhalo["enabled"] = bool(declaration["subhalo"])
    subhalo["mass"] = float(injection["injection_mass_msun"])
    subhalo["position"] = {
        "type": "direct",
        "centre": [
            float(injection["position_yx_arcsec"][0]),
            float(injection["position_yx_arcsec"][1]),
        ],
    }
    return config


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Restamped staged ladder configuration")
    parser.add_argument(
        "positions", help="The member's injection_position.json"
    )
    parser.add_argument("arm", choices=sorted(ARMS), help="Validation arm")
    parser.add_argument("output_dir", help="Directory for fit outputs")
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
        "--force",
        action="store_true",
        help="Replace an existing arm artifact",
    )
    return parser


def main(argv=None) -> None:
    """Run one validation arm's fit pair and write its artifact."""
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    artifact_path = output_dir/f"nonlinear_validation_{args.arm}.json"
    if artifact_path.exists() and not args.force:
        raise ValueError(
            f"Refusing to overwrite {artifact_path}; pass --force to replace it"
        )

    with open(args.config, encoding="utf-8") as stream:
        staged_config = yaml.safe_load(stream)
    with open(args.positions, encoding="utf-8") as stream:
        injection = json.load(stream)

    system_id_value = str(staged_config["run_name"])
    if str(injection["system_id"]) != system_id_value:
        raise ValueError(
            f"Positions artifact belongs to {injection['system_id']!r}, "
            f"configuration to {system_id_value!r}"
        )

    declaration = ARMS[args.arm]
    seed = derive_sampler_seed(
        system_index(system_id_value), declaration["arm_index"]
    )

    from run_stage0_observation import (
        _verify_code_revision,
        _verify_source_asset,
    )
    import run_ladder

    run_ladder._verify_psf_state(staged_config)
    revision = _verify_code_revision(staged_config)
    asset_sha256 = _verify_source_asset(staged_config)

    from hwoslaps.lensing.generator import generate_lensing_system
    from hwoslaps.modeling.nonlinear.autolens_runner import (
        AutoLensFitRunner,
        NonlinearSearchSettings,
    )
    from hwoslaps.modeling.nonlinear.mass_mapping import (
        build_mass_mapping_context,
    )
    from hwoslaps.modeling.nonlinear.psf_mismatch import run_psf_mismatch_case
    from hwoslaps.modeling.nonlinear.trial import (
        trial_from_fisher_map_position,
    )
    from hwoslaps.modeling.nonlinear.validator import NonlinearMetricValidator
    from hwoslaps.observation.generator import generate_observation
    from hwoslaps.provenance import config_hash
    from hwoslaps.psf.generator import generate_psf_system

    timings = {}
    start = time.time()

    injected_config = build_arm_config(
        staged_config, "asimov_injected", injection
    )
    arm_config = (
        injected_config
        if declaration["subhalo"]
        else build_arm_config(staged_config, args.arm, injection)
    )

    lensing_injected = generate_lensing_system(
        injected_config["lensing"], full_config=injected_config
    )
    lensing_for_data = (
        lensing_injected
        if declaration["subhalo"]
        else generate_lensing_system(
            arm_config["lensing"], full_config=arm_config
        )
    )
    psf_data = generate_psf_system(
        arm_config["psf"], full_config=arm_config
    )
    run_ladder._verify_psf_rms(psf_data)
    observation = generate_observation(
        lensing_data=lensing_for_data,
        psf_data=psf_data,
        observation_config=arm_config["observation"],
        full_config=arm_config,
    )
    timings["scene_psf_observation_s"] = time.time() - start

    trial = trial_from_fisher_map_position(
        injected_config,
        lensing_injected,
        float(injection["injection_mass_msun"]),
        (
            float(injection["position_yx_arcsec"][0]),
            float(injection["position_yx_arcsec"][1]),
        ),
        fisher_q=float(injection["q_at_position"]),
        case_id=f"{system_id_value}_{args.arm}",
    )
    mass_context = build_mass_mapping_context(
        injected_config, log10_m200_range=LOG10_M200_RANGE
    )

    runner = AutoLensFitRunner(
        NonlinearSearchSettings(
            n_live_smooth=args.n_live_smooth,
            n_live_subhalo_search=args.n_live_subhalo,
            number_of_cores=1,
            maxcall=args.maxcall,
            seed=seed,
            path_prefix=f"{system_id_value}_{args.arm}",
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
        arm_config,
        trial,
        fit_mode="freed",
        dataset_kind=declaration["dataset_kind"],
        mass_context=mass_context,
    )
    timings["fit_pair_s"] = time.time() - start

    case = result.case
    payload = {
        "schema_version": 1,
        "artifact": artifact_path.name,
        "system_id": system_id_value,
        "tier": str(injection["tier"]),
        "arm": args.arm,
        "arm_index": declaration["arm_index"],
        "dataset_kind": declaration["dataset_kind"],
        "subhalo_in_truth": declaration["subhalo"],
        "sampler_seed": seed,
        "seed_entropy": SEED_ENTROPY,
        "seed_spawn_key": [
            SAMPLER_SPAWN_KEY,
            system_index(system_id_value),
            declaration["arm_index"],
        ],
        "injection_logm": float(injection["injection_logm"]),
        "injection_mass_msun": float(injection["injection_mass_msun"]),
        "censored": bool(injection["censored"]),
        "position_yx_arcsec": [
            float(injection["position_yx_arcsec"][0]),
            float(injection["position_yx_arcsec"][1]),
        ],
        "fisher_q_at_position": float(injection["q_at_position"]),
        "ladder_campaign_uuid": str(injection["ladder_campaign_uuid"]),
        "ladder_config_hash": str(injection["ladder_config_hash"]),
        "staged_config_hash": config_hash(staged_config),
        "arm_config_hash": config_hash(arm_config),
        "source_asset_sha256": asset_sha256,
        "code_revision": revision,
        "n_live_smooth": args.n_live_smooth,
        "n_live_subhalo": args.n_live_subhalo,
        "jax_n_batch": args.jax_n_batch,
        "maxcall": args.maxcall,
        "log10_m200_range": list(LOG10_M200_RANGE),
        "nautilus_training_workers": os.environ.get(
            "HWOSLAPS_NAUTILUS_TRAINING_WORKERS"
        ),
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
        "trial": trial.to_dict(),
        "measured_truth_total_rms_nm": psf_data.total_rms_nm,
        "kernel_sha256": result.kernel_sha256,
        "truth_kernel_sha256": result.truth_kernel_sha256,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "campaign_uuid": os.environ.get("HWOSLAPS_CAMPAIGN_UUID", ""),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result.write_json(output_dir/f"case_{args.arm}.json")
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Nonlinear validation artifact: {artifact_path}\n"
        f"  {system_id_value} {args.arm}: q_fit {result.q_fit!r}, "
        f"dlogZ {result.delta_log_evidence!r}, statuses "
        f"{result.smooth_status}/{result.subhalo_status}, pair "
        f"{timings['fit_pair_s']:.0f} s"
    )


if __name__ == "__main__":
    main()
