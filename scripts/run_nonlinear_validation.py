#!/usr/bin/env python
"""Run one nonlinear-validation arm of one production ladder member.

One invocation is one smooth/subhalo Nautilus fit pair under the
DesignFreeze v3 ``nonlinear_validation`` protocol. Every protocol
setting is read from the freeze itself: the arm table (dataset kind,
truth subhalo, fit mode, rung, eligibility), the fit settings, the
kernel declaration and the sampler seed rule. Nothing about the
protocol is CLI-overridable, so a job cannot silently run off-protocol.

The member's staged ladder configuration is rendered at the declared
fit kernel with its trial subhalo injected (or withheld, for the
control arm) at the rung and support-matched position the
injection-position artifact declares. Before any fit, the runner
verifies the code revision, the source asset, the PSF state, the staged
kernel against the declaration, the declared training-worker
environment, and that the trial position lies inside the nonlinear
dataset's PSF-border-valid support with a non-degenerate mask.

The job artifact is ``nonlinear_validation_<arm>.json`` under
``--output-dir``; it embeds the complete case record (both fit
summaries with error strings, freed-recovery values, diagnostics and
quality flags).
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

DESIGN_FREEZE_PATH = REPO_ROOT/"configs"/"design"/"design_freeze_v1.yaml"

SAMPLER_SPAWN_KEY = 5
"""Leading spawn key of the declared sampler stream (`int`)."""


def load_protocol(path=DESIGN_FREEZE_PATH) -> dict:
    """Load the declared nonlinear-validation protocol block.

    Parameters
    ----------
    path : path-like, optional
        Design freeze artifact to read.

    Returns
    -------
    protocol : `dict`
        The validated ``nonlinear_validation`` block.
    """
    from hwoslaps.campaign.design_freeze import load_design_freeze

    return load_design_freeze(path)["nonlinear_validation"]


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


def derive_sampler_seed(entropy: int, index: int, arm_index: int) -> int:
    """Derive one arm's declared Nautilus sampler seed.

    Parameters
    ----------
    entropy : `int`
        The freeze seed entropy.
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
        entropy=int(entropy),
        spawn_key=(SAMPLER_SPAWN_KEY, int(index), int(arm_index)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def build_arm_config(
    staged_config: dict,
    arm_declaration: dict,
    rung_payload: dict,
    fit_block: dict,
) -> dict:
    """Build the rendering configuration of one validation arm.

    Parameters
    ----------
    staged_config : `dict`
        The member's restamped staged ladder configuration.
    arm_declaration : `dict`
        The arm's declaration from the freeze protocol.
    rung_payload : `dict`
        The rung block of the member's injection-position artifact.
    fit_block : `dict`
        The freeze protocol's ``fit`` block.

    Returns
    -------
    config : `dict`
        Full configuration the arm's scene and fits are built from.

    Raises
    ------
    ValueError
        Raised when the staged kernel disagrees with the declaration.
    """
    config = copy.deepcopy(staged_config)
    config.pop("provenance_note", None)
    config["plotting"] = {"enabled": False}
    staged_kernel = list(config["psf"]["kernel"]["shape_native"])
    declared_kernel = list(fit_block["kernel_shape_native"])
    if staged_kernel != declared_kernel:
        raise ValueError(
            f"Staged kernel {staged_kernel} is not the declared fit kernel "
            f"{declared_kernel}"
        )
    config["modeling"]["fit_psf"] = {
        "mode": "delta",
        "delta": dict(fit_block["fit_psf"]),
    }
    subhalo = config["lensing"]["subhalo"]
    subhalo["enabled"] = bool(arm_declaration["subhalo_in_truth"])
    subhalo["mass"] = float(rung_payload["mass_msun"])
    subhalo["position"] = {
        "type": "direct",
        "centre": [
            float(rung_payload["position_yx_arcsec"][0]),
            float(rung_payload["position_yx_arcsec"][1]),
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
    parser.add_argument("arm", help="Declared validation arm name")
    parser.add_argument("output_dir", help="Directory for fit outputs")
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

    protocol = load_protocol()
    fit_block = protocol["fit"]
    arms = protocol["arms"]
    if args.arm not in arms:
        raise ValueError(
            f"Arm {args.arm!r} is not declared; declared arms: "
            f"{sorted(arms)}"
        )
    declaration = arms[args.arm]

    declared_workers = str(fit_block["nautilus_training_workers"])
    effective_workers = os.environ.get("HWOSLAPS_NAUTILUS_TRAINING_WORKERS")
    if effective_workers != declared_workers:
        raise ValueError(
            "HWOSLAPS_NAUTILUS_TRAINING_WORKERS is "
            f"{effective_workers!r} but the protocol declares "
            f"{declared_workers!r}"
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
    if list(injection["fit_kernel_shape_native"]) != list(
        fit_block["kernel_shape_native"]
    ):
        raise ValueError(
            "Positions artifact was extracted for kernel "
            f"{injection['fit_kernel_shape_native']}, protocol declares "
            f"{fit_block['kernel_shape_native']}"
        )

    rung_name = str(declaration["rung"])
    if rung_name not in injection["rungs"]:
        raise ValueError(
            f"Positions artifact carries no {rung_name!r} rung for "
            f"{system_id_value} (censored: {injection['censored']})"
        )
    rung_payload = injection["rungs"][rung_name]

    seed = derive_sampler_seed(
        int(protocol["seeds"]["entropy"]),
        system_index(system_id_value),
        int(declaration["arm_index"]),
    )

    from run_stage0_observation import (
        _verify_code_revision,
        _verify_source_asset,
    )
    import run_ladder
    from extract_injection_positions import support_half_widths

    run_ladder._verify_psf_state(staged_config)
    revision = _verify_code_revision(staged_config)
    asset_sha256 = _verify_source_asset(staged_config)

    from hwoslaps.lensing.generator import generate_lensing_system
    from hwoslaps.modeling.nonlinear.autolens_runner import (
        AutoLensFitRunner,
        NonlinearSearchSettings,
    )
    from hwoslaps.modeling.nonlinear.dataset_builder import (
        _exclude_psf_edge_pixels,
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

    injected_declaration = dict(declaration)
    injected_declaration["subhalo_in_truth"] = True
    injected_config = build_arm_config(
        staged_config, injected_declaration, rung_payload, fit_block
    )
    arm_config = (
        injected_config
        if declaration["subhalo_in_truth"]
        else build_arm_config(
            staged_config, declaration, rung_payload, fit_block
        )
    )

    lensing_injected = generate_lensing_system(
        injected_config["lensing"], full_config=injected_config
    )
    lensing_for_data = (
        lensing_injected
        if declaration["subhalo_in_truth"]
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

    kernel_shape = tuple(fit_block["kernel_shape_native"])
    image_shape = tuple(
        np.asarray(observation.data.native, dtype=float).shape
    )
    use_mask = _exclude_psf_edge_pixels(
        np.ones(image_shape, dtype=bool), psf_shape=kernel_shape
    )
    n_unmasked_pixels = int(np.count_nonzero(use_mask))
    if n_unmasked_pixels == 0:
        raise ValueError(
            f"The PSF border of kernel {kernel_shape} leaves no valid "
            f"pixels on an image of shape {image_shape}"
        )
    half_widths = support_half_widths(
        image_shape, float(observation.pixel_scale), kernel_shape
    )
    position = rung_payload["position_yx_arcsec"]
    if (
        abs(float(position[0])) > half_widths[0]
        or abs(float(position[1])) > half_widths[1]
    ):
        raise ValueError(
            f"Trial position {position} lies outside the PSF-border-valid "
            f"support half-widths {half_widths}"
        )

    trial = trial_from_fisher_map_position(
        injected_config,
        lensing_injected,
        float(rung_payload["mass_msun"]),
        (float(position[0]), float(position[1])),
        fisher_q=float(rung_payload["q_f_matched"]),
        case_id=f"{system_id_value}_{args.arm}",
    )
    # The M200 mapping context exists only for the freed search; the
    # model builder requires None for the fixed-template mode.
    mass_context = (
        build_mass_mapping_context(
            injected_config,
            log10_m200_range=tuple(fit_block["log10_m200_range"]),
        )
        if str(declaration["fit_mode"]) == "freed"
        else None
    )

    runner = AutoLensFitRunner(
        NonlinearSearchSettings(
            n_live_smooth=int(fit_block["n_live_smooth"]),
            n_live_subhalo_search=int(fit_block["n_live_subhalo_search"]),
            n_live_subhalo_fixed=int(fit_block["n_live_subhalo_fixed"]),
            number_of_cores=int(fit_block["number_of_cores"]),
            maxcall=int(fit_block["maxcall"]),
            seed=seed,
            path_prefix=f"{system_id_value}_{args.arm}",
            use_jax=True,
            jax_n_batch=int(fit_block["jax_n_batch"]),
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
        fit_mode=str(declaration["fit_mode"]),
        dataset_kind=str(declaration["dataset_kind"]),
        mass_context=mass_context,
    )
    timings["fit_pair_s"] = time.time() - start

    case = result.case
    delta_log_likelihood = None
    if (
        case.subhalo_fit.log_likelihood_max is not None
        and case.smooth_fit.log_likelihood_max is not None
    ):
        delta_log_likelihood = float(
            case.subhalo_fit.log_likelihood_max
            - case.smooth_fit.log_likelihood_max
        )
    payload = {
        "schema_version": 2,
        "artifact": artifact_path.name,
        "system_id": system_id_value,
        "tier": str(injection["tier"]),
        "arm": args.arm,
        "arm_declaration": dict(declaration),
        "sampler_seed": seed,
        "seed_entropy": int(protocol["seeds"]["entropy"]),
        "seed_spawn_key": [
            SAMPLER_SPAWN_KEY,
            system_index(system_id_value),
            int(declaration["arm_index"]),
        ],
        "rung": dict(rung_payload),
        "censored": bool(injection["censored"]),
        "ladder_campaign_uuid": str(injection["ladder_campaign_uuid"]),
        "ladder_config_hash": str(injection["ladder_config_hash"]),
        "staged_config_hash": config_hash(staged_config),
        "arm_config_hash": config_hash(arm_config),
        "source_asset_sha256": asset_sha256,
        "code_revision": revision,
        "fit_settings": {
            key: fit_block[key]
            for key in (
                "kernel_shape_native",
                "n_live_smooth",
                "n_live_subhalo_search",
                "n_live_subhalo_fixed",
                "maxcall",
                "jax_n_batch",
                "number_of_cores",
                "log10_m200_range",
                "nautilus_training_workers",
            )
        },
        "n_unmasked_pixels": n_unmasked_pixels,
        "image_shape": list(image_shape),
        "support_half_widths_arcsec": [half_widths[0], half_widths[1]],
        "timings": timings,
        "q_fit": result.q_fit,
        "delta_log_evidence": result.delta_log_evidence,
        "delta_log_likelihood": delta_log_likelihood,
        "smooth_status": result.smooth_status,
        "subhalo_status": result.subhalo_status,
        "quality_flags": list(case.quality_flags),
        "case": case.to_dict(),
        "trial": trial.to_dict(),
        "measured_truth_total_rms_nm": psf_data.total_rms_nm,
        "kernel_sha256": result.kernel_sha256,
        "truth_kernel_sha256": result.truth_kernel_sha256,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "campaign_uuid": os.environ.get("HWOSLAPS_CAMPAIGN_UUID", ""),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
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
