#!/usr/bin/env python
"""Run one nonlinear-validation arm of one production ladder member.

One invocation is one smooth/subhalo Nautilus fit pair under the
DesignFreeze v4 ``nonlinear_validation`` protocol. Every protocol
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
import hashlib
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
V7_OBJECTIVE_VERSION = "consistent_sampling_v2"
V7_PROFILE_PROCEDURE = "fresh_profile_v1"

SAMPLER_SPAWN_KEY = 5
"""Leading spawn key of the declared sampler stream (`int`)."""

NULL_NOISE_SPAWN_KEY = 6
"""Leading spawn key of the declared null-noise stream (`int`)."""

PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY = 7
"""Leading spawn key of the PSF knowledge direction stream (`int`)."""


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


def derive_noise_seed(entropy: int, replicate: int, index: int) -> int:
    """Derive one declared operational null-noise seed.

    Parameters
    ----------
    entropy : `int`
        The freeze seed entropy.
    replicate : `int`
        Positive null replicate index ``k``.
    index : `int`
        System index ``i``.

    Returns
    -------
    seed : `int`
        The 32-bit observation noise seed of the freeze's null-noise stream.
    """
    sequence = np.random.SeedSequence(
        entropy=int(entropy),
        spawn_key=(NULL_NOISE_SPAWN_KEY, int(replicate), int(index)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def derive_direction_seed(entropy: int, direction: int, index: int) -> int:
    """Derive one declared PSF knowledge-error direction seed.

    Parameters
    ----------
    entropy : `int`
        The freeze seed entropy.
    direction : `int`
        Direction index ``d`` in the PSF knowledge direction stream.
    index : `int`
        System index ``i``.

    Returns
    -------
    seed : `int`
        The 32-bit direction seed of the freeze's PSF knowledge stream.
    """
    sequence = np.random.SeedSequence(
        entropy=int(entropy),
        spawn_key=(PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY, int(direction), int(index)),
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def validate_direction_argument(
    declaration: dict,
    direction: int | None,
) -> None:
    """Validate the optional direction against one arm declaration.

    Parameters
    ----------
    declaration : `dict`
        Arm declaration from the nonlinear-validation protocol.
    direction : `int` or `None`
        Direction supplied by the queue, or no direction.

    Raises
    ------
    ValueError
        Raised when the direction presence or value disagrees with the arm.
    """
    carries_delta = "fit_psf_delta" in declaration
    if carries_delta and direction is None:
        raise ValueError(
            "A fit_psf_delta arm requires --direction"
        )
    if not carries_delta and direction is not None:
        raise ValueError(
            "--direction is only valid for an arm carrying fit_psf_delta"
        )
    if direction is None:
        return
    if isinstance(direction, bool) or not isinstance(direction, int):
        raise ValueError(f"direction must be an integer, got {direction!r}")
    declared_directions = declaration["fit_psf_delta"].get("directions")
    if direction not in declared_directions:
        raise ValueError(
            f"direction {direction} is not declared for this arm; declared "
            f"directions are {declared_directions!r}"
        )


def apply_direction_override(
    arm_config: dict,
    declaration: dict,
    entropy: int,
    index: int,
    direction: int,
) -> tuple[dict, int]:
    """Apply one PSF knowledge direction to a private arm configuration.

    Parameters
    ----------
    arm_config : `dict`
        Configuration already built for the validation arm.
    declaration : `dict`
        Arm declaration carrying ``fit_psf_delta``.
    entropy : `int`
        The freeze seed entropy.
    index : `int`
        System index ``i``.
    direction : `int`
        Declared direction index ``d``.

    Returns
    -------
    config : `dict`
        Deep-copied arm configuration carrying the declared amplitude and
        direction seed.
    seed : `int`
        Direction seed written into the private configuration.

    Raises
    ------
    ValueError
        Raised when the arm has no compatible fit-PSF delta declaration.
    """
    validate_direction_argument(declaration, direction)
    if "fit_psf_delta" not in declaration:
        raise ValueError(
            "Cannot apply a PSF knowledge direction to a non-delta arm"
        )
    config = copy.deepcopy(arm_config)
    seed = derive_direction_seed(entropy, direction, index)
    delta = config["modeling"]["fit_psf"]["delta"]
    delta["amplitude_rms_nm"] = float(
        declaration["fit_psf_delta"]["amplitude_rms_nm"]
    )
    delta["seed"] = seed
    return config, seed


def apply_noise_replicate(
    arm_config: dict,
    declaration: dict,
    entropy: int,
    index: int,
) -> tuple[dict, int, int, list | None]:
    """Apply a declared null-noise replicate to an arm configuration.

    Parameters
    ----------
    arm_config : `dict`
        Configuration already built for the arm.
    declaration : `dict`
        Arm declaration from the nonlinear-validation protocol.
    entropy : `int`
        The freeze seed entropy.
    index : `int`
        System index ``i``.

    Returns
    -------
    config : `dict`
        Private arm configuration carrying the selected noise seed.
    noise_seed : `int`
        The seed used to generate the observation.
    noise_replicate : `int`
        Zero for the primary realization or the declared replicate index.
    noise_spawn_key : `list` or `None`
        The declared spawn key for a replicate, or `None` for replicate zero.

    Raises
    ------
    ValueError
        Raised when the arm configuration has no valid integer global seed,
        or when an invalid replicate is attached to an arm.
    """
    config = copy.deepcopy(arm_config)
    staged_seed = config.get("global_seed")
    if (
        isinstance(staged_seed, bool)
        or not isinstance(staged_seed, int)
    ):
        raise ValueError("arm_config.global_seed must be an int")
    if "noise_replicate" not in declaration:
        return config, int(staged_seed), 0, None

    replicate = declaration["noise_replicate"]
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or replicate < 1
    ):
        raise ValueError(
            f"noise_replicate must be a positive integer, got {replicate!r}"
        )
    if declaration.get("dataset_kind") == "asimov":
        raise ValueError("noise_replicate is invalid for an asimov arm")
    if declaration.get("subhalo_in_truth") is True:
        raise ValueError(
            "noise_replicate is invalid when subhalo_in_truth is true"
        )
    noise_seed = derive_noise_seed(entropy, replicate, index)
    config["global_seed"] = noise_seed
    return (
        config,
        noise_seed,
        replicate,
        [NULL_NOISE_SPAWN_KEY, replicate, int(index)],
    )


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
    from hwoslaps.modeling.nonlinear.trial import subhalo_truth_config

    return subhalo_truth_config(
        config,
        float(rung_payload["mass_msun"]),
        rung_payload["position_yx_arcsec"],
        enabled=bool(arm_declaration["subhalo_in_truth"]),
    )


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
    parser.add_argument(
        "--direction",
        type=int,
        help="Declared PSF knowledge-error direction index",
    )
    parser.add_argument(
        "--freeze-path",
        help="Explicit consumed DesignFreeze path; omitted for the legacy default",
    )
    parser.add_argument(
        "--release-freeze-path",
        help="Explicit additive v7 release declaration (required for v7)",
    )
    parser.add_argument(
        "--objective-version",
        choices=("legacy_ring1_v1", V7_OBJECTIVE_VERSION),
        help="Opt-in objective identity; legacy remains the default",
    )
    parser.add_argument(
        "--procedure",
        choices=("legacy", V7_PROFILE_PROCEDURE),
        help="Opt-in local-profile procedure; legacy remains the default",
    )
    parser.add_argument(
        "--h1-anchor",
        help="JSON zero-residual H1 anchor for an explicit bracket invocation",
    )
    parser.add_argument(
        "--bracket-rung",
        default="above_upper_rung",
        help="Synthetic positions rung used by an H0 bracket invocation",
    )
    parser.add_argument(
        "--bracket-arm-index",
        type=int,
        default=24,
        help="Declared sampler spawn index for an H0 bracket invocation",
    )
    parser.add_argument(
        "--compute-tangent-comparator",
        action="store_true",
        help="Attach the likelihood-matched tangent diagnostic for selected cases",
    )
    parser.add_argument(
        "--compute-bracket-fisher-q",
        action="store_true",
        help="Evaluate the established 999x999 Fisher q for a bracket point",
    )
    return parser


def _execution_contract(args):
    """Resolve the consumed freeze and explicit v7 procedure."""
    objective = args.objective_version or "legacy_ring1_v1"
    procedure = args.procedure or "legacy"
    wants_v7 = (
        objective == V7_OBJECTIVE_VERSION
        or procedure == V7_PROFILE_PROCEDURE
        or args.release_freeze_path is not None
        or args.h1_anchor is not None
    )
    if not wants_v7:
        freeze_path = Path(args.freeze_path or DESIGN_FREEZE_PATH).resolve()
        return {
            "freeze_path": freeze_path,
            "protocol": load_protocol(freeze_path),
            "release": None,
            "objective_version": objective,
            "procedure": procedure,
            "anchor": None,
            "compute_comparator": False,
            "compute_bracket_fisher_q": False,
        }
    if args.release_freeze_path is None:
        raise ValueError(
            "The v7 objective/procedure requires an explicit --release-freeze-path"
        )
    from hwoslaps.campaign.design_freeze import load_release_freeze

    release_path = Path(args.release_freeze_path).resolve()
    release = load_release_freeze(release_path)
    consumed = (REPO_ROOT / release["consumed_freeze"]["path"]).resolve()
    if args.freeze_path is not None and Path(args.freeze_path).resolve() != consumed:
        raise ValueError("--freeze-path does not match release consumed_freeze.path")
    release_identity = release["protocol"]["identity"]
    declared_objective = release_identity["objective_version"]
    if objective != declared_objective:
        raise ValueError(
            f"v7 release declares objective {declared_objective!r}, got {objective!r}"
        )
    if procedure != V7_PROFILE_PROCEDURE:
        raise ValueError("v7 release requires --procedure fresh_profile_v1")
    anchor = None
    if args.h1_anchor is not None:
        with Path(args.h1_anchor).open(encoding="utf-8") as stream:
            anchor = json.load(stream)
        if not isinstance(anchor, dict):
            raise ValueError("--h1-anchor must contain a JSON object")
    return {
        "freeze_path": consumed,
        "protocol": load_protocol(consumed),
        "release": release,
        "release_path": release_path,
        "objective_version": objective,
        "procedure": procedure,
        "anchor": anchor,
        "compute_comparator": bool(args.compute_tangent_comparator),
        "compute_bracket_fisher_q": bool(args.compute_bracket_fisher_q),
    }


def _v7_retention_contract(case, bracket_mode):
    """Report per role whether retained sampler state meets the v7 contract.

    A standard pair must retain the raw state of both fresh searches. A
    bracket retains only the H0 search; its verified H1 anchor is not a
    sampler output and must claim no retained state. Artifact completeness
    is reported separately from numerical acceptance.
    """
    roles = {}
    for role, fit in (("smooth", case.smooth_fit), ("subhalo", case.subhalo_fit)):
        payload = fit.search_internal_payload or {}
        sampler_state_required = not (bracket_mode and role == "subhalo")
        if sampler_state_required:
            complete = bool(
                fit.status == "success"
                and fit.search_internal_retention_requested is True
                and fit.search_internal_retained is True
                and payload.get("missing_required") == []
                and payload.get("bound_to_result_path") is not False
            )
        else:
            complete = bool(
                fit.search_internal_retention_requested is False
                and fit.search_internal_retained is False
                and fit.search_engine == "VerifiedZeroResidualAnchor"
            )
        roles[role] = {
            "sampler_state_required": sampler_state_required,
            "fit_status": fit.status,
            "retention_requested": fit.search_internal_retention_requested,
            "retained": fit.search_internal_retained,
            "route": payload.get("route"),
            "files": payload.get("files"),
            "missing_required": payload.get("missing_required"),
            "bound_to_result_path": payload.get("bound_to_result_path"),
            "complete": complete,
        }
    return {
        "policy": (
            "standard pairs retain both fresh searches; brackets retain H0 only "
            "and the verified H1 anchor has no sampler state"
        ),
        "roles": roles,
        "complete": all(role["complete"] for role in roles.values()),
    }


def _v7_sampler_settings(protocol, release):
    """Build the explicit v7 search settings from the additive declaration."""
    if release is None:
        raise ValueError("v7 sampler settings require an additive release declaration")
    sampler = release["protocol"]["sampler"]
    fit = protocol["fit"]
    expected = {
        "n_live_smooth": 100,
        "n_live_subhalo_search": 200,
        "n_live_subhalo_fixed": 100,
    }
    for key, value in expected.items():
        if int(fit[key]) != value or int(sampler[key]) != value:
            raise ValueError(f"v7 sampler contract mismatch for {key}")
    if (
        int(sampler["n_eff"]) != 500
        or sampler.get("f_live") != 0.01
        or sampler.get("retain_sampler_internals") is not True
    ):
        raise ValueError("v7 sampler contract must pin n_eff=500 and retention")
    if sampler.get("n_shell") != 1 or sampler.get("discard_exploration") is not False:
        raise ValueError("v7 sampler contract must explicitly pin n_shell=1 and discard_exploration=false")
    return {
        "n_eff": sampler["n_eff"],
        "n_shell": sampler["n_shell"],
        "f_live": sampler["f_live"],
        "discard_exploration": sampler["discard_exploration"],
        "retain_search_internal": True,
        "sampler_contract": {
            "n_eff": 500,
            "n_shell": 1,
            "f_live": 0.01,
            "discard_exploration": False,
            "n_live_by_fit_mode": {
                "smooth": int(sampler["n_live_smooth"]),
                "freed": int(sampler["n_live_subhalo_search"]),
                "fixed_template": int(sampler["n_live_subhalo_fixed"]),
            },
        },
    }


def _require_fresh_namespace(output_dir: Path) -> None:
    """Fail before rendering/fitting if a v7 case output is non-empty."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            f"v7 fresh-search namespace is not empty: {output_dir}"
        )


def bracket_arm_declaration(rung: str, arm_index: int) -> dict:
    """Declare injected noiseless data whose null model is searched afresh."""
    return {
        "arm_index": int(arm_index),
        "dataset_kind": "asimov",
        "subhalo_in_truth": True,
        "fit_mode": "fixed_template",
        "rung": str(rung),
        "sample": "selected12_bracket",
        "purpose": "fresh H0 profile with verified zero-residual H1 anchor",
    }


def v7_runner_factory(profile_settings, anchor):
    """Build the v7 fresh-profile runner factory the route calls.

    The route constructs its runner as ``factory(settings, output_dir=...)``,
    the ``AutoLensFitRunner`` signature, so the factory must accept exactly
    that call. An H0 bracket wraps the fresh runner in the zero-residual
    anchor check.
    """
    from hwoslaps.modeling.nonlinear.fresh_profile import (
        FreshProfileRunner,
        ZeroResidualAnchorRunner,
    )

    def factory(settings, output_dir):
        runner = FreshProfileRunner(
            settings, output_dir, profile_settings=profile_settings
        )
        if anchor is None:
            return runner
        return ZeroResidualAnchorRunner(runner, anchor)

    return factory


def main(argv=None, *, runner_factory=None, validator_factory=None,
         artifact_prefix="nonlinear_validation") -> None:
    """Run one validation arm; optional factories support versioned replay.

    Default execution and every preparation/validation guard remain unchanged.
    """
    if artifact_prefix not in {"nonlinear_validation", "profile_protocol"}:
        raise ValueError("Unsupported validation artifact prefix")
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    execution = _execution_contract(args)
    protocol = execution["protocol"]
    objective_version = execution["objective_version"]
    procedure = execution["procedure"]
    release = execution["release"]
    if release is not None:
        _require_fresh_namespace(output_dir)
    if execution["anchor"] is not None and args.arm != "h0_bracket":
        raise ValueError("an H1 anchor is valid only for the h0_bracket adapter")
    fit_block = protocol["fit"]
    arms = protocol["arms"]
    bracket_mode = args.arm == "h0_bracket" and args.h1_anchor is not None
    if args.arm not in arms and not bracket_mode:
        raise ValueError(
            f"Arm {args.arm!r} is not declared; declared arms: "
            f"{sorted(arms)}"
        )
    declaration = (
        bracket_arm_declaration(args.bracket_rung, args.bracket_arm_index)
        if bracket_mode
        else arms[args.arm]
    )
    validate_direction_argument(declaration, args.direction)
    artifact_suffix = (
        "" if args.direction is None else f"_dir{args.direction}"
    )
    artifact_path = output_dir/f"{artifact_prefix}_{args.arm}{artifact_suffix}.json"
    if artifact_path.exists() and not args.force:
        raise ValueError(
            f"Refusing to overwrite {artifact_path}; pass --force to replace it"
        )

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
    positions_artifact_sha256 = hashlib.sha256(
        Path(args.positions).read_bytes()
    ).hexdigest()

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
    if objective_version == V7_OBJECTIVE_VERSION:
        current_rendering = arm_config.get("nonlinear_rendering")
        if current_rendering not in (
            None,
            {"objective_version": V7_OBJECTIVE_VERSION},
        ):
            raise ValueError(
                "v7 objective selection conflicts with config rendering identity"
            )
        arm_config = copy.deepcopy(arm_config)
        arm_config["nonlinear_rendering"] = {
            "objective_version": V7_OBJECTIVE_VERSION,
        }
    arm_config, noise_seed, noise_replicate, noise_spawn_key = (
        apply_noise_replicate(
            arm_config,
            declaration,
            int(protocol["seeds"]["entropy"]),
            system_index(system_id_value),
        )
    )
    direction_seed = None
    if args.direction is not None:
        arm_config, direction_seed = apply_direction_override(
            arm_config,
            declaration,
            int(protocol["seeds"]["entropy"]),
            system_index(system_id_value),
            args.direction,
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

    bracket_fisher_q = None
    if bracket_mode and execution["compute_bracket_fisher_q"]:
        from hwoslaps.modeling.nonlinear.fresh_profile import (
            evaluate_established_fisher_q,
        )

        bracket_fisher_q = evaluate_established_fisher_q(
            config=staged_config,
            position_yx_arcsec=position,
            log10_m200=float(rung_payload["logm"]),
            kernel_shape_native=(999, 999),
        )

    trial = trial_from_fisher_map_position(
        injected_config,
        lensing_injected,
        float(rung_payload["mass_msun"]),
        (float(position[0]), float(position[1])),
        fisher_q=(
            bracket_fisher_q["q_f_production_at_position"]
            if bracket_fisher_q is not None
            else None
            if bracket_mode
            else float(rung_payload["q_f_matched"])
        ),
        case_id=f"{system_id_value}_{args.arm}{artifact_suffix}",
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

    settings_kwargs = {
        "n_live_smooth": int(fit_block["n_live_smooth"]),
        "n_live_subhalo_search": int(fit_block["n_live_subhalo_search"]),
        "n_live_subhalo_fixed": int(fit_block["n_live_subhalo_fixed"]),
        "number_of_cores": int(fit_block["number_of_cores"]),
        "maxcall": int(fit_block["maxcall"]),
        "seed": seed,
        "path_prefix": f"{system_id_value}_{args.arm}{artifact_suffix}",
        "use_jax": True,
        "jax_n_batch": int(fit_block["jax_n_batch"]),
    }
    selected_runner_factory = runner_factory
    selected_validator_factory = validator_factory
    if release is not None:
        settings_kwargs.update(_v7_sampler_settings(protocol, release))
        from hwoslaps.modeling.nonlinear.fresh_profile import (
            FreshProfileSettings,
            FreshProfileValidator,
        )

        profile_settings = FreshProfileSettings.from_release_protocol(release)
        if selected_runner_factory is None:
            selected_runner_factory = v7_runner_factory(
                profile_settings, execution["anchor"]
            )
        if selected_validator_factory is None:
            def selected_validator_factory(configured_runner):
                return FreshProfileValidator(
                    configured_runner,
                    compute_comparator=execution["compute_comparator"],
                )
    settings = NonlinearSearchSettings(**settings_kwargs)
    runner = (selected_runner_factory or AutoLensFitRunner)(
        settings,
        output_dir=str(output_dir),
    )
    validator = (selected_validator_factory or NonlinearMetricValidator)(runner)

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
    profile_records = getattr(runner, "profile_records", {})
    profile_role_statuses = {
        role: record.get("candidate_acceptance_status")
        for role, record in profile_records.items()
        if isinstance(record, dict)
    }
    numerical_role_ok = {
        "accepted_repeatable_profile",
        "verified_zero_residual_anchor",
    }
    numerical_status = None
    profile_decision = None
    marginal_q_flag = None
    retention_contract = None
    if release is not None:
        retention_contract = _v7_retention_contract(case, bracket_mode)
        numerical_status = (
            "accepted"
            if all(
                profile_role_statuses.get(role) in numerical_role_ok
                for role in ("smooth", "subhalo")
            )
            else "unresolved"
        )
        if numerical_status != "accepted":
            if "fresh_profile_unresolved" not in case.quality_flags:
                case.quality_flags.append("fresh_profile_unresolved")
        signed_q = None if delta_log_likelihood is None else 2.0 * delta_log_likelihood
        if signed_q is not None:
            marginal_q_flag = bool(abs(signed_q - 10.0) < 1.0)
            if numerical_status == "accepted":
                profile_decision = bool(signed_q >= 10.0)
    payload = {
        "schema_version": 3,
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
        "noise_seed": noise_seed,
        "noise_replicate": noise_replicate,
        "noise_spawn_key": noise_spawn_key,
        "rung": dict(rung_payload),
        "positions_artifact_sha256": positions_artifact_sha256,
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
    if release is not None:
        payload.update(
            {
                "freeze_protocol_path": str(execution["freeze_path"]),
                "objective_version": objective_version,
                "procedure_version": getattr(
                    getattr(runner, "profile_settings", None),
                    "version",
                    procedure,
                ),
                "fresh_search": True,
                "numerical_status": numerical_status,
                "retention_contract": retention_contract,
                "artifact_completeness_status": (
                    "complete" if retention_contract["complete"] else "incomplete"
                ),
                "profile_role_statuses": profile_role_statuses,
                "profile_decision": profile_decision,
                "marginal_q_flag": marginal_q_flag,
                "sampler_pair_status": {
                    "smooth": case.smooth_fit.status,
                    "subhalo": case.subhalo_fit.status,
                },
                "bracket_fisher_q": bracket_fisher_q,
            }
        )
        payload["freeze_protocol_sha256"] = hashlib.sha256(
            Path(execution["freeze_path"]).read_bytes()
        ).hexdigest()
        payload["release_freeze_path"] = str(execution["release_path"])
        payload["release_freeze_sha256"] = hashlib.sha256(
            Path(execution["release_path"]).read_bytes()
        ).hexdigest()
        payload["consumed_freeze_sha256"] = release["consumed_freeze"]["sha256"]
        payload["release_protocol"] = release["protocol"]
        payload["fit_settings"].update(
            {
                "n_eff": settings.n_eff,
                "n_shell": settings.n_shell,
                "discard_exploration": settings.discard_exploration,
                "retain_search_internal": settings.retain_search_internal,
                "sampler_contract": settings.sampler_contract,
            }
        )
        if hasattr(runner, "profile_records"):
            payload["fresh_profile_records"] = runner.profile_records
            payload["likelihood_matched_tangent"] = runner.profile_records.get(
                "likelihood_matched_tangent"
            )
        if execution["anchor"] is not None:
            payload["h1_anchor"] = {
                "sampler_executed": False,
                "evidence_claim": False,
                "anchor_source": execution["anchor"].get("source"),
            }
    if hasattr(runner, "procedure"):
        payload["artifact_role"] = "nonlinear_profile_diagnostic"
        payload["inference_procedure"] = runner.procedure
        payload["sampler_executed"] = False
        payload["profile_decision"] = runner.records.get("profile_decision")
    if args.direction is None:
        payload["fit_psf_delta"] = None
    else:
        delta = declaration["fit_psf_delta"]
        payload["fit_psf_delta"] = {
            "amplitude_rms_nm": float(delta["amplitude_rms_nm"]),
            "direction": int(args.direction),
            "seed": int(direction_seed),
            "seed_spawn_key": [
                PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY,
                int(args.direction),
                system_index(system_id_value),
            ],
            "delta_id": result.delta_id,
            "requested_draw_rms_nm": result.requested_amplitude_rms_nm,
            "measured_draw_rms_nm": result.measured_draw_rms_nm,
            "fit_kernel_sha256": result.kernel_sha256,
            "truth_kernel_sha256": result.truth_kernel_sha256,
            "fit_psf_config_hash": result.fit_psf_config_hash,
            "truth_psf_config_hash": result.truth_psf_config_hash,
            "lensing_pixel_scale": float(
                arm_config["lensing"]["grid"]["pixel_scale"]
            ),
            "prior_table_sha256": result.prior_table_sha256,
            "family": result.family,
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
