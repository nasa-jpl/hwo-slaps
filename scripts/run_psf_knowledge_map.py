#!/usr/bin/env python
"""Run one Fisher PSF knowledge-error job across its three mass rungs.

The staged ladder configuration supplies the member scene and the
production ladder artifact supplies the three D-K5 rung choices. A delta
job builds one detector for one paired residual direction, then retargets
that detector across the selected rungs. The delta-0 job leaves the staged
configuration matched and is the production-path receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))
if str(REPO_ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"scripts"))

import run_ladder  # noqa: E402
from run_nonlinear_validation import (  # noqa: E402
    PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY,
    derive_direction_seed,
    system_index,
)

DESIGN_FREEZE_PATH = REPO_ROOT/"configs"/"design"/"design_freeze_v1.yaml"
MAP_ARTIFACT_SCHEMA_VERSION = 1
CELL_AREA_ARCSEC2 = run_ladder.NODE_SPACING_ARCSEC**2


def _scalar(record: dict, name: str):
    """Return one scalar artifact member or raise a named error."""
    if name not in record:
        raise ValueError(f"Artifact is missing required member {name!r}")
    value = np.asarray(record[name])
    if value.ndim != 0:
        raise ValueError(f"Artifact member {name!r} is not a scalar")
    return value.item()


def load_npz_record(path: Path) -> dict:
    """Load an NPZ record into memory and close its file handle.

    Parameters
    ----------
    path : pathlib.Path
        NPZ file to read.

    Returns
    -------
    record : dict
        Independent NumPy arrays keyed by the archive member names.
    """
    with np.load(path, allow_pickle=False) as stored:
        return {
            name: np.array(stored[name], copy=True)
            for name in stored.files
        }


def validate_job_coordinates(
    delta: float,
    direction: int,
    knowledge: dict,
) -> tuple[float, int]:
    """Validate one map job's declared delta and direction coordinates.

    Parameters
    ----------
    delta : float
        Requested residual RMS in nanometers.
    direction : int
        Direction index, with zero reserved for the matched job.
    knowledge : dict
        Validated psf_knowledge_error freeze block.

    Returns
    -------
    coordinates : tuple
        Canonical delta and direction values.

    Raises
    ------
    ValueError
        Raised when either coordinate is outside the frozen protocol.
    """
    if isinstance(delta, (bool, np.bool_)) or not isinstance(
        delta, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"delta must be a number, got {delta!r}")
    delta_value = float(delta)
    if not np.isfinite(delta_value):
        raise ValueError(f"delta must be finite, got {delta!r}")
    residual = knowledge["residual_model"]
    rungs = [float(value) for value in residual["amplitude_rms_nm_rungs"]]
    if delta_value not in rungs:
        raise ValueError(
            f"delta {delta_value:g} is not a declared PSF knowledge-error "
            f"rung: {rungs!r}"
        )
    if isinstance(direction, (bool, np.bool_)) or not isinstance(
        direction, (int, np.integer)
    ):
        raise ValueError(f"direction must be an integer, got {direction!r}")
    direction_value = int(direction)
    directions = int(residual["directions"])
    if direction_value < 0 or direction_value > directions:
        raise ValueError(
            f"direction {direction_value} is outside the declared range "
            f"0..{directions}"
        )
    if delta_value == 0.0 and direction_value != 0:
        raise ValueError("direction must be 0 when delta is 0")
    if delta_value != 0.0 and direction_value == 0:
        raise ValueError("direction must be nonzero when delta is positive")
    return delta_value, direction_value


def map_artifact_name(logm: float, delta: float, direction: int) -> str:
    """Return the stable per-rung PSF knowledge map artifact name.

    Parameters
    ----------
    logm : float
        Logarithmic mass rung.
    delta : float
        Residual RMS in nanometers.
    direction : int
        Direction index.

    Returns
    -------
    name : str
        Artifact filename using the frozen two-decimal mass and percent-g
        delta rendering conventions.
    """
    return (
        f"psf_knowledge_map_m{float(logm):.2f}_delta{float(delta):g}"
        f"_dir{int(direction)}.npz"
    )


def job_summary_name(delta: float, direction: int) -> str:
    """Return the stable job-summary filename for one map job."""
    return f"psf_knowledge_job_delta{float(delta):g}_dir{int(direction)}.json"


def _bracket_top(record: dict, name: str) -> float:
    """Return and validate one crossing's upper bracket rung."""
    value = float(_scalar(record, name))
    if not np.isfinite(value):
        raise ValueError(f"Artifact {name} is NaN or non-finite")
    bracket_name = f"{name}_bracket_logm"
    if bracket_name not in record:
        raise ValueError(f"Artifact is missing required member {bracket_name!r}")
    bracket = np.asarray(record[bracket_name], dtype=float)
    if bracket.shape != (2,) or not np.all(np.isfinite(bracket)):
        raise ValueError(
            f"Artifact {bracket_name} must be a finite two-element bracket"
        )
    return float(bracket[1])


def select_knowledge_rungs(record: dict) -> list[dict]:
    """Select the three D-K5 upper-bracket rungs from a ladder artifact.

    Parameters
    ----------
    record : dict
        Loaded production ladder artifact.

    Returns
    -------
    rungs : list
        Ascending unique rung records with logm, classes and source index.

    Raises
    ------
    ValueError
        Raised when a crossing is non-finite, a bracket is malformed, or a
        selected upper bracket is not a walked production rung.
    """
    if "rung_logm" not in record:
        raise ValueError("Artifact is missing required member 'rung_logm'")
    rung_logm = np.asarray(record["rung_logm"], dtype=float)
    if rung_logm.ndim != 1 or not rung_logm.size:
        raise ValueError(
            "Artifact rung_logm must be a non-empty one-dimensional array"
        )
    if not np.all(np.isfinite(rung_logm)):
        raise ValueError("Artifact rung_logm contains non-finite values")
    selected = []
    for class_name in ("m_best", "m10", "m50"):
        logm = _bracket_top(record, class_name)
        matches = np.isclose(rung_logm, logm, rtol=0.0, atol=1.0e-9)
        count = int(np.count_nonzero(matches))
        if count != 1:
            raise ValueError(
                f"D-K5 rung {logm} for {class_name} matches {count} "
                "walked artifact rungs, expected 1"
            )
        index = int(np.flatnonzero(matches)[0])
        for rung in selected:
            if np.isclose(rung["logm"], logm, rtol=0.0, atol=1.0e-9):
                rung["classes"].append(class_name)
                break
        else:
            selected.append({
                "logm": logm,
                "classes": [class_name],
                "artifact_index": index,
            })
    selected.sort(key=lambda rung: rung["logm"])
    return selected


def production_rung_reference(record: dict, logm: float) -> dict:
    """Return production q-max, area and cell count at one walked rung.

    Parameters
    ----------
    record : dict
        Loaded production ladder artifact.
    logm : float
        Walked mass rung.

    Returns
    -------
    reference : dict
        Production q_max, clipped area and integer cell count.

    Raises
    ------
    ValueError
        Raised when the rung is absent or its area is not an integer cell
        multiple at the production spacing.
    """
    rung_logm = np.asarray(record["rung_logm"], dtype=float)
    matches = np.isclose(rung_logm, float(logm), rtol=0.0, atol=1.0e-9)
    if int(np.count_nonzero(matches)) != 1:
        raise ValueError(
            f"Rung {logm} matches {int(np.count_nonzero(matches))} "
            "production artifact rungs, expected 1"
        )
    area = float(np.asarray(record["rung_detectable_area_arcsec2"])[matches][0])
    cells_float = area/CELL_AREA_ARCSEC2
    cells = int(round(cells_float))
    if not np.isclose(cells_float, cells, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"Production rung area {area!r} is not an integer number of "
            f"{CELL_AREA_ARCSEC2} arcsec2 cells"
        )
    q_max = float(np.asarray(record["rung_q_max"])[matches][0])
    return {
        "production_q_max": q_max,
        "production_detectable_area_arcsec2": area,
        "production_cells": cells,
    }


def verify_ladder_artifact_identity(
    record: dict,
    config: dict,
    knowledge: dict,
    aperture: dict | None = None,
) -> None:
    """Verify the production ladder identity bound to one map job.

    Parameters
    ----------
    record : dict
        Loaded production ladder artifact.
    config : dict
        Restamped staged ladder configuration.
    knowledge : dict
        Validated PSF knowledge-error freeze block.
    aperture : dict, optional
        Verified ladder aperture. Defaults to config ladder aperture.

    Raises
    ------
    ValueError
        Raised when any system, tier, state, kernel, aperture or campaign
        identity member disagrees.
    """
    expected_system = str(config["run_name"])
    actual_system = str(_scalar(record, "system_id"))
    if actual_system != expected_system:
        raise ValueError(
            f"Artifact system id {actual_system!r} does not match "
            f"configuration run_name {expected_system!r}"
        )
    if str(_scalar(record, "tier")) != "selected":
        raise ValueError("PSF knowledge map requires a selected ladder artifact")
    actual_state = str(_scalar(record, "psf_state"))
    if actual_state != run_ladder.PSF_STATE:
        raise ValueError(
            f"Artifact psf_state {actual_state!r} is not "
            f"{run_ladder.PSF_STATE!r}"
        )
    shape = [
        int(value)
        for value in np.asarray(
            record["psf_kernel_shape_native"], dtype=int
        )
    ]
    if shape != list(run_ladder.KERNEL_SHAPE_NATIVE):
        raise ValueError(
            f"Artifact psf kernel shape {shape!r} is not "
            f"{run_ladder.KERNEL_SHAPE_NATIVE!r}"
        )
    expected_uuid = str(knowledge["member_set"]["source_campaign_uuid"])
    actual_uuid = str(_scalar(record, "campaign_uuid"))
    if actual_uuid != expected_uuid:
        raise ValueError(
            f"Artifact campaign uuid {actual_uuid!r} does not match the "
            f"selected ladder campaign {expected_uuid!r}"
        )
    declared_aperture = aperture or config["ladder"]["aperture"]
    for artifact_name, config_name in (
        ("aperture_sha256", "stage0_aperture_sha256"),
        ("contour_sha256", "stage0_contour_sha256"),
    ):
        actual = str(_scalar(record, artifact_name))
        expected = str(declared_aperture[config_name])
        if actual != expected:
            raise ValueError(
                f"Artifact {artifact_name} {actual!r} does not match the "
                f"configuration aperture digest {expected!r}"
            )


def verify_truth_kernel_digest(actual: str, expected: str) -> None:
    """Require a regenerated truth kernel to match a ladder artifact.

    Parameters
    ----------
    actual : str
        Digest of the regenerated truth kernel.
    expected : str
        Digest recorded by the production ladder.

    Raises
    ------
    ValueError
        Raised when the two kernel digests differ.
    """
    if str(actual) != str(expected):
        raise ValueError(
            f"Regenerated truth kernel digest {actual} does not match the "
            f"production artifact digest {expected}"
        )


def _mask_sha256(mask: np.ndarray) -> str:
    """Return the digest of one unpacked boolean mask."""
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _pack_mask(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Pack one boolean mask and retain its shape and unpacked digest."""
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    return (
        np.asarray(np.packbits(values), dtype=np.uint8),
        np.asarray(values.shape, dtype=np.int64),
        _mask_sha256(values),
    )


def _aperture_inside_mask(
    y_coords,
    x_coords,
    centre_arcsec,
    radius_arcsec: float,
) -> np.ndarray:
    """Return the grid-node mask of the closed D-F7 aperture."""
    y_values = np.asarray(y_coords, dtype=float)
    x_values = np.asarray(x_coords, dtype=float)
    return (
        (y_values[:, None] - float(centre_arcsec[0]))**2
        + (x_values[None, :] - float(centre_arcsec[1]))**2
        <= float(radius_arcsec)**2
    )


def _job_identity(
    campaign_uuid: str,
    config_hash_value: str,
    system_id: str,
    revision: dict,
    source_asset_sha256: str,
    ladder_artifact: dict,
    ladder_artifact_sha256: str,
    truth_kernel_sha256: str,
    fit_kernel_sha256: str,
    psf_state_rms_nm: float,
    aperture: dict,
) -> dict:
    """Build identity fields shared by map artifacts and summaries."""
    return {
        "campaign_uuid": np.asarray(campaign_uuid),
        "config_hash": np.asarray(config_hash_value),
        "system_id": np.asarray(system_id),
        "code_revision_sha256": np.asarray(str(revision["sha256"])),
        "code_git_hash": np.asarray(str(revision["git_hash"])),
        "code_git_dirty": np.asarray(str(revision["git_dirty"])),
        "ladder_campaign_uuid": np.asarray(
            str(_scalar(ladder_artifact, "campaign_uuid"))
        ),
        "ladder_config_hash": np.asarray(
            str(_scalar(ladder_artifact, "config_hash"))
        ),
        "ladder_artifact_sha256": np.asarray(ladder_artifact_sha256),
        "source_asset_sha256": np.asarray(source_asset_sha256),
        "aperture_sha256": np.asarray(
            str(aperture["stage0_aperture_sha256"])
        ),
        "contour_sha256": np.asarray(
            str(aperture["stage0_contour_sha256"])
        ),
        "truth_kernel_sha256": np.asarray(truth_kernel_sha256),
        "fit_kernel_sha256": np.asarray(fit_kernel_sha256),
        "psf_state": np.asarray(run_ladder.PSF_STATE),
        "psf_state_rms_nm": np.asarray(float(psf_state_rms_nm)),
        "psf_kernel_shape_native": np.asarray(
            [int(value) for value in run_ladder.KERNEL_SHAPE_NATIVE]
        ),
    }


def _write_json(path: Path, payload: dict) -> None:
    """Write one JSON payload atomically through a same-directory file."""
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _load_yaml(path: Path) -> dict:
    """Load one YAML mapping from a staged configuration path."""
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration {path} must be a mapping")
    return config


def _map_payload(
    identity: dict,
    logm: float,
    rung_classes: list[str],
    delta: float,
    direction: int,
    seed: int,
    seed_spawn_key: list[int],
    delta_id: str,
    requested_draw_rms_nm: float,
    measured_draw_rms_nm: float,
    prior_table_sha256: str,
    family: str,
    production: dict,
    matched_metrics: dict,
    mismatch_metrics: dict | None,
    spurious_metrics: dict | None,
    grid_map,
    aperture_mask: np.ndarray,
    detector_build_seconds: float,
    map_wall_seconds: float,
    truth_psf_config_hash: str,
    fit_psf_config_hash: str,
    lensing_pixel_scale: float,
) -> dict:
    """Build one per-rung map artifact payload."""
    matched_mask = np.asarray(grid_map.detectable_mask_2d, dtype=bool)
    if mismatch_metrics is None:
        mismatch_mask = np.zeros_like(matched_mask, dtype=bool)
        spurious_mask = np.zeros_like(matched_mask, dtype=bool)
    else:
        mismatch_mask = np.asarray(
            grid_map.mismatch_detectable_mask_2d, dtype=bool
        )
        spurious_mask = np.asarray(
            grid_map.false_positive_mask_2d, dtype=bool
        )
    packed_masks = {}
    for prefix, mask in (
        ("detectable_mask", matched_mask),
        ("mismatch_detectable_mask", mismatch_mask),
        ("false_positive_mask", spurious_mask),
        ("aperture_mask", np.asarray(aperture_mask, dtype=bool)),
    ):
        packed, shape, digest = _pack_mask(mask)
        packed_masks[f"{prefix}_packed"] = packed
        packed_masks[f"{prefix}_shape"] = shape
        packed_masks[f"{prefix}_sha256"] = np.asarray(digest)
    mismatch_values = mismatch_metrics or {
        "q_max": np.nan,
        "detectable_area_arcsec2": np.nan,
    }
    spurious_values = spurious_metrics or {
        "q_max": np.nan,
        "detectable_area_arcsec2": np.nan,
    }
    mismatch_cells = (
        int(
            round(
                mismatch_values["detectable_area_arcsec2"]
                / grid_map.spacing_arcsec**2
            )
        )
        if mismatch_metrics is not None
        else -1
    )
    spurious_cells = (
        int(
            round(
                spurious_values["detectable_area_arcsec2"]
                / grid_map.spacing_arcsec**2
            )
        )
        if spurious_metrics is not None
        else -1
    )
    payload = {
        "schema_version": np.asarray(MAP_ARTIFACT_SCHEMA_VERSION),
        **identity,
        "logm": np.asarray(float(logm)),
        "rung_classes": np.asarray(rung_classes, dtype=np.str_),
        "delta_nm": np.asarray(float(delta)),
        "direction": np.asarray(int(direction)),
        "seed": np.asarray(int(seed)),
        "seed_spawn_key": np.asarray(seed_spawn_key, dtype=np.int64),
        "delta_id": np.asarray(delta_id),
        "requested_draw_rms_nm": np.asarray(float(requested_draw_rms_nm)),
        "measured_draw_rms_nm": np.asarray(float(measured_draw_rms_nm)),
        "prior_table_sha256": np.asarray(prior_table_sha256),
        "family": np.asarray(family),
        "truth_psf_config_hash": np.asarray(truth_psf_config_hash),
        "fit_psf_config_hash": np.asarray(fit_psf_config_hash),
        "lensing_pixel_scale": np.asarray(float(lensing_pixel_scale)),
        "production_q_max": np.asarray(float(production["production_q_max"])),
        "production_detectable_area_arcsec2": np.asarray(
            float(production["production_detectable_area_arcsec2"])
        ),
        "production_cells": np.asarray(int(production["production_cells"])),
        "matched_q_max": np.asarray(float(matched_metrics["q_max"])),
        "matched_cells": np.asarray(
            int(
                round(
                    matched_metrics["detectable_area_arcsec2"]
                    / grid_map.spacing_arcsec**2
                )
            )
        ),
        "matched_area_arcsec2": np.asarray(
            float(matched_metrics["detectable_area_arcsec2"])
        ),
        "matched_aperture_fraction": np.asarray(
            float(matched_metrics["aperture_fraction"])
        ),
        "matched_perimeter_clipped": np.asarray(
            bool(matched_metrics["perimeter_clipped"])
        ),
        "mismatch_q_max": np.asarray(float(mismatch_values["q_max"])),
        "mismatch_cells": np.asarray(mismatch_cells),
        "mismatch_area_arcsec2": np.asarray(
            float(mismatch_values["detectable_area_arcsec2"])
        ),
        "spurious_q_max": np.asarray(float(spurious_values["q_max"])),
        "spurious_cells": np.asarray(spurious_cells),
        "spurious_area_arcsec2": np.asarray(
            float(spurious_values["detectable_area_arcsec2"])
        ),
        "detectable_area_arcsec2": np.asarray(
            float(grid_map.detectable_area_arcsec2)
        ),
        "mismatch_detectable_area_arcsec2": np.asarray(
            float(grid_map.mismatch_detectable_area_arcsec2)
            if mismatch_metrics is not None
            else np.nan
        ),
        "false_positive_area_arcsec2": np.asarray(
            float(grid_map.false_positive_area_arcsec2)
            if spurious_metrics is not None
            else np.nan
        ),
        "num_detectable": np.asarray(int(grid_map.num_detectable)),
        "num_mismatch_detectable": np.asarray(
            int(grid_map.num_mismatch_detectable)
            if mismatch_metrics is not None
            else -1
        ),
        "num_false_positive": np.asarray(
            int(grid_map.num_false_positive)
            if spurious_metrics is not None
            else -1
        ),
        "max_z_spurious": np.asarray(
            float(grid_map.max_z_spurious)
            if spurious_metrics is not None
            else np.nan
        ),
        "nodes_inside_aperture": np.asarray(
            int(np.count_nonzero(aperture_mask))
        ),
        "spacing_arcsec": np.asarray(float(grid_map.spacing_arcsec)),
        "detector_build_seconds": np.asarray(float(detector_build_seconds)),
        "map_wall_seconds": np.asarray(float(map_wall_seconds)),
    }
    payload.update(packed_masks)
    return payload


def _summary_scalar_payload(record: dict) -> dict:
    """Extract scalar map fields used in a resume summary."""
    return {
        "logm": float(_scalar(record, "logm")),
        "rung_classes": [
            str(value) for value in np.atleast_1d(record["rung_classes"])
        ],
        "matched_cells": int(_scalar(record, "matched_cells")),
        "mismatch_cells": int(_scalar(record, "mismatch_cells")),
        "spurious_cells": int(_scalar(record, "spurious_cells")),
        "detector_build_seconds": float(
            _scalar(record, "detector_build_seconds")
        ),
        "map_wall_seconds": float(_scalar(record, "map_wall_seconds")),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Restamped staged ladder configuration")
    parser.add_argument("ladder_artifact", help="Production ladder_result.npz")
    parser.add_argument("delta", type=float, help="Declared residual RMS in nm")
    parser.add_argument("direction", type=int, help="Declared direction index")
    parser.add_argument("output_dir", help="Directory for map artifacts")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute existing map artifacts instead of resuming them",
    )
    return parser


def main(argv=None) -> None:
    """Run one PSF knowledge-error map job and write its artifacts."""
    args = _build_parser().parse_args(argv)
    from hwoslaps.campaign.design_freeze import load_design_freeze

    freeze = load_design_freeze(DESIGN_FREEZE_PATH)
    knowledge = freeze["psf_knowledge_error"]
    delta, direction = validate_job_coordinates(
        args.delta, args.direction, knowledge
    )
    config_path = Path(args.config).expanduser().resolve()
    ladder_artifact_path = Path(args.ladder_artifact).expanduser().resolve()
    config = _load_yaml(config_path)
    ladder_artifact = load_npz_record(ladder_artifact_path)

    ladder = run_ladder._verify_ladder_block(config)
    run_ladder._verify_psf_state(config)
    run_ladder._enable_float64()
    run_ladder._enable_jax_compilation_cache()

    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.provenance import config_hash
    from hwoslaps.psf import generate_psf_system
    from hwoslaps.psf.mismatch import (
        _canonical_psf_config,
        _kernel_sha256,
    )
    from hwoslaps.psf.utils import pyauto_kernel_native

    revision = run_ladder._verify_code_revision(config)
    source_asset_sha256 = run_ladder._verify_source_asset(config)
    extraction = run_ladder._extract_theta_e_eff(config)
    aperture = run_ladder._verify_aperture(config, extraction)
    verify_ladder_artifact_identity(
        ladder_artifact,
        config,
        knowledge,
        aperture,
    )
    selected_rungs = select_knowledge_rungs(ladder_artifact)
    for rung in selected_rungs:
        production_rung_reference(ladder_artifact, rung["logm"])

    rung_config = run_ladder._rung_config(config, ladder, aperture)
    system_id = str(config["run_name"])
    index = system_index(system_id)
    if delta > 0.0:
        seed = derive_direction_seed(
            int(freeze["seeds"]["entropy"]), direction, index
        )
        rung_config["modeling"]["fit_psf"] = {
            "mode": "delta",
            "delta": {
                "prior_table": knowledge["residual_model"]["prior_table"],
                "family": "combined",
                "seed": seed,
                "amplitude_rms_nm": delta,
            },
        }
        seed_spawn_key = [PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY, direction, index]
    else:
        seed = -1
        seed_spawn_key = []
    validate_or_raise(rung_config)

    psf_data = generate_psf_system(
        rung_config["psf"], full_config=rung_config
    )
    measured_truth_rms = run_ladder._verify_psf_rms(psf_data)
    truth_kernel_sha256 = _kernel_sha256(
        pyauto_kernel_native(psf_data.kernel)
    )
    production_kernel_sha256 = str(
        _scalar(ladder_artifact, "psf_kernel_sha256")
    )
    verify_truth_kernel_digest(truth_kernel_sha256, production_kernel_sha256)

    detector_start = perf_counter()
    detector = run_ladder._build_detector(rung_config, psf_data)
    detector_build_seconds = perf_counter() - detector_start
    if delta > 0.0:
        fit_delta = detector.fit_psf_delta
        if not isinstance(fit_delta, dict):
            raise ValueError(
                "Delta detector did not expose fit_psf_delta metadata"
            )
        measured_draw = float(fit_delta["measured_draw_rms_nm"])
        if abs(measured_draw - delta) > 1.0e-9*max(1.0, delta):
            raise ValueError(
                f"Measured draw RMS {measured_draw!r} does not match "
                f"declared delta {delta!r}"
            )
        if str(fit_delta["truth_kernel_sha256"]) != production_kernel_sha256:
            raise ValueError(
                "Delta detector truth kernel digest does not match the "
                "production ladder artifact"
            )
        if str(fit_delta["prior_table_sha256"]) != str(
            knowledge["residual_model"]["prior_table_sha256"]
        ):
            raise ValueError(
                "Delta detector prior table digest does not match the "
                "freeze-bound digest"
            )
        if int(fit_delta["seed"]) != seed:
            raise ValueError(
                f"Delta detector seed {fit_delta['seed']!r} does not match "
                f"the derived direction seed {seed}"
            )
        delta_id = str(fit_delta["delta_id"])
        requested_draw = float(fit_delta["requested_amplitude_rms_nm"])
        if abs(requested_draw - delta) > 1.0e-9*max(1.0, delta):
            raise ValueError(
                f"Delta detector requested draw RMS {requested_draw!r} does "
                f"not match declared delta {delta!r}"
            )
        prior_table_sha256 = str(fit_delta["prior_table_sha256"])
        family = str(fit_delta["family"])
        if family != "combined":
            raise ValueError(
                f"Delta detector family {family!r} is not the frozen 'combined'"
            )
        fit_kernel_sha256 = str(fit_delta["fit_kernel_sha256"])
        truth_psf_config_hash = str(fit_delta["truth_psf_config_hash"])
        fit_psf_config_hash = str(fit_delta["fit_psf_config_hash"])
    else:
        delta_id = ""
        requested_draw = 0.0
        measured_draw = 0.0
        prior_table_sha256 = str(
            knowledge["residual_model"]["prior_table_sha256"]
        )
        family = "combined"
        fit_kernel_sha256 = production_kernel_sha256
        truth_config = _canonical_psf_config(rung_config["psf"])
        truth_psf_config_hash = config_hash(truth_config)
        fit_psf_config_hash = truth_psf_config_hash

    campaign_uuid = os.environ.get("HWOSLAPS_CAMPAIGN_UUID", "").strip()
    if not campaign_uuid:
        raise ValueError(
            "HWOSLAPS_CAMPAIGN_UUID must identify the PSF knowledge campaign"
        )
    ladder_artifact_sha256 = hashlib.sha256(
        ladder_artifact_path.read_bytes()
    ).hexdigest()
    config_hash_value = config_hash(config)
    output_dir = Path(args.output_dir)
    summary_path = output_dir/job_summary_name(delta, direction)

    identity = _job_identity(
        campaign_uuid,
        config_hash_value,
        system_id,
        revision,
        source_asset_sha256,
        ladder_artifact,
        ladder_artifact_sha256,
        production_kernel_sha256,
        fit_kernel_sha256,
        measured_truth_rms,
        aperture,
    )
    centre = extraction.aperture.centre_arcsec
    radius = extraction.aperture.radius_arcsec
    map_summaries = []
    total_map_wall = 0.0
    for rung in selected_rungs:
        logm = float(rung["logm"])
        artifact_path = output_dir/map_artifact_name(logm, delta, direction)
        if artifact_path.exists() and not args.force:
            stored = load_npz_record(artifact_path)
            summary = _summary_scalar_payload(stored)
            summary["artifact"] = str(artifact_path)
            map_summaries.append(summary)
            total_map_wall += summary["map_wall_seconds"]
            print(
                f"logm {logm:.2f}: matched cells "
                f"{summary['matched_cells']}, mismatch cells "
                f"{summary['mismatch_cells']}, spurious cells "
                f"{summary['spurious_cells']}, wall "
                f"{summary['map_wall_seconds']:.3f} s (skipped)",
                flush=True,
            )
            continue
        production = production_rung_reference(ladder_artifact, logm)
        start = perf_counter()
        run_ladder._point_detector_at_rung(detector, logm)
        grid_map = detector.compute_grid_map()
        map_wall_seconds = perf_counter() - start
        matched_metrics = run_ladder._rung_metrics(
            grid_map.y_coords,
            grid_map.x_coords,
            grid_map.q_asimov_2d,
            grid_map.detectable_mask_2d,
            grid_map.spacing_arcsec,
            centre,
            radius,
        )
        mismatch_metrics = None
        spurious_metrics = None
        if delta > 0.0:
            if any(
                value is None
                for value in (
                    grid_map.q_mismatch_2d,
                    grid_map.mismatch_detectable_mask_2d,
                    grid_map.q_spurious_2d,
                    grid_map.false_positive_mask_2d,
                )
            ):
                raise ValueError(
                    "Delta Fisher grid map did not expose all mismatch arrays"
                )
            mismatch_metrics = run_ladder._rung_metrics(
                grid_map.y_coords,
                grid_map.x_coords,
                grid_map.q_mismatch_2d,
                grid_map.mismatch_detectable_mask_2d,
                grid_map.spacing_arcsec,
                centre,
                radius,
            )
            spurious_metrics = run_ladder._rung_metrics(
                grid_map.y_coords,
                grid_map.x_coords,
                grid_map.q_spurious_2d,
                grid_map.false_positive_mask_2d,
                grid_map.spacing_arcsec,
                centre,
                radius,
            )
        aperture_mask = _aperture_inside_mask(
            grid_map.y_coords,
            grid_map.x_coords,
            centre,
            radius,
        )
        payload = _map_payload(
            identity,
            logm,
            list(rung["classes"]),
            delta,
            direction,
            seed,
            seed_spawn_key,
            delta_id,
            requested_draw,
            measured_draw,
            prior_table_sha256,
            family,
            production,
            matched_metrics,
            mismatch_metrics,
            spurious_metrics,
            grid_map,
            aperture_mask,
            detector_build_seconds,
            map_wall_seconds,
            truth_psf_config_hash,
            fit_psf_config_hash,
            float(rung_config["lensing"]["grid"]["pixel_scale"]),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        run_ladder._write_artifact(artifact_path, payload)
        map_summaries.append({
            "logm": logm,
            "rung_classes": list(rung["classes"]),
            "artifact": str(artifact_path),
            "matched_cells": int(round(
                matched_metrics["detectable_area_arcsec2"]
                / grid_map.spacing_arcsec**2
            )),
            "mismatch_cells": (
                int(round(
                    mismatch_metrics["detectable_area_arcsec2"]
                    / grid_map.spacing_arcsec**2
                ))
                if mismatch_metrics is not None
                else -1
            ),
            "spurious_cells": (
                int(round(
                    spurious_metrics["detectable_area_arcsec2"]
                    / grid_map.spacing_arcsec**2
                ))
                if spurious_metrics is not None
                else -1
            ),
            "detector_build_seconds": detector_build_seconds,
            "map_wall_seconds": map_wall_seconds,
        })
        total_map_wall += map_wall_seconds
        print(
            f"logm {logm:.2f}: matched cells "
            f"{map_summaries[-1]['matched_cells']}, mismatch cells "
            f"{map_summaries[-1]['mismatch_cells']}, spurious cells "
            f"{map_summaries[-1]['spurious_cells']}, wall "
            f"{map_wall_seconds:.3f} s",
            flush=True,
        )

    summary_identity = {
        key: (
            (value.tolist() if value.ndim else str(value.item()))
            if isinstance(value, np.ndarray)
            else value
        )
        for key, value in identity.items()
    }
    summary = {
        "schema_version": MAP_ARTIFACT_SCHEMA_VERSION,
        **summary_identity,
        "truth_psf_config_hash": truth_psf_config_hash,
        "fit_psf_config_hash": fit_psf_config_hash,
        "lensing_pixel_scale": float(
            rung_config["lensing"]["grid"]["pixel_scale"]
        ),
        "delta_nm": delta,
        "direction": direction,
        "seed": seed,
        "seed_spawn_key": seed_spawn_key,
        "delta_id": delta_id,
        "requested_draw_rms_nm": requested_draw,
        "measured_draw_rms_nm": measured_draw,
        "prior_table_sha256": prior_table_sha256,
        "family": family,
        "rungs": map_summaries,
        "artifacts": [entry["artifact"] for entry in map_summaries],
        "artifact_names": [
            Path(entry["artifact"]).name for entry in map_summaries
        ],
        "walls": [
            {
                "logm": entry["logm"],
                "detector_build_seconds": entry["detector_build_seconds"],
                "map_wall_seconds": entry["map_wall_seconds"],
            }
            for entry in map_summaries
        ],
        "total_wall_seconds": total_map_wall + detector_build_seconds,
        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_path, summary)
    print(f"PSF knowledge job summary: {summary_path}")


if __name__ == "__main__":
    main()
