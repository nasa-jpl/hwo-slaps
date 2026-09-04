#!/usr/bin/env python
"""Harvest Fisher PSF knowledge-error maps and compute frozen estimands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))
if str(REPO_ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"scripts"))

from extract_injection_positions import Q_MAX_RELATIVE_TOLERANCE  # noqa: E402
from run_nonlinear_validation import (  # noqa: E402
    PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY,
    derive_direction_seed,
    system_index,
)
from run_psf_knowledge_map import (  # noqa: E402
    load_npz_record,
    map_artifact_name,
)

FISHER_CAMPAIGN_NAME = "psf_knowledge_fisher_v1"
MAP_ARTIFACT_SCHEMA_VERSION = 1
NODE_SPACING_ARCSEC = 0.05
CELL_AREA_ARCSEC2 = NODE_SPACING_ARCSEC**2


def _scalar(record: dict, name: str):
    """Return one scalar map-artifact member."""
    if name not in record:
        raise ValueError(f"Map artifact is missing required member {name!r}")
    value = np.asarray(record[name])
    if value.ndim != 0:
        raise ValueError(f"Map artifact member {name!r} is not scalar")
    return value.item()


def _string_list(record: dict, name: str) -> list[str]:
    """Return one string-array map-artifact member."""
    if name not in record:
        raise ValueError(f"Map artifact is missing required member {name!r}")
    return [str(value) for value in np.atleast_1d(record[name])]


def _relative_difference(actual: float, expected: float) -> float:
    """Return a stable relative difference for a receipt comparison."""
    denominator = abs(float(expected))
    if denominator == 0.0:
        return abs(float(actual) - float(expected))
    return abs(float(actual) - float(expected))/denominator


def matched_receipt_findings(
    artifact: dict,
    manifest_rung: dict,
    label: str = "map",
) -> list[str]:
    """Return findings for one delta-0 matched-reference receipt.

    Parameters
    ----------
    artifact : dict
        Loaded delta-0 map artifact.
    manifest_rung : dict
        Manifest rung carrying production cells and q-max.
    label : str, optional
        Artifact label used in finding messages.

    Returns
    -------
    findings : list
        Receipt findings, empty when cells and q-max pass.
    """
    findings = []
    matched_cells = int(_scalar(artifact, "matched_cells"))
    production_cells = int(manifest_rung["production_cells"])
    if matched_cells != production_cells:
        findings.append(
            f"{label}: matched cells {matched_cells} do not equal "
            f"production cells {production_cells}"
        )
    matched_q = float(_scalar(artifact, "matched_q_max"))
    production_q = float(manifest_rung["production_q_max"])
    if _relative_difference(matched_q, production_q) > Q_MAX_RELATIVE_TOLERANCE:
        findings.append(
            f"{label}: matched q_max {matched_q!r} does not reproduce "
            f"production q_max {production_q!r} to relative "
            f"{Q_MAX_RELATIVE_TOLERANCE}"
        )
    return findings


def _mask_sha256(mask: np.ndarray) -> str:
    """Return the SHA-256 digest of an unpacked boolean mask."""
    values = np.ascontiguousarray(np.asarray(mask, dtype=np.bool_))
    import hashlib

    return hashlib.sha256(values.tobytes()).hexdigest()


def unpack_record_mask(
    artifact: dict,
    prefix: str,
    label: str,
) -> np.ndarray:
    """Unpack and verify one stored map mask.

    Parameters
    ----------
    artifact : dict
        Loaded map artifact.
    prefix : str
        Mask prefix, such as detectable_mask.
    label : str
        Artifact label used in failure messages.

    Returns
    -------
    mask : numpy.ndarray
        Unpacked boolean mask with its recorded shape.

    Raises
    ------
    ValueError
        Raised when the packed bits, shape or digest is malformed.
    """
    packed_name = f"{prefix}_packed"
    shape_name = f"{prefix}_shape"
    digest_name = f"{prefix}_sha256"
    for name in (packed_name, shape_name, digest_name):
        if name not in artifact:
            raise ValueError(f"{label}: missing mask member {name}")
    shape = np.asarray(artifact[shape_name], dtype=int)
    if shape.ndim != 1 or shape.size != 2 or np.any(shape < 0):
        raise ValueError(f"{label}: mask {prefix} shape is malformed")
    shape_tuple = (int(shape[0]), int(shape[1]))
    packed = np.asarray(artifact[packed_name])
    if packed.dtype != np.uint8:
        raise ValueError(f"{label}: mask {prefix} is not uint8 packed data")
    unpacked = np.unpackbits(packed).astype(bool)
    size = shape_tuple[0]*shape_tuple[1]
    if unpacked.size < size:
        raise ValueError(f"{label}: mask {prefix} has too few packed bits")
    mask = unpacked[:size].reshape(shape_tuple)
    observed = _mask_sha256(mask)
    expected = str(_scalar(artifact, digest_name))
    if observed != expected:
        raise ValueError(
            f"{label}: mask {prefix} sha256 {observed} does not match "
            f"recorded {expected}"
        )
    return mask


def reconcile_mask_cells(artifact: dict, label: str) -> list[str]:
    """Recount the aperture-clipped cells from the stored masks.

    Parameters
    ----------
    artifact : dict
        Loaded map artifact carrying the four packed masks.
    label : str
        Artifact label used in finding messages.

    Returns
    -------
    findings : list
        One finding per scalar cell count that the masks do not reproduce.
    """
    findings = []
    try:
        inside = unpack_record_mask(artifact, "aperture_mask", label)
    except ValueError as exc:
        return [str(exc)]
    nodes_inside = int(np.count_nonzero(inside))
    if nodes_inside != int(_scalar(artifact, "nodes_inside_aperture")):
        findings.append(
            f"{label}: nodes_inside_aperture "
            f"{int(_scalar(artifact, 'nodes_inside_aperture'))} does not "
            f"equal the aperture mask count {nodes_inside}"
        )
    delta = float(_scalar(artifact, "delta_nm"))
    checks = [("detectable_mask", "matched_cells")]
    if delta > 0.0:
        checks.extend([
            ("mismatch_detectable_mask", "mismatch_cells"),
            ("false_positive_mask", "spurious_cells"),
        ])
    for prefix, scalar_name in checks:
        try:
            mask = unpack_record_mask(artifact, prefix, label)
        except ValueError as exc:
            findings.append(str(exc))
            continue
        if mask.shape != inside.shape:
            findings.append(
                f"{label}: mask {prefix} shape {mask.shape} differs from "
                f"the aperture mask shape {inside.shape}"
            )
            continue
        recount = int(np.count_nonzero(mask & inside))
        recorded = int(_scalar(artifact, scalar_name))
        if recount != recorded:
            findings.append(
                f"{label}: {scalar_name} {recorded} does not equal the "
                f"aperture-clipped {prefix} count {recount}"
            )
    spacing = float(_scalar(artifact, "spacing_arcsec"))
    if spacing != NODE_SPACING_ARCSEC:
        findings.append(
            f"{label}: spacing_arcsec {spacing!r} is not the frozen "
            f"{NODE_SPACING_ARCSEC}"
        )
    for scalar_name, area_name in (
        ("matched_cells", "matched_area_arcsec2"),
        ("mismatch_cells", "mismatch_area_arcsec2"),
        ("spurious_cells", "spurious_area_arcsec2"),
    ):
        if delta == 0.0 and scalar_name != "matched_cells":
            continue
        expected_area = int(_scalar(artifact, scalar_name))*NODE_SPACING_ARCSEC**2
        recorded_area = float(_scalar(artifact, area_name))
        if not np.isfinite(recorded_area) or abs(
            recorded_area - expected_area
        ) > 1.0e-12*max(1.0, expected_area):
            findings.append(
                f"{label}: {area_name} {recorded_area!r} is not "
                f"{scalar_name} times the cell area ({expected_area!r})"
            )
    return findings


def _quantiles(values: list[float]) -> dict:
    """Return 10th, 50th and 90th percentiles for finite values."""
    if not values:
        return {"q10": None, "q50": None, "q90": None}
    return {
        "q10": float(np.percentile(values, 10)),
        "q50": float(np.percentile(values, 50)),
        "q90": float(np.percentile(values, 90)),
    }


def _delta_key(delta: float) -> str:
    """Return the manifest's compact delta key."""
    value = float(delta)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _rung_key(system_id: str, logm: float) -> tuple[str, float]:
    """Return a stable system and rung grouping key."""
    return str(system_id), float(logm)


def _artifact_row(
    artifact: dict,
    job: dict,
    delta: float,
    direction: int,
    artifact_path: Path,
) -> dict:
    """Reduce one verified map artifact to a scalar harvest row."""
    return {
        "artifact": str(artifact_path),
        "system_id": str(_scalar(artifact, "system_id")),
        "template": job["template"],
        "golden": bool(job["golden"]),
        "tier": job["tier"],
        "logm": float(_scalar(artifact, "logm")),
        "rung_classes": _string_list(artifact, "rung_classes"),
        "delta_nm": float(delta),
        "direction": int(direction),
        "seed": int(_scalar(artifact, "seed")),
        "delta_id": str(_scalar(artifact, "delta_id")),
        "requested_draw_rms_nm": float(
            _scalar(artifact, "requested_draw_rms_nm")
        ),
        "measured_draw_rms_nm": float(
            _scalar(artifact, "measured_draw_rms_nm")
        ),
        "family": str(_scalar(artifact, "family")),
        "truth_psf_config_hash": str(
            _scalar(artifact, "truth_psf_config_hash")
        ),
        "fit_psf_config_hash": str(_scalar(artifact, "fit_psf_config_hash")),
        "lensing_pixel_scale": float(
            _scalar(artifact, "lensing_pixel_scale")
        ),
        "production_q_max": float(_scalar(artifact, "production_q_max")),
        "production_detectable_area_arcsec2": float(
            _scalar(artifact, "production_detectable_area_arcsec2")
        ),
        "production_cells": int(_scalar(artifact, "production_cells")),
        "matched_q_max": float(_scalar(artifact, "matched_q_max")),
        "matched_cells": int(_scalar(artifact, "matched_cells")),
        "matched_area_arcsec2": float(
            _scalar(artifact, "matched_area_arcsec2")
        ),
        "matched_aperture_fraction": float(
            _scalar(artifact, "matched_aperture_fraction")
        ),
        "matched_perimeter_clipped": bool(
            _scalar(artifact, "matched_perimeter_clipped")
        ),
        "mismatch_q_max": float(_scalar(artifact, "mismatch_q_max")),
        "mismatch_cells": int(_scalar(artifact, "mismatch_cells")),
        "mismatch_area_arcsec2": float(
            _scalar(artifact, "mismatch_area_arcsec2")
        ),
        "spurious_q_max": float(_scalar(artifact, "spurious_q_max")),
        "spurious_cells": int(_scalar(artifact, "spurious_cells")),
        "spurious_area_arcsec2": float(
            _scalar(artifact, "spurious_area_arcsec2")
        ),
        "detectable_area_arcsec2": float(
            _scalar(artifact, "detectable_area_arcsec2")
        ),
        "mismatch_detectable_area_arcsec2": float(
            _scalar(artifact, "mismatch_detectable_area_arcsec2")
        ),
        "false_positive_area_arcsec2": float(
            _scalar(artifact, "false_positive_area_arcsec2")
        ),
        "num_detectable": int(_scalar(artifact, "num_detectable")),
        "num_mismatch_detectable": int(
            _scalar(artifact, "num_mismatch_detectable")
        ),
        "num_false_positive": int(_scalar(artifact, "num_false_positive")),
        "max_z_spurious": float(_scalar(artifact, "max_z_spurious")),
        "nodes_inside_aperture": int(
            _scalar(artifact, "nodes_inside_aperture")
        ),
        "spacing_arcsec": float(_scalar(artifact, "spacing_arcsec")),
        "detector_build_seconds": float(
            _scalar(artifact, "detector_build_seconds")
        ),
        "map_wall_seconds": float(_scalar(artifact, "map_wall_seconds")),
        "code_revision_sha256": str(
            _scalar(artifact, "code_revision_sha256")
        ),
        "config_hash": str(_scalar(artifact, "config_hash")),
        "campaign_uuid": str(_scalar(artifact, "campaign_uuid")),
        "ladder_campaign_uuid": str(
            _scalar(artifact, "ladder_campaign_uuid")
        ),
        "ladder_config_hash": str(_scalar(artifact, "ladder_config_hash")),
        "ladder_artifact_sha256": str(
            _scalar(artifact, "ladder_artifact_sha256")
        ),
        "truth_kernel_sha256": str(
            _scalar(artifact, "truth_kernel_sha256")
        ),
        "fit_kernel_sha256": str(_scalar(artifact, "fit_kernel_sha256")),
        "source_asset_sha256": str(
            _scalar(artifact, "source_asset_sha256")
        ),
    }


def verify_map_artifact(
    artifact: dict,
    artifact_path: Path,
    job: dict,
    manifest: dict,
    manifest_rung: dict,
    freeze: dict,
    delta: float,
    direction: int,
) -> tuple[dict | None, list[str]]:
    """Verify one map artifact and return its scalar harvest row.

    Parameters
    ----------
    artifact : dict
        Loaded map artifact.
    artifact_path : pathlib.Path
        Artifact path used in finding messages.
    job : dict
        Fisher campaign manifest job entry.
    manifest : dict
        Fisher campaign manifest.
    manifest_rung : dict
        Expected production rung record.
    freeze : dict
        Validated design freeze document.
    delta : float
        Queue delta coordinate.
    direction : int
        Queue direction coordinate.

    Returns
    -------
    row : dict or None
        Scalar harvest row when all scalar members are present.
    findings : list
        Integrity findings for this artifact.
    """
    label = str(artifact_path)
    findings = []
    required = (
        "schema_version",
        "campaign_uuid",
        "code_revision_sha256",
        "config_hash",
        "system_id",
        "psf_state",
        "psf_kernel_shape_native",
        "ladder_campaign_uuid",
        "ladder_config_hash",
        "ladder_artifact_sha256",
        "truth_kernel_sha256",
        "fit_kernel_sha256",
        "delta_nm",
        "direction",
        "seed",
        "seed_spawn_key",
        "delta_id",
        "measured_draw_rms_nm",
        "requested_draw_rms_nm",
        "prior_table_sha256",
        "family",
        "truth_psf_config_hash",
        "fit_psf_config_hash",
        "lensing_pixel_scale",
        "logm",
        "rung_classes",
        "production_q_max",
        "production_detectable_area_arcsec2",
        "production_cells",
        "matched_q_max",
        "matched_cells",
        "matched_area_arcsec2",
        "matched_aperture_fraction",
        "matched_perimeter_clipped",
        "mismatch_q_max",
        "mismatch_cells",
        "mismatch_area_arcsec2",
        "spurious_q_max",
        "spurious_cells",
        "spurious_area_arcsec2",
        "detectable_area_arcsec2",
        "mismatch_detectable_area_arcsec2",
        "false_positive_area_arcsec2",
        "num_detectable",
        "num_mismatch_detectable",
        "num_false_positive",
        "max_z_spurious",
        "nodes_inside_aperture",
        "spacing_arcsec",
        "detector_build_seconds",
        "map_wall_seconds",
        "source_asset_sha256",
        "aperture_sha256",
        "contour_sha256",
    )
    missing = [name for name in required if name not in artifact]
    if missing:
        return None, [f"{label}: missing required members {missing}"]
    if int(_scalar(artifact, "schema_version")) != MAP_ARTIFACT_SCHEMA_VERSION:
        findings.append(f"{label}: unexpected schema_version")
    if str(_scalar(artifact, "campaign_uuid")) != manifest["campaign_uuid"]:
        findings.append(f"{label}: campaign_uuid does not match manifest")
    if str(_scalar(artifact, "code_revision_sha256")) != manifest[
        "code_revision"
    ]["sha256"]:
        findings.append(f"{label}: code_revision_sha256 does not match manifest")
    if str(_scalar(artifact, "config_hash")) != job["restamped_config_hash"]:
        findings.append(f"{label}: config_hash does not match restamped_config_hash")
    if str(_scalar(artifact, "system_id")) != job["system_id"]:
        findings.append(f"{label}: system_id does not match manifest job")
    if str(_scalar(artifact, "psf_state")) != "science35":
        findings.append(f"{label}: psf_state is not science35")
    if list(np.asarray(artifact["psf_kernel_shape_native"], dtype=int)) != [999, 999]:
        findings.append(f"{label}: psf kernel shape is not [999, 999]")
    for name, expected in (
        ("ladder_campaign_uuid", job["ladder_campaign_uuid"]),
        ("ladder_config_hash", job["ladder_config_hash"]),
        ("ladder_artifact_sha256", job["ladder_artifact_sha256"]),
        ("truth_kernel_sha256", job["psf_kernel_sha256"]),
    ):
        if str(_scalar(artifact, name)) != str(expected):
            findings.append(f"{label}: {name} does not match manifest job")
    actual_delta = float(_scalar(artifact, "delta_nm"))
    if actual_delta != float(delta):
        findings.append(f"{label}: delta_nm {actual_delta!r} does not match queue")
    actual_direction = int(_scalar(artifact, "direction"))
    if actual_direction != int(direction):
        findings.append(
            f"{label}: direction {actual_direction!r} does not match queue"
        )
    expected_seed = -1
    expected_spawn_key = []
    if delta > 0.0:
        expected_seed = derive_direction_seed(
            int(freeze["seeds"]["entropy"]),
            direction,
            system_index(job["system_id"]),
        )
        expected_spawn_key = [
            PSF_KNOWLEDGE_DIRECTION_SPAWN_KEY,
            int(direction),
            system_index(job["system_id"]),
        ]
    if int(_scalar(artifact, "seed")) != expected_seed:
        findings.append(
            f"{label}: seed does not match the re-derived {expected_seed}"
        )
    if list(np.asarray(artifact["seed_spawn_key"], dtype=int)) != expected_spawn_key:
        findings.append(
            f"{label}: seed_spawn_key does not match {expected_spawn_key!r}"
        )
    measured = float(_scalar(artifact, "measured_draw_rms_nm"))
    if abs(measured - delta) > 1.0e-9*max(1.0, delta):
        findings.append(f"{label}: measured_draw_rms_nm does not match delta")
    requested = float(_scalar(artifact, "requested_draw_rms_nm"))
    if abs(requested - delta) > 1.0e-9*max(1.0, delta):
        findings.append(f"{label}: requested_draw_rms_nm does not match delta")
    prior_digest = freeze["psf_knowledge_error"]["residual_model"][
        "prior_table_sha256"
    ]
    if str(_scalar(artifact, "prior_table_sha256")) != prior_digest:
        findings.append(f"{label}: prior_table_sha256 does not match the freeze")
    if str(_scalar(artifact, "family")) != "combined":
        findings.append(f"{label}: family is not combined")
    if delta == 0.0 and str(_scalar(artifact, "delta_id")) != "":
        findings.append(f"{label}: delta-0 artifact carries a delta_id")
    if delta > 0.0:
        if str(_scalar(artifact, "family")) != "combined":
            findings.append(f"{label}: positive delta family is not combined")
        if not np.isfinite(float(_scalar(artifact, "requested_draw_rms_nm"))):
            findings.append(f"{label}: requested_draw_rms_nm is not finite")
        from hwoslaps.psf.mismatch import _identity_from_payload

        expected_delta_id = _identity_from_payload({
            "schema": "psf_mismatch_delta_v1",
            "prior_table_sha256": str(_scalar(
                artifact, "prior_table_sha256"
            )),
            "amplitude_rms_nm": actual_delta,
            "seed": expected_seed,
            "family": "combined",
            "truth_psf_config_hash": str(_scalar(
                artifact, "truth_psf_config_hash"
            )),
            "lensing_pixel_scale": float(_scalar(
                artifact, "lensing_pixel_scale"
            )),
        })
        if str(_scalar(artifact, "delta_id")) != expected_delta_id:
            findings.append(f"{label}: delta_id does not re-derive from payload")
    if delta == 0.0 and str(_scalar(artifact, "fit_kernel_sha256")) != str(
        _scalar(artifact, "truth_kernel_sha256")
    ):
        findings.append(f"{label}: delta-0 fit kernel is not the truth kernel")
    actual_logm = float(_scalar(artifact, "logm"))
    if not np.isclose(actual_logm, float(manifest_rung["logm"]), rtol=0.0, atol=1.0e-9):
        findings.append(f"{label}: logm is not the manifest rung")
    if _string_list(artifact, "rung_classes") != list(manifest_rung["classes"]):
        findings.append(f"{label}: rung_classes do not match the manifest rung")
    if int(_scalar(artifact, "production_cells")) != int(
        manifest_rung["production_cells"]
    ):
        findings.append(f"{label}: production_cells do not match the manifest rung")
    if not int(_scalar(artifact, "nodes_inside_aperture")) > 0:
        findings.append(f"{label}: nodes_inside_aperture is not positive")
    for prefix in (
        "detectable_mask",
        "mismatch_detectable_mask",
        "false_positive_mask",
    ):
        try:
            unpack_record_mask(artifact, prefix, label)
        except ValueError as exc:
            findings.append(str(exc))
    findings.extend(reconcile_mask_cells(artifact, label))
    if delta == 0.0:
        findings.extend(matched_receipt_findings(artifact, manifest_rung, label))
    try:
        row = _artifact_row(artifact, job, delta, direction, artifact_path)
    except ValueError as exc:
        findings.append(f"{label}: {exc}")
        row = None
    return row, findings


def _group_rows(rows: list[dict]) -> dict[tuple[str, float], list[dict]]:
    """Group map rows by system and mass rung."""
    groups = {}
    for row in rows:
        groups.setdefault(_rung_key(row["system_id"], row["logm"]), []).append(row)
    return groups


def _direction_estimand_rows(
    rows: list[dict],
    matched_cells: int,
    floor_cells: int,
) -> tuple[list[dict], int, int]:
    """Build per-direction retention and spurious rows at one denominator."""
    estimands = []
    below_floor = 0
    zero_area = 0
    for row in sorted(rows, key=lambda item: item["direction"]):
        mismatch_cells = int(row["mismatch_cells"])
        spurious_cells = int(row["spurious_cells"])
        if matched_cells == 0:
            zero_area += 1
            retention = None
            spurious = None
            below = True
            mismatch_reported = None
            mismatch_area = None
        elif matched_cells < floor_cells:
            below_floor += 1
            retention = None
            spurious = None
            below = True
            mismatch_reported = mismatch_cells
            mismatch_area = float(row["mismatch_area_arcsec2"])
        else:
            retention = mismatch_cells/matched_cells
            spurious = spurious_cells/matched_cells
            below = False
            mismatch_reported = mismatch_cells
            mismatch_area = float(row["mismatch_area_arcsec2"])
        estimands.append({
            "direction": int(row["direction"]),
            "seed": int(row["seed"]),
            "mismatch_cells": mismatch_reported,
            "spurious_cells": spurious_cells,
            "mismatch_area_arcsec2": mismatch_area,
            "spurious_area_arcsec2": float(row["spurious_area_arcsec2"]),
            "R": retention,
            "F": spurious,
            "below_ratio_floor": below,
        })
    return estimands, below_floor, zero_area


def _delta_estimand_summary(
    rows: list[dict],
    matched_cells: int,
    floor_cells: int,
    delta: float,
) -> dict:
    """Summarize one system-rung delta over its direction rows."""
    if delta == 0.0:
        return {
            "delta_nm": 0.0,
            "n_directions": 1,
            "directions": [],
            "quantiles": {
                "R": _quantiles([]),
                "F": _quantiles([]),
                "mismatch_area_arcsec2": _quantiles([]),
                "spurious_area_arcsec2": _quantiles([]),
            },
            "below_ratio_floor": matched_cells < floor_cells,
            "zero_area_exclusion_count": int(matched_cells == 0),
            "endpoint_anchor": False,
        }
    directions, below_floor, zero_area = _direction_estimand_rows(
        rows, matched_cells, floor_cells
    )
    if float(delta) == 35.0:
        for entry in directions:
            entry["R"] = None
            entry["F"] = None
    retention = [entry["R"] for entry in directions if entry["R"] is not None]
    spurious = [entry["F"] for entry in directions if entry["F"] is not None]
    mismatch_areas = [
        entry["mismatch_area_arcsec2"]
        for entry in directions
        if entry["mismatch_area_arcsec2"] is not None
    ]
    spurious_areas = [entry["spurious_area_arcsec2"] for entry in directions]
    return {
        "delta_nm": float(delta),
        "n_directions": len(directions),
        "directions": directions,
        "quantiles": {
            "R": _quantiles(retention),
            "F": _quantiles(spurious),
            "mismatch_area_arcsec2": _quantiles(mismatch_areas),
            "spurious_area_arcsec2": _quantiles(spurious_areas),
        },
        "below_ratio_floor": bool(matched_cells < floor_cells),
        "below_ratio_floor_direction_count": below_floor,
        "zero_area_exclusion_count": zero_area,
        "endpoint_anchor": float(delta) == 35.0,
    }


def _delta_star(
    summaries: dict[float, dict],
    retention_gate: float,
    spurious_gate: float,
    endpoint: float,
) -> dict:
    """Return the largest passing delta and its gate result."""
    passing = []
    for delta, summary in summaries.items():
        if float(delta) == float(endpoint):
            continue
        q_r = summary["quantiles"]["R"]["q10"]
        q_f = summary["quantiles"]["F"]["q90"]
        if (
            q_r is not None
            and q_f is not None
            and q_r >= retention_gate
            and q_f <= spurious_gate
        ):
            passing.append(float(delta))
    return {
        "delta_star": max(passing) if passing else None,
        "none_passes": not bool(passing),
        "retention_q10_min": float(retention_gate),
        "spurious_q90_max": float(spurious_gate),
        "passing_deltas": sorted(passing),
    }


def first_spurious_delta(summaries: dict[float, dict]) -> float | None:
    """Return the smallest delta with a positive spurious cell count."""
    for delta in sorted(summaries, key=float):
        if any(
            int(direction["spurious_cells"]) > 0
            for direction in summaries[delta].get("directions", [])
        ):
            return float(delta)
    return None


def compute_fisher_science(rows: list[dict], manifest: dict, freeze: dict) -> dict:
    """Compute Fisher retention, spurious and delta-star summaries.

    Parameters
    ----------
    rows : `list` [`dict`]
        Scalar rows from verified map artifacts.
    manifest : `dict`
        Fisher campaign manifest.
    freeze : `dict`
        Validated design freeze document.

    Returns
    -------
    science : `dict`
        System-rung estimands, tier summaries, template splits and receipts.
    """
    knowledge = freeze["psf_knowledge_error"]
    residual = knowledge["residual_model"]
    deltas = [float(value) for value in residual["amplitude_rms_nm_rungs"]]
    endpoint = float(residual["endpoint_anchor_nm"])
    floor_cells = int(knowledge["ratio_floor"]["cells"])
    gates = [
        (
            "default",
            float(knowledge["gates"]["retention_q10_min"]),
            float(knowledge["gates"]["spurious_q90_max"]),
        )
    ]
    gates.extend(
        (
            f"sensitivity_{float(pair[0]):g}_{float(pair[1]):g}",
            float(pair[0]),
            float(pair[1]),
        )
        for pair in knowledge["gates"]["sensitivity"]
    )
    grouped = _group_rows(rows)
    per_system = {}
    receipt = []
    pairing = []
    for job in manifest["jobs"]:
        system_id = job["system_id"]
        for manifest_rung in job["rungs"]:
            logm = float(manifest_rung["logm"])
            key = _rung_key(system_id, logm)
            rung_rows = grouped.get(key, [])
            matched_rows = [row for row in rung_rows if row["delta_nm"] == 0.0]
            matched_cells = (
                matched_rows[0]["matched_cells"] if matched_rows else None
            )
            if len(matched_rows) == 1:
                receipt_findings = []
                if matched_cells != manifest_rung["production_cells"]:
                    receipt_findings.append("matched cell count differs")
                if _relative_difference(
                    matched_rows[0]["matched_q_max"],
                    manifest_rung["production_q_max"],
                ) > Q_MAX_RELATIVE_TOLERANCE:
                    receipt_findings.append("matched q_max differs")
                receipt.append({
                    "system_id": system_id,
                    "logm": logm,
                    "production_cells": manifest_rung["production_cells"],
                    "matched_cells": matched_rows[0]["matched_cells"],
                    "production_q_max": manifest_rung["production_q_max"],
                    "matched_q_max": matched_rows[0]["matched_q_max"],
                    "findings": receipt_findings,
                })
            delta_summaries = {}
            for delta in deltas:
                delta_rows = [
                    row for row in rung_rows if row["delta_nm"] == delta
                ]
                delta_summaries[delta] = _delta_estimand_summary(
                    delta_rows,
                    matched_cells if matched_cells is not None else 0,
                    floor_cells,
                    delta,
                )
                if delta > 0.0:
                    for direction in sorted(
                        delta_rows, key=lambda row: row["direction"]
                    ):
                        pairing.append({
                            "system_id": system_id,
                            "logm": logm,
                            "direction": direction["direction"],
                            "delta_nm": delta,
                            "seed": direction["seed"],
                            "measured_draw_rms_nm": direction[
                                "measured_draw_rms_nm"
                            ],
                        })
            class_entry = {
                "system_id": system_id,
                "template": job["template"],
                "logm": logm,
                "classes": list(manifest_rung["classes"]),
                "matched_cells": matched_cells,
                "matched_area_arcsec2": (
                    None
                    if matched_cells is None
                    else matched_cells*CELL_AREA_ARCSEC2
                ),
                "per_delta": {
                    _delta_key(delta): summary
                    for delta, summary in delta_summaries.items()
                },
                "first_spurious_delta": first_spurious_delta(delta_summaries),
                "delta_star": {
                    name: _delta_star(
                        delta_summaries,
                        retention_gate,
                        spurious_gate,
                        endpoint,
                    )
                    for name, retention_gate, spurious_gate in gates
                },
                "below_ratio_floor": (
                    matched_cells is None or matched_cells < floor_cells
                ),
                "zero_area_exclusion_count": sum(
                    summary["zero_area_exclusion_count"]
                    for delta, summary in delta_summaries.items()
                    if delta > 0.0
                ),
            }
            for class_name in manifest_rung["classes"]:
                per_system.setdefault(class_name, []).append(class_entry)
    tier_summary = {}
    for class_name in sorted(per_system):
        entries = per_system[class_name]
        class_summary = {}
        for gate_name, _, _ in gates:
            values = [
                entry["delta_star"][gate_name]["delta_star"]
                for entry in entries
                if entry["delta_star"][gate_name]["delta_star"] is not None
            ]
            with_ratios = sum(
                any(
                    direction["R"] is not None
                    for direction in entry["per_delta"]["1"]["directions"]
                )
                for entry in entries
                if "1" in entry["per_delta"]
            )
            class_summary[gate_name] = {
                "n_with_delta_star": len(values),
                "n_with_ratios": with_ratios,
                "n_none_passes": with_ratios - len(values),
                "median_delta_star": (
                    float(np.median(values)) if values else None
                ),
                "range_delta_star": (
                    [min(values), max(values)] if values else None
                ),
            }
        class_summary["n_system_rungs"] = len(entries)
        class_summary["n_below_ratio_floor"] = sum(
            int(entry["below_ratio_floor"]) for entry in entries
        )
        default = class_summary["default"]
        class_summary["requirement_sentence"] = (
            "For " + class_name + ", the largest drift-shaped residual "
            "with Q10[R] >= "
            f"{knowledge['gates']['retention_q10_min']:g} and Q90[F] <= "
            f"{knowledge['gates']['spurious_q90_max']:g} has median "
            f"delta* {default['median_delta_star']!r} nm RMS over "
            f"{default['n_with_delta_star']} system rungs."
        )
        tier_summary[class_name] = class_summary
    per_template = {}
    for class_name, entries in per_system.items():
        by_template = {}
        for entry in entries:
            by_template.setdefault(entry["template"], []).append(entry)
        per_template[class_name] = {
            template: {
                "n": len(template_entries),
                "median_delta_star": (
                    float(np.median([
                        item["delta_star"]["default"]["delta_star"]
                        for item in template_entries
                        if item["delta_star"]["default"]["delta_star"] is not None
                    ]))
                    if any(
                        item["delta_star"]["default"]["delta_star"] is not None
                        for item in template_entries
                    )
                    else None
                ),
            }
            for template, template_entries in sorted(by_template.items())
        }
    pairing_groups = {}
    for entry in pairing:
        key = (entry["system_id"], entry["logm"], entry["direction"])
        group = pairing_groups.setdefault(key, {})
        group[_delta_key(entry["delta_nm"])] = {
            "seed": entry["seed"],
            "measured_draw_rms_nm": entry["measured_draw_rms_nm"],
        }
    pairing_receipt = []
    for (system_id, logm, direction), by_delta in sorted(pairing_groups.items()):
        seeds = {value["seed"] for value in by_delta.values()}
        pairing_receipt.append({
            "system_id": system_id,
            "logm": logm,
            "direction": direction,
            "by_delta": by_delta,
            "seed_same_across_deltas": len(seeds) == 1,
        })
    return {
        "gates": {
            "default": [
                knowledge["gates"]["retention_q10_min"],
                knowledge["gates"]["spurious_q90_max"],
            ],
            "sensitivity": knowledge["gates"]["sensitivity"],
        },
        "ratio_floor": {
            "cells": floor_cells,
            "arcsec2": knowledge["ratio_floor"]["arcsec2"],
        },
        "per_system_rung_class": per_system,
        "tier_summary": tier_summary,
        "per_template": per_template,
        "receipt": receipt,
        "direction_pairing": pairing_receipt,
        "endpoint_anchor_nm": endpoint,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", help="Fisher campaign directory")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write the partial harvest without failing on missing maps",
    )
    return parser


def main(argv=None) -> None:
    """Harvest every expected Fisher map and write review artifacts."""
    args = _build_parser().parse_args(argv)
    campaign_dir = Path(args.campaign_dir)
    manifest_path = campaign_dir/"manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"Fisher manifest {manifest_path} must be a mapping")
    if manifest.get("name") != FISHER_CAMPAIGN_NAME:
        raise ValueError(
            f"Fisher harvest requires {FISHER_CAMPAIGN_NAME!r}, got "
            f"{manifest.get('name')!r}"
        )
    from hwoslaps.campaign.design_freeze import load_design_freeze

    freeze = load_design_freeze()
    knowledge = freeze["psf_knowledge_error"]
    residual = knowledge["residual_model"]
    deltas = [float(value) for value in residual["amplitude_rms_nm_rungs"]]
    directions = [int(value) for value in residual["direction_indices"]]
    rows = []
    missing = []
    findings = []
    expected_maps = 0
    manifest_jobs = manifest.get("jobs")
    if not isinstance(manifest_jobs, list):
        raise ValueError(f"Fisher manifest {manifest_path} jobs must be a list")
    for job in manifest_jobs:
        if not isinstance(job, dict):
            findings.append("manifest contains a malformed job")
            continue
        for manifest_rung in job.get("rungs", []):
            if not isinstance(manifest_rung, dict):
                findings.append(
                    f"{job.get('run_name', '<job>')}: malformed manifest rung"
                )
                continue
            logm = float(manifest_rung["logm"])
            for delta in deltas:
                direction_values = [0] if delta == 0.0 else directions
                for direction in direction_values:
                    expected_maps += 1
                    artifact_path = (
                        Path(job["output_dir"])
                        / map_artifact_name(logm, delta, direction)
                    )
                    if not artifact_path.is_file():
                        missing.append(str(artifact_path))
                        continue
                    artifact = load_npz_record(artifact_path)
                    row, artifact_findings = verify_map_artifact(
                        artifact,
                        artifact_path,
                        job,
                        manifest,
                        manifest_rung,
                        freeze,
                        delta,
                        direction,
                    )
                    findings.extend(artifact_findings)
                    if row is not None:
                        rows.append(row)
    declared_maps = manifest.get("n_maps")
    if declared_maps != expected_maps:
        findings.append(
            f"manifest n_maps {declared_maps!r} does not equal expected "
            f"{expected_maps}"
        )
    declared_jobs = manifest.get("n_jobs")
    expected_jobs = len(manifest_jobs)*49
    if declared_jobs != expected_jobs:
        findings.append(
            f"manifest n_jobs {declared_jobs!r} does not equal expected "
            f"{expected_jobs}"
        )
    science = compute_fisher_science(rows, manifest, freeze)
    for receipt in science["direction_pairing"]:
        if not receipt["seed_same_across_deltas"]:
            findings.append(
                f"{receipt['system_id']}/{receipt['logm']}/dir"
                f"{receipt['direction']}: direction seed differs across deltas"
            )
    verdict = "CLEAN"
    if missing:
        verdict = "INCOMPLETE"
    elif findings:
        verdict = "FINDINGS"
    review = {
        "schema_version": MAP_ARTIFACT_SCHEMA_VERSION,
        "campaign_uuid": manifest["campaign_uuid"],
        "code_revision": manifest["code_revision"],
        "expected_maps": expected_maps,
        "missing": missing,
        "integrity_findings": findings,
        "receipt": science["receipt"],
        "science": science,
        "verdict": verdict,
        "total_map_wall_hours": float(
            sum(row["map_wall_seconds"] for row in rows)/3600.0
        ),
    }
    harvest_dir = campaign_dir/"harvest"
    harvest_dir.mkdir(parents=True, exist_ok=True)
    (harvest_dir/"harvest.json").write_text(
        json.dumps({
            "schema_version": MAP_ARTIFACT_SCHEMA_VERSION,
            "campaign_uuid": manifest["campaign_uuid"],
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (harvest_dir/"review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(review, indent=2, sort_keys=True))
    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"Campaign incomplete: {len(missing)} map artifacts missing"
        )
    if findings:
        raise SystemExit(f"{len(findings)} integrity findings")


if __name__ == "__main__":
    main()
