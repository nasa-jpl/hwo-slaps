#!/usr/bin/env python
"""Extract one ladder member's nonlinear-validation injection point.

The DesignFreeze v3 ``nonlinear_validation`` protocol injects each
system's trial subhalo at the bracket-top rung of its measured ladder
(the first refined rung at which ``q_F`` reached the declared threshold)
and at the position where that rung's Fisher grid map peaks inside the
D-F7 aperture. The ladder artifact stores per-rung aggregates only, so
this runner recomputes the one injection rung's grid map with the same
staged configuration, kernel, engine and detector construction as the
production ladder, and records the aperture argmax.

The recomputed aperture ``q_max`` must agree with the artifact's
recorded rung ``q_max`` to a relative 1.0e-6, which makes every
extraction a determinism canary of the production campaign; any
disagreement fails closed.

Right-censored members (``m_best`` NaN) take the ladder ceiling rung,
logm 9.5, where their expected nonlinear outcome is non-detection.

The artifact is ``injection_position.json`` under ``--output-dir``,
carrying the injection rung, the position, the recomputed and recorded
``q_max``, and the identity of the ladder artifact it derives from.
"""

from __future__ import annotations

import argparse
import json
import math
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

ARTIFACT_NAME = "injection_position.json"

CENSORED_LOGM = 9.5
"""Injection rung of a right-censored member (`float`)."""

Q_MAX_RELATIVE_TOLERANCE = 1.0e-6
"""Largest accepted recomputed-versus-recorded ``q_max`` departure (`float`)."""


def injection_logm(
    m_best: float,
    m_best_bracket_logm,
    ceiling: float = CENSORED_LOGM,
):
    """Select the injection rung of one ladder artifact.

    Parameters
    ----------
    m_best : `float`
        The artifact's ``m_best`` estimand, NaN when right-censored.
    m_best_bracket_logm : `numpy.ndarray`
        The artifact's two bracketing rungs, NaN when right-censored.
    ceiling : `float`, optional
        Ladder ceiling rung used for censored members.

    Returns
    -------
    logm : `float`
        Injection rung in log10 solar masses.
    censored : `bool`
        Whether the member is right-censored.

    Raises
    ------
    ValueError
        Raised when ``m_best`` is finite but its bracket is not.
    """
    if math.isnan(float(m_best)):
        return float(ceiling), True
    bracket_top = float(np.asarray(m_best_bracket_logm, dtype=float)[1])
    if math.isnan(bracket_top):
        raise ValueError(
            "m_best is finite but m_best_bracket_logm[1] is NaN; the "
            "artifact is inconsistent"
        )
    return bracket_top, False


def argmax_inside_aperture(
    y_coords,
    x_coords,
    q_2d,
    centre_arcsec,
    radius_arcsec: float,
):
    """Locate the grid-map maximum inside the D-F7 aperture.

    Parameters
    ----------
    y_coords, x_coords : `numpy.ndarray`
        Grid node coordinates of the map in arcseconds.
    q_2d : `numpy.ndarray`
        Per-node ``q_F``.
    centre_arcsec : `tuple`
        Aperture centre as ``(y, x)``.
    radius_arcsec : `float`
        Aperture radius.

    Returns
    -------
    position_yx_arcsec : `tuple` [`float`, `float`]
        Coordinates of the peak node.
    q_max : `float`
        ``q_F`` at the peak node.

    Raises
    ------
    ValueError
        Raised when no node lies inside the aperture or the aperture
        holds a non-finite ``q_F``.
    """
    y_values = np.asarray(y_coords, dtype=float)
    x_values = np.asarray(x_coords, dtype=float)
    q_values = np.asarray(q_2d, dtype=float)
    offsets_y = y_values[:, None] - float(centre_arcsec[0])
    offsets_x = x_values[None, :] - float(centre_arcsec[1])
    inside = offsets_y**2 + offsets_x**2 <= float(radius_arcsec)**2
    if not inside.any():
        raise ValueError(
            f"The grid map holds no node inside the D-F7 aperture of radius "
            f"{radius_arcsec} arcsec about {tuple(centre_arcsec)}"
        )
    if not np.all(np.isfinite(q_values[inside])):
        raise ValueError(
            "The grid map leaves non-finite q_F inside the D-F7 aperture"
        )
    masked = np.where(inside, q_values, -np.inf)
    flat_index = int(np.argmax(masked))
    iy, ix = np.unravel_index(flat_index, masked.shape)
    return (
        (float(y_values[iy]), float(x_values[ix])),
        float(q_values[iy, ix]),
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Restamped staged ladder configuration")
    parser.add_argument("artifact", help="The member's ladder_result.npz")
    parser.add_argument("output_dir", help="Directory for the position artifact")
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Replace an existing {ARTIFACT_NAME}",
    )
    return parser


def main(argv=None) -> None:
    """Recompute one member's injection rung and write its position."""
    args = _build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    artifact_path = output_dir/ARTIFACT_NAME
    if artifact_path.exists() and not args.force:
        raise ValueError(
            f"Refusing to overwrite {artifact_path}; pass --force to replace it"
        )

    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    ladder_artifact = np.load(args.artifact, allow_pickle=False)

    system_id = str(ladder_artifact["system_id"])
    if str(config["run_name"]) != system_id:
        raise ValueError(
            f"Configuration run_name {config['run_name']!r} does not match "
            f"artifact system_id {system_id!r}"
        )

    logm, censored = injection_logm(
        float(ladder_artifact["m_best"]),
        ladder_artifact["m_best_bracket_logm"],
    )
    rung_logm = np.asarray(ladder_artifact["rung_logm"], dtype=float)
    rung_q_max = np.asarray(ladder_artifact["rung_q_max"], dtype=float)
    matches = np.isclose(rung_logm, logm, rtol=0.0, atol=1.0e-9)
    if int(np.count_nonzero(matches)) != 1:
        raise ValueError(
            f"Injection rung {logm} matches "
            f"{int(np.count_nonzero(matches))} artifact rungs, expected 1"
        )
    recorded_q_max = float(rung_q_max[matches][0])

    import run_ladder

    ladder = run_ladder._verify_ladder_block(config)
    run_ladder._verify_psf_state(config)
    run_ladder._enable_float64()
    run_ladder._enable_jax_compilation_cache()

    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.provenance import config_hash
    from hwoslaps.psf import generate_psf_system
    from run_stage0_observation import (
        _extract_theta_e_eff,
        _verify_code_revision,
        _verify_source_asset,
    )

    revision = _verify_code_revision(config)
    asset_sha256 = _verify_source_asset(config)
    extraction = _extract_theta_e_eff(config)
    aperture = run_ladder._verify_aperture(config, extraction)

    rung_config = run_ladder._rung_config(config, ladder, aperture)
    validate_or_raise(rung_config)

    psf_data = generate_psf_system(
        rung_config["psf"], full_config=rung_config
    )
    run_ladder._verify_psf_rms(psf_data)
    detector = run_ladder._build_detector(rung_config, psf_data)

    start = perf_counter()
    run_ladder._point_detector_at_rung(detector, logm)
    grid_map = detector.compute_grid_map()
    position_yx, q_max = argmax_inside_aperture(
        grid_map.y_coords,
        grid_map.x_coords,
        grid_map.q_asimov_2d,
        extraction.aperture.centre_arcsec,
        extraction.aperture.radius_arcsec,
    )
    wall_seconds = perf_counter() - start

    relative = abs(q_max - recorded_q_max)/abs(recorded_q_max)
    if relative > Q_MAX_RELATIVE_TOLERANCE:
        raise ValueError(
            f"Recomputed aperture q_max {q_max!r} disagrees with the "
            f"artifact's recorded rung q_max {recorded_q_max!r} by a "
            f"relative {relative:.3e}, beyond {Q_MAX_RELATIVE_TOLERANCE}; "
            "the recomputation does not reproduce the campaign"
        )

    payload = {
        "schema_version": 1,
        "artifact": ARTIFACT_NAME,
        "system_id": system_id,
        "tier": str(ladder_artifact["tier"]),
        "injection_logm": logm,
        "injection_mass_msun": float(10.0**logm),
        "censored": censored,
        "position_yx_arcsec": [position_yx[0], position_yx[1]],
        "q_at_position": q_max,
        "recorded_rung_q_max": recorded_q_max,
        "q_max_relative_difference": relative,
        "m_best": float(ladder_artifact["m_best"]),
        "m_best_bracket_logm": [
            float(value) for value in ladder_artifact["m_best_bracket_logm"]
        ],
        "stop_reason": str(ladder_artifact["stop_reason"]),
        "ladder_campaign_uuid": str(ladder_artifact["campaign_uuid"]),
        "ladder_config_hash": str(ladder_artifact["config_hash"]),
        "config_hash": config_hash(config),
        "source_asset_sha256": asset_sha256,
        "aperture_centre_arcsec": [
            float(extraction.aperture.centre_arcsec[0]),
            float(extraction.aperture.centre_arcsec[1]),
        ],
        "aperture_radius_arcsec": float(extraction.aperture.radius_arcsec),
        "aperture_sha256": str(ladder_artifact["aperture_sha256"]),
        "node_spacing_arcsec": float(ladder["node_spacing_arcsec"]),
        "code_revision": revision,
        "wall_seconds": wall_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Injection position artifact: {artifact_path}\n"
        f"  {system_id}: logm {logm:.2f}"
        f"{' (censored ceiling)' if censored else ''}, position "
        f"({position_yx[0]:+.4f}, {position_yx[1]:+.4f}) arcsec, q_max "
        f"{q_max:.4f} (recorded {recorded_q_max:.4f}, "
        f"relative {relative:.2e}), {wall_seconds:.0f} s"
    )


if __name__ == "__main__":
    main()
