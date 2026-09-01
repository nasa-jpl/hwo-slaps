#!/usr/bin/env python
"""Extract one ladder member's nonlinear-validation injection points.

The DesignFreeze v3 ``nonlinear_validation`` protocol fits each system's
trial subhalo at up to two rungs of its measured ladder: the bracket-top
rung (the first refined rung at which the production ``q_F`` reached the
declared threshold; censored members take the 9.5 ceiling) and, for
non-censored members, the bracket-bottom rung below threshold. The
ladder artifact stores per-rung aggregates only, so this runner
recomputes each used rung's Fisher grid map and records the trial
positions.

Two maps are computed per rung, per the protocol's injection rule:

- The production-kernel (999 x 999) map, whose aperture maximum must
  reproduce the artifact's recorded rung ``q_max`` to a relative 1.0e-6
  (fails closed otherwise), making every extraction a determinism canary
  of the production campaign.
- A support-matched map at the nonlinear arm's own kernel (51 x 51),
  restricted to grid nodes inside both the D-F7 aperture and the
  nonlinear dataset's PSF-border-valid region. Its argmax is the trial
  position, its value there is the comparison statistic ``q_F(k51)``,
  and the production-kernel value at the same node is recorded for
  linkage.

The artifact is ``injection_position.json`` under ``--output-dir``.
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

FIT_KERNEL_SHAPE = [51, 51]
"""Nonlinear-arm kernel the support-matched map renders at (`list`)."""


def injection_logm(
    m_best: float,
    m_best_bracket_logm,
    ceiling: float = CENSORED_LOGM,
):
    """Select the bracket-top injection rung of one ladder artifact.

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


def below_logm(m_best_bracket_logm) -> float:
    """Select the bracket-bottom rung of a non-censored artifact.

    Parameters
    ----------
    m_best_bracket_logm : `numpy.ndarray`
        The artifact's two bracketing rungs.

    Returns
    -------
    logm : `float`
        The bracket-bottom rung in log10 solar masses.

    Raises
    ------
    ValueError
        Raised when the bracket bottom is NaN.
    """
    bottom = float(np.asarray(m_best_bracket_logm, dtype=float)[0])
    if math.isnan(bottom):
        raise ValueError("m_best_bracket_logm[0] is NaN")
    return bottom


def support_half_widths(
    image_shape,
    pixel_scale: float,
    kernel_shape,
):
    """Half-widths of the nonlinear dataset's PSF-border-valid region.

    The nonlinear dataset excludes pixels within half a kernel of the
    image edge, so a grid node is support-valid only inside the box
    these half-widths describe about the image centre. One extra pixel
    is removed on every side so node-to-pixel rounding can never admit
    an excluded pixel.

    Parameters
    ----------
    image_shape : `tuple` [`int`, `int`]
        Rendered image shape in pixels.
    pixel_scale : `float`
        Image pixel scale in arcseconds.
    kernel_shape : `tuple` [`int`, `int`]
        Native fit-kernel shape.

    Returns
    -------
    half_widths : `tuple` [`float`, `float`]
        Valid ``(y, x)`` half-widths in arcseconds.

    Raises
    ------
    ValueError
        Raised when the border removes the whole image.
    """
    halves = []
    for axis in (0, 1):
        pixels = int(image_shape[axis])//2 - int(kernel_shape[axis])//2 - 1
        if pixels <= 0:
            raise ValueError(
                f"The PSF border of kernel {tuple(kernel_shape)} leaves no "
                f"valid pixels on an image of shape {tuple(image_shape)}"
            )
        halves.append(pixels*float(pixel_scale))
    return tuple(halves)


def argmax_inside_aperture(
    y_coords,
    x_coords,
    q_2d,
    centre_arcsec,
    radius_arcsec: float,
    support_half_widths_arcsec=None,
):
    """Locate the grid-map maximum inside the aperture and valid support.

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
    support_half_widths_arcsec : `tuple`, optional
        Valid ``(y, x)`` half-widths about the image centre; when given,
        nodes outside that box are excluded and the aperture's support
        coverage is reported.

    Returns
    -------
    position_yx_arcsec : `tuple` [`float`, `float`]
        Coordinates of the peak node.
    q_max : `float`
        ``q_F`` at the peak node.
    indices : `tuple` [`int`, `int`]
        Row and column of the peak node.
    aperture_support_fraction : `float`
        Fraction of aperture nodes inside the valid support (1.0 when
        no support box is given).

    Raises
    ------
    ValueError
        Raised when no node lies inside the selection region or the
        aperture holds a non-finite ``q_F``.
    """
    y_values = np.asarray(y_coords, dtype=float)
    x_values = np.asarray(x_coords, dtype=float)
    q_values = np.asarray(q_2d, dtype=float)
    offsets_y = y_values[:, None] - float(centre_arcsec[0])
    offsets_x = x_values[None, :] - float(centre_arcsec[1])
    aperture = offsets_y**2 + offsets_x**2 <= float(radius_arcsec)**2
    if not aperture.any():
        raise ValueError(
            f"The grid map holds no node inside the D-F7 aperture of radius "
            f"{radius_arcsec} arcsec about {tuple(centre_arcsec)}"
        )
    if not np.all(np.isfinite(q_values[aperture])):
        raise ValueError(
            "The grid map leaves non-finite q_F inside the D-F7 aperture"
        )
    selection = aperture
    support_fraction = 1.0
    if support_half_widths_arcsec is not None:
        y_half, x_half = support_half_widths_arcsec
        supported = (
            (np.abs(y_values)[:, None] <= float(y_half))
            & (np.abs(x_values)[None, :] <= float(x_half))
        )
        selection = aperture & supported
        support_fraction = float(
            np.count_nonzero(selection)/np.count_nonzero(aperture)
        )
        if not selection.any():
            raise ValueError(
                "No aperture node lies inside the PSF-border-valid support"
            )
    masked = np.where(selection, q_values, -np.inf)
    flat_index = int(np.argmax(masked))
    iy, ix = np.unravel_index(flat_index, masked.shape)
    return (
        (float(y_values[iy]), float(x_values[ix])),
        float(q_values[iy, ix]),
        (int(iy), int(ix)),
        support_fraction,
    )


def _recorded_rung_q_max(ladder_artifact, logm: float) -> float:
    """Return the artifact's recorded aperture ``q_max`` at one rung."""
    rung_logm = np.asarray(ladder_artifact["rung_logm"], dtype=float)
    rung_q_max = np.asarray(ladder_artifact["rung_q_max"], dtype=float)
    matches = np.isclose(rung_logm, logm, rtol=0.0, atol=1.0e-9)
    if int(np.count_nonzero(matches)) != 1:
        raise ValueError(
            f"Rung {logm} matches {int(np.count_nonzero(matches))} artifact "
            "rungs, expected 1"
        )
    return float(rung_q_max[matches][0])


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
    """Recompute one member's injection rungs and write its positions."""
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

    top_logm, censored = injection_logm(
        float(ladder_artifact["m_best"]),
        ladder_artifact["m_best_bracket_logm"],
    )
    rungs = {"top": top_logm}
    if not censored:
        rungs["below"] = below_logm(ladder_artifact["m_best_bracket_logm"])

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

    production_config = run_ladder._rung_config(config, ladder, aperture)
    validate_or_raise(production_config)
    from copy import deepcopy

    matched_config = deepcopy(production_config)
    matched_config["psf"]["kernel"]["shape_native"] = list(FIT_KERNEL_SHAPE)
    validate_or_raise(matched_config)

    half_widths = support_half_widths(
        config["lensing"]["grid"]["shape"],
        float(config["lensing"]["grid"]["pixel_scale"]),
        FIT_KERNEL_SHAPE,
    )

    detectors = {}
    for label, rung_config in (
        ("production", production_config),
        ("matched", matched_config),
    ):
        psf_data = generate_psf_system(
            rung_config["psf"], full_config=rung_config
        )
        run_ladder._verify_psf_rms(psf_data)
        detectors[label] = run_ladder._build_detector(rung_config, psf_data)

    centre = extraction.aperture.centre_arcsec
    radius = extraction.aperture.radius_arcsec
    start = perf_counter()
    rung_payloads = {}
    for rung_name, logm in rungs.items():
        maps = {}
        for label, detector in detectors.items():
            run_ladder._point_detector_at_rung(detector, logm)
            maps[label] = detector.compute_grid_map()
        production_map = maps["production"]
        matched_map = maps["matched"]
        if not (
            np.array_equal(production_map.y_coords, matched_map.y_coords)
            and np.array_equal(production_map.x_coords, matched_map.x_coords)
        ):
            raise ValueError(
                "Production and support-matched grid maps disagree on node "
                "coordinates"
            )

        _, production_aperture_max, _, _ = argmax_inside_aperture(
            production_map.y_coords,
            production_map.x_coords,
            production_map.q_asimov_2d,
            centre,
            radius,
        )
        recorded = _recorded_rung_q_max(ladder_artifact, logm)
        relative = abs(production_aperture_max - recorded)/abs(recorded)
        if relative > Q_MAX_RELATIVE_TOLERANCE:
            raise ValueError(
                f"Recomputed production aperture q_max "
                f"{production_aperture_max!r} at rung {logm} disagrees with "
                f"the artifact's recorded {recorded!r} by a relative "
                f"{relative:.3e}, beyond {Q_MAX_RELATIVE_TOLERANCE}; the "
                "recomputation does not reproduce the campaign"
            )

        position, q_matched, indices, support_fraction = (
            argmax_inside_aperture(
                matched_map.y_coords,
                matched_map.x_coords,
                matched_map.q_asimov_2d,
                centre,
                radius,
                support_half_widths_arcsec=half_widths,
            )
        )
        q_production_at_position = float(
            np.asarray(production_map.q_asimov_2d, dtype=float)[indices]
        )
        rung_payloads[rung_name] = {
            "logm": float(logm),
            "mass_msun": float(10.0**float(logm)),
            "position_yx_arcsec": [position[0], position[1]],
            "q_f_matched": q_matched,
            "q_f_production_at_position": q_production_at_position,
            "recomputed_production_aperture_q_max": production_aperture_max,
            "recorded_rung_q_max": recorded,
            "q_max_relative_difference": relative,
            "aperture_support_fraction": support_fraction,
        }
    wall_seconds = perf_counter() - start

    payload = {
        "schema_version": 2,
        "artifact": ARTIFACT_NAME,
        "system_id": system_id,
        "tier": str(ladder_artifact["tier"]),
        "censored": censored,
        "rungs": rung_payloads,
        "m_best": float(ladder_artifact["m_best"]),
        "m_best_bracket_logm": [
            float(value) for value in ladder_artifact["m_best_bracket_logm"]
        ],
        "stop_reason": str(ladder_artifact["stop_reason"]),
        "fit_kernel_shape_native": list(FIT_KERNEL_SHAPE),
        "support_half_widths_arcsec": [half_widths[0], half_widths[1]],
        "ladder_campaign_uuid": str(ladder_artifact["campaign_uuid"]),
        "ladder_config_hash": str(ladder_artifact["config_hash"]),
        "config_hash": config_hash(config),
        "source_asset_sha256": asset_sha256,
        "aperture_centre_arcsec": [float(centre[0]), float(centre[1])],
        "aperture_radius_arcsec": float(radius),
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
    summary = ", ".join(
        f"{name} logm {block['logm']:.2f} q51 {block['q_f_matched']:.3f} "
        f"(k999 {block['q_f_production_at_position']:.3f}, canary rel "
        f"{block['q_max_relative_difference']:.1e})"
        for name, block in rung_payloads.items()
    )
    print(
        f"Injection position artifact: {artifact_path}\n"
        f"  {system_id}{' (censored ceiling)' if censored else ''}: "
        f"{summary}, {wall_seconds:.0f} s"
    )


if __name__ == "__main__":
    main()
