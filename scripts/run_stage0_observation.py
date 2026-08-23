#!/usr/bin/env python
"""Render one Stage 0 no-subhalo observation and its selection observables.

The S1-lite runner for the Stage 0 pool. It takes one staged campaign
configuration, renders the no-subhalo observation with the production
constructors, extracts ``theta_E_eff`` and the D-F7 aperture with the
frozen algorithm, and computes the pre-registered selection statistics on
the noiseless PSF-convolved lensed-source electrons inside that aperture.

The job artifact is ``stage0_observation.npz`` under the job output
directory. It carries the mandatory ``campaign_uuid`` and ``config_hash``
identity members the campaign layer validates, the selection
observables, the aperture census and the theta_E provenance hashes. The
electron maps themselves are not stored: Stage 0 exists to produce
selection statistics and every later stage re-renders from the staged
configuration.

The realized unlensed detected rate is measured on this system's own
grid and size scale and stored beside its target, so the P0-3 discrete
detected-rate contract is checked at production geometry rather than
assumed from the canonical solve.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))

ARTIFACT_NAME = "stage0_observation.npz"


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Staged Stage 0 campaign configuration")
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Replace an existing {ARTIFACT_NAME}",
    )
    return parser


def _output_dir(config: dict) -> Path:
    """Return the job output directory of one staged configuration."""
    root = Path(config["plotting"]["output_dir"]).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT/root
    return root/str(config["run_name"])


def _contract_grid_side(config: dict) -> int:
    """Return the grid side that contains this system's scaled stamp.

    The rate contract measures the normalization, so it must not be
    confounded by grid truncation. The production grid is theta_E driven
    and a large sampled source can be wider than it, so the contract is
    evaluated on a grid sized to hold the whole scaled stamp about the
    sampled source centre, at the production pixel scale.
    """
    from hwoslaps.lensing import load_source_image_asset

    light = config["lensing"]["source_galaxy"]["light"]
    asset = load_source_image_asset(light["asset_path"])
    pixel_scale = float(config["lensing"]["grid"]["pixel_scale"])
    stamp_arcsec = (
        max(asset.sb.shape)*float(asset.pixel_scale_arcsec)
        * float(light["size_scale"])
    )
    offset_arcsec = max(abs(float(value)) for value in light["centre"])
    side = int(np.ceil((stamp_arcsec + 2.0*offset_arcsec)/pixel_scale)) + 2
    return side + side % 2


def _unlensed_rates_e_per_s(config: dict, lensing_data) -> tuple:
    """Return the unlensed source rate on the contract and production grids.

    The rate contract convention: the discrete sum of the unlensed source
    surface brightness rendered through the production constructors is
    the detected rate in electrons per second. The first return value is
    that sum on the containing contract grid, which is what the contract
    is gated on; the second is the same sum on this system's production
    grid, recorded as a truncation diagnostic.
    """
    from hwoslaps.lensing.generator import _create_grid, _create_source_galaxy

    source = _create_source_galaxy(config["lensing"]["source_galaxy"])
    side = _contract_grid_side(config)
    contract_grid = _create_grid({
        "shape": [side, side],
        "pixel_scale": float(config["lensing"]["grid"]["pixel_scale"]),
    })
    contract_rate = float(
        np.sum(np.asarray(source.image_2d_from(grid=contract_grid)))
    )
    production_rate = float(
        np.sum(np.asarray(source.image_2d_from(grid=lensing_data.grid)))
    )
    return contract_rate, production_rate, side


CONVOLUTION_ROUNDOFF_TOLERANCE = 1.0e-9
"""Largest negative PSF-convolution excursion accepted, as a fraction of
the map peak (`float`)."""


def _clip_convolution_roundoff(source_electrons):
    """Clip the PSF convolution's negative round-off to zero.

    A noiseless PSF-convolved source map is non-negative in exact
    arithmetic, but the Fourier convolution leaves excursions at the
    floating-point round-off level. The pre-registered expected-variance
    map ``sigma^2 = s + B`` requires a non-negative map, and the noisy
    branch of the same pre-registration already uses ``max(s, 0) + B``,
    so the round-off is clipped here on the same convention.

    Anything larger than `CONVOLUTION_ROUNDOFF_TOLERANCE` times the map
    peak is not round-off and fails closed rather than being clipped
    away.
    """
    values = np.asarray(source_electrons, dtype=float)
    minimum = float(np.min(values))
    peak = float(np.max(values))
    if minimum < 0.0 and abs(minimum) > CONVOLUTION_ROUNDOFF_TOLERANCE*max(peak, 1.0):
        raise ValueError(
            f"The noiseless source map reaches {minimum} electrons against a "
            f"peak of {peak}, far beyond the declared convolution round-off "
            f"tolerance {CONVOLUTION_ROUNDOFF_TOLERANCE}; this is not round-off"
        )
    return np.clip(values, 0.0, None), minimum


def _selection_observables(config: dict, observation, extraction) -> dict:
    """Compute the pre-registered selection statistics of one system."""
    from hwoslaps.analysis import selection_score as score

    detector = config["observation"]["detector"]
    exposure = float(config["observation"]["exposure_time"])
    blank_variance = score.blank_variance_e2(
        float(detector["sky_background"]),
        float(detector["dark_current"]),
        float(detector["read_noise"]),
        exposure,
    )
    source_e, raw_minimum = _clip_convolution_roundoff(
        observation.source_electrons
    )
    variance = score.expected_variance_e2(source_e, blank_variance)

    grid = np.asarray(observation.imaging.data.mask.derive_grid.all_false.native)
    mask = score.aperture_mask(
        grid[..., 0],
        grid[..., 1],
        extraction.aperture.radius_arcsec,
        centre_arcsec=tuple(extraction.aperture.centre_arcsec),
    )
    pixel_scale = float(config["lensing"]["grid"]["pixel_scale"])
    arc_snr = score.arc_snr(source_e, variance, mask=mask)
    gradient_power = score.gradient_power(
        source_e, variance, pixel_scale, mask=mask
    )
    theta_res = score.diffraction_scale_arcsec(
        float(config["psf"]["hres_psf"]["wavelength"]),
        float(config["psf"]["telescope"]["pupil_diameter"]),
    )
    return {
        "blank_variance_e2": blank_variance,
        "arc_snr": arc_snr,
        "gradient_power_arcsec_m2": gradient_power,
        "theta_res_arcsec": theta_res,
        "complexity": score.complexity(gradient_power, arc_snr, theta_res),
        "aperture_pixels": int(np.count_nonzero(mask)),
        "grid_pixels": int(mask.size),
        "source_electrons_aperture": float(np.sum(source_e[mask])),
        "source_electrons_total": float(np.sum(source_e)),
        "source_electrons_min_before_clip": raw_minimum,
    }


def main(argv=None) -> None:
    """Render one Stage 0 observation and write its selection artifact."""
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    output_dir = _output_dir(config)
    artifact_path = output_dir/ARTIFACT_NAME
    if artifact_path.exists() and not args.force:
        raise ValueError(
            f"Refusing to overwrite {artifact_path}; pass --force to replace it"
        )

    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.lensing import critical_curve as cc
    from hwoslaps.observation import generate_observation
    from hwoslaps.provenance import config_hash, write_provenance
    from hwoslaps.psf import generate_psf_system

    validate_or_raise(config)
    if config["lensing"]["subhalo"].get("enabled", False):
        raise ValueError(
            "Stage 0 renders no-subhalo observations only; "
            f"{config_path} enables a subhalo"
        )

    lensing_data = generate_lensing_system(config["lensing"], config)
    psf_data = generate_psf_system(config["psf"], config)
    observation = generate_observation(
        lensing_data, psf_data, config["observation"], config
    )
    extraction = cc.extract_theta_e_from_lens_config(
        config["lensing"]["lens_galaxy"]
    )
    observables = _selection_observables(config, observation, extraction)

    exposure = float(config["observation"]["exposure_time"])
    unlensed_rate, production_rate, contract_side = _unlensed_rates_e_per_s(
        config, lensing_data
    )
    stage0 = config["stage0"]
    target_rate = float(stage0["target_unlensed_rate_e_per_s"])
    rate_ratio = unlensed_rate/target_rate
    tolerance = float(stage0["rate_contract_tolerance"])
    if abs(rate_ratio - 1.0) > tolerance:
        raise ValueError(
            f"System {stage0['system_id']} realizes an unlensed detected rate "
            f"of {unlensed_rate} e-/s against the design target {target_rate} "
            f"e-/s, a ratio of {rate_ratio} outside the declared "
            f"rate-contract tolerance {tolerance}"
        )
    lensed_rate = observables["source_electrons_total"]/exposure

    output_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(output_dir/"provenance.yaml", config=config, command=sys.argv)

    payload = {
        "campaign_uuid": np.asarray(os.environ.get("HWOSLAPS_CAMPAIGN_UUID", "")),
        "config_hash": np.asarray(config_hash(config)),
        "system_id": np.asarray(str(config["run_name"])),
        "theta_e_eff_arcsec": np.asarray(extraction.theta_e_eff_arcsec),
        "aperture_radius_arcsec": np.asarray(extraction.aperture.radius_arcsec),
        "contour_sha256": np.asarray(extraction.contour_sha256),
        "aperture_sha256": np.asarray(extraction.aperture.sha256),
        "theta_e_provenance_json": np.asarray(
            json.dumps(extraction.to_provenance_dict(), sort_keys=True)
        ),
        "exposure_time_s": np.asarray(exposure),
        "unlensed_rate_e_per_s": np.asarray(unlensed_rate),
        "unlensed_rate_target_e_per_s": np.asarray(target_rate),
        "unlensed_rate_ratio": np.asarray(rate_ratio),
        "unlensed_rate_contract_grid_side": np.asarray(contract_side),
        "unlensed_rate_on_production_grid_e_per_s": np.asarray(production_rate),
        "production_grid_flux_fraction": np.asarray(
            production_rate/unlensed_rate
        ),
        "lensed_rate_e_per_s": np.asarray(lensed_rate),
        "magnification_realized": np.asarray(lensed_rate/unlensed_rate),
    }
    for name, value in observables.items():
        payload[name] = np.asarray(value)

    tmp_path = artifact_path.with_name(artifact_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, artifact_path)
    print(f"Stage 0 artifact: {artifact_path}")
    print(
        f"  theta_E_eff {extraction.theta_e_eff_arcsec:.6f} arcsec, "
        f"S {observables['arc_snr']:.3f}, C {observables['complexity']:.6e}"
    )


if __name__ == "__main__":
    main()
