#!/usr/bin/env python
"""Walk one member's complete adaptive Fisher mass ladder.

The S1-lite runner for the layer 3 and layer 4 ladder campaigns. One job
is one member's whole ladder: the Fisher detector is built once from the
member's staged configuration and every rung reuses it, so the detector
setup is amortized across the walk and the adaptivity lives inside the
job while the manifest stays static.

Every fail-closed verification of the Stage 0 runner is inherited by
importing it, so the template asset bytes, the source revision and the
frozen ``theta_E`` extraction are held to the same declarations by the
same code. On top of those the ladder block's own declarations are
verified: the PSF state, the kernel, the engine, the mask policy, the
node spacing, the detection threshold and the D-F7 aperture the ladder
is bound to.

The walk itself is `hwoslaps.campaign.ladder_walk`, which imports no
engine: this module renders the rungs that module asks for and reduces
each grid map to the four numbers the ladder records.

The job artifact is ``ladder_result.npz`` under the job output
directory. It carries the mandatory ``campaign_uuid`` and ``config_hash``
identity members, the verified asset digest and source revision, the
detector and PSF provenance, the per-rung table with its wall seconds,
the estimands with their bracketing rungs, the walk's stop reason and
the clipping flags. No electron map and no q map is stored: every later
figure re-renders from the staged configuration.

The job consumes no random stream. Fisher ladders are deterministic, the
manifest declares that they draw nothing, and this runner constructs no
generator. The seed of the science35 PSF state is part of that state's
frozen definition rather than a campaign stream, and reaches the render
as the frozen aberration coefficients the staged configuration carries.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
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

from hwoslaps.campaign.ladder_walk import (  # noqa: E402
    M10_LEVEL,
    M50_LEVEL,
    Q_THRESHOLD,
    THRESHOLD_DECLARATION,
    aperture_fraction_crossing,
    log_linear_crossing,
    next_rung,
    policy_from_mass_ladder,
)
from run_stage0_observation import (  # noqa: E402
    _extract_theta_e_eff,
    _verify_code_revision,
    _verify_source_asset,
)

ARTIFACT_NAME = "ladder_result.npz"

PSF_STATE = "science35"
"""Truth PSF state label the ladder renders at (`str`)."""

PSF_STATE_SEED = 20260835
"""Seed of the science35 state's frozen definition (`int`)."""

PSF_STATE_RMS_NM = 35.0
"""Measured piston-removed aperture RMS of the science35 state (`float`)."""

PSF_STATE_RMS_TOLERANCE_NM = 1.0e-6
"""Largest accepted departure from `PSF_STATE_RMS_NM` (`float`).

The state is realized to its measured RMS exactly and its coefficients
travel as decimal text, so anything beyond serialization round-off means
the staged aberrations are a different state.
"""

KERNEL_LABEL = "k999"
"""Detector kernel the ladder renders at (`str`)."""

KERNEL_SHAPE_NATIVE = [999, 999]
"""Native kernel shape `KERNEL_LABEL` denotes (`list`)."""

ENGINE = "jax"
"""Fisher grid template engine the ladder runs on (`str`)."""

MASK_MODE = "all_pixels"
"""Fisher pixel mask policy the ladder runs under (`str`)."""

NODE_SPACING_ARCSEC = 0.05
"""Fisher grid node spacing the A2 ruling pins (`float`)."""

APERTURE_THETA_E_FACTOR = 2.0
"""D-F7 aperture radius in units of ``theta_E_eff`` (`float`)."""

TIERS = ("parent", "selected")
"""Tiers a ladder job may belong to (`tuple`)."""

JAX_CACHE_DIR_ENV = "HWOSLAPS_JAX_CACHE_DIR"
"""Environment variable holding an optional JAX compilation cache (`str`)."""


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Staged ladder campaign configuration")
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


def _enable_float64() -> None:
    """Establish JAX 64-bit mode before anything imports AutoLens.

    The grid template engine runs in float64 and the flag is only
    honoured while no array has been made, so it is set here rather than
    inside the first map. The check mirrors
    `hwoslaps.modeling.nonlinear.autolens_runner.ensure_jax_x64`; it is
    written out rather than imported because that module lives inside the
    nonlinear package, which this layer does not depend on.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError(
            "The ladder renders in float64, but JAX 64-bit mode could not be "
            "enabled"
        )


def _enable_jax_compilation_cache() -> None:
    """Point JAX at a persistent compilation cache when one is asked for.

    The grid engine compiles one batch executable per distinct batch
    shape, which is cold work every job repeats from scratch. Setting
    ``HWOSLAPS_JAX_CACHE_DIR`` lets jobs on one machine share those
    executables; leaving it unset keeps JAX's default of no on-disk cache,
    so a production run is unaffected by this hook.
    """
    cache_dir = os.environ.get(JAX_CACHE_DIR_ENV, "").strip()
    if not cache_dir:
        return

    import jax

    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Every executable the ladder compiles is worth keeping; JAX otherwise
    # caches only those that took longer than a wall-clock threshold, which
    # would make the cache's contents depend on machine load.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


def _verify_ladder_block(config: dict) -> dict:
    """Verify the staged ladder declarations against this checkout.

    The ladder block travels inside the staged configuration and is
    covered by its hash, so a job whose declared PSF state, kernel,
    engine, mask policy, node spacing or detection threshold has moved
    away from what this runner implements fails before anything is
    rendered rather than producing a plausible artifact under the wrong
    conditions.

    Parameters
    ----------
    config : `dict`
        Staged ladder campaign configuration.

    Returns
    -------
    ladder : `dict`
        The verified ``ladder`` block.
    """
    ladder = config["ladder"]
    for name, declared, implemented in (
        ("psf_state", str(ladder["psf_state"]), PSF_STATE),
        ("kernel", str(ladder["kernel"]), KERNEL_LABEL),
        ("engine", str(ladder["engine"]), ENGINE),
        ("mask_mode", str(ladder["mask_mode"]), MASK_MODE),
        ("threshold", str(ladder["threshold"]).strip(), THRESHOLD_DECLARATION),
    ):
        if declared != implemented:
            raise ValueError(
                f"The campaign declares ladder {name} {declared!r} but this "
                f"checkout implements {implemented!r}"
            )
    spacing = float(ladder["node_spacing_arcsec"])
    if spacing != NODE_SPACING_ARCSEC:
        raise ValueError(
            f"The campaign declares a ladder node spacing of {spacing} arcsec "
            f"but this checkout implements {NODE_SPACING_ARCSEC}; the A2 "
            "ruling pins the spacing that carries the declared "
            "spatial_sampling_qmax systematic"
        )
    tier = str(ladder["tier"])
    if tier not in TIERS:
        raise ValueError(
            f"The campaign declares ladder tier {tier!r}, which is not one of "
            f"{list(TIERS)}"
        )
    for name in ("golden", "parent_overlap"):
        if not isinstance(ladder.get(name), bool):
            raise ValueError(
                f"The campaign declares ladder {name} "
                f"{ladder.get(name)!r}; the ladder block carries it as a "
                "boolean on every job"
            )
    fit_psf = config["modeling"].get("fit_psf")
    if fit_psf is not None:
        mode = str(fit_psf.get("mode", "matched")).lower()
        if mode != "matched":
            raise ValueError(
                f"The staged configuration fits PSF mode {mode!r}, but the "
                "ladder runs a matched fit PSF: the fit PSF is the truth PSF"
            )
    return ladder


def _verify_psf_state(config: dict) -> None:
    """Fail closed unless the staged aberrations are the science35 state.

    The science35 truth state is a frozen coefficient realization, not a
    label the runner can expand: it is the combined global and segment
    content drawn from the ``jwst_wss_static_v1`` prior at
    `PSF_STATE_RMS_NM` under seed `PSF_STATE_SEED`, realized in the
    aperture basis and renormalized exactly. The campaign writer stages
    those coefficients into ``psf.aberrations`` so they are covered by
    the configuration hash; this checks the staged content declares the
    state's shape, and `_verify_psf_rms` checks it realizes its RMS.

    Parameters
    ----------
    config : `dict`
        Staged ladder campaign configuration.
    """
    aberrations = config["psf"]["aberrations"]
    for name, expected in (
        ("enable_segment_pistons", False),
        ("enable_segment_tiptilts", False),
        ("enable_segment_hexikes", True),
        ("enable_global_zernikes", True),
    ):
        if bool(aberrations.get(name, False)) is not expected:
            raise ValueError(
                f"The campaign declares psf_state {PSF_STATE} but the staged "
                f"psf.aberrations set {name} to {aberrations.get(name)!r} "
                f"rather than {expected}; {PSF_STATE} is the combined "
                "global and segment state drawn from the jwst_wss_static_v1 "
                f"prior at {PSF_STATE_RMS_NM} nm under seed {PSF_STATE_SEED} "
                "and its coefficients must be staged into the configuration"
            )
    for name in ("segment_hexikes", "global_zernikes"):
        if not aberrations.get(name):
            raise ValueError(
                f"The campaign declares psf_state {PSF_STATE} but the staged "
                f"psf.aberrations carry no {name} coefficients"
            )


def _verify_psf_rms(psf_data) -> float:
    """Verify the generated PSF realizes the science35 measured RMS.

    Parameters
    ----------
    psf_data : `hwoslaps.psf.PSFData`
        PSF system generated from the staged configuration.

    Returns
    -------
    measured_rms_nm : `float`
        The verified piston-removed aperture RMS, recorded in the job
        artifact.
    """
    measured = float(psf_data.total_rms_nm)
    if abs(measured - PSF_STATE_RMS_NM) > PSF_STATE_RMS_TOLERANCE_NM:
        raise ValueError(
            f"The staged psf.aberrations realize a piston-removed aperture "
            f"RMS of {measured!r} nm, but psf_state {PSF_STATE} is defined at "
            f"{PSF_STATE_RMS_NM} nm; the staged state is not {PSF_STATE}"
        )
    return measured


def _verify_aperture(config: dict, extraction) -> dict:
    """Verify the ladder is bound to the aperture the selection used.

    The ladder's aperture is the member's realized D-F7 aperture, and the
    Stage 0 extraction this runner re-runs already fails closed against
    the digests the Stage 0 campaign recorded. Comparing the ladder's own
    declaration with that realized extraction is what binds the ladder,
    the selection and the map extent to one aperture.

    Parameters
    ----------
    config : `dict`
        Staged ladder campaign configuration.
    extraction : `hwoslaps.lensing.critical_curve.ThetaEExtraction`
        Realized extraction of this member's ``theta_E_eff``.

    Returns
    -------
    aperture : `dict`
        The verified ``ladder.aperture`` block.
    """
    declared = config["ladder"]["aperture"]
    realized = extraction.aperture
    if float(declared["theta_e_factor"]) != APERTURE_THETA_E_FACTOR:
        raise ValueError(
            f"The campaign declares a ladder aperture factor of "
            f"{declared['theta_e_factor']} but the D-F7 ruling pins "
            f"{APERTURE_THETA_E_FACTOR}"
        )
    for name, realized_value, declared_value in (
        (
            "theta_e_factor",
            realized.theta_e_factor,
            float(declared["theta_e_factor"]),
        ),
        (
            "theta_e_eff_arcsec",
            realized.theta_e_eff_arcsec,
            float(declared["theta_e_eff_arcsec"]),
        ),
        (
            "radius_arcsec",
            realized.radius_arcsec,
            float(declared["radius_arcsec"]),
        ),
        (
            "required_map_half_width_arcsec",
            realized.required_map_half_width_arcsec,
            float(declared["required_map_half_width_arcsec"]),
        ),
    ):
        if float(realized_value) != declared_value:
            raise ValueError(
                f"The realized D-F7 aperture {name} is {float(realized_value)} "
                f"but the ladder campaign declares {declared_value}; the "
                "ladder is not bound to the aperture the selection used"
            )
    for name, realized_digest, declared_digest in (
        (
            "contour",
            extraction.contour_sha256,
            declared["stage0_contour_sha256"],
        ),
        ("aperture", realized.sha256, declared["stage0_aperture_sha256"]),
    ):
        if str(realized_digest) != str(declared_digest):
            raise ValueError(
                f"The realized {name} hashes to {realized_digest} but the "
                f"ladder campaign declares {declared_digest}; the ladder is "
                "not bound to the aperture the selection used"
            )
    return declared


def _rung_config(config: dict, ladder: dict, aperture: dict) -> dict:
    """Build the rendering configuration the ladder's rungs are mapped on.

    The staged configuration is the member's Stage 0 configuration plus
    the ladder block, so it still carries the Stage 0 observation-layer
    settings. The ladder's own declarations are applied here, once, and
    the resulting configuration is what the single reused Fisher detector
    is built from.

    Parameters
    ----------
    config : `dict`
        Staged ladder campaign configuration.
    ladder : `dict`
        Verified ``ladder`` block.
    aperture : `dict`
        Verified ``ladder.aperture`` block.

    Returns
    -------
    rung_config : `dict`
        Configuration for the member's Fisher grid maps.
    """
    rung_config = deepcopy(config)
    rung_config["psf"]["kernel"]["shape_native"] = list(KERNEL_SHAPE_NATIVE)
    rung_config["lensing"]["subhalo"]["enabled"] = True
    modeling = rung_config["modeling"]
    modeling["enabled"] = True
    fisher = modeling["fisher"]
    fisher["mode"] = "map"
    fisher["mask_mode"] = MASK_MODE
    fisher["map"]["type"] = "grid"
    fisher["map"]["engine"] = ENGINE
    fisher["map"]["detection_q_threshold"] = Q_THRESHOLD
    fisher["map"]["grid"] = {
        "spacing_arcsec": float(ladder["node_spacing_arcsec"]),
        "half_width_arcsec": float(aperture["required_map_half_width_arcsec"]),
        "annulus": None,
    }
    return rung_config


def _build_detector(rung_config: dict, psf_data):
    """Build the one Fisher detector every rung of this member reuses.

    The detector's construction cost is baseline work: the mean image,
    the noise whitening, the nuisance derivative images and the profile
    likelihood workspace all describe the no-subhalo scene and none of
    them depends on the injected mass.

    Parameters
    ----------
    rung_config : `dict`
        Configuration the member's grid maps are rendered on.
    psf_data : `hwoslaps.psf.PSFData`
        Truth PSF, which is also the matched fit PSF.

    Returns
    -------
    detector : `hwoslaps.modeling.fisher_detector.FisherDetector`
        Detector bound to this member's baseline scene.
    """
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.modeling.fisher_detector import FisherDetector
    from hwoslaps.observation import generate_observation

    baseline_config = deepcopy(rung_config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    lensing_baseline = generate_lensing_system(
        baseline_config["lensing"], full_config=baseline_config
    )
    observation_baseline = generate_observation(
        lensing_data=lensing_baseline,
        psf_data=psf_data,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )
    return FisherDetector(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=rung_config,
        fisher_config=rung_config["modeling"]["fisher"],
    )


def _point_detector_at_rung(detector, logm: float) -> None:
    """Point the reused detector at the next mass rung.

    Only the template configurations and the grid engine's radial deflection
    table carry the injected mass, so a rung is advanced by rewriting the
    templates and retargeting the engine. Everything the detector and the
    engine spent their construction on is baseline work and survives.

    Parameters
    ----------
    detector : `hwoslaps.modeling.fisher_detector.FisherDetector`
        Detector to advance.
    logm : `float`
        Log-mass of the rung to render next.
    """
    mass = float(10.0**float(logm))
    for template in (
        detector.full_config,
        detector.map_config_template,
        detector.map_config_template_truth,
    ):
        template["lensing"]["subhalo"]["mass"] = mass
    detector.retarget_grid_engine()


def _rung_metrics(
    y_coords,
    x_coords,
    q_asimov_2d,
    detectable_mask_2d,
    spacing_arcsec: float,
    centre_arcsec,
    radius_arcsec: float,
) -> dict:
    """Reduce one Fisher grid map to the four numbers a rung records.

    The estimands are all aperture quantities, so ``q_max`` and the
    detected area are taken inside the closed D-F7 aperture rather than
    over the whole map, which reaches past it by the computational
    margin. The clipping flag is the map-perimeter one: a detected
    region touching the map edge makes the whole-map area a lower bound.

    Parameters
    ----------
    y_coords, x_coords : `numpy.ndarray`
        Grid node coordinates of the map in arcseconds.
    q_asimov_2d : `numpy.ndarray`
        Per-node ``q_F``.
    detectable_mask_2d : `numpy.ndarray`
        Per-node detection mask at the declared threshold.
    spacing_arcsec : `float`
        Node spacing, whose square is the cell area.
    centre_arcsec : `tuple`
        Aperture centre as ``(y, x)``.
    radius_arcsec : `float`
        Aperture radius.

    Returns
    -------
    metrics : `dict`
        ``q_max``, ``detectable_area_arcsec2``, ``aperture_fraction`` and
        ``perimeter_clipped``.
    """
    y_values = np.asarray(y_coords, dtype=float)
    x_values = np.asarray(x_coords, dtype=float)
    q_values = np.asarray(q_asimov_2d, dtype=float)
    detectable = np.asarray(detectable_mask_2d, dtype=bool)
    offsets_y = y_values[:, None] - float(centre_arcsec[0])
    offsets_x = x_values[None, :] - float(centre_arcsec[1])
    inside = offsets_y**2 + offsets_x**2 <= float(radius_arcsec)**2
    nodes_inside = int(np.count_nonzero(inside))
    if nodes_inside == 0:
        raise ValueError(
            f"The grid map holds no node inside the D-F7 aperture of radius "
            f"{radius_arcsec} arcsec about {tuple(centre_arcsec)}"
        )
    if not np.all(np.isfinite(q_values[inside])):
        raise ValueError(
            "The grid map leaves non-finite q_F inside the D-F7 aperture, so "
            "the aperture estimands are not measurable at this rung"
        )
    detected = int(np.count_nonzero(detectable & inside))
    return {
        "q_max": float(np.max(q_values[inside])),
        "detectable_area_arcsec2": detected*float(spacing_arcsec)**2,
        "aperture_fraction": detected/nodes_inside,
        "perimeter_clipped": bool(
            detectable[0, :].any() or detectable[-1, :].any()
            or detectable[:, 0].any() or detectable[:, -1].any()
        ),
    }


def _crossing_members(name: str, crossing) -> dict:
    """Return one estimand's artifact members, null when never crossed.

    Parameters
    ----------
    name : `str`
        Estimand name the members are prefixed with.
    crossing : `hwoslaps.campaign.ladder_walk.Crossing` or `None`
        Interpolated crossing, or `None` when the ladder never crossed.

    Returns
    -------
    members : `dict`
        Artifact member name to value. A ladder that never crossed
        records not-a-number: the curve stands as measured and the
        crossing is a finding, never an extrapolation.
    """
    if crossing is None:
        return {
            name: np.asarray(np.nan),
            f"{name}_bracket_logm": np.asarray([np.nan, np.nan]),
            f"{name}_bracket_value": np.asarray([np.nan, np.nan]),
        }
    return {
        name: np.asarray(crossing.logm),
        f"{name}_bracket_logm": np.asarray(
            [crossing.lower_logm, crossing.upper_logm]
        ),
        f"{name}_bracket_value": np.asarray(
            [crossing.lower_value, crossing.upper_value]
        ),
    }


def _artifact_payload(
    *,
    campaign_uuid: str,
    config_hash_value: str,
    system_id: str,
    revision: dict,
    source_asset_path: str,
    source_asset_sha256: str,
    ladder: dict,
    aperture: dict,
    psf: dict,
    table,
    estimands: dict,
    stop_reason: str,
) -> dict:
    """Build the ``ladder_result.npz`` payload.

    Parameters
    ----------
    campaign_uuid : `str`
        Campaign identity member the campaign layer validates.
    config_hash_value : `str`
        Staged configuration hash the campaign layer validates.
    system_id : `str`
        Job identity.
    revision : `dict`
        Verified ``git_hash``, ``git_dirty`` and ``sha256``.
    source_asset_path, source_asset_sha256 : `str`
        Verified template asset declaration.
    ladder : `dict`
        Verified ``ladder`` block.
    aperture : `dict`
        Verified ``ladder.aperture`` block.
    psf : `dict`
        ``state``, ``rms_nm``, ``kernel_shape_native`` and
        ``kernel_sha256`` of the truth and matched fit PSF.
    table : `Sequence`
        Per-rung rows in the order they were measured.
    estimands : `dict`
        Estimand name to `Crossing` or `None`.
    stop_reason : `str`
        Why the coarse ascent stopped.

    Returns
    -------
    payload : `dict`
        Artifact member name to array. No electron map and no q map is
        stored.
    """
    payload = {
        "campaign_uuid": np.asarray(campaign_uuid),
        "config_hash": np.asarray(config_hash_value),
        "system_id": np.asarray(system_id),
        "code_revision_sha256": np.asarray(str(revision["sha256"])),
        "code_git_hash": np.asarray(str(revision["git_hash"])),
        "code_git_dirty": np.asarray(str(revision["git_dirty"])),
        "source_asset_path": np.asarray(str(source_asset_path)),
        "source_asset_sha256": np.asarray(str(source_asset_sha256)),
        "tier": np.asarray(str(ladder["tier"])),
        "golden": np.asarray(bool(ladder["golden"])),
        "parent_overlap": np.asarray(bool(ladder["parent_overlap"])),
        "psf_state": np.asarray(str(psf["state"])),
        "psf_state_rms_nm": np.asarray(float(psf["rms_nm"])),
        "psf_kernel_shape_native": np.asarray(
            [int(value) for value in psf["kernel_shape_native"]]
        ),
        "psf_kernel_sha256": np.asarray(str(psf["kernel_sha256"])),
        "node_spacing_arcsec": np.asarray(
            float(ladder["node_spacing_arcsec"])
        ),
        "theta_e_eff_arcsec": np.asarray(
            float(aperture["theta_e_eff_arcsec"])
        ),
        "aperture_radius_arcsec": np.asarray(float(aperture["radius_arcsec"])),
        "map_half_width_arcsec": np.asarray(
            float(aperture["required_map_half_width_arcsec"])
        ),
        "contour_sha256": np.asarray(str(aperture["stage0_contour_sha256"])),
        "aperture_sha256": np.asarray(
            str(aperture["stage0_aperture_sha256"])
        ),
        "rung_logm": np.asarray([float(row["logm"]) for row in table]),
        "rung_q_max": np.asarray([float(row["q_max"]) for row in table]),
        "rung_detectable_area_arcsec2": np.asarray(
            [float(row["detectable_area_arcsec2"]) for row in table]
        ),
        "rung_aperture_fraction": np.asarray(
            [float(row["aperture_fraction"]) for row in table]
        ),
        "rung_perimeter_clipped": np.asarray(
            [bool(row["perimeter_clipped"]) for row in table], dtype=bool
        ),
        "rung_wall_seconds": np.asarray(
            [float(row["wall_seconds"]) for row in table]
        ),
        "stop_reason": np.asarray(str(stop_reason)),
        "any_perimeter_clipped": np.asarray(
            bool(any(row["perimeter_clipped"] for row in table))
        ),
        "perimeter_cap_flag": np.asarray(
            bool(aperture["perimeter_cap_flag"])
        ),
    }
    for name, crossing in estimands.items():
        payload.update(_crossing_members(name, crossing))
    return payload


def _write_artifact(artifact_path: Path, payload: dict) -> None:
    """Write one artifact through a same-directory temporary file."""
    tmp_path = artifact_path.with_name(artifact_path.name + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, artifact_path)


def main(argv=None) -> None:
    """Walk one member's mass ladder and write its result artifact."""
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

    ladder = _verify_ladder_block(config)
    policy = policy_from_mass_ladder(ladder["mass_ladder"])
    _verify_psf_state(config)

    _enable_float64()
    _enable_jax_compilation_cache()

    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.provenance import config_hash, write_provenance
    from hwoslaps.psf import generate_psf_system
    from hwoslaps.psf.mismatch import _kernel_sha256
    from hwoslaps.psf.utils import pyauto_kernel_native

    revision = _verify_code_revision(config)
    asset_sha256 = _verify_source_asset(config)
    extraction = _extract_theta_e_eff(config)
    aperture = _verify_aperture(config, extraction)

    rung_config = _rung_config(config, ladder, aperture)
    validate_or_raise(rung_config)

    psf_data = generate_psf_system(rung_config["psf"], full_config=rung_config)
    psf = {
        "state": PSF_STATE,
        "rms_nm": _verify_psf_rms(psf_data),
        "kernel_shape_native": KERNEL_SHAPE_NATIVE,
        "kernel_sha256": _kernel_sha256(pyauto_kernel_native(psf_data.kernel)),
    }
    detector = _build_detector(rung_config, psf_data)

    centre_arcsec = extraction.aperture.centre_arcsec
    radius_arcsec = extraction.aperture.radius_arcsec
    table = []
    while True:
        step = next_rung(table, policy)
        if step.logm is None:
            stop_reason = step.stop_reason
            break
        start = perf_counter()
        _point_detector_at_rung(detector, step.logm)
        grid_map = detector.compute_grid_map()
        row = {"logm": step.logm}
        row.update(
            _rung_metrics(
                grid_map.y_coords,
                grid_map.x_coords,
                grid_map.q_asimov_2d,
                grid_map.detectable_mask_2d,
                grid_map.spacing_arcsec,
                centre_arcsec,
                radius_arcsec,
            )
        )
        row["wall_seconds"] = perf_counter() - start
        table.append(row)
        print(
            f"  rung {row['logm']:.2f} ({step.phase}): q_max "
            f"{row['q_max']:.4g}, aperture fraction "
            f"{row['aperture_fraction']:.4f} ({row['wall_seconds']:.0f} s)",
            flush=True,
        )

    estimands = {
        "m_best": log_linear_crossing(table, policy),
        "m10": aperture_fraction_crossing(table, policy, M10_LEVEL),
        "m50": aperture_fraction_crossing(table, policy, M50_LEVEL),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(output_dir/"provenance.yaml", config=config, command=sys.argv)
    _write_artifact(
        artifact_path,
        _artifact_payload(
            campaign_uuid=os.environ.get("HWOSLAPS_CAMPAIGN_UUID", ""),
            config_hash_value=config_hash(config),
            system_id=str(config["run_name"]),
            revision=revision,
            source_asset_path=str(config["stage0"]["source_asset_path"]),
            source_asset_sha256=asset_sha256,
            ladder=ladder,
            aperture=aperture,
            psf=psf,
            table=table,
            estimands=estimands,
            stop_reason=stop_reason,
        ),
    )
    reported = {
        name: "unbracketed" if crossing is None else f"{crossing.logm:.4f}"
        for name, crossing in estimands.items()
    }
    print(f"Ladder artifact: {artifact_path}")
    print(
        f"  {len(table)} rungs, stopped on {stop_reason}, "
        f"M_best {reported['m_best']}, M10 {reported['m10']}, "
        f"M50 {reported['m50']}"
    )


if __name__ == "__main__":
    main()
