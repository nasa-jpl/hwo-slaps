"""Stage 0 pool sampler and S1-lite campaign generator.

The code path that turns ``configs/design/design_freeze_v1.yaml`` into a
runnable Stage 0 campaign: the declared number of no-subhalo
observations, sampled deterministically from the embedded parent design,
emitted as an S1-lite manifest beside a design catalogue that records
everything the manifest schema has no room for.

Two layers, deliberately separated.

The sampler is pure numpy, scipy and astropy. It draws one system per
index from its own ``SeedSequence`` spawn key, in the frozen draw order,
one inverse-CDF uniform per variable, so a system depends only on the
entropy, its index and the draw order. Nothing about pool size, job
ordering or parallel decomposition can move it, which is what makes
Stage 0 resumable under a hard wall-clock stop.

The campaign builder additionally needs the lensing engine: it converts
the sampled axis ratio and position angle with the library conversion,
extracts ``theta_E_eff`` and the aperture with the frozen D-F7 algorithm,
and sizes each system's grid from the aperture's own margin rule. Its
two outputs are byte-deterministic for a given freeze and output root.

The integrity chain is closed at every link. The consumed freeze mapping
is proved equal to the freeze file whose digest is recorded, the
manifest binds the catalogue and the freeze by digest and
`validate_stage0_manifest` re-hashes both, and every job configuration
carries the template asset digest, the frozen extraction settings and
the source revision the campaign was generated at, so all three travel
inside the staged configuration hash the executor already validates.

Outputs
-------
``manifest.yaml``
    A valid S1-lite campaign manifest. Its ``seed_policy`` block carries
    the design provenance and the catalogue digest, so the catalogue is
    inside the frozen manifest's hash chain.
``stage0_catalogue.json``
    Per-system sampled parameters, template assignment, the theta_E
    extraction record with its contour and aperture hashes, the grid
    plan with any cap, the declared selection observables, and the pool
    summary.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from . import _common
from .design_freeze import (
    file_sha256,
    load_design_freeze,
    repo_root,
)


__all__ = [
    "SPEED_OF_LIGHT_KM_S",
    "Stage0Error",
    "assign_templates",
    "build_stage0_campaign",
    "einstein_radius_arcsec",
    "engine_noise_seed",
    "grid_plan",
    "macro_einstein_radius_arcsec",
    "pool_summary",
    "sample_pool",
    "sample_system",
    "selection_observable_plan",
    "sie_area_equivalent_factor",
    "system_id",
    "validate_stage0_manifest",
    "write_stage0_campaign",
]


SPEED_OF_LIGHT_KM_S = 299792.458
"""Speed of light in kilometres per second (`float`)."""

_ARCSEC_PER_RADIAN = 180.0*3600.0/math.pi

_MANIFEST_NAME = "manifest.yaml"
_CATALOGUE_NAME = "stage0_catalogue.json"

_HEX_DIGITS = frozenset("0123456789abcdef")


class Stage0Error(ValueError):
    """Raised for any Stage 0 sampling or manifest generation failure."""


def system_id(index: int, n_systems: int) -> str:
    """Return the campaign job identifier of one pool member.

    The identifier is zero padded to the width of the pool so job ids
    sort lexicographically in index order, and it matches the S1-lite
    ``[a-z0-9_]+`` identifier pattern.

    Parameters
    ----------
    index : `int`
        Zero-based system index.
    n_systems : `int`
        Pool size, which fixes the zero padding width.

    Returns
    -------
    system_id : `str`
        Identifier such as ``sys0042``.
    """
    if index < 0 or index >= n_systems:
        raise Stage0Error(f"System index {index} is outside a pool of {n_systems}")
    width = max(4, len(str(n_systems - 1)))
    return f"sys{index:0{width}d}"


def _generator(entropy: int, spawn_key: tuple) -> np.random.Generator:
    """Return the PCG64 generator of one declared spawn key."""
    sequence = np.random.SeedSequence(entropy=entropy, spawn_key=tuple(spawn_key))
    return np.random.Generator(np.random.PCG64(sequence))


def _uniform(rng: np.random.Generator) -> float:
    """Draw one uniform variate on ``[0, 1)``."""
    return float(rng.uniform())


def _normal_cdf(value: float) -> float:
    """Return the standard normal CDF."""
    from scipy.special import ndtr

    return float(ndtr(value))


def _normal_ppf(value: float) -> float:
    """Return the standard normal quantile function."""
    from scipy.special import ndtri

    return float(ndtri(value))


def _truncated_normal(u: float, mu: float, sigma: float, low: float, high: float) -> float:
    """Return the truncated-normal quantile at one uniform variate."""
    if not high > low:
        raise Stage0Error(f"Truncated normal bounds are not ordered: [{low}, {high}]")
    lower = _normal_cdf((low - mu)/sigma)
    upper = _normal_cdf((high - mu)/sigma)
    value = mu + sigma*_normal_ppf(lower + u*(upper - lower))
    return float(min(max(value, low), high))


def _truncated_lognormal(
    u: float, median: float, sigma_ln: float, low: float, high: float
) -> float:
    """Return the truncated-lognormal quantile at one uniform variate."""
    value = _truncated_normal(
        u, math.log(median), sigma_ln, math.log(low), math.log(high)
    )
    return float(min(max(math.exp(value), low), high))


def _loguniform(u: float, low: float, high: float) -> float:
    """Return the log-uniform quantile at one uniform variate."""
    value = math.exp(math.log(low) + u*(math.log(high) - math.log(low)))
    return float(min(max(value, low), high))


def _uniform_between(u: float, low: float, high: float) -> float:
    """Return the uniform quantile at one uniform variate."""
    return float(low + u*(high - low))


def sample_system(freeze: dict, index: int) -> dict:
    """Draw one Stage 0 system from the frozen parent design.

    Every variable of ``seeds.draw_order`` consumes exactly one uniform
    variate from ``Generator(PCG64(SeedSequence(entropy, (0, index))))``
    and is mapped through its inverse CDF, so the draw order is part of
    the design rather than an implementation detail.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    index : `int`
        Zero-based system index.

    Returns
    -------
    system : `dict`
        Sampled parameters plus the deterministic quantities that follow
        from them: the design ``theta_E``, the isothermal macro
        parameter, the source-plane centre, the template size scale and
        the engine noise seed.

    Raises
    ------
    Stage0Error
        Raised when a declared distribution is missing, malformed, or
        leaves an empty support for this system.
    """
    seeds = freeze["seeds"]
    design = freeze["parent_design"]["distributions"]
    entropy = int(seeds["entropy"])
    stream = tuple(seeds["streams"]["parent_design"]["spawn_key"])
    rng = _generator(entropy, stream + (int(index),))

    order = list(seeds["draw_order"])
    expected = [
        "z_lens",
        "z_source",
        "sigma_v",
        "lens_axis_ratio",
        "lens_position_angle_deg",
        "caustic_offset_fraction",
        "caustic_offset_azimuth_deg",
        "source_magnitude_ab",
        "source_half_light_radius_arcsec",
        "source_rotation_deg",
    ]
    if order != expected:
        raise Stage0Error(
            f"seeds.draw_order {order} is not the implemented draw order "
            f"{expected}; the sampler and the freeze must be amended together"
        )

    z_lens_spec = design["z_lens"]
    z_lens = _truncated_normal(
        _uniform(rng),
        float(z_lens_spec["mu"]),
        float(z_lens_spec["sigma"]),
        float(z_lens_spec["low"]),
        float(z_lens_spec["high"]),
    )

    z_source_spec = design["z_source"]
    source_low = max(
        float(z_source_spec["low_floor"]),
        z_lens + float(z_source_spec["min_separation"]),
    )
    z_source = _truncated_normal(
        _uniform(rng),
        float(z_source_spec["mu"]),
        float(z_source_spec["sigma"]),
        source_low,
        float(z_source_spec["high"]),
    )

    sigma_spec = design["sigma_v"]
    sigma_v = _truncated_lognormal(
        _uniform(rng),
        float(sigma_spec["median"]),
        float(sigma_spec["sigma_ln"]),
        float(sigma_spec["low"]),
        float(sigma_spec["high"]),
    )

    axis_spec = design["lens_axis_ratio"]
    axis_ratio = _truncated_normal(
        _uniform(rng),
        float(axis_spec["mu"]),
        float(axis_spec["sigma"]),
        float(axis_spec["low"]),
        float(axis_spec["high"]),
    )

    angle_bounds = design["lens_position_angle_deg"]["bounds"]
    position_angle_deg = _uniform_between(
        _uniform(rng), float(angle_bounds[0]), float(angle_bounds[1])
    )

    offset_spec = design["caustic_offset_fraction"]
    offset_fraction = _loguniform(
        _uniform(rng), float(offset_spec["low"]), float(offset_spec["high"])
    )

    azimuth_bounds = design["caustic_offset_azimuth_deg"]["bounds"]
    azimuth_deg = _uniform_between(
        _uniform(rng), float(azimuth_bounds[0]), float(azimuth_bounds[1])
    )

    magnitude_spec = design["source_magnitude_ab"]
    source_magnitude_ab = _truncated_normal(
        _uniform(rng),
        float(magnitude_spec["mu"]),
        float(magnitude_spec["sigma"]),
        float(magnitude_spec["low"]),
        float(magnitude_spec["high"]),
    )

    radius_spec = design["source_half_light_radius_arcsec"]
    half_light_arcsec = _truncated_lognormal(
        _uniform(rng),
        float(radius_spec["median"]),
        float(radius_spec["sigma_ln"]),
        float(radius_spec["low"]),
        float(radius_spec["high"]),
    )

    rotation_bounds = design["source_rotation_deg"]["bounds"]
    source_rotation_deg = _uniform_between(
        _uniform(rng), float(rotation_bounds[0]), float(rotation_bounds[1])
    )

    theta_e = einstein_radius_arcsec(sigma_v, z_lens, z_source)
    area_factor = sie_area_equivalent_factor(axis_ratio)
    beta = offset_fraction*theta_e
    azimuth_rad = math.radians(azimuth_deg)
    canonical_half_light = float(freeze["templates"]["canonical_half_light_arcsec"])

    return {
        "index": int(index),
        "z_lens": z_lens,
        "z_source": z_source,
        "sigma_v_km_s": sigma_v,
        "lens_axis_ratio": axis_ratio,
        "lens_position_angle_deg": position_angle_deg,
        "caustic_offset_fraction": offset_fraction,
        "caustic_offset_azimuth_deg": azimuth_deg,
        "source_magnitude_ab": source_magnitude_ab,
        "source_half_light_radius_arcsec": half_light_arcsec,
        "source_rotation_deg": source_rotation_deg,
        "theta_e_design_arcsec": theta_e,
        "sie_area_equivalent_factor": area_factor,
        "macro_einstein_radius_arcsec": macro_einstein_radius_arcsec(
            theta_e, axis_ratio
        ),
        "source_offset_arcsec": beta,
        "source_centre_arcsec": [
            beta*math.sin(azimuth_rad),
            beta*math.cos(azimuth_rad),
        ],
        "source_size_scale": half_light_arcsec/canonical_half_light,
        "engine_noise_seed": engine_noise_seed(freeze, index),
    }


def _cosmology():
    """Return the Planck15 cosmology object."""
    from astropy.cosmology import Planck15

    return Planck15


def einstein_radius_arcsec(sigma_v_km_s: float, z_lens: float, z_source: float) -> float:
    """Return the singular isothermal Einstein radius.

    ``theta_E = 4 pi (sigma_v / c)^2 D_ls / D_s`` with angular diameter
    distances in the frozen Planck15 cosmology.

    Parameters
    ----------
    sigma_v_km_s : `float`
        Lens velocity dispersion in kilometres per second.
    z_lens, z_source : `float`
        Lens and source redshifts, with ``z_source > z_lens``.

    Returns
    -------
    theta_e : `float`
        Einstein radius in arcseconds.
    """
    if not z_source > z_lens:
        raise Stage0Error(
            f"z_source {z_source} does not exceed z_lens {z_lens}; the design "
            "guarantees a minimum separation"
        )
    cosmology = _cosmology()
    d_s = float(cosmology.angular_diameter_distance(z_source).value)
    d_ls = float(cosmology.angular_diameter_distance_z1z2(z_lens, z_source).value)
    radians = 4.0*math.pi*(sigma_v_km_s/SPEED_OF_LIGHT_KM_S)**2*(d_ls/d_s)
    return float(radians*_ARCSEC_PER_RADIAN)


def sie_area_equivalent_factor(axis_ratio: float) -> float:
    """Return the isothermal area-equivalent Einstein radius factor.

    For the isothermal ellipsoid the area enclosed by the tangential
    critical curve gives ``theta_E_eff = einstein_radius * 2 sqrt(q) /
    (1 + q)``. The factor is purely geometric, is 1 in the circular
    limit, and falls to 0.943 at the design's flattest allowed axis
    ratio, which is why the macro parameter is solved for the sampled
    ``theta_E`` rather than set equal to it.

    Parameters
    ----------
    axis_ratio : `float`
        Lens axis ratio in ``(0, 1]``.

    Returns
    -------
    factor : `float`
        ``theta_E_eff / einstein_radius``.
    """
    if not 0.0 < axis_ratio <= 1.0:
        raise Stage0Error(f"Axis ratio {axis_ratio} is outside (0, 1]")
    return float(2.0*math.sqrt(axis_ratio)/(1.0 + axis_ratio))


def macro_einstein_radius_arcsec(theta_e_arcsec: float, axis_ratio: float) -> float:
    """Return the isothermal macro parameter realizing one ``theta_E``.

    Parameters
    ----------
    theta_e_arcsec : `float`
        Target area-equivalent Einstein radius.
    axis_ratio : `float`
        Lens axis ratio.

    Returns
    -------
    einstein_radius : `float`
        Value written to ``lensing.lens_galaxy.mass.einstein_radius``.
    """
    return float(theta_e_arcsec/sie_area_equivalent_factor(axis_ratio))


def engine_noise_seed(freeze: dict, index: int) -> int:
    """Return the engine ``global_seed`` of one system.

    The engine takes a single integer noise seed, so the declared
    ``primary_noise`` stream is narrowed to 32 bits exactly once, here,
    by the rule the freeze records.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    index : `int`
        Zero-based system index.

    Returns
    -------
    seed : `int`
        Non-negative 32-bit engine noise seed.
    """
    seeds = freeze["seeds"]
    stream = tuple(seeds["streams"]["primary_noise"]["spawn_key"])
    sequence = np.random.SeedSequence(
        entropy=int(seeds["entropy"]), spawn_key=stream + (int(index),)
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def assign_templates(freeze: dict, n_systems: Optional[int] = None) -> tuple[str, ...]:
    """Assign the declared templates in exactly balanced allocation.

    One seeded permutation of the pool labels under the declared
    ``template_permutation`` spawn key, so every level receives exactly
    ``n_systems / n_levels`` members and the morphology factor is
    orthogonal to pool size.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    n_systems : `int`, optional
        Pool size. Defaults to ``stage0.n_systems``.

    Returns
    -------
    templates : `tuple` [`str`]
        Template identifier of every system, in index order.

    Raises
    ------
    Stage0Error
        Raised when the pool size is not divisible by the level count,
        which balanced allocation requires.
    """
    levels = [level["id"] for level in freeze["templates"]["levels"]]
    size = int(n_systems if n_systems is not None else freeze["stage0"]["n_systems"])
    if size % len(levels):
        raise Stage0Error(
            f"Stage 0 pool size {size} is not divisible by the {len(levels)} "
            "declared template levels, so balanced allocation is impossible"
        )
    labels = np.repeat(np.arange(len(levels)), size//len(levels))
    stream = tuple(freeze["seeds"]["streams"]["template_permutation"]["spawn_key"])
    rng = _generator(int(freeze["seeds"]["entropy"]), stream)
    rng.shuffle(labels)
    return tuple(levels[int(label)] for label in labels)


def sample_pool(freeze: dict, n_systems: Optional[int] = None) -> tuple[dict, ...]:
    """Sample the whole Stage 0 pool without touching the engine.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    n_systems : `int`, optional
        Pool size. Defaults to ``stage0.n_systems``.

    Returns
    -------
    pool : `tuple` [`dict`]
        One record per system, in index order, each carrying its system
        id and template assignment alongside the sampled parameters.
    """
    size = int(n_systems if n_systems is not None else freeze["stage0"]["n_systems"])
    templates = assign_templates(freeze, size)
    pool = []
    for index in range(size):
        record = sample_system(freeze, index)
        record["system_id"] = system_id(index, size)
        record["source_template"] = templates[index]
        pool.append(record)
    return tuple(pool)


def _quantiles(values, points=(1, 5, 16, 50, 84, 95, 99)) -> dict:
    """Return the named percentiles of one sampled quantity."""
    array = np.asarray(values, dtype=float)
    return {f"p{point}": float(np.percentile(array, point)) for point in points}


def pool_summary(freeze: dict, pool) -> dict:
    """Summarize one sampled pool against the design sanity table.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    pool : sequence of `dict`
        Records from `sample_pool`.

    Returns
    -------
    summary : `dict`
        Induced quantiles of every sampled quantity, the template
        balance, and the survival fraction of the ``theta_E`` floor cut.
        The arc signal-to-noise floor cannot be evaluated before the
        observations are rendered and is therefore reported as pending.
    """
    records = list(pool)
    floor = float(freeze["selection"]["floor_cuts"]["theta_e_arcsec_min"])
    theta_e = np.asarray(
        [record["theta_e_design_arcsec"] for record in records], dtype=float
    )
    balance: dict = {level["id"]: 0 for level in freeze["templates"]["levels"]}
    for record in records:
        balance[record["source_template"]] += 1
    quantities = (
        "z_lens",
        "z_source",
        "sigma_v_km_s",
        "lens_axis_ratio",
        "caustic_offset_fraction",
        "source_magnitude_ab",
        "source_half_light_radius_arcsec",
        "theta_e_design_arcsec",
        "macro_einstein_radius_arcsec",
    )
    survivors = int(np.count_nonzero(theta_e > floor))
    return {
        "n_systems": len(records),
        "quantiles": {
            name: _quantiles([record[name] for record in records])
            for name in quantities
        },
        "theta_e_design_arcsec_mean": float(np.mean(theta_e)),
        "theta_e_design_arcsec_max": float(np.max(theta_e)),
        "theta_e_floor_arcsec": floor,
        "theta_e_floor_survivors": survivors,
        "theta_e_floor_survival_fraction": float(survivors/len(records)),
        "arc_snr_floor_status": (
            "pending: the arc signal-to-noise floor is evaluated on the "
            "rendered no-subhalo observations, not at manifest time"
        ),
        "template_balance": balance,
    }


def grid_plan(freeze: dict, aperture) -> dict:
    """Size one system's production grid from its aperture.

    The extent comes from the aperture's own margin rule, so the map is
    guaranteed to carry the aperture rim the mask machinery evaluates. A
    system whose required side exceeds the declared maximum is capped and
    flagged rather than silently truncated.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.
    aperture : `hwoslaps.lensing.critical_curve.ApertureDefinition`
        Aperture of the extracted ``theta_E_eff``.

    Returns
    -------
    plan : `dict`
        Realized side in pixels, the required side before any cap, the
        cap flag, and the realized aperture coverage in units of
        ``theta_E_eff``.
    """
    pixel_scale = float(freeze["grid_sizing"]["pixel_scale_arcsec"])
    maximum = int(freeze["grid_sizing"]["max_side_px"])
    extent = float(aperture.required_map_extent_arcsec)
    required = int(math.ceil(extent/pixel_scale))
    required += required % 2
    side = min(required, maximum)
    side -= side % 2
    half_width = 0.5*side*pixel_scale
    return {
        "shape": [side, side],
        "pixel_scale_arcsec": pixel_scale,
        "required_map_extent_arcsec": extent,
        "required_side_px": required,
        "max_side_px": maximum,
        "grid_capped": bool(required > maximum),
        "realized_half_width_arcsec": half_width,
        "realized_coverage_theta_e": float(
            half_width/aperture.theta_e_eff_arcsec
        ),
        "requested_coverage_theta_e": float(
            aperture.required_map_half_width_arcsec/aperture.theta_e_eff_arcsec
        ),
    }


def selection_observable_plan(freeze: dict) -> dict:
    """Return the declaration of the observables Stage 0 must compute.

    The values themselves only exist after the observations are
    rendered. This is the frozen declaration of what will be computed,
    with what module, on what plane, so the catalogue states it before
    any number exists.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.

    Returns
    -------
    plan : `dict`
        Module, plane, aperture rule, statistic definitions and the
        floor cuts that will be applied to them.
    """
    selection = freeze["selection"]
    return {
        "module": selection["module"],
        "computed_by": freeze["stage0"]["runner"],
        "computed_when": "after observation generation, per job, inside the job",
        "plane": selection["observables"]["plane"],
        "aperture": selection["observables"]["aperture"],
        "statistics": {
            "arc_snr": selection["observables"]["arc_snr"],
            "gradient_power": selection["observables"]["gradient_power"],
            "diffraction_scale": selection["observables"]["diffraction_scale"],
            "complexity": selection["observables"]["complexity"],
            "blank_variance_e2": selection["observables"]["blank_variance_e2"],
            "pixel_variance_e2": selection["observables"]["pixel_variance_e2"],
        },
        "score": selection["score"]["expression"],
        "standardization": selection["score"]["standardization"],
        "floor_cuts": {
            "theta_e_arcsec_min": selection["floor_cuts"]["theta_e_arcsec_min"],
            "arc_snr_min": selection["floor_cuts"]["arc_snr_min"],
            "strict": selection["floor_cuts"]["strict"],
            "theta_e_used": selection["floor_cuts"]["theta_e_used"],
        },
        "pre_registration": selection["pre_registration"],
    }


def _verified_freeze_artifact(freeze: dict, freeze_path) -> Path:
    """Prove the consumed freeze mapping is the artifact on disk.

    Every campaign artifact records the digest of the freeze *file*, so
    a mapping that has drifted from the file would travel under a digest
    describing a different design. The file is re-read through
    `load_design_freeze`, which also verifies the artifacts the freeze
    binds by hash, and compared with the consumed mapping for exact
    structural equality.

    Parameters
    ----------
    freeze : `dict`
        Freeze mapping handed to the builder.
    freeze_path : path-like or `None`
        Freeze artifact whose digest is recorded. `None` means the
        committed freeze.

    Returns
    -------
    resolved : `pathlib.Path`
        The freeze artifact the campaign is built from.

    Raises
    ------
    Stage0Error
        Raised when the mapping does not equal the file, which includes
        every case where it never went through the verifying loader.
    """
    resolved = _common._freeze_artifact_path(freeze_path)
    if freeze != load_design_freeze(resolved):
        raise Stage0Error(
            "The design freeze mapping handed to the campaign builder is not "
            f"the content of {resolved}; the recorded design freeze digest "
            "would describe a different design than the one being built"
        )
    return resolved


def _verified_runner_command(freeze: dict, runner_command) -> list:
    """Return the runner command, proved to be the frozen runner.

    ``stage0.runner`` is the design's declaration of what renders a
    Stage 0 job, so a manifest that points somewhere else is refused
    rather than recorded.
    """
    parts = [str(part) for part in runner_command]
    declared = str(freeze["stage0"]["runner"])
    if not any(
        part == declared or part.endswith("/" + declared) for part in parts
    ):
        raise Stage0Error(
            f"The runner command {parts} does not invoke the frozen Stage 0 "
            f"runner {declared!r}; the freeze declares the runner and no "
            "driver may substitute another"
        )
    return parts


def _extraction_settings(freeze: dict) -> dict:
    """Return the frozen ``theta_E`` extraction settings as one record.

    The generator and every job configuration read this single record,
    so a runner cannot silently fall back on a module default the design
    never declared.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze.

    Returns
    -------
    settings : `dict`
        Algorithm identities, the extraction grid, the aperture factor
        and margin, and the contour guards.
    """
    algorithm = freeze["aperture"]["theta_e_algorithm"]
    grid = algorithm["extraction_grid"]
    guards = algorithm["guards"]
    return {
        "algorithm_id": str(algorithm["algorithm_id"]),
        "choice_rule_id": str(algorithm["choice_rule_id"]),
        "extraction_grid": {
            "pixel_scale_arcsec": float(grid["pixel_scale_arcsec"]),
            "half_width_factor": float(grid["half_width_factor"]),
        },
        "theta_e_factor": float(freeze["aperture"]["theta_e_factor"]),
        "computational_margin_fraction": float(
            freeze["aperture"]["computational_margin_fraction"]
        ),
        "guards": {
            "closure_tolerance_pixels": float(guards["closure_tolerance_pixels"]),
            "border_margin_pixels": float(guards["border_margin_pixels"]),
            "min_contour_vertices": int(guards["min_contour_vertices"]),
        },
    }


def _template_index(freeze: dict) -> dict:
    """Return the template bank keyed by identifier."""
    return {level["id"]: level for level in freeze["templates"]["levels"]}


def _magnitude_scale(freeze: dict, magnitude_ab: float) -> float:
    """Return the flux ratio of a sampled magnitude to the D1 anchor."""
    anchor = float(
        freeze["observing"]["r_arms"]["arms"]["R0"]["source_magnitude_ab"]
    )
    return float(10.0**(-0.4*(magnitude_ab - anchor)))


def _target_rate_e_per_s(freeze: dict, magnitude_ab: float) -> float:
    """Return the detected unlensed rate a sampled magnitude implies."""
    anchor_rate = float(
        freeze["observing"]["r_arms"]["arms"]["R0"]["detected_rate_e_per_s"]
    )
    return float(anchor_rate*_magnitude_scale(freeze, magnitude_ab))


def _source_total_flux(
    freeze: dict, template: dict, magnitude_ab: float, size_scale: float
) -> float:
    """Return the ``total_flux`` realizing one sampled magnitude and size.

    The ``Image`` source evaluates ``total_flux * sb(theta / size_scale)``,
    so stretching the stamp over a larger angular area multiplies its
    integrated flux by ``size_scale ** 2``. The design samples an
    intrinsic magnitude and an intrinsic half-light radius
    independently, so the size factor is divided out and the source
    carries the sampled magnitude's total flux at any sampled size.
    """
    canonical = float(template["canonical_total_flux"])
    scale = _magnitude_scale(freeze, magnitude_ab)
    return float(canonical*scale/size_scale**2)


def _extract_theta_e(freeze: dict, record: dict, ell_comps) -> Any:
    """Extract ``theta_E_eff`` and the aperture of one sampled system."""
    import autolens as al

    from hwoslaps.lensing import critical_curve as cc

    settings = _extraction_settings(freeze)
    extraction_grid = settings["extraction_grid"]
    guards = settings["guards"]
    galaxy = al.Galaxy(
        redshift=record["z_lens"],
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=record["macro_einstein_radius_arcsec"],
            ell_comps=tuple(ell_comps),
        ),
    )
    grid = cc.CriticalCurveGrid(
        requested_half_width_arcsec=(
            extraction_grid["half_width_factor"]
            * record["macro_einstein_radius_arcsec"]
        ),
        pixel_scale_arcsec=extraction_grid["pixel_scale_arcsec"],
    )
    return cc.extract_theta_e(
        galaxy,
        lens_centre_arcsec=(0.0, 0.0),
        grid=grid,
        theta_e_factor=settings["theta_e_factor"],
        computational_margin_fraction=settings["computational_margin_fraction"],
        closure_tolerance_pixels=guards["closure_tolerance_pixels"],
        border_margin_pixels=guards["border_margin_pixels"],
        min_contour_vertices=guards["min_contour_vertices"],
    )


def _resolve_system(freeze: dict, record: dict) -> dict:
    """Add the engine-derived members of one sampled system."""
    import autogalaxy as ag

    ell_comps = ag.convert.ell_comps_from(
        axis_ratio=record["lens_axis_ratio"],
        angle=record["lens_position_angle_deg"],
    )
    extraction = _extract_theta_e(freeze, record, ell_comps)
    tolerance = float(freeze["derived"]["einstein_radius"]["verification"][
        "tolerance_fractional"
    ])
    realized = float(extraction.theta_e_eff_arcsec)
    design = float(record["theta_e_design_arcsec"])
    if abs(realized/design - 1.0) > tolerance:
        raise Stage0Error(
            f"System {record['system_id']} realizes theta_E_eff {realized} "
            f"against the design theta_E {design}, outside the declared "
            f"fractional tolerance {tolerance}"
        )
    resolved = dict(record)
    resolved["lens_ell_comps"] = [float(ell_comps[0]), float(ell_comps[1])]
    resolved["theta_e_eff_arcsec"] = realized
    resolved["theta_e_realized_over_design"] = float(realized/design)
    resolved["theta_e_extraction"] = extraction.to_provenance_dict()
    resolved["grid"] = grid_plan(freeze, extraction.aperture)
    return resolved


def _job_overrides(
    freeze: dict, resolved: dict, template: dict, code_revision: dict
) -> dict:
    """Build the S1-lite job overrides of one resolved system.

    The ``stage0`` block carries everything the runner must be held to
    rather than allowed to re-derive: the template asset digest, the
    extraction settings, the realized ``theta_E_eff`` with its
    verification tolerance, the exact contour and aperture digests the
    generator extracted, and the source revision the campaign was
    generated at. All of it is covered by the staged configuration hash
    the executor validates against every job artifact.
    """
    stage0 = freeze["stage0"]
    return {
        "global_seed": int(resolved["engine_noise_seed"]),
        "stage0": {
            "system_id": str(resolved["system_id"]),
            "source_template": str(resolved["source_template"]),
            "source_magnitude_ab": float(resolved["source_magnitude_ab"]),
            "theta_e_design_arcsec": float(resolved["theta_e_design_arcsec"]),
            "target_unlensed_rate_e_per_s": _target_rate_e_per_s(
                freeze, resolved["source_magnitude_ab"]
            ),
            "rate_contract_tolerance": float(
                freeze["templates"]["rate_contract_production_tolerance"]
            ),
            "source_asset_path": str(template["asset_path"]),
            "source_asset_sha256": str(template["sha256"]),
            "theta_e_contour_sha256": str(
                resolved["theta_e_extraction"]["contour_sha256"]
            ),
            "theta_e_aperture_sha256": str(
                resolved["theta_e_extraction"]["aperture_sha256"]
            ),
            "theta_e_eff_arcsec": float(resolved["theta_e_eff_arcsec"]),
            "theta_e_eff_tolerance_fractional": float(
                freeze["derived"]["einstein_radius"]["verification"][
                    "tolerance_fractional"
                ]
            ),
            "theta_e_extraction": _extraction_settings(freeze),
            "code_revision": dict(code_revision),
        },
        "lensing": {
            "grid": {
                "shape": list(resolved["grid"]["shape"]),
                "pixel_scale": float(resolved["grid"]["pixel_scale_arcsec"]),
            },
            "lens_galaxy": {
                "redshift": float(resolved["z_lens"]),
                "mass": {
                    "type": "Isothermal",
                    "einstein_radius": float(
                        resolved["macro_einstein_radius_arcsec"]
                    ),
                    "centre": [0.0, 0.0],
                    "ell_comps": list(resolved["lens_ell_comps"]),
                },
            },
            "source_galaxy": {
                "redshift": float(resolved["z_source"]),
                "light": {
                    "type": "Image",
                    "asset_path": template["asset_path"],
                    "centre": list(resolved["source_centre_arcsec"]),
                    "rotation_deg": float(resolved["source_rotation_deg"]),
                    "total_flux": _source_total_flux(
                        freeze,
                        template,
                        resolved["source_magnitude_ab"],
                        resolved["source_size_scale"],
                    ),
                    "flux_scale": 1.0,
                    "size_scale": float(resolved["source_size_scale"]),
                },
            },
            "subhalo": {"enabled": False},
            "cosmology": str(stage0["cosmology"]),
        },
        "psf": {
            "kernel": {
                "shape_native": list(stage0["psf_kernel_shape_native"]),
            },
        },
        "observation": {
            "exposure_time": float(stage0["exposure_time_s"]),
        },
        "modeling": {"enabled": False},
    }


def build_stage0_campaign(
    freeze: dict,
    output_root: str,
    runner_command,
    freeze_path=None,
    n_systems: Optional[int] = None,
    campaign_name: str = "stage0_pool",
    campaign_uuid: Optional[str] = None,
    root=None,
    progress=None,
    allow_unfrozen_pool: bool = False,
) -> dict:
    """Build the Stage 0 manifest and catalogue from the design freeze.

    The freeze mapping is proved to be the content of ``freeze_path``
    before its digest is recorded, so the digest every artifact carries
    always describes the design that was actually built.

    Parameters
    ----------
    freeze : `dict`
        Validated design freeze, which must equal the content of
        ``freeze_path``.
    output_root : `str`
        Campaign output root written into the manifest verbatim.
    runner_command : sequence of `str`
        S1-lite runner command, which must carry the ``{config}``
        placeholder and must invoke the frozen ``stage0.runner``.
    freeze_path : path-like, optional
        Freeze artifact whose digest is recorded. Defaults to the
        committed freeze.
    n_systems : `int`, optional
        Pool size override, used by tests. Defaults to
        ``stage0.n_systems``. Any other size is refused unless
        ``allow_unfrozen_pool`` is passed; an allowed deviation records
        both numbers in its ``seed_policy`` so a reader can see at once
        that it is not the frozen pool.
    campaign_name : `str`, optional
        S1-lite campaign name.
    campaign_uuid : `str`, optional
        Pinned campaign UUID. Left unset the layer generates one at
        freeze time.
    root : path-like, optional
        Repository root the freeze's repo-relative scene and reference
        paths are resolved against. S1-lite resolves manifest paths
        against the manifest's own directory, and a generated campaign
        does not live in the repository, so the manifest carries the
        resolved absolute paths.
    progress : callable, optional
        Called as ``progress(done, total)`` after each system resolves.
        It has no effect on the emitted bytes.
    allow_unfrozen_pool : `bool`, optional
        Permit a pool size other than the frozen ``stage0.n_systems``.
        This exists for engineering probes and tests; the freeze is the
        authority on the production pool, so a deviation must be asked
        for by name rather than passed as a number.

    Returns
    -------
    built : `dict`
        ``manifest``, ``catalogue``, ``summary`` and ``pool``. The
        catalogue digest is injected into the manifest by
        `write_stage0_campaign`, which is what binds the two.

    Raises
    ------
    Stage0Error
        Raised when the freeze mapping is not the content of
        ``freeze_path``, when the runner command does not invoke the
        frozen Stage 0 runner, or when a pool size other than the
        frozen one is requested without ``allow_unfrozen_pool``.
    """
    freeze_artifact = _verified_freeze_artifact(freeze, freeze_path)
    command = _verified_runner_command(freeze, runner_command)
    code_revision = _common._code_revision_record()
    base = Path(root if root is not None else repo_root()).resolve()
    frozen_size = int(freeze["stage0"]["n_systems"])
    size = int(n_systems if n_systems is not None else frozen_size)
    if size != frozen_size and not allow_unfrozen_pool:
        raise Stage0Error(
            f"A pool of {size} systems is not the frozen stage0.n_systems "
            f"{frozen_size}; the freeze declares the pool and an engineering "
            "deviation must pass allow_unfrozen_pool explicitly"
        )
    templates = _template_index(freeze)
    pool = sample_pool(freeze, size)

    systems = []
    jobs = []
    for record in pool:
        resolved = _resolve_system(freeze, record)
        template = templates[resolved["source_template"]]
        overrides = _job_overrides(freeze, resolved, template, code_revision)
        resolved["source_total_flux"] = overrides["lensing"]["source_galaxy"][
            "light"
        ]["total_flux"]
        resolved["source_target_rate_e_per_s"] = overrides["stage0"][
            "target_unlensed_rate_e_per_s"
        ]
        resolved["source_asset_path"] = template["asset_path"]
        resolved["source_asset_sha256"] = template["sha256"]
        systems.append(resolved)
        jobs.append({
            "job_id": resolved["system_id"],
            "scene": str(freeze["stage0"]["base_scene_label"]),
            "overrides": overrides,
        })
        if progress is not None:
            progress(len(jobs), size)

    summary = pool_summary(freeze, pool)
    summary["grid"] = {
        "side_px": _quantiles([system["grid"]["shape"][0] for system in systems]),
        "min_side_px": min(system["grid"]["shape"][0] for system in systems),
        "max_side_px": max(system["grid"]["shape"][0] for system in systems),
        "capped_systems": sorted(
            system["system_id"] for system in systems if system["grid"]["grid_capped"]
        ),
        "declared_max_side_px": int(freeze["grid_sizing"]["max_side_px"]),
    }
    summary["theta_e_eff"] = {
        "quantiles": _quantiles(
            [system["theta_e_eff_arcsec"] for system in systems]
        ),
        "min_realized_over_design": min(
            system["theta_e_realized_over_design"] for system in systems
        ),
        "max_realized_over_design": max(
            system["theta_e_realized_over_design"] for system in systems
        ),
    }

    digest = file_sha256(freeze_artifact)
    catalogue = {
        "schema_version": int(freeze["schema_version"]),
        "campaign_name": campaign_name,
        "design_freeze": {
            "path": freeze_artifact.name,
            "sha256": digest,
            "status": str(freeze["freeze"]["status"]),
            "provisional_items": [
                item["id"] for item in freeze["provisional_items"]
            ],
        },
        "code_revision": dict(code_revision),
        "claim_labels": dict(freeze["claim_labels"]),
        "foreground_free_ceiling": bool(freeze["foreground_free_ceiling"]),
        "n_systems": size,
        "seeds": {
            "entropy": int(freeze["seeds"]["entropy"]),
            "streams": dict(freeze["seeds"]["streams"]),
            "draw_order": list(freeze["seeds"]["draw_order"]),
        },
        "selection_observable_plan": selection_observable_plan(freeze),
        "grid_sizing": dict(freeze["grid_sizing"]),
        "aperture": {
            "theta_e_factor": float(freeze["aperture"]["theta_e_factor"]),
            "computational_margin_fraction": float(
                freeze["aperture"]["computational_margin_fraction"]
            ),
            "algorithm_id": freeze["aperture"]["theta_e_algorithm"]["algorithm_id"],
            "choice_rule_id": freeze["aperture"]["theta_e_algorithm"][
                "choice_rule_id"
            ],
        },
        "summary": summary,
        "systems": systems,
    }

    manifest = {
        "campaign": {
            "name": campaign_name,
            "output_root": str(output_root),
            "runner_command": command,
            "base_scene_configs": {
                str(freeze["stage0"]["base_scene_label"]): str(
                    base/freeze["stage0"]["base_scene_config"]
                ),
            },
            "observing_reference": str(base/freeze["observing"]["reference"]["path"]),
            "expected_artifacts": [str(freeze["stage0"]["artifact"])],
            "expected_job_count": len(jobs),
            "seed_policy": {
                "design_freeze_path": str(freeze_artifact),
                "design_freeze_sha256": digest,
                "design_freeze_status": str(freeze["freeze"]["status"]),
                "code_revision_sha256": str(code_revision["sha256"]),
                "stage0_n_systems": size,
                "stage0_n_systems_frozen": int(freeze["stage0"]["n_systems"]),
                "entropy": int(freeze["seeds"]["entropy"]),
                "parent_design_spawn_key": list(
                    freeze["seeds"]["streams"]["parent_design"]["spawn_key"]
                ),
                "primary_noise_spawn_key": list(
                    freeze["seeds"]["streams"]["primary_noise"]["spawn_key"]
                ),
                "template_permutation_spawn_key": list(
                    freeze["seeds"]["streams"]["template_permutation"]["spawn_key"]
                ),
                "engine_seed_rule": str(
                    freeze["seeds"]["streams"]["primary_noise"]["engine_seed_rule"]
                ),
                "draw_order": list(freeze["seeds"]["draw_order"]),
                "catalogue": _CATALOGUE_NAME,
                "catalogue_sha256": None,
                "foreground_free_ceiling": bool(freeze["foreground_free_ceiling"]),
            },
            "jobs": jobs,
        }
    }
    if campaign_uuid is not None:
        manifest["campaign"]["campaign_uuid"] = str(campaign_uuid)
    return {
        "manifest": manifest,
        "catalogue": catalogue,
        "summary": summary,
        "pool": pool,
    }


def _catalogue_bytes(catalogue: dict) -> bytes:
    """Render the catalogue to its canonical bytes."""
    return (json.dumps(catalogue, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_stage0_campaign(
    directory,
    freeze: dict,
    output_root: str,
    runner_command,
    freeze_path=None,
    n_systems: Optional[int] = None,
    campaign_name: str = "stage0_pool",
    campaign_uuid: Optional[str] = None,
    root=None,
    progress=None,
    allow_unfrozen_pool: bool = False,
) -> dict:
    """Write the Stage 0 catalogue and manifest into one directory.

    The catalogue is written first and its digest is injected into the
    manifest's ``seed_policy``, so the manifest, and therefore the frozen
    manifest S1-lite derives from it, covers the catalogue's bytes.

    Parameters
    ----------
    directory : path-like
        Destination directory, created if absent.
    freeze : `dict`
        Validated design freeze.
    output_root : `str`
        Campaign output root written into the manifest verbatim.
    runner_command : sequence of `str`
        S1-lite runner command.
    freeze_path : path-like, optional
        Freeze artifact whose digest is recorded.
    n_systems : `int`, optional
        Pool size override, refused without ``allow_unfrozen_pool``.
    campaign_name : `str`, optional
        S1-lite campaign name.
    campaign_uuid : `str`, optional
        Pinned campaign UUID.
    root : path-like, optional
        Repository root the freeze's repo-relative paths resolve against.
    progress : callable, optional
        Called as ``progress(done, total)`` after each system resolves.
    allow_unfrozen_pool : `bool`, optional
        Permit a pool size other than the frozen ``stage0.n_systems``.

    Returns
    -------
    written : `dict`
        Manifest path, catalogue path, their digests, and the pool
        summary.
    """
    built = build_stage0_campaign(
        freeze,
        output_root=output_root,
        runner_command=runner_command,
        freeze_path=freeze_path,
        n_systems=n_systems,
        campaign_name=campaign_name,
        campaign_uuid=campaign_uuid,
        root=root,
        progress=progress,
        allow_unfrozen_pool=allow_unfrozen_pool,
    )
    target = Path(directory).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    catalogue_payload = _catalogue_bytes(built["catalogue"])
    catalogue_path = target/_CATALOGUE_NAME
    catalogue_path.write_bytes(catalogue_payload)

    manifest = built["manifest"]
    manifest["campaign"]["seed_policy"]["catalogue_sha256"] = hashlib.sha256(
        catalogue_payload
    ).hexdigest()
    manifest_payload = _common._manifest_bytes(manifest)
    manifest_path = target/_MANIFEST_NAME
    manifest_path.write_bytes(manifest_payload)

    return {
        "manifest_path": manifest_path,
        "catalogue_path": catalogue_path,
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "catalogue_sha256": manifest["campaign"]["seed_policy"]["catalogue_sha256"],
        "summary": built["summary"],
        "n_jobs": len(manifest["campaign"]["jobs"]),
    }


def _policy_member(policy: dict, key: str) -> Any:
    """Return one required ``seed_policy`` member or raise."""
    if key not in policy:
        raise Stage0Error(
            f"campaign.seed_policy is missing '{key}'; a Stage 0 manifest "
            "binds its catalogue and its design freeze by digest"
        )
    return policy[key]


def _require_digest(value: Any, description: str) -> str:
    """Return a full lowercase SHA-256 digest string or raise."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise Stage0Error(
            f"{description} must be a 64-character lowercase sha256 digest, "
            f"got {value!r}"
        )
    return value


def _verify_bound_file(path: Path, expected: str, description: str) -> str:
    """Hash one bound file and compare it with the recorded digest."""
    if not path.is_file():
        raise Stage0Error(
            f"The manifest binds {description} {path}, which does not exist"
        )
    digest = file_sha256(path)
    if digest != expected:
        raise Stage0Error(
            f"The manifest binds {description} {path} at {expected} but its "
            f"bytes hash to {digest}"
        )
    return digest


def validate_stage0_manifest(manifest_path) -> dict:
    """Validate one written Stage 0 manifest and the files it binds.

    The S1-lite schema check is the first half. The second half opens
    the catalogue and the design freeze the manifest claims to bind,
    hashes their bytes and compares them with the recorded digests, so a
    manifest cannot carry a digest for a file that is missing, has moved
    or has been rewritten. The catalogue's own record of the design
    freeze digest is cross-checked against the manifest's, which closes
    the loop between the two written artifacts.

    Parameters
    ----------
    manifest_path : path-like
        Manifest written by `write_stage0_campaign`. The catalogue is
        resolved against this file's own directory, as S1-lite resolves
        every other manifest-relative path.

    Returns
    -------
    normalized : `dict`
        The S1-lite normalized manifest.

    Raises
    ------
    Stage0Error
        Raised when a bound digest is malformed, its file is missing, or
        its bytes hash to a different value.
    hwoslaps.campaign.s1_lite.CampaignError
        Raised for any schema violation. Nothing is repaired.
    """
    from .s1_lite import validate_campaign_manifest

    path = Path(manifest_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        normalized = validate_campaign_manifest(yaml.safe_load(stream))

    policy = normalized["campaign"]["seed_policy"]
    catalogue_name = _policy_member(policy, "catalogue")
    if not isinstance(catalogue_name, str) or not catalogue_name:
        raise Stage0Error(
            f"campaign.seed_policy.catalogue must name a file, got "
            f"{catalogue_name!r}"
        )
    catalogue_digest = _require_digest(
        _policy_member(policy, "catalogue_sha256"),
        "campaign.seed_policy.catalogue_sha256",
    )
    catalogue_path = path.parent/catalogue_name
    _verify_bound_file(catalogue_path, catalogue_digest, "the design catalogue")

    freeze_digest = _require_digest(
        _policy_member(policy, "design_freeze_sha256"),
        "campaign.seed_policy.design_freeze_sha256",
    )
    freeze_artifact = Path(
        str(_policy_member(policy, "design_freeze_path"))
    ).expanduser()
    _verify_bound_file(freeze_artifact, freeze_digest, "the design freeze")

    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    block = catalogue.get("design_freeze") if isinstance(catalogue, dict) else None
    recorded = block.get("sha256") if isinstance(block, dict) else None
    if recorded != freeze_digest:
        raise Stage0Error(
            f"The design catalogue {catalogue_path} records design freeze "
            f"digest {recorded!r} but the manifest records {freeze_digest!r}"
        )
    return normalized
