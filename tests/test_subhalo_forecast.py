"""Tests for the Item 10 dark-matter subhalo mass-function fold."""

from __future__ import annotations

import copy
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hwoslaps.analysis.subhalo_forecast as sf
from hwoslaps.modeling.utils_fisher import (
    FisherGridMapData,
    load_fisher_grid_map_npz,
    save_fisher_grid_map_npz,
)
from hwoslaps.provenance import config_hash


def _snapshot(mass_msun, run_name, output_dir):
    """Build the minimal runner snapshot needed by the fold gates."""
    return {
        "run_name": run_name,
        "plotting": {"output_dir": str(output_dir), "enabled": False},
        "lensing": {
            "cosmology": "Planck15",
            "lens_galaxy": {
                "redshift": 0.5,
                "mass": {"einstein_radius": 1.0},
            },
            "source_galaxy": {"redshift": 1.5},
            "subhalo": {
                "enabled": True,
                "model": "NFW",
                "mass": float(mass_msun),
                "concentration": {
                    "model": "moline2017_eq7",
                    "x_sub": 1.0,
                },
                "position": {"type": "direct", "centre": [1.0, 0.0]},
            },
        },
        "modeling": {
            "fisher": {
                "map": {
                    "type": "grid",
                    "grid": {"spacing_arcsec": 0.1},
                    "detection_q_threshold": 10.0,
                }
            }
        },
    }


def _grid_map(
    mass_msun,
    detectable_nodes,
    statistic="matched",
    embedded_config_hash=None,
    sign_structure=False,
):
    """Build one valid synthetic Fisher grid map."""
    shape = (5, 5)
    evaluated = np.ones(shape, dtype=bool)
    q_asimov = np.full(shape, 1.0, dtype=float)
    q_asimov.flat[:detectable_nodes] = 20.0
    detectable = evaluated & (q_asimov >= 10.0)
    finite = np.ones(shape, dtype=float)
    keywords = {}
    if statistic == "mismatch":
        amplitude_hat = np.ones(shape, dtype=float)
        if sign_structure:
            amplitude_hat.flat[0] = -1.0
        q_mismatch = np.array(q_asimov, copy=True)
        mismatch_mask = (
            evaluated & (amplitude_hat > 0.0) & (q_mismatch >= 10.0)
        )
        zeros = np.zeros(shape, dtype=float)
        false_mask = np.zeros(shape, dtype=bool)
        keywords = {
            "mismatch_enabled": True,
            "amplitude_hat_2d": amplitude_hat,
            "q_mismatch_2d": q_mismatch,
            "z_mismatch_2d": np.sqrt(q_mismatch),
            "mismatch_detectable_mask_2d": mismatch_mask,
            "mismatch_detectable_area_arcsec2": float(
                np.count_nonzero(mismatch_mask)*(0.1*0.1)
            ),
            "num_mismatch_detectable": int(np.count_nonzero(mismatch_mask)),
            "amplitude_spurious_2d": zeros,
            "q_spurious_2d": zeros,
            "z_spurious_2d": zeros,
            "false_positive_mask_2d": false_mask,
            "false_positive_area_arcsec2": 0.0,
            "num_false_positive": 0,
            "max_z_spurious": 0.0,
        }
    return FisherGridMapData(
        y_coords=np.linspace(-0.2, 0.2, shape[0]),
        x_coords=np.linspace(-0.2, 0.2, shape[1]),
        spacing_arcsec=0.1,
        centre_yx=(0.0, 0.0),
        detection_q_threshold=10.0,
        evaluated_mask_2d=evaluated,
        detectable_mask_2d=detectable,
        q_asimov_2d=q_asimov,
        z_asimov_2d=np.sqrt(q_asimov),
        fisher_raw_2d=finite,
        fisher_profiled_2d=finite,
        sigma_amplitude_profiled_2d=finite,
        degradation_2d=finite,
        absorbed_fraction_2d=zeros_like(shape),
        num_positions_evaluated=int(np.count_nonzero(evaluated)),
        num_detectable=int(np.count_nonzero(detectable)),
        detectable_area_arcsec2=float(np.count_nonzero(detectable)*(0.1*0.1)),
        max_z_asimov=float(np.max(np.sqrt(q_asimov))),
        median_z_asimov=float(np.median(np.sqrt(q_asimov))),
        subhalo_mass=float(mass_msun),
        subhalo_model="NFW",
        lens_einstein_radius=1.0,
        config_hash=embedded_config_hash,
        git_hash="0123456789abcdef",
        **keywords,
    )


def zeros_like(shape):
    """Return a zero float array for a synthetic map shape."""
    return np.zeros(shape, dtype=float)


def _write_ladder(
    tmp_path,
    masses=(1.0e6, 2.0e6, 4.0e6),
    detectable_nodes=(2, 4, 8),
    statistic="matched",
    snapshots=True,
    embedded_hashes=True,
    sign_structure=False,
):
    """Write a valid synthetic map ladder and optional snapshots."""
    paths = []
    snapshot_paths = []
    for index, (mass, nodes) in enumerate(zip(masses, detectable_nodes)):
        run_dir = tmp_path / f"run-{index}"
        modeling_dir = run_dir / "modeling"
        modeling_dir.mkdir(parents=True)
        snapshot = _snapshot(mass, f"run-{index}", run_dir)
        snapshot_path = run_dir / "config_used.yaml"
        if snapshots:
            with snapshot_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(snapshot, stream, sort_keys=False)
        embedded = config_hash(snapshot) if embedded_hashes else None
        grid_map = _grid_map(
            mass,
            nodes,
            statistic=statistic,
            embedded_config_hash=embedded,
            sign_structure=sign_structure,
        )
        path = save_fisher_grid_map_npz(
            grid_map,
            modeling_dir / "fisher_grid_map.npz",
        )
        paths.append(path)
        snapshot_paths.append(snapshot_path)
    return paths, snapshot_paths


def _fold_config(paths, masses, statistic="matched", allow_unverified=False):
    """Build one canonical direct-normalization fold specification."""
    return {
        "subhalo_forecast": {
            "maps": [
                {"path": str(path), "mass_msun": float(mass)}
                for path, mass in zip(paths, masses)
            ],
            "statistic": statistic,
            "detection_q_threshold": 10.0,
            "allow_unverified_maps": allow_unverified,
            "lens_plane": {
                "lens_redshift": 0.5,
                "cosmology": "Planck15",
            },
            "shmf": {
                "slope": -1.9,
                "pivot_mass_msun": 1.0e8,
                "normalization": {"sigma_sub_kpc2": 0.012},
            },
            "wdm": {
                "suppression": "lovell20_bound",
                "half_mode_mass_grid": {
                    "log10_min_msun": 6.0,
                    "log10_max_msun": 8.0,
                    "num": 5,
                },
            },
            "integration": {"samples_per_bin": 128},
            "discrimination": {"delta_logl_threshold": 5.0},
            "robustness": {"mass_axis_shift_dex": 0.0},
        }
    }


def _from_f_sub_config(paths, masses):
    """Build a fold specification using the host-normalized path."""
    config = _fold_config(paths, masses)
    config["subhalo_forecast"]["shmf"]["normalization"] = {
        "from_f_sub": {
            "preset": "hydro_dv17",
            "mass_range_msun": [1.0e6, 1.0e11],
            "aperture_factor": 2.0,
            "host_slope": 2.0,
            "source_redshift": 1.5,
            "einstein_radius_arcsec": 1.0,
        }
    }
    return config


def _corrupt_npz(
    source,
    destination=None,
    updates=None,
    delete=(),
    recompute_content_digest=False,
):
    """Mutate NPZ members to construct a deliberately malformed fixture."""
    source = Path(source)
    destination = source if destination is None else Path(destination)
    with np.load(source, allow_pickle=False) as stored:
        payload = {name: np.array(stored[name], copy=True) for name in stored.files}
    for name in delete:
        payload.pop(name, None)
    if updates:
        payload.update(updates)
    if recompute_content_digest:
        payload.pop("content_digest", None)
        payload["content_digest"] = np.asarray(sf._content_digest(payload))
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    return destination


def _set_path(mapping, path, value):
    """Set one dotted configuration path in a copied mapping."""
    keys = path.split(".")
    target = mapping
    for key in keys[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    if isinstance(target, list):
        target[int(keys[-1])] = value
    else:
        target[keys[-1]] = value


def _assert_forecasts_equal(first, second):
    """Assert field-wise equality for array-bearing forecast dataclasses."""
    for field in fields(first):
        left = getattr(first, field.name)
        right = getattr(second, field.name)
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict) and field.name == "robustness":
            assert set(left) == set(right)
            for name in left:
                if isinstance(left[name], np.ndarray):
                    np.testing.assert_array_equal(left[name], right[name])
                else:
                    assert left[name] == right[name]
        else:
            assert left == right


# T1: schema validation.


@pytest.mark.parametrize(
    "path",
    [
        "config",
        "subhalo_forecast",
        "subhalo_forecast.maps.0",
        "subhalo_forecast.lens_plane",
        "subhalo_forecast.shmf",
        "subhalo_forecast.shmf.normalization",
        "subhalo_forecast.shmf.normalization.from_f_sub",
        "subhalo_forecast.wdm",
        "subhalo_forecast.wdm.half_mode_mass_grid",
        "subhalo_forecast.integration",
        "subhalo_forecast.discrimination",
        "subhalo_forecast.robustness",
    ],
)
def test_schema_rejects_unknown_keys_at_every_level(path):
    """Reject an unknown key with its full containing path."""
    if path == "subhalo_forecast.shmf.normalization.from_f_sub":
        config = _from_f_sub_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    else:
        config = _fold_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    target = config
    if path != "config":
        for key in path.split("."):
            target = target[int(key)] if isinstance(target, list) else target[key]
    target["unexpected"] = 1
    display_path = path.replace(".maps.0", ".maps[0]")

    with pytest.raises(ValueError, match=re.escape(display_path)):
        sf.validate_subhalo_forecast_config(config)


def test_schema_requires_exactly_one_normalization_mode():
    """Reject both and neither SHMF normalization block."""
    config = _fold_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    normalization = config["subhalo_forecast"]["shmf"]["normalization"]
    normalization["from_f_sub"] = {
        "preset": "hydro_dv17",
        "mass_range_msun": [1.0, 2.0],
        "aperture_factor": 2.0,
        "host_slope": 2.0,
        "source_redshift": 1.5,
        "einstein_radius_arcsec": 1.0,
    }
    with pytest.raises(ValueError, match="exactly one"):
        sf.validate_subhalo_forecast_config(config)

    normalization.clear()
    with pytest.raises(ValueError, match="exactly one"):
        sf.validate_subhalo_forecast_config(config)


def test_schema_enforces_preset_xor_explicit_f_sub():
    """Require exactly one preset or explicit fraction."""
    config = _from_f_sub_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    block = config["subhalo_forecast"]["shmf"]["normalization"]["from_f_sub"]
    block["f_sub"] = 0.01
    with pytest.raises(ValueError, match="exactly one"):
        sf.validate_subhalo_forecast_config(config)

    block.pop("preset")
    block.pop("f_sub")
    with pytest.raises(ValueError, match="exactly one"):
        sf.validate_subhalo_forecast_config(config)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("preset", [], r"from_f_sub\.preset"),
        ("suppression", {}, r"wdm\.suppression"),
    ],
)
def test_schema_rejects_non_string_enum_values(field, value, message):
    """Reject unhashable enum containers with path-qualified ValueError."""
    if field == "preset":
        config = _from_f_sub_config(["a", "b", "c"], [1.0, 2.0, 3.0])
        host = config["subhalo_forecast"]["shmf"]["normalization"][
            "from_f_sub"
        ]
        host["preset"] = value
    else:
        config = _fold_config(["a", "b", "c"], [1.0, 2.0, 3.0])
        config["subhalo_forecast"]["wdm"]["suppression"] = value
    with pytest.raises(ValueError, match=message):
        sf.validate_subhalo_forecast_config(config)


@pytest.mark.parametrize(
    "suppression,custom_abc,message",
    [
        ("custom", None, "custom_abc"),
        ("lovell20_bound", [1.0, 1.0, -1.0], "custom_abc"),
        ("custom", [np.nan, 1.0, -1.0], r"custom_abc\[0\].*finite"),
        ("custom", [0.0, 1.0, -1.0], r"custom_abc\[0\]"),
        ("custom", [1.0, 0.0, -1.0], r"custom_abc\[1\]"),
        ("custom", [1.0, 1.0, 0.0], r"custom_abc\[2\]"),
    ],
)
def test_schema_enforces_custom_suppression_domain(
    suppression,
    custom_abc,
    message,
):
    """Require custom coefficients exactly for a valid custom preset."""
    config = _fold_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    wdm = config["subhalo_forecast"]["wdm"]
    wdm["suppression"] = suppression
    if custom_abc is not None:
        wdm["custom_abc"] = custom_abc
    with pytest.raises(ValueError, match=message):
        sf.validate_subhalo_forecast_config(config)


@pytest.mark.parametrize(
    "path,value,message",
    [
        ("subhalo_forecast.maps.0.mass_msun", 0.0, "mass_msun"),
        ("subhalo_forecast.maps.0.mass_msun", np.nan, "finite"),
        ("subhalo_forecast.detection_q_threshold", False, "numeric"),
        ("subhalo_forecast.detection_q_threshold", np.inf, "finite"),
        ("subhalo_forecast.detection_q_threshold", 0.0, "positive"),
        ("subhalo_forecast.lens_plane.lens_redshift", 0.0, "positive"),
        ("subhalo_forecast.lens_plane.lens_redshift", np.nan, "finite"),
        ("subhalo_forecast.shmf.slope", -3.0, "strictly between"),
        ("subhalo_forecast.shmf.slope", -1.0, "strictly between"),
        ("subhalo_forecast.shmf.slope", np.nan, "finite"),
        ("subhalo_forecast.shmf.pivot_mass_msun", 0.0, "positive"),
        ("subhalo_forecast.shmf.pivot_mass_msun", np.inf, "finite"),
        (
            "subhalo_forecast.shmf.normalization.sigma_sub_kpc2",
            0.0,
            "positive",
        ),
        (
            "subhalo_forecast.shmf.normalization.sigma_sub_kpc2",
            np.nan,
            "finite",
        ),
        (
            "subhalo_forecast.wdm.half_mode_mass_grid.log10_min_msun",
            np.nan,
            "finite",
        ),
        (
            "subhalo_forecast.wdm.half_mode_mass_grid.log10_max_msun",
            np.inf,
            "finite",
        ),
        ("subhalo_forecast.wdm.half_mode_mass_grid.num", 1, "at least 2"),
        ("subhalo_forecast.wdm.half_mode_mass_grid.num", 2.0, "integer"),
        ("subhalo_forecast.integration.samples_per_bin", True, "integer"),
        ("subhalo_forecast.integration.samples_per_bin", 1, "at least 2"),
        (
            "subhalo_forecast.discrimination.delta_logl_threshold",
            0.0,
            "positive",
        ),
        (
            "subhalo_forecast.discrimination.delta_logl_threshold",
            np.nan,
            "finite",
        ),
        ("subhalo_forecast.robustness.mass_axis_shift_dex", -0.1, "non-negative"),
        ("subhalo_forecast.robustness.mass_axis_shift_dex", np.inf, "finite"),
    ],
)
def test_schema_rejects_each_direct_numeric_domain(path, value, message):
    """Reject each invalid direct-normalization numeric domain in isolation."""
    config = _fold_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    _set_path(config, path, value)
    with pytest.raises(ValueError, match=message):
        sf.validate_subhalo_forecast_config(config)


@pytest.mark.parametrize(
    "path,value,message",
    [
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.f_sub",
            1.0,
            "strictly between 0 and 1",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.f_sub",
            np.nan,
            "finite",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.mass_range_msun.0",
            0.0,
            "positive",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.mass_range_msun.1",
            np.inf,
            "finite",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.aperture_factor",
            0.0,
            "positive",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.aperture_factor",
            np.nan,
            "finite",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.host_slope",
            1.0,
            "strictly between 1 and 3",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.host_slope",
            3.0,
            "strictly between 1 and 3",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.host_slope",
            np.nan,
            "finite",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.source_redshift",
            0.5,
            "greater than",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.source_redshift",
            np.inf,
            "finite",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.einstein_radius_arcsec",
            0.0,
            "positive",
        ),
        (
            "subhalo_forecast.shmf.normalization.from_f_sub.einstein_radius_arcsec",
            np.nan,
            "finite",
        ),
    ],
)
def test_schema_rejects_each_host_normalization_domain(path, value, message):
    """Reject each invalid host-normalization numeric domain in isolation."""
    config = _from_f_sub_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    block = config["subhalo_forecast"]["shmf"]["normalization"]["from_f_sub"]
    block.pop("preset")
    block["f_sub"] = 0.01
    _set_path(config, path, value)
    with pytest.raises(ValueError, match=message):
        sf.validate_subhalo_forecast_config(config)


def test_schema_rejects_unordered_ranges_and_canonicalizes_numbers():
    """Reject reversed ranges and return canonical floats and lists."""
    config = _from_f_sub_config(["a", "b", "c"], [1, 2, 3])
    block = config["subhalo_forecast"]["shmf"]["normalization"]["from_f_sub"]
    block["mass_range_msun"] = [2.0, 1.0]
    with pytest.raises(ValueError, match="strictly increasing"):
        sf.validate_subhalo_forecast_config(config)

    direct = _fold_config(["a", "b", "c"], [1.0, 2.0, 3.0])
    grid = direct["subhalo_forecast"]["wdm"]["half_mode_mass_grid"]
    grid["log10_max_msun"] = grid["log10_min_msun"]
    with pytest.raises(ValueError, match="log10_min_msun < log10_max_msun"):
        sf.validate_subhalo_forecast_config(direct)

    block["mass_range_msun"] = (1, 2)
    normalized = sf.validate_subhalo_forecast_config(config)
    forecast = normalized["subhalo_forecast"]
    assert forecast["maps"][0]["mass_msun"] == 1.0
    assert isinstance(forecast["maps"][0]["mass_msun"], float)
    assert forecast["shmf"]["normalization"]["from_f_sub"][
        "mass_range_msun"
    ] == [1.0, 2.0]


# T2: closed-form folds and quadrature convergence.


def _power_law_integral(sigma_sub, pivot, slope, area_scale, area_power, lo, hi):
    """Return the analytic integral for a power-law area and SHMF."""
    exponent = slope + area_power + 1.0
    coefficient = (
        sigma_sub*area_scale/(pivot**(slope + 1.0)*lo**area_power)
    )
    return coefficient*(hi**exponent - lo**exponent)/exponent


@pytest.mark.parametrize(
    "nodes,area_power",
    [((4, 4, 4), 0.0), ((1, 2, 4), 1.0)],
)
def test_closed_form_power_law_folds_and_resolution_convergence(
    tmp_path,
    nodes,
    area_power,
):
    """Match exact interpolants and converge quadrature to analytic folds."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path, masses, nodes)
    config = _fold_config(paths, masses)
    normalized = sf.validate_subhalo_forecast_config(config)
    data = sf.run_subhalo_forecast(normalized)

    midpoint = np.sqrt(masses[0]*masses[1])
    interpolated = sf._interpolate_area(
        np.asarray([midpoint]),
        masses[0],
        masses[1],
        data.detectable_area_kpc2[0],
        data.detectable_area_kpc2[1],
    )[0]
    exact_midpoint = data.detectable_area_kpc2[0]*(midpoint/masses[0])**area_power
    assert interpolated == pytest.approx(exact_midpoint, rel=1.0e-10)

    area_scale = data.detectable_area_kpc2[0]
    analytic = _power_law_integral(
        data.sigma_sub_kpc2,
        data.pivot_mass_msun,
        data.shmf_slope,
        area_scale,
        area_power,
        masses[0],
        masses[-1],
    )
    assert data.mu_cdm == pytest.approx(analytic, rel=1.0e-4)

    errors = []
    for samples in (32, 64):
        candidate = copy.deepcopy(config)
        candidate["subhalo_forecast"]["integration"]["samples_per_bin"] = samples
        result = sf.run_subhalo_forecast(candidate)
        errors.append(abs(result.mu_cdm - analytic))
    assert errors[1] < errors[0]


# T3: f_sub normalization.


class _FakeCosmology:
    """Fixed cosmology values for hand-computed normalization tests."""

    def kpc_per_arcsec_from(self, redshift):
        """Return a fixed angular scale."""
        assert redshift == 0.5
        return 7.0

    def critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        self,
        redshift_0,
        redshift_1,
    ):
        """Return a fixed critical surface density."""
        assert (redshift_0, redshift_1) == (0.5, 1.5)
        return 2.0e9


def test_sigma_sub_from_f_sub_matches_hand_algebra_and_stable_slope_limit(
    monkeypatch,
):
    """Match host normalization, linearity, gamma sign, and slope limits."""
    import hwoslaps.lensing.generator as generator

    monkeypatch.setattr(generator, "_get_cosmology", lambda name: _FakeCosmology())
    common = dict(
        mass_range_msun=[1.0e6, 1.0e11],
        aperture_factor=2.0,
        host_slope=2.0,
        lens_redshift=0.5,
        source_redshift=1.5,
        einstein_radius_arcsec=1.0,
        cosmology_name="Planck15",
        slope=-2.0,
        pivot_mass_msun=1.0e8,
    )
    result = sf.sigma_sub_from_f_sub(f_sub=0.005, **common)
    mass_integral = 1.0e8*np.log(1.0e11/1.0e6)
    expected = 0.005*(0.5*2.0e9)/mass_integral
    assert result == pytest.approx(expected, rel=1.0e-14)
    assert sf.sigma_sub_from_f_sub(f_sub=0.01, **common) == pytest.approx(
        2.0*result
    )

    steeper = sf.sigma_sub_from_f_sub(
        f_sub=0.005,
        **{**common, "host_slope": 2.5},
    )
    assert steeper < result
    for slope in (-2.0 - 1.0e-12, -2.0 + 1.0e-12):
        near = sf.sigma_sub_from_f_sub(
            f_sub=0.005,
            **{**common, "slope": slope},
        )
        assert near == pytest.approx(result, rel=1.0e-9)


def test_sigma_sub_planck15_canonical_scene_sanity_window():
    """Keep the canonical Planck15 normalization in its physical window."""
    result = sf.sigma_sub_from_f_sub(
        f_sub=0.005,
        mass_range_msun=[1.0e6, 1.0e11],
        aperture_factor=2.0,
        host_slope=2.0,
        lens_redshift=0.5,
        source_redshift=1.5,
        einstein_radius_arcsec=1.0,
        cosmology_name="Planck15",
        slope=-1.9,
        pivot_mass_msun=1.0e8,
    )
    # kappa_bar = (theta_E/R)^(gamma-1) = 0.5;
    # Sigma_crit(0.5, 1.5) = 2266568131.0659657 Msun kpc^-2 (measured same
    # session); Sigma_host = 1.1332840655e9; I_m at slope -1.9 =
    # 1.36444e9 Msun; Sigma_sub = 0.005*Sigma_host/I_m.
    assert 0.001 < result < 0.1
    assert result == pytest.approx(0.004153338476539621, rel=1.0e-9)


# T4-T6: WDM, conversions, and discrimination.


def test_wdm_limit_monotonicity_presets_and_fold_bound(tmp_path):
    """Recover CDM exactly and enforce the named suppression behavior."""
    masses = np.asarray([1.0e6, 1.0e7, 1.0e8])
    exact = sf.wdm_suppression(masses, 0.0, 4.2, 2.5, -0.2)
    np.testing.assert_array_equal(exact, np.ones_like(masses))
    values = [sf.wdm_suppression(masses, mhm, 4.2, 2.5, -0.2) for mhm in (1.0e6, 1.0e7)]
    assert np.all(values[1] < values[0])
    assert sf._SUPPRESSION_PRESETS == {
        "lovell20_bound": (4.2, 2.5, -0.2),
        "lovell14": (1.0, 1.0, -1.3),
        "oriordan23_mmax": (1.1, 1.0, -0.5),
    }

    paths, _ = _write_ladder(tmp_path)
    data = sf.run_subhalo_forecast(_fold_config(paths, (1.0e6, 2.0e6, 4.0e6)))
    assert np.all(data.mu_wdm <= data.mu_cdm)


def test_half_mode_thermal_conversion_exact_anchor_and_roundtrip():
    """Pin the 3.3 keV anchor, inverse, and zero-mass reporting rule."""
    assert sf.half_mode_mass_from_thermal_kev(3.3) == 3.0e8
    for thermal_mass in (1.5, 3.3, 8.0):
        recovered = sf.thermal_kev_from_half_mode_mass(
            sf.half_mode_mass_from_thermal_kev(thermal_mass)
        )
        assert recovered == pytest.approx(thermal_mass, rel=1.0e-12)
    assert np.isnan(sf.thermal_kev_from_half_mode_mass(0.0))


def test_binned_poisson_discrimination_and_zero_guards():
    """Match hand KL, information scaling, and bin-merging inequality."""
    cdm = np.asarray([2.0, 3.0])
    wdm = np.asarray([[1.0], [2.0]])
    result = sf._poisson_discrimination(cdm, wdm, 5.0)
    expected = (1.0 - 2.0 + 2.0*np.log(2.0)) + (
        2.0 - 3.0 + 3.0*np.log(3.0/2.0)
    )
    assert result[0][0] == pytest.approx(expected)
    assert result[1][0] == pytest.approx(5.0/expected)
    finite = np.isfinite(result[1])
    np.testing.assert_array_equal(result[2][finite], np.ceil(result[1][finite]))

    zero = sf._poisson_discrimination(
        np.asarray([0.0, 0.0]),
        np.asarray([[1.0], [0.0]]),
        5.0,
    )
    assert zero[0][0] == 1.0
    assert not np.isnan(zero[0][0])
    identical = sf._poisson_discrimination(cdm, cdm[:, None], 5.0)
    assert identical[0][0] == 0.0
    assert np.isinf(identical[1][0])
    assert np.isinf(identical[2][0])
    assert sf._n_req_from_divergence(np.asarray([2.0]), 5.0)[0] == pytest.approx(2.5)
    assert sf._n_req_from_divergence(np.asarray([4.0]), 5.0)[0] == pytest.approx(1.25)

    tilted = sf._poisson_discrimination(
        np.asarray([4.0, 1.0]),
        np.asarray([[1.0], [2.0]]),
        5.0,
    )
    assert tilted[3][0] >= tilted[1][0]


@pytest.mark.parametrize("epsilon", [1.0e-8, 1.0e-12])
def test_poisson_discrimination_is_stable_near_cdm(epsilon):
    """Retain positive second-order KL information near the CDM limit."""
    wdm_value = 1.0 - epsilon
    delta = wdm_value - 1.0
    expected = 0.5*delta*delta
    result = sf._poisson_discrimination(
        np.asarray([1.0]),
        np.asarray([[wdm_value]]),
        5.0,
    )
    assert result[0][0] == pytest.approx(expected, rel=1.0e-3, abs=0.0)
    assert result[0][0] > 0.0
    assert 5.0/result[3][0] == pytest.approx(expected, rel=1.0e-3, abs=0.0)


def test_ceil_finite_ceils_only_finite_values():
    """Ceil fractional finites while preserving integers and infinity."""
    values = np.asarray([1.2, 2.0, np.inf])
    np.testing.assert_array_equal(
        sf._ceil_finite(values),
        np.asarray([2.0, 2.0, np.inf]),
    )


# T7-T10: integrity gates and routing.


@pytest.mark.parametrize(
    "member,value,message",
    [
        ("q_asimov_2d", np.ones((4, 5)), "G0.*shape"),
        ("evaluated_mask_2d", np.ones((5, 5), dtype=np.int64), "G0.*boolean"),
        (
            "q_asimov_2d",
            np.full((5, 5), np.nan),
            "G0.*finite.*evaluated",
        ),
        (
            "evaluated_mask_2d",
            np.eye(5, dtype=bool),
            "G0.*NaN exactly off evaluated",
        ),
        ("x_coords", np.asarray([0.0, 0.1, 0.1, 0.2, 0.3]), "G0.*monotone"),
        (
            "y_coords",
            np.asarray([-0.2, -0.1, 0.0, 0.1, 0.25]),
            "G0.*coordinate steps.*spacing_arcsec",
        ),
        ("spacing_arcsec", np.asarray(0.0), "G0.*spacing_arcsec"),
    ],
)
def test_g0_rejects_malformed_map_structure(tmp_path, member, value, message):
    """Reject malformed shapes, mask dtypes, NaNs, coordinates, and spacing."""
    paths, _ = _write_ladder(tmp_path)
    _corrupt_npz(paths[0], updates={member: value})
    with pytest.raises(ValueError, match=message):
        sf.run_subhalo_forecast(_fold_config(paths, (1.0e6, 2.0e6, 4.0e6)))


def test_g0_binds_coordinate_steps_to_spacing(tmp_path):
    """Reject coordinated spacing and stored-area corruption."""
    paths, _ = _write_ladder(tmp_path)
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            count = np.count_nonzero(stored["detectable_mask_2d"])
        _corrupt_npz(
            path,
            updates={
                "spacing_arcsec": np.asarray(0.2),
                "detectable_area_arcsec2": np.asarray(count*(0.2*0.2)),
            },
        )
    with pytest.raises(ValueError, match="G0.*coordinate steps.*spacing_arcsec"):
        sf.run_subhalo_forecast(
            _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
        )


def test_g0_validates_mismatch_amplitude_nan_structure(tmp_path):
    """Require finite mismatch amplitudes at every evaluated grid node."""
    paths, _ = _write_ladder(tmp_path, statistic="mismatch")
    with np.load(paths[0], allow_pickle=False) as stored:
        amplitude = np.array(stored["amplitude_hat_2d"], copy=True)
    amplitude.flat[0] = np.nan
    _corrupt_npz(paths[0], updates={"amplitude_hat_2d": amplitude})
    config = _fold_config(
        paths,
        (1.0e6, 2.0e6, 4.0e6),
        statistic="mismatch",
    )
    with pytest.raises(ValueError, match="G0.*amplitude_hat_2d.*finite"):
        sf.run_subhalo_forecast(config)


def test_g1_rejects_duplicate_path_and_file_hash(tmp_path):
    """Reject repeated paths and byte-identical files at distinct paths."""
    paths, _ = _write_ladder(tmp_path / "path")
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    config["subhalo_forecast"]["maps"][1]["path"] = str(paths[0])
    with pytest.raises(ValueError, match="G1.*resolved paths"):
        sf.run_subhalo_forecast(config)

    paths, _ = _write_ladder(tmp_path / "hash")
    duplicate = paths[1]
    duplicate.write_bytes(paths[0].read_bytes())
    with pytest.raises(ValueError, match="G1.*sha256"):
        sf.run_subhalo_forecast(
            _fold_config(paths, (1.0e6, 1.0e6, 4.0e6))
        )


@pytest.mark.parametrize(
    "stored_masses,declared_masses,message",
    [
        ((1.0e6, 1.0e6, 4.0e6), (1.0e6, 1.0e6, 4.0e6), "strictly increasing"),
        ((1.0e6, 2.0e6, 4.0e6), (1.0e6, 2.1e6, 4.0e6), "declared mass"),
    ],
)
def test_g1_rejects_mass_ladder_errors(
    tmp_path,
    stored_masses,
    declared_masses,
    message,
):
    """Reject non-increasing stored masses and declared-mass mismatches."""
    paths, _ = _write_ladder(tmp_path, masses=stored_masses)
    with pytest.raises(ValueError, match=message):
        sf.run_subhalo_forecast(_fold_config(paths, declared_masses))


@pytest.mark.parametrize(
    "updates,message",
    [
        ({"x_coords": np.linspace(-0.3, 0.1, 5)}, "x_coords"),
        ({"y_coords": np.linspace(-0.1, 0.3, 5)}, "y_coords"),
        (
            {
                "evaluated_mask_2d": np.eye(5, dtype=bool),
                "q_asimov_2d": np.where(
                    np.eye(5, dtype=bool),
                    1.0,
                    np.nan,
                ),
            },
            "evaluated_mask_2d",
        ),
        ({"centre_yx": np.asarray([0.1, 0.0])}, "centre_yx"),
        ({"subhalo_model": np.str_("PointMass")}, "subhalo_model"),
        ({"lens_einstein_radius": np.asarray(1.1)}, "lens_einstein_radius"),
        (
            {"source_image_asset_sha256_16": np.str_("a"*16)},
            "source_image_asset_sha256_16",
        ),
        (
            {"source_image_asset_path": np.str_("assets/other.npz")},
            "source_image_asset_path",
        ),
    ],
)
def test_g2_rejects_cross_map_incompatibilities(tmp_path, updates, message):
    """Reject every incompatible geometry or lens-model identity field."""
    paths, _ = _write_ladder(tmp_path)
    _corrupt_npz(paths[1], updates=updates)
    with pytest.raises(ValueError, match=f"G2.*{message}"):
        sf.run_subhalo_forecast(_fold_config(paths, (1.0e6, 2.0e6, 4.0e6)))


def test_g2_rejects_from_f_sub_einstein_radius_mismatch(tmp_path):
    """Bind the host normalization aperture to the map Einstein radius."""
    paths, _ = _write_ladder(tmp_path)
    config = _from_f_sub_config(paths, (1.0e6, 2.0e6, 4.0e6))
    block = config["subhalo_forecast"]["shmf"]["normalization"]["from_f_sub"]
    block["einstein_radius_arcsec"] = 1.1
    with pytest.raises(ValueError, match="G2.*einstein_radius_arcsec"):
        sf.run_subhalo_forecast(config)


@pytest.mark.parametrize(
    "member",
    ["mismatch_enabled", "q_mismatch_2d", "amplitude_hat_2d"],
)
def test_g3_requires_mismatch_siblings(tmp_path, member):
    """Require the mismatch flag, q grid, and one-sided amplitude grid."""
    paths, _ = _write_ladder(tmp_path, statistic="mismatch")
    _corrupt_npz(paths[0], delete=(member,))
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6), statistic="mismatch")
    with pytest.raises(ValueError, match=f"G3.*{member}"):
        sf.run_subhalo_forecast(config)


def test_g4_rejects_embedded_hash_mismatch(tmp_path):
    """Reject a map that is not bound to its adjacent snapshot."""
    paths, _ = _write_ladder(tmp_path)
    _corrupt_npz(paths[0], updates={"config_hash": np.str_("badbadbadbadbadb")})
    with pytest.raises(ValueError, match="G4.*embedded config_hash"):
        sf.run_subhalo_forecast(_fold_config(paths, (1.0e6, 2.0e6, 4.0e6)))


def test_g4_binds_stored_ladder_mass_to_snapshot(tmp_path):
    """Reject a mutable stored mass that disagrees with its bound snapshot."""
    paths, _ = _write_ladder(tmp_path)
    _corrupt_npz(paths[1], updates={"subhalo_mass": np.asarray(3.0e6)})
    config = _fold_config(paths, (1.0e6, 3.0e6, 4.0e6))
    with pytest.raises(ValueError, match="G4.*subhalo mass.*stored map mass"):
        sf.run_subhalo_forecast(config)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("spacing_arcsec", 0.2, "spacing_arcsec"),
        ("detection_q_threshold", 11.0, "detection_q_threshold"),
        ("type", "explicit", "map type must be grid"),
    ],
)
def test_g4_binds_map_geometry_and_threshold_to_snapshot(
    tmp_path,
    field,
    value,
    message,
):
    """Reject snapshot map settings inconsistent with the stored maps."""
    paths, snapshots = _write_ladder(tmp_path)
    for path, snapshot_path in zip(paths, snapshots):
        with snapshot_path.open("r", encoding="utf-8") as stream:
            snapshot = yaml.safe_load(stream)
        map_block = snapshot["modeling"]["fisher"]["map"]
        if field == "spacing_arcsec":
            map_block["grid"][field] = value
        else:
            map_block[field] = value
        with snapshot_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(snapshot, stream, sort_keys=False)
        _corrupt_npz(
            path,
            updates={"config_hash": np.str_(config_hash(snapshot))},
        )
    with pytest.raises(ValueError, match=f"G4.*{message}"):
        sf.run_subhalo_forecast(
            _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
        )


def test_g4_mixed_git_revisions_require_unverified_flag(tmp_path):
    """Reject or downgrade a ladder assembled from multiple revisions."""
    paths, _ = _write_ladder(tmp_path)
    _corrupt_npz(paths[1], updates={"git_hash": np.str_("f"*40)})
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    with pytest.raises(ValueError, match="G4.*one code revision"):
        sf.run_subhalo_forecast(config)

    config["subhalo_forecast"]["allow_unverified_maps"] = True
    assert sf.run_subhalo_forecast(config).inputs_verified is False


def test_g4_rejects_nonredacted_congruence_change(tmp_path):
    """Reject a concentration change while allowing the three redactions."""
    paths, snapshots = _write_ladder(tmp_path)
    with snapshots[1].open("r", encoding="utf-8") as stream:
        changed = yaml.safe_load(stream)
    changed["lensing"]["subhalo"]["concentration"]["x_sub"] = 1.2
    with snapshots[1].open("w", encoding="utf-8") as stream:
        yaml.safe_dump(changed, stream, sort_keys=False)
    _corrupt_npz(
        paths[1],
        updates={"config_hash": np.str_(config_hash(changed))},
    )
    with pytest.raises(ValueError, match="G4.*congruence"):
        sf.run_subhalo_forecast(_fold_config(paths, (1.0e6, 2.0e6, 4.0e6)))


def test_g4_redaction_allowlist_and_verified_flag(tmp_path):
    """Allow only run name, output directory, and subhalo-mass differences."""
    paths, _ = _write_ladder(tmp_path)
    data = sf.run_subhalo_forecast(
        _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    )
    assert data.inputs_verified is True


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("cosmology", "Other", "cosmology"),
        ("lens_redshift", 0.6, "lens_redshift"),
        ("source_redshift", 1.6, "source_redshift"),
    ],
)
def test_g4_rejects_snapshot_fold_spec_inconsistency(
    tmp_path,
    key,
    value,
    message,
):
    """Reject snapshot cosmology and redshifts inconsistent with the fold."""
    paths, snapshots = _write_ladder(tmp_path)
    config = _from_f_sub_config(paths, (1.0e6, 2.0e6, 4.0e6))
    for path, map_path in zip(snapshots, paths):
        with path.open("r", encoding="utf-8") as stream:
            snapshot = yaml.safe_load(stream)
        if key == "cosmology":
            snapshot["lensing"]["cosmology"] = value
        elif key == "lens_redshift":
            snapshot["lensing"]["lens_galaxy"]["redshift"] = value
        else:
            snapshot["lensing"]["source_galaxy"]["redshift"] = value
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(snapshot, stream, sort_keys=False)
        _corrupt_npz(
            map_path,
            updates={"config_hash": np.str_(config_hash(snapshot))},
        )
    with pytest.raises(ValueError, match=f"G4.*{message}"):
        sf.run_subhalo_forecast(config)


def test_g4_rejects_partial_snapshot_presence(tmp_path):
    """Reject a ladder with only some adjacent runner snapshots."""
    paths, snapshots = _write_ladder(tmp_path)
    snapshots[0].unlink()
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6), allow_unverified=True)
    with pytest.raises(ValueError, match="G4.*partial snapshot presence"):
        sf.run_subhalo_forecast(config)


def test_g4_unverified_path_requires_flag_and_records_false(tmp_path):
    """Require permission for a wholly unverified old-format ladder."""
    paths, _ = _write_ladder(
        tmp_path,
        snapshots=False,
        embedded_hashes=False,
    )
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    with pytest.raises(ValueError, match="allow_unverified_maps"):
        sf.run_subhalo_forecast(config)

    config["subhalo_forecast"]["allow_unverified_maps"] = True
    assert sf.run_subhalo_forecast(config).inputs_verified is False


def test_g4_rejects_embedded_hashes_without_snapshots(tmp_path):
    """Reject bound maps when all adjacent snapshots are missing."""
    paths, _ = _write_ladder(
        tmp_path,
        snapshots=False,
        embedded_hashes=True,
    )
    config = _fold_config(
        paths,
        (1.0e6, 2.0e6, 4.0e6),
        allow_unverified=True,
    )
    with pytest.raises(ValueError, match="G4.*embedded config_hash.*no adjacent"):
        sf.run_subhalo_forecast(config)


def test_g4_snapshots_with_legacy_maps_require_unverified_flag(tmp_path):
    """Treat snapshots with absent embedded hashes as unverified inputs."""
    paths, _ = _write_ladder(
        tmp_path,
        snapshots=True,
        embedded_hashes=False,
    )
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    with pytest.raises(ValueError, match="G4.*without embedded hashes"):
        sf.run_subhalo_forecast(config)

    config["subhalo_forecast"]["allow_unverified_maps"] = True
    assert sf.run_subhalo_forecast(config).inputs_verified is False


@pytest.mark.parametrize("statistic", ["matched", "mismatch"])
def test_g5_reproduces_stored_threshold_products(tmp_path, statistic):
    """Reproduce the stored threshold mask and area for both statistics."""
    paths, _ = _write_ladder(tmp_path, statistic=statistic, sign_structure=True)
    config = _fold_config(
        paths,
        (1.0e6, 2.0e6, 4.0e6),
        statistic=statistic,
    )
    sf.run_subhalo_forecast(config)


def test_g5_rejects_corrupted_stored_mask(tmp_path):
    """Reject a stored threshold mask that disagrees with its q grid."""
    paths, _ = _write_ladder(tmp_path)
    _corrupt_npz(
        paths[0],
        updates={"detectable_mask_2d": np.zeros((5, 5), dtype=bool)},
    )
    with pytest.raises(ValueError, match="G5.*detectable_mask_2d"):
        sf.run_subhalo_forecast(_fold_config(paths, (1.0e6, 2.0e6, 4.0e6)))


def test_hybrid_interpolation_dense_reference_and_zero_branch(tmp_path):
    """Exercise hybrid branches, continuity, range, and dense reference."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path, masses, (0, 4, 8))
    config = _fold_config(paths, masses)
    config["subhalo_forecast"]["integration"]["samples_per_bin"] = 1024
    data = sf.run_subhalo_forecast(config)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        error_filtered_bins = sf._integrate_mass_bins(
            data.ladder_masses_msun,
            data.detectable_area_kpc2,
            data.sigma_sub_kpc2,
            data.shmf_slope,
            data.pivot_mass_msun,
            data.mhm_grid_msun,
            (4.2, 2.5, -0.2),
            1024,
        )
    np.testing.assert_array_equal(error_filtered_bins[0], data.mu_per_bin_cdm)
    np.testing.assert_array_equal(error_filtered_bins[1], data.mu_per_bin_wdm)
    assert np.all(np.isfinite(data.mu_per_bin_cdm))
    assert data.mass_range_folded_msun == masses[::2]

    first_mid = sf._interpolate_area(
        np.asarray([np.sqrt(masses[0]*masses[1])]),
        masses[0],
        masses[1],
        0.0,
        data.detectable_area_kpc2[1],
    )[0]
    assert first_mid == pytest.approx(0.5*data.detectable_area_kpc2[1])
    second_mid = sf._interpolate_area(
        np.asarray([np.sqrt(masses[1]*masses[2])]),
        masses[1],
        masses[2],
        data.detectable_area_kpc2[1],
        data.detectable_area_kpc2[2],
    )[0]
    assert second_mid == pytest.approx(
        np.sqrt(data.detectable_area_kpc2[1]*data.detectable_area_kpc2[2])
    )
    for index in range(1):
        boundary = masses[index + 1]
        left = sf._interpolate_area(
            np.asarray([boundary]),
            masses[index],
            masses[index + 1],
            data.detectable_area_kpc2[index],
            data.detectable_area_kpc2[index + 1],
        )[0]
        right = sf._interpolate_area(
            np.asarray([boundary]),
            masses[index + 1],
            masses[min(index + 2, 2)],
            data.detectable_area_kpc2[index + 1],
            data.detectable_area_kpc2[min(index + 2, 2)],
        )[0]
        assert left == pytest.approx(right, rel=1.0e-12, abs=0.0)

    reference = sf._integrate_mass_bins(
        np.asarray(masses),
        data.detectable_area_kpc2,
        data.sigma_sub_kpc2,
        data.shmf_slope,
        data.pivot_mass_msun,
        np.asarray([0.0]),
        (4.2, 2.5, -0.2),
        102400,
    )[0]
    assert data.mu_cdm == pytest.approx(np.sum(reference), rel=1.0e-6)


def test_statistic_routing_applies_one_sided_mismatch_gate(tmp_path):
    """Fold mismatch q with amplitude sign while matched ignores siblings."""
    paths, _ = _write_ladder(
        tmp_path / "mismatch",
        detectable_nodes=(3, 3, 3),
        statistic="mismatch",
        sign_structure=True,
    )
    mismatch = sf.run_subhalo_forecast(
        _fold_config(paths, (1.0e6, 2.0e6, 4.0e6), statistic="mismatch")
    )
    assert np.allclose(mismatch.detectable_area_arcsec2, 0.02)

    matched = sf.run_subhalo_forecast(
        _fold_config(paths, (1.0e6, 2.0e6, 4.0e6), statistic="matched")
    )
    assert np.allclose(matched.detectable_area_arcsec2, 0.03)


# T11-T15: artifacts, imports, figures, robustness, and CLI.


def test_artifact_roundtrip_member_set_identity_and_manifest(tmp_path):
    """Round-trip exactly, reject member drift, and verify input digests."""
    paths, _ = _write_ladder(tmp_path / "inputs")
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    first = sf.run_subhalo_forecast(config)
    artifact = sf.save_subhalo_forecast_npz(first, tmp_path / "forecast.npz")
    loaded = sf.load_subhalo_forecast_npz(artifact)
    _assert_forecasts_equal(first, loaded)

    with np.load(artifact, allow_pickle=False) as stored:
        assert set(stored.files) == sf._BASE_ARTIFACT_MEMBERS
        assert "content_digest" in stored.files
    for path, entry in zip(paths, first.map_manifest):
        assert entry["sha256"] == hashlib.sha256(Path(path).read_bytes()).hexdigest()
        grid = load_fisher_grid_map_npz(path)
        assert entry["q_grid_digest"] == sf._q_grid_digest(grid.q_asimov_2d)

    extra = _corrupt_npz(
        artifact,
        tmp_path / "extra.npz",
        updates={"extra": np.asarray(1)},
    )
    with pytest.raises(ValueError, match="unexpected.*extra"):
        sf.load_subhalo_forecast_npz(extra)
    missing = _corrupt_npz(
        artifact,
        tmp_path / "missing.npz",
        delete=("mu_wdm",),
    )
    with pytest.raises(ValueError, match="missing.*mu_wdm"):
        sf.load_subhalo_forecast_npz(missing)
    tampered = _corrupt_npz(
        artifact,
        tmp_path / "tampered.npz",
        updates={"detection_q_threshold": np.asarray(11.0)},
        recompute_content_digest=True,
    )
    with pytest.raises(ValueError, match="forecast_id mismatch"):
        sf.load_subhalo_forecast_npz(tampered)


def test_artifact_content_digest_rejects_payload_tampering(tmp_path):
    """Reject forged science output, verification, and robustness members."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path / "verified")
    data = sf.run_subhalo_forecast(_fold_config(paths, masses))
    artifact = sf.save_subhalo_forecast_npz(data, tmp_path / "forecast.npz")
    doubled = _corrupt_npz(
        artifact,
        tmp_path / "doubled.npz",
        updates={"mu_wdm": 2.0*data.mu_wdm},
    )
    with pytest.raises(ValueError, match="content digest mismatch"):
        sf.load_subhalo_forecast_npz(doubled)

    unverified_paths, _ = _write_ladder(
        tmp_path / "unverified",
        snapshots=False,
        embedded_hashes=False,
    )
    unverified_config = _fold_config(
        unverified_paths,
        masses,
        allow_unverified=True,
    )
    unverified = sf.run_subhalo_forecast(unverified_config)
    assert unverified.inputs_verified is False
    unverified_artifact = sf.save_subhalo_forecast_npz(
        unverified,
        tmp_path / "unverified.npz",
    )
    forged_verified = _corrupt_npz(
        unverified_artifact,
        tmp_path / "forged-verified.npz",
        updates={"inputs_verified": np.asarray(True, dtype=np.bool_)},
    )
    with pytest.raises(ValueError, match="content digest mismatch"):
        sf.load_subhalo_forecast_npz(forged_verified)

    shifted_config = _fold_config(paths, masses)
    shifted_config["subhalo_forecast"]["robustness"][
        "mass_axis_shift_dex"
    ] = 0.25
    shifted = sf.run_subhalo_forecast(shifted_config)
    shifted_artifact = sf.save_subhalo_forecast_npz(
        shifted,
        tmp_path / "shifted.npz",
    )
    stripped = _corrupt_npz(
        shifted_artifact,
        tmp_path / "stripped.npz",
        updates={"robustness_present": np.asarray(False, dtype=np.bool_)},
        delete=sf._ROBUSTNESS_FIELDS,
    )
    with pytest.raises(ValueError):
        sf.load_subhalo_forecast_npz(stripped)


def test_content_digest_recipe_matches_independent_oracle():
    """Pin the digest to sorted names with dtype, shape, and raw bytes."""
    payload = {
        "b_values": np.asarray([1.0, 2.0]),
        "a_flag": np.asarray(True),
    }
    oracle = hashlib.sha256()
    for name in ("a_flag", "b_values"):
        value = np.asarray(payload[name])
        oracle.update(name.encode())
        oracle.update(b":")
        oracle.update(value.dtype.str.encode())
        oracle.update(b":")
        oracle.update(str(value.shape).encode())
        oracle.update(b":")
        oracle.update(np.ascontiguousarray(value).tobytes())
    assert sf._content_digest(payload) == oracle.hexdigest()

    reordered = dict(reversed(list(payload.items())))
    assert sf._content_digest(reordered) == sf._content_digest(payload)
    assert sf._content_digest(
        {"m": np.zeros(2, dtype=np.int64)}
    ) != sf._content_digest({"m": np.zeros(2, dtype=np.uint64)})
    assert sf._content_digest(
        {"m": np.zeros(4, dtype=np.float64)}
    ) != sf._content_digest({"m": np.zeros((2, 2), dtype=np.float64)})


def test_artifact_rejects_malformed_json_member_by_name(tmp_path):
    """Name the offending member when stored JSON fails to parse."""
    paths, _ = _write_ladder(tmp_path / "inputs")
    data = sf.run_subhalo_forecast(
        _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    )
    artifact = sf.save_subhalo_forecast_npz(data, tmp_path / "forecast.npz")
    corrupted = _corrupt_npz(
        artifact,
        tmp_path / "malformed.npz",
        updates={"config_json": np.asarray("{")},
        recompute_content_digest=True,
    )
    with pytest.raises(ValueError, match="config_json is invalid"):
        sf.load_subhalo_forecast_npz(corrupted)


@pytest.mark.parametrize(
    "member",
    [
        "normalization_preset_json",
        "from_f_sub_json",
        "map_manifest_json",
        "source_digests_json",
        "revision_provenance_json",
        "config_json",
    ],
)
def test_artifact_requires_canonical_raw_json(tmp_path, member):
    """Reject non-canonical JSON text even when its content is unchanged."""
    paths, _ = _write_ladder(tmp_path / "inputs")
    data = sf.run_subhalo_forecast(
        _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    )
    artifact = sf.save_subhalo_forecast_npz(data, tmp_path / "forecast.npz")
    with np.load(artifact, allow_pickle=False) as stored:
        raw = str(stored[member])
    assert json.loads(" " + raw) == json.loads(raw)
    noncanonical = " " + raw
    corrupted = _corrupt_npz(
        artifact,
        tmp_path / f"noncanonical-{member}.npz",
        updates={member: np.asarray(noncanonical)},
        recompute_content_digest=True,
    )
    with pytest.raises(ValueError, match=member):
        sf.load_subhalo_forecast_npz(corrupted)


def test_artifact_robustness_presence_follows_config(tmp_path):
    """Require robustness members exactly when the shift is positive."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path / "inputs")
    zero = sf.run_subhalo_forecast(_fold_config(paths, masses))
    zero_path = sf.save_subhalo_forecast_npz(zero, tmp_path / "zero.npz")

    shifted_config = _fold_config(paths, masses)
    shifted_config["subhalo_forecast"]["robustness"][
        "mass_axis_shift_dex"
    ] = 0.25
    shifted = sf.run_subhalo_forecast(shifted_config)
    shifted_path = sf.save_subhalo_forecast_npz(
        shifted,
        tmp_path / "shifted.npz",
    )
    missing = _corrupt_npz(
        shifted_path,
        tmp_path / "missing-robustness.npz",
        updates={"robustness_present": np.asarray(False, dtype=np.bool_)},
        delete=sf._ROBUSTNESS_FIELDS,
        recompute_content_digest=True,
    )
    with pytest.raises(ValueError, match="robustness block inconsistent"):
        sf.load_subhalo_forecast_npz(missing)

    with np.load(shifted_path, allow_pickle=False) as stored:
        robustness_members = {
            name: np.array(stored[name], copy=True)
            for name in sf._ROBUSTNESS_FIELDS
        }
    unexpected = _corrupt_npz(
        zero_path,
        tmp_path / "unexpected-robustness.npz",
        updates={
            "robustness_present": np.asarray(True, dtype=np.bool_),
            **robustness_members,
        },
        recompute_content_digest=True,
    )
    with pytest.raises(ValueError, match="robustness block inconsistent"):
        sf.load_subhalo_forecast_npz(unexpected)

    invalid_dtype = _corrupt_npz(
        zero_path,
        tmp_path / "invalid-robustness-dtype.npz",
        updates={"robustness_present": np.asarray(0, dtype=np.int64)},
        recompute_content_digest=True,
    )
    with pytest.raises(ValueError, match="robustness_present.*boolean dtype"):
        sf.load_subhalo_forecast_npz(invalid_dtype)


def test_forecast_identity_excludes_relocated_map_paths(tmp_path):
    """Keep the forecast identity stable when the same run dirs move."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path / "original")
    original = sf.run_subhalo_forecast(_fold_config(paths, masses))
    relocated_root = tmp_path / "relocated"
    relocated_paths = []
    for index, path in enumerate(paths):
        copied_run = relocated_root / f"run-{index}"
        shutil.copytree(path.parent.parent, copied_run)
        relocated_paths.append(copied_run / "modeling" / path.name)
    relocated = sf.run_subhalo_forecast(
        _fold_config(relocated_paths, masses)
    )
    assert relocated.forecast_id == original.forecast_id


def test_double_run_is_byte_identical_and_robustness_members_are_exact(tmp_path):
    """Produce deterministic bytes and the exact enabled robustness set."""
    paths, _ = _write_ladder(tmp_path / "inputs")
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    first = sf.run_subhalo_forecast(config)
    second = sf.run_subhalo_forecast(config)
    first_path = sf.save_subhalo_forecast_npz(first, tmp_path / "first.npz")
    second_path = sf.save_subhalo_forecast_npz(second, tmp_path / "second.npz")
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.robustness is None

    config["subhalo_forecast"]["robustness"]["mass_axis_shift_dex"] = 0.25
    shifted = sf.run_subhalo_forecast(config)
    assert set(shifted.robustness) == set(sf._ROBUSTNESS_FIELDS)
    shifted_path = sf.save_subhalo_forecast_npz(shifted, tmp_path / "shifted.npz")
    with np.load(shifted_path, allow_pickle=False) as stored:
        assert set(stored.files) == sf._BASE_ARTIFACT_MEMBERS | set(
            sf._ROBUSTNESS_FIELDS
        )
    _assert_forecasts_equal(shifted, sf.load_subhalo_forecast_npz(shifted_path))


@pytest.mark.parametrize(
    "statement",
    ["import hwoslaps.analysis", "import hwoslaps.analysis.subhalo_forecast"],
)
def test_analysis_imports_stay_light_in_fresh_subprocess(statement):
    """Keep both analysis import forms free of plotting and engine packages."""
    code = (
        "import sys; "
        f"{statement}; "
        "assert 'matplotlib' not in sys.modules; "
        "assert 'autolens' not in sys.modules; "
        "assert 'hcipy' not in sys.modules"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_forecast_figures_write_nonempty_files(tmp_path):
    """Write both forecast figures through the noninteractive backend."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    paths, _ = _write_ladder(tmp_path / "inputs")
    data = sf.run_subhalo_forecast(
        _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    )
    from hwoslaps.plotting.subhalo_forecast import (
        plot_expected_detections_vs_mhm,
        plot_lenses_to_discriminate,
    )

    outputs = (
        plot_expected_detections_vs_mhm(data, tmp_path / "mu.png"),
        plot_lenses_to_discriminate(data, tmp_path / "nreq.png"),
    )
    assert all(path.stat().st_size > 0 for path in outputs)


def test_robustness_shift_zero_and_closed_form_mass_scaling(tmp_path):
    """Disable zero shift and match constant-area mass relabeling exactly."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path, masses, (4, 4, 4))
    config = _fold_config(paths, masses)
    assert sf.run_subhalo_forecast(config).robustness is None

    shift = 0.25
    config["subhalo_forecast"]["robustness"]["mass_axis_shift_dex"] = shift
    data = sf.run_subhalo_forecast(config)
    expected_plus = 10.0**(shift*(data.shmf_slope + 1.0))
    expected_minus = 10.0**(-shift*(data.shmf_slope + 1.0))
    assert data.robustness["mu_cdm_shift_plus"] == pytest.approx(
        data.mu_cdm*expected_plus,
        rel=1.0e-10,
    )
    assert data.robustness["mu_cdm_shift_minus"] == pytest.approx(
        data.mu_cdm*expected_minus,
        rel=1.0e-10,
    )
    # Downward relabeling raises n at each rung by 10**(+1.9d), while
    # shrinking dm by 10**(-d); the net constant-area fold rises by 10**(0.9d).
    assert data.robustness["mu_cdm_shift_minus"] > data.mu_cdm
    assert data.robustness["mu_cdm_shift_plus"] < data.mu_cdm


def _run_cli(spec_path, output_dir, *extra):
    """Run the forecast CLI in a fresh subprocess."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC_ROOT)
    environment["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_subhalo_forecast.py"),
            str(spec_path),
            str(output_dir),
            *extra,
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_cli_success_refusal_force_and_invalid_inputs(tmp_path):
    """Exercise all CLI outputs, overwrite policy, and validation failures."""
    paths, _ = _write_ladder(tmp_path / "inputs")
    config = _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    spec_path = tmp_path / "fold.yaml"
    with spec_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    output_dir = tmp_path / "output"

    completed = _run_cli(spec_path, output_dir)
    assert completed.returncode == 0, completed.stderr
    expected = {
        "subhalo_forecast.npz",
        "expected_detections_vs_mhm.png",
        "lenses_to_discriminate.png",
        "provenance.yaml",
    }
    assert expected <= {path.name for path in output_dir.iterdir()}
    with (output_dir / "provenance.yaml").open("r", encoding="utf-8") as stream:
        provenance = yaml.safe_load(stream)
    assert provenance["command"][0].endswith("run_subhalo_forecast.py")

    refused = _run_cli(spec_path, output_dir)
    assert refused.returncode != 0
    assert "--force" in refused.stderr
    (output_dir / "subhalo_forecast.npz").write_bytes(b"corrupt")
    forced = _run_cli(spec_path, output_dir, "--force")
    assert forced.returncode == 0, forced.stderr
    sf.load_subhalo_forecast_npz(output_dir / "subhalo_forecast.npz")

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("subhalo_forecast: [", encoding="utf-8")
    malformed = _run_cli(invalid_yaml, tmp_path / "bad-yaml")
    assert malformed.returncode != 0
    assert "expected" in malformed.stderr.lower()

    invalid_config = tmp_path / "invalid-config.yaml"
    invalid = copy.deepcopy(config)
    invalid["subhalo_forecast"]["detection_q_threshold"] = 0.0
    with invalid_config.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(invalid, stream)
    rejected = _run_cli(invalid_config, tmp_path / "bad-config")
    assert rejected.returncode != 0
    assert "detection_q_threshold must be positive" in rejected.stderr


def test_forecast_ratio_compatibility_and_inf_awareness(tmp_path):
    """Return self ratios of one and reject incompatible forecast axes."""
    paths, _ = _write_ladder(tmp_path)
    data = sf.run_subhalo_forecast(
        _fold_config(paths, (1.0e6, 2.0e6, 4.0e6))
    )
    ratio = sf.forecast_ratio(data, data)
    np.testing.assert_array_equal(ratio["mu_ratio"], np.ones_like(data.mu_wdm))
    np.testing.assert_array_equal(
        ratio["n_req_ratio"],
        np.ones_like(data.N_req),
    )
    assert ratio["mu_cdm_ratio"] == 1.0

    incompatible = replace(data, detection_q_threshold=11.0)
    with pytest.raises(ValueError, match="detection threshold"):
        sf.forecast_ratio(incompatible, data)


@pytest.mark.parametrize(
    "path,value",
    [
        (
            "subhalo_forecast.discrimination.delta_logl_threshold",
            7.0,
        ),
        ("subhalo_forecast.integration.samples_per_bin", 64),
        ("subhalo_forecast.lens_plane.lens_redshift", 0.6),
    ],
)
def test_forecast_ratio_requires_identical_fold_settings(
    tmp_path,
    path,
    value,
):
    """Reject ratios that differ in any output-defining fold setting."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(
        tmp_path,
        snapshots=False,
        embedded_hashes=False,
    )
    baseline_config = _fold_config(
        paths,
        masses,
        allow_unverified=True,
    )
    numerator_config = copy.deepcopy(baseline_config)
    _set_path(numerator_config, path, value)
    baseline = sf.run_subhalo_forecast(baseline_config)
    numerator = sf.run_subhalo_forecast(numerator_config)

    with pytest.raises(ValueError, match="identical fold settings"):
        sf.forecast_ratio(numerator, baseline)


def test_forecast_ratio_pins_infinity_zero_and_nan_branches(tmp_path):
    """Pin every finite, infinite, zero, and undefined ratio branch."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path)
    config = _fold_config(paths, masses)
    config["subhalo_forecast"]["wdm"]["half_mode_mass_grid"]["num"] = 6
    data = sf.run_subhalo_forecast(config)
    baseline = replace(
        data,
        N_req=np.asarray([2.0, np.inf, np.inf, 4.0, 0.0, 0.0]),
        mu_wdm=np.asarray([1.0, 0.0, 2.0, 3.0, 4.0, 5.0]),
        mu_cdm=0.0,
    )
    numerator = replace(
        data,
        N_req=np.asarray([6.0, np.inf, 5.0, np.inf, 0.0, 3.0]),
        mu_wdm=np.asarray([2.0, 7.0, 1.0, 6.0, 8.0, 10.0]),
        mu_cdm=1.0,
    )

    ratio = sf.forecast_ratio(numerator, baseline)
    np.testing.assert_array_equal(
        ratio["n_req_ratio"],
        np.asarray([3.0, 1.0, 0.0, np.inf, 1.0, np.inf]),
    )
    assert np.isnan(ratio["mu_ratio"][1])
    np.testing.assert_array_equal(
        ratio["mu_ratio"][[0, 2, 3, 4, 5]],
        np.asarray([2.0, 0.5, 2.0, 2.0, 2.0]),
    )
    assert np.isnan(ratio["mu_cdm_ratio"])
    directional = sf.forecast_ratio(
        replace(data, mu_cdm=1.0),
        replace(data, mu_cdm=4.0),
    )
    assert directional["mu_cdm_ratio"] == 0.25


def test_forecast_ratio_ignores_map_paths_and_unverified_policy(tmp_path):
    """Allow ratios across relocated maps and differing verification policy."""
    masses = (1.0e6, 2.0e6, 4.0e6)
    paths, _ = _write_ladder(tmp_path / "original")
    baseline = sf.run_subhalo_forecast(_fold_config(paths, masses))
    relocated_root = tmp_path / "relocated"
    relocated_paths = []
    for index, path in enumerate(paths):
        copied_run = relocated_root / f"run-{index}"
        shutil.copytree(path.parent.parent, copied_run)
        relocated_paths.append(copied_run / "modeling" / path.name)
    numerator = sf.run_subhalo_forecast(
        _fold_config(relocated_paths, masses, allow_unverified=True)
    )
    numerator_config_before = copy.deepcopy(numerator.config)
    baseline_config_before = copy.deepcopy(baseline.config)

    ratio = sf.forecast_ratio(numerator, baseline)
    np.testing.assert_array_equal(
        ratio["n_req_ratio"],
        np.ones_like(baseline.N_req),
    )
    assert numerator.config == numerator_config_before
    assert baseline.config == baseline_config_before
