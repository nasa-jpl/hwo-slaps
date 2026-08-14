"""Tests for nonlinear and Fisher PSF-mismatch fitting."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("autolens")
pytest.importorskip("hcipy")

from hwoslaps.config.validation import validate_or_raise
from hwoslaps.lensing import generate_lensing_system
from hwoslaps.modeling.fisher_detector import FisherDetector
from hwoslaps.modeling.generator_fisher import perform_fisher_detection
from hwoslaps.modeling.nonlinear.autolens_runner import (
    _array_hash,
    analysis_key_from,
)
from hwoslaps.modeling.nonlinear.dataset_builder import (
    NonlinearDatasetMetadata,
    fitted_kernel_sha256,
    imaging_from_observation,
)
from hwoslaps.modeling.nonlinear.psf_mismatch import (
    run_psf_mismatch_case,
)
from hwoslaps.modeling.nonlinear.psf_bank import (
    build_psf_bank,
    run_psf_bank_case,
)
from hwoslaps.modeling.nonlinear.validator import NonlinearMetricValidator
from hwoslaps.modeling.utils_fisher import print_fisher_summary
from hwoslaps.observation import generate_observation
from hwoslaps.psf.aberration_models import (
    apply_global_zernikes,
    apply_segment_zernikes,
)
from hwoslaps.psf.generator import generate_psf_system
from hwoslaps.psf.mismatch import (
    _aberrations_from_wire,
    _canonical_aberrations,
    _flat_int_map_from_wire,
    _kernel_sha256,
    _nested_int_map_from_wire,
    build_psf_mismatch_spec,
    generate_fit_psf,
)
from hwoslaps.psf.utils import (
    make_pyauto_convolver,
    make_pyauto_kernel,
    pyauto_kernel_native,
    pyauto_kernel_pixel_scales,
)
from hwoslaps.provenance import config_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def prior_table(tmp_path) -> Path:
    """Write a tiny two-sided mode-weight prior table."""
    path = tmp_path / "prior.yaml"
    path.write_text(
        yaml.safe_dump({
            "name": "tiny",
            "segment_variance_fraction": 0.4,
            "global_weights": {4: 1.0, 5: 0.5},
            "segment_weights": {1: 1.0, 2: 0.5},
            "metadata": {"basis_convention": "test"},
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def compact_config(prior_table, tmp_path) -> dict:
    """Load a compact scene with a nonzero truth PSF and delta fit."""
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    config["run_name"] = "item9-test"
    config["plotting"] = {
        "enabled": False,
        "output_dir": str(tmp_path),
    }
    config["lensing"]["grid"] = {"shape": [15, 15], "pixel_scale": 0.1}
    config["lensing"]["lens_galaxy"]["mass"] = {
        "type": "Isothermal",
        "centre": [0.0, 0.0],
        "ell_comps": [0.05, -0.02],
        "einstein_radius": 0.5,
    }
    config["lensing"]["lens_galaxy"].pop("shear", None)
    config["lensing"]["source_galaxy"]["light"].update({
        "centre": [0.02, 0.03],
        "ell_comps": [0.03, -0.01],
        "intensity": 4.0,
        "effective_radius": 0.16,
    })
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "model": "PointMass",
        "mass": 1.0e8,
        "position": {"type": "direct", "centre": [0.1, 0.0]},
    }
    config["psf"]["telescope"]["num_rings"] = 1
    config["psf"]["telescope"]["supersampling_factor"] = 1
    config["psf"]["hres_psf"].update({
        "num_pix": 64,
        "num_airy": 4,
        "sampling": 5,
        "save_highres_psf_npy": False,
    })
    config["psf"]["kernel"]["shape_native"] = [7, 7]
    config["psf"]["aberrations"] = {
        "enable_segment_pistons": True,
        "enable_segment_tiptilts": True,
        "enable_segment_hexikes": False,
        "enable_global_zernikes": True,
        "segment_pistons": {0: 0.5, 1: -0.5},
        "segment_tiptilts": {0: [0.01, -0.02]},
        "segment_hexikes": {0: {1: 3.0, 2: -1.0}},
        "global_zernikes": {4: 3.0, 6: -1.0},
    }
    config["modeling"]["fit_psf"] = {
        "mode": "delta",
        "delta": {
            "prior_table": str(prior_table),
            "amplitude_rms_nm": 5.0,
            "seed": 20260814,
            "family": "combined",
        },
    }
    config["modeling"]["fisher"].update({
        "mode": "local",
        "mask_mode": "all_pixels",
        "include_background_offset": False,
        "include_psf_nuisance": False,
        "compute_psf_mode_scan": False,
        "psf_basis": {"global_zernikes": {"mode_nolls": [4]}},
        "map": {
            "type": "grid",
            "grid": {
                "spacing_arcsec": 0.1,
                "half_width_arcsec": 0.1,
                "annulus": None,
            },
            "detection_q_threshold": 10.0,
            "num_workers": 1,
            "engine": "reference",
        },
    })
    validate_or_raise(config)
    return config


def _quiet_call(function, *args, **kwargs):
    """Call one optical constructor without progress output."""
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            return function(*args, **kwargs)


def _build(config):
    """Build one mismatch specification quietly."""
    return _quiet_call(build_psf_mismatch_spec, config)


def _psf(config, psf_config=None):
    """Generate one compact PSF quietly."""
    selected = config["psf"] if psf_config is None else psf_config
    return _quiet_call(
        generate_psf_system,
        copy.deepcopy(selected),
        full_config=config,
    )


def _truth_as_applied(aberrations: dict) -> dict:
    """Return the coefficient maps enabled by generator semantics."""
    applied = copy.deepcopy(aberrations)
    for flag, map_name in (
        ("enable_segment_pistons", "segment_pistons"),
        ("enable_segment_tiptilts", "segment_tiptilts"),
        ("enable_segment_hexikes", "segment_hexikes"),
        ("enable_global_zernikes", "global_zernikes"),
    ):
        if not applied[flag]:
            applied[map_name] = {}
    return applied


def _subtract_flat(left: dict, right: dict) -> dict:
    """Subtract scalar maps over their key union."""
    return {
        key: float(left.get(key, 0.0)) - float(right.get(key, 0.0))
        for key in set(left) | set(right)
    }


def _subtract_nested(left: dict, right: dict) -> dict:
    """Subtract nested scalar maps over both key unions."""
    return {
        segment: _subtract_flat(
            left.get(segment, {}),
            right.get(segment, {}),
        )
        for segment in set(left) | set(right)
    }


def _strip_untouched_flat(difference: dict, draw: dict) -> dict:
    """Drop exact-zero entries for modes absent from the draw."""
    return {
        key: value
        for key, value in difference.items()
        if key in draw or value != 0.0
    }


def _strip_untouched_nested(difference: dict, draw: dict) -> dict:
    """Drop exact-zero nested entries for modes absent from the draw."""
    stripped = {}
    for segment, modes in difference.items():
        kept = _strip_untouched_flat(modes, draw.get(segment, {}))
        if kept or segment in draw:
            stripped[segment] = kept
    return stripped


def _phase_screen_rms_nm(telescope_data, segment_maps, global_map) -> float:
    """Measure piston-removed RMS from the applied phase-screen difference."""
    wavelength = telescope_data["wavelength"]
    phase = np.zeros_like(np.asarray(telescope_data["pupil_grid"].zeros()))
    if segment_maps:
        segment_phase, _ = apply_segment_zernikes(
            segment_maps,
            telescope_data,
            wavelength,
        )
        phase += np.asarray(segment_phase)
    if global_map:
        phase += np.asarray(apply_global_zernikes(
            global_map,
            telescope_data,
            wavelength,
        ))
    valid = np.asarray(telescope_data["aper"]) > 0.5
    opd = phase[valid] * wavelength / (2.0*np.pi)
    opd -= np.mean(opd)
    return float(np.sqrt(np.mean(opd**2))*1.0e9)


@pytest.mark.parametrize("family", ["combined", "global", "segment"])
def test_additive_fit_minus_truth_maps_equal_draw(compact_config, family):
    """Add the realized draw to truth-as-applied mode by mode."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["delta"]["family"] = family
    spec = _build(config)
    truth = _truth_as_applied(config["psf"]["aberrations"])
    fit = spec.fit_psf_config["aberrations"]
    draw = spec.draw_aberrations

    assert draw is not None
    segment_difference = _strip_untouched_nested(
        _subtract_nested(
            fit["segment_hexikes"],
            truth["segment_hexikes"],
        ),
        draw["segment_hexikes"],
    )
    assert set(segment_difference) == set(draw["segment_hexikes"])
    for segment, modes in draw["segment_hexikes"].items():
        assert segment_difference[segment] == pytest.approx(
            modes, rel=1.0e-9, abs=1.0e-12
        )
    global_difference = _strip_untouched_flat(
        _subtract_flat(
            fit["global_zernikes"],
            truth["global_zernikes"],
        ),
        draw["global_zernikes"],
    )
    assert global_difference == pytest.approx(
        draw["global_zernikes"], rel=1.0e-9, abs=1.0e-12
    )
    assert fit["segment_pistons"] == truth["segment_pistons"]
    assert fit["segment_tiptilts"] == truth["segment_tiptilts"]
    assert fit["enable_segment_pistons"] is True
    assert fit["enable_segment_tiptilts"] is True
    assert fit["enable_segment_hexikes"] is bool(
        fit["segment_hexikes"]
    )
    assert fit["enable_global_zernikes"] is bool(
        fit["global_zernikes"]
    )
    if family == "global":
        assert fit["segment_hexikes"] == {}
    if family == "segment":
        assert fit["global_zernikes"] == truth["global_zernikes"]


@pytest.mark.parametrize(
    "family,amplitude",
    [
        ("combined", 5.0),
        ("global", 5.0),
        ("segment", 5.0),
        ("combined", 0.0),
    ],
)
def test_fit_minus_truth_applied_opd_has_requested_rms(
    compact_config,
    family,
    amplitude,
):
    """Measure the requested RMS on the applied fit-minus-truth OPD."""
    config = copy.deepcopy(compact_config)
    delta = config["modeling"]["fit_psf"]["delta"]
    delta["family"] = family
    delta["amplitude_rms_nm"] = amplitude
    spec = _build(config)
    telescope_data = _psf(config).telescope_data
    truth = _truth_as_applied(config["psf"]["aberrations"])
    fit = _truth_as_applied(spec.fit_psf_config["aberrations"])
    segment_difference = _subtract_nested(
        fit["segment_hexikes"],
        truth["segment_hexikes"],
    )
    global_difference = _subtract_flat(
        fit["global_zernikes"],
        truth["global_zernikes"],
    )
    measured = _phase_screen_rms_nm(
        telescope_data,
        segment_difference,
        global_difference,
    )

    assert measured == pytest.approx(amplitude, rel=1.0e-9, abs=1.0e-9)
    if amplitude == 0.0:
        assert measured == 0.0


@pytest.mark.parametrize("family", ["combined", "global", "segment"])
def test_measured_draw_rms_matches_requested_amplitude(compact_config, family):
    """Record the exact physical RMS for every supported draw family."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["delta"]["family"] = family
    spec = _build(config)

    assert spec.measured_draw_rms_nm == pytest.approx(
        spec.requested_amplitude_rms_nm,
        rel=1.0e-9,
        abs=1.0e-9,
    )


def test_corrupted_realized_draw_raises_exact_rms_error(
    compact_config,
    monkeypatch,
):
    """Reject realized maps that violate the exact-RMS contract."""
    import hwoslaps.psf.families as families

    original = families.realize_weighted_draw

    def scaled_realize(*args, **kwargs):
        segment, global_map = original(*args, **kwargs)
        return (
            {
                segment_id: {
                    mode: 0.5*value for mode, value in modes.items()
                }
                for segment_id, modes in segment.items()
            },
            {mode: 0.5*value for mode, value in global_map.items()},
        )

    monkeypatch.setattr(families, "realize_weighted_draw", scaled_realize)
    with pytest.raises(ValueError, match="measured.*amplitude"):
        _build(compact_config)


def test_effective_delta_rejects_catastrophic_truth_addition(compact_config):
    """Reject a requested delta erased by float64 truth coefficients."""
    config = copy.deepcopy(compact_config)
    config["psf"]["aberrations"]["global_zernikes"] = {4: 2.0**53}
    config["modeling"]["fit_psf"]["delta"].update({
        "amplitude_rms_nm": 1.0,
        "family": "global",
    })

    with pytest.raises(
        ValueError,
        match="floating-point addition.*destroyed the requested delta",
    ):
        _build(config)


def test_delta_identity_is_deterministic_canonical_and_colon_labeled(
    compact_config,
):
    """Canonicalize numeric spellings and use the colon label format."""
    first = _build(compact_config)
    reordered = copy.deepcopy(compact_config)
    reordered["psf"]["aberrations"]["global_zernikes"] = {
        6: -1,
        4: 3,
    }
    reordered["modeling"]["fit_psf"]["delta"][
        "amplitude_rms_nm"
    ] = 5
    second = _build(reordered)

    assert first.delta_id == second.delta_id
    assert len(first.delta_id) == 16
    assert int(first.delta_id, 16) >= 0
    assert f"{first.mode}:{first.delta_id}" == f"delta:{first.delta_id}"

    negative_zero = copy.deepcopy(compact_config)
    negative_zero["modeling"]["fit_psf"]["delta"][
        "amplitude_rms_nm"
    ] = -0.0
    positive_zero = copy.deepcopy(negative_zero)
    positive_zero["modeling"]["fit_psf"]["delta"][
        "amplitude_rms_nm"
    ] = 0.0
    assert _build(negative_zero).delta_id == _build(positive_zero).delta_id


def test_delta_identity_preserves_large_integer_seed_bits(compact_config):
    """Keep distinct valid seeds above float64's exact integer range."""
    first_config = copy.deepcopy(compact_config)
    second_config = copy.deepcopy(compact_config)
    first_config["modeling"]["fit_psf"]["delta"]["seed"] = 2**53
    second_config["modeling"]["fit_psf"]["delta"]["seed"] = 2**53 + 1
    first = _build(first_config)
    second = _build(second_config)

    assert first.delta_id != second.delta_id
    assert first.draw_aberrations != second.draw_aberrations


def test_delta_identity_changes_with_every_generation_input(
    compact_config,
    tmp_path,
):
    """Bind identity to content, amplitude, seed, family, truth, scale."""
    base = _build(compact_config)
    variants = []
    amplitude = copy.deepcopy(compact_config)
    amplitude["modeling"]["fit_psf"]["delta"]["amplitude_rms_nm"] = 6.0
    variants.append(amplitude)
    seed = copy.deepcopy(compact_config)
    seed["modeling"]["fit_psf"]["delta"]["seed"] += 1
    variants.append(seed)
    family = copy.deepcopy(compact_config)
    family["modeling"]["fit_psf"]["delta"]["family"] = "global"
    variants.append(family)
    truth = copy.deepcopy(compact_config)
    truth["psf"]["aberrations"]["global_zernikes"][4] += 1.0
    variants.append(truth)
    scale = copy.deepcopy(compact_config)
    scale["lensing"]["grid"]["pixel_scale"] = 0.11
    variants.append(scale)
    changed_table = tmp_path / "changed.yaml"
    changed_table.write_text(
        yaml.safe_dump({
            "name": "tiny",
            "segment_variance_fraction": 0.4,
            "global_weights": {4: 1.0, 5: 0.75},
            "segment_weights": {1: 1.0, 2: 0.5},
        }),
        encoding="utf-8",
    )
    table = copy.deepcopy(compact_config)
    table["modeling"]["fit_psf"]["delta"]["prior_table"] = str(
        changed_table
    )
    variants.append(table)

    variant_ids = {_build(config).delta_id for config in variants}
    assert base.delta_id not in variant_ids
    assert len(variant_ids) == len(variants)


def test_delta_identity_uses_prior_content_not_path(compact_config, tmp_path):
    """Keep delta identity stable when identical prior bytes move."""
    original = Path(
        compact_config["modeling"]["fit_psf"]["delta"]["prior_table"]
    )
    copied = tmp_path / "same-content.yaml"
    copied.write_bytes(original.read_bytes())
    moved = copy.deepcopy(compact_config)
    moved["modeling"]["fit_psf"]["delta"]["prior_table"] = str(copied)

    assert _build(moved).delta_id == _build(compact_config).delta_id


def test_delta_zero_still_requires_resolvable_prior_table(compact_config):
    """Keep zero-amplitude identity total by resolving its prior table."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["delta"].update({
        "amplitude_rms_nm": 0.0,
        "prior_table": "missing-item9-prior.yaml",
    })

    with pytest.raises(FileNotFoundError, match="missing-item9-prior.yaml"):
        _build(config)


def test_mismatch_builder_rejects_modes_outside_delta_and_explicit(
    compact_config,
):
    """Name both accepted builder modes in unsupported-mode errors."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"] = {"mode": "matched"}

    with pytest.raises(ValueError, match="'delta'.*'explicit'"):
        _build(config)


def test_explicit_identity_changes_with_fit_psf_config(compact_config):
    """Bind explicit identity to canonical truth and fit PSF configurations."""
    config = copy.deepcopy(compact_config)
    fit = copy.deepcopy(config["psf"])
    fit["aberrations"]["global_zernikes"][4] = 7
    config["modeling"]["fit_psf"] = {"mode": "explicit", "psf": fit}
    first = _build(config)
    same = copy.deepcopy(config)
    same["modeling"]["fit_psf"]["psf"]["aberrations"][
        "global_zernikes"
    ][4] = 7.0
    changed = copy.deepcopy(config)
    changed["modeling"]["fit_psf"]["psf"]["aberrations"][
        "global_zernikes"
    ][4] = 8.0

    assert first.mode == "explicit"
    assert first.delta_id == _build(same).delta_id
    assert first.delta_id != _build(changed).delta_id
    assert first.requested_amplitude_rms_nm is None
    assert first.draw_aberrations is None


class _Observation:
    """Small observation seam accepted by the mismatch executor."""

    def __init__(self, psf, pixel_scale):
        shape = (15, 15)
        self.psf = psf
        self.pixel_scale = float(pixel_scale)
        self.noiseless_source_eps = np.ones(shape, dtype=float)
        self.data = SimpleNamespace(native=np.ones(shape, dtype=float))
        self.noise_map = SimpleNamespace(native=np.ones(shape, dtype=float))
        self.gain = 1.0
        self.exposure_time = 1.0
        self.sky_electrons_per_pixel = 0.0
        self.dark_electrons_per_pixel = 0.0


def _observation(config, psf_config=None):
    """Return an observation seam carrying one generated kernel."""
    psf_data = _psf(config, psf_config=psf_config)
    return _Observation(
        psf_data.kernel,
        config["lensing"]["grid"]["pixel_scale"],
    )


def _trial():
    """Return one compact trial seam for stubbed executor tests."""
    return SimpleNamespace(case_id="item9-case")


def test_dataset_metadata_records_actual_matched_and_supplied_kernel_hashes(
    compact_config,
):
    """Record supplied state and actual fit-kernel identity unconditionally."""
    observation = _observation(compact_config)
    matched_dataset, matched_metadata = imaging_from_observation(
        observation,
        psf_for_fit=None,
    )
    supplied_dataset, supplied_metadata = imaging_from_observation(
        observation,
        psf_for_fit=observation.psf,
        psf_fit_label="delta:test",
    )

    assert matched_metadata.psf_fit_supplied is False
    assert supplied_metadata.psf_fit_supplied is True
    assert matched_metadata.psf_fit_sha256 == _kernel_sha256(
        pyauto_kernel_native(matched_dataset.psf)
    )
    assert supplied_metadata.psf_fit_sha256 == _kernel_sha256(
        pyauto_kernel_native(supplied_dataset.psf)
    )
    assert matched_metadata.to_dict()["psf_fit_sha256"]


class _FakeValidator:
    """Return deterministic fit summaries without starting searches."""

    def __init__(
        self,
        smooth_status="success",
        subhalo_status="success",
        smooth_log_l=-10.0,
        subhalo_log_l=-4.0,
        smooth_logz=-12.0,
        subhalo_logz=-5.0,
    ):
        self.calls = []
        self.smooth_status = smooth_status
        self.subhalo_status = subhalo_status
        self.smooth_log_l = smooth_log_l
        self.subhalo_log_l = subhalo_log_l
        self.smooth_logz = smooth_logz
        self.subhalo_logz = subhalo_logz

    def validate_case(
        self,
        dataset,
        dataset_metadata,
        full_config,
        trial,
        fit_mode="fixed_template",
        psf_case="nominal",
        priors_config=None,
        mass_context=None,
        clumpy_fit_parameterization="host_free",
        smooth_result=None,
        expected_psf_fit_sha256=None,
    ):
        """Record the executor call and return one case-shaped object."""
        del (
            full_config,
            priors_config,
            mass_context,
            clumpy_fit_parameterization,
        )
        smooth = SimpleNamespace(
            status=self.smooth_status,
            log_likelihood_max=self.smooth_log_l,
            log_evidence=self.smooth_logz,
        )
        subhalo = SimpleNamespace(
            status=self.subhalo_status,
            log_likelihood_max=self.subhalo_log_l,
            log_evidence=self.subhalo_logz,
        )
        case = SimpleNamespace(
            case_id=trial.case_id,
            psf_case=psf_case,
            smooth_fit=smooth,
            subhalo_fit=subhalo,
        )
        self.calls.append({
            "dataset": dataset,
            "metadata": dataset_metadata,
            "fit_mode": fit_mode,
            "psf_case": psf_case,
            "smooth_result": smooth_result,
            "expected_psf_fit_sha256": expected_psf_fit_sha256,
        })
        return case


def test_delta_zero_kernel_matches_truth_and_observation_bytes(compact_config):
    """Reduce zero mismatch to one byte-identical truth kernel."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["delta"]["amplitude_rms_nm"] = 0.0
    spec = _build(config)
    truth_kernel, truth_scale, _ = _quiet_call(
        generate_fit_psf,
        spec.fit_psf_config,
        config,
    )
    observation = _observation(config)
    observation_native = np.ascontiguousarray(
        pyauto_kernel_native(observation.psf),
        dtype=np.float64,
    )

    assert spec.fit_psf_config == _canonical_aberrations_config(
        config["psf"]
    )
    assert truth_scale == observation.pixel_scale
    assert truth_kernel.tobytes() == observation_native.tobytes()

    result = run_psf_mismatch_case(
        _FakeValidator(),
        observation,
        config,
        _trial(),
        fit_mode="fixed_template",
    )
    assert result.kernel_sha256 == result.truth_kernel_sha256


def _canonical_aberrations_config(psf_config: dict) -> dict:
    """Canonicalize the aberrations block of a PSF configuration."""
    canonical = copy.deepcopy(psf_config)
    canonical["aberrations"] = _canonical_aberrations(
        canonical["aberrations"]
    )
    return canonical


def _metadata(
    label,
    supplied,
    psf_fit_sha256="0"*64,
) -> NonlinearDatasetMetadata:
    """Return metadata suitable for guard-only validation calls."""
    return NonlinearDatasetMetadata(
        dataset_kind="asimov",
        data_units="e_per_s",
        background_treatment="subtract_known",
        sky_dark_background_adu=0.0,
        mask_name="test",
        n_unmasked_pixels=1,
        psf_truth_label="truth",
        psf_fit_label=label,
        psf_fit_supplied=supplied,
        psf_fit_sha256=psf_fit_sha256,
    )


def _guard_kernel() -> np.ndarray:
    """Return a small deterministic kernel for guard-only calls."""
    return np.linspace(0.0, 1.0, 49, dtype=np.float64).reshape(7, 7)


@pytest.mark.parametrize(
    "mode,label,supplied,passes",
    [
        ("matched", "fit", False, True),
        ("matched", "delta:wrong", False, False),
        ("matched", "fit", True, False),
        ("bank", "fit", False, False),
        ("bank", "delta:wrong", True, False),
        ("bank", "bank:right", True, True),
        ("delta", "fit", False, False),
        ("delta", "delta:right", False, False),
        ("delta", "bank:wrong", True, False),
        ("delta", "explicit:wrong", True, False),
        ("delta", "delta:right", True, True),
        ("explicit", "fit", False, False),
        ("explicit", "explicit:right", False, False),
        ("explicit", "bank:wrong", True, False),
        ("explicit", "delta:wrong", True, False),
        ("explicit", "explicit:right", True, True),
    ],
)
def test_validator_guard_rejects_mode_label_and_supplied_incoherence(
    compact_config,
    monkeypatch,
    mode,
    label,
    supplied,
    passes,
):
    """Enforce the configured mode, mismatch prefix, and supplied flag."""
    import hwoslaps.modeling.nonlinear.validator as validator_module

    config = copy.deepcopy(compact_config)
    if mode == "matched":
        config["modeling"]["fit_psf"] = {"mode": "matched"}
    elif mode == "bank":
        config["modeling"]["fit_psf"] = {
            "mode": "bank",
            "bank": {"kind": "explicit", "candidates": [
                copy.deepcopy(config["psf"]["aberrations"])
            ]},
        }
    elif mode == "explicit":
        config["modeling"]["fit_psf"] = {
            "mode": "explicit",
            "psf": copy.deepcopy(config["psf"]),
        }

    class ReachedModelBuild(Exception):
        """Signal that guard validation passed."""

    def reached(*args, **kwargs):
        raise ReachedModelBuild

    monkeypatch.setattr(
        validator_module,
        "smooth_model_spec_from_config",
        reached,
    )
    validator = NonlinearMetricValidator(SimpleNamespace())
    kernel = _guard_kernel()
    digest = _kernel_sha256(kernel)
    resolved_label = label
    if passes and mode in {"delta", "explicit"}:
        resolved_label = f"{mode}:{_build(config).delta_id}"

    def call():
        metadata = _metadata(resolved_label, supplied, digest)
        expected_digest = (
            metadata.psf_fit_sha256
            if mode in {"bank", "delta", "explicit"}
            else None
        )
        return validator.validate_case(
            SimpleNamespace(psf=kernel),
            metadata,
            config,
            _trial(),
            psf_case=resolved_label,
            expected_psf_fit_sha256=expected_digest,
        )
    if passes:
        with pytest.raises(ReachedModelBuild):
            call()
    else:
        with pytest.raises(ValueError, match=f"mode is '{mode}'"):
            call()


def test_validator_guard_treats_absent_fit_psf_as_matched(
    compact_config,
    monkeypatch,
):
    """Treat a missing fit-PSF block as matched configuration."""
    import hwoslaps.modeling.nonlinear.validator as validator_module

    config = copy.deepcopy(compact_config)
    config["modeling"].pop("fit_psf")

    class ReachedModelBuild(Exception):
        """Signal that guard validation passed."""

    monkeypatch.setattr(
        validator_module,
        "smooth_model_spec_from_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(ReachedModelBuild()),
    )
    with pytest.raises(ReachedModelBuild):
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            SimpleNamespace(),
            _metadata("fit", False),
            config,
            _trial(),
        )
    with pytest.raises(ValueError, match="must be None"):
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            SimpleNamespace(),
            _metadata("fit", False),
            config,
            _trial(),
            expected_psf_fit_sha256="0"*64,
        )


def test_validator_guard_requires_executor_kernel_digest(compact_config):
    """Reject mismatch-mode datasets not bound to an executor kernel."""
    with pytest.raises(
        ValueError,
        match=(
            "mismatch-mode datasets must be executed through "
            "run_psf_mismatch_case / run_psf_bank_case"
        ),
    ):
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            SimpleNamespace(),
            _metadata("delta:forged", True),
            compact_config,
            _trial(),
        )


def test_validator_guard_rejects_forged_label_truth_kernel(compact_config):
    """Reject a truth kernel supplied under a forged mismatch label."""
    observation = _observation(compact_config)
    _, metadata = imaging_from_observation(
        observation,
        psf_for_fit=observation.psf,
        psf_fit_label="delta:forged",
    )
    fit_kernel, _, _ = _quiet_call(
        generate_fit_psf,
        _build(compact_config).fit_psf_config,
        compact_config,
    )
    expected_digest = _kernel_sha256(fit_kernel)
    assert expected_digest != metadata.psf_fit_sha256

    with pytest.raises(ValueError) as error:
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            SimpleNamespace(),
            metadata,
            compact_config,
            _trial(),
            expected_psf_fit_sha256=expected_digest,
        )
    assert expected_digest in str(error.value)
    assert metadata.psf_fit_sha256 in str(error.value)


def test_validator_guard_rejects_swapped_dataset_pair(compact_config):
    """Reject a dataset whose PSF is not the digested executor kernel."""
    spec = _build(compact_config)
    label = f"delta:{spec.delta_id}"
    kernel = _guard_kernel()
    digest = _kernel_sha256(kernel)
    metadata = _metadata(label, True, digest)

    with pytest.raises(
        ValueError,
        match="does not match the executor kernel digest",
    ):
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            SimpleNamespace(psf=kernel + 1.0),
            metadata,
            compact_config,
            _trial(),
            psf_case=label,
            expected_psf_fit_sha256=digest,
        )


def test_validator_guard_rejects_truth_dataset_with_fit_metadata(
    compact_config,
):
    """Reject a truth-PSF dataset paired with mismatch-build metadata."""
    observation = _observation(compact_config)
    spec = _build(compact_config)
    label = f"delta:{spec.delta_id}"
    fit_kernel, _, _ = _quiet_call(
        generate_fit_psf,
        spec.fit_psf_config,
        compact_config,
    )
    truth_dataset, _ = imaging_from_observation(
        observation,
        psf_for_fit=observation.psf,
        psf_fit_label=label,
    )
    _, fit_metadata = imaging_from_observation(
        observation,
        psf_for_fit=fit_kernel,
        psf_fit_label=label,
    )
    assert fit_metadata.psf_fit_sha256 != _kernel_sha256(
        pyauto_kernel_native(truth_dataset.psf)
    )

    with pytest.raises(
        ValueError,
        match="does not match the executor kernel digest",
    ):
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            truth_dataset,
            fit_metadata,
            compact_config,
            _trial(),
            psf_case=label,
            expected_psf_fit_sha256=fit_metadata.psf_fit_sha256,
        )


def test_validator_guard_requires_psf_case_label_congruence(compact_config):
    """Require the recorded psf_case to equal the guarded dataset label."""
    spec = _build(compact_config)
    label = f"delta:{spec.delta_id}"
    kernel = _guard_kernel()
    digest = _kernel_sha256(kernel)
    metadata = _metadata(label, True, digest)

    with pytest.raises(
        ValueError,
        match="does not match the dataset label",
    ):
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            SimpleNamespace(psf=kernel),
            metadata,
            compact_config,
            _trial(),
            psf_case="delta:other",
            expected_psf_fit_sha256=digest,
        )


def test_validator_guard_rejects_arbitrary_delta_identity(compact_config):
    """Reject delta labels that are not the configured identity."""
    kernel = _guard_kernel()
    digest = _kernel_sha256(kernel)
    label = "delta:not-the-configured-id"
    metadata = _metadata(label, True, digest)

    with pytest.raises(
        ValueError,
        match="recomputed from modeling.fit_psf",
    ):
        NonlinearMetricValidator(SimpleNamespace()).validate_case(
            SimpleNamespace(psf=kernel),
            metadata,
            compact_config,
            _trial(),
            psf_case=label,
            expected_psf_fit_sha256=digest,
        )


@pytest.mark.parametrize("mode", ["bank", "delta"])
def test_validator_matched_control_contract(compact_config, monkeypatch, mode):
    """Permit only truth-kernel matched references under mismatch configs."""
    import hwoslaps.modeling.nonlinear.validator as validator_module

    config = copy.deepcopy(compact_config)
    if mode == "bank":
        config["modeling"]["fit_psf"] = {
            "mode": "bank",
            "bank": {"kind": "explicit", "candidates": [
                copy.deepcopy(config["psf"]["aberrations"])
            ]},
        }

    class ReachedModelBuild(Exception):
        """Signal that guard validation passed."""

    monkeypatch.setattr(
        validator_module,
        "smooth_model_spec_from_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(ReachedModelBuild()),
    )
    validator = NonlinearMetricValidator(SimpleNamespace())
    with pytest.raises(ReachedModelBuild):
        validator.validate_case(
            SimpleNamespace(),
            _metadata("fit", False),
            config,
            _trial(),
            matched_control=True,
        )
    with pytest.raises(ValueError, match="matched_control=True"):
        validator.validate_case(
            SimpleNamespace(),
            _metadata(f"{mode}:wrong", True),
            config,
            _trial(),
            matched_control=True,
        )
    with pytest.raises(ValueError, match="must be None"):
        validator.validate_case(
            SimpleNamespace(),
            _metadata("fit", False),
            config,
            _trial(),
            matched_control=True,
            expected_psf_fit_sha256="0"*64,
        )


def _fisher_products(config):
    """Generate the tiny truth scene products used for Fisher parity."""
    baseline_config = copy.deepcopy(config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    psf_data = _psf(config)
    baseline = _quiet_call(
        generate_lensing_system,
        baseline_config["lensing"],
        full_config=baseline_config,
    )
    test = _quiet_call(
        generate_lensing_system,
        config["lensing"],
        full_config=config,
    )
    observation_baseline = _quiet_call(
        generate_observation,
        baseline,
        psf_data,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )
    observation_test = _quiet_call(
        generate_observation,
        test,
        psf_data,
        observation_config=config["observation"],
        full_config=config,
    )
    return psf_data, baseline, test, observation_baseline, observation_test


def test_fisher_delta_matches_explicit_reduction(compact_config):
    """Match Fisher mismatch statistics after explicit reduction."""
    products = _fisher_products(compact_config)
    psf_data, baseline, test, observation_baseline, observation_test = products
    delta_detector = _quiet_call(
        FisherDetector,
        observation_baseline=observation_baseline,
        lensing_baseline=baseline,
        psf_data=psf_data,
        full_config=compact_config,
        fisher_config=compact_config["modeling"]["fisher"],
    )
    delta_local = delta_detector.compute_local(observation_test, test)
    spec = _build(compact_config)
    explicit = copy.deepcopy(compact_config)
    explicit["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": copy.deepcopy(spec.fit_psf_config),
    }
    explicit_detector = _quiet_call(
        FisherDetector,
        observation_baseline=observation_baseline,
        lensing_baseline=baseline,
        psf_data=psf_data,
        full_config=explicit,
        fisher_config=explicit["modeling"]["fisher"],
    )
    explicit_local = explicit_detector.compute_local(observation_test, test)

    for field in (
        "q_mismatch",
        "q_spurious",
        "amplitude_hat_mismatch",
        "amplitude_spurious",
    ):
        assert getattr(delta_local, field) == pytest.approx(
            getattr(explicit_local, field),
            rel=1.0e-12,
            abs=1.0e-12,
        )
    assert delta_detector.fit_psf_mode == "delta"
    assert delta_detector.fit_psf_delta["delta_id"] == spec.delta_id


def test_fisher_delta_rejects_different_same_scale_truth_psf(compact_config):
    """Bind Fisher delta construction to the supplied native truth kernel."""
    products = _fisher_products(compact_config)
    truth_psf, baseline, _, observation_baseline, _ = products
    other_psf_config = copy.deepcopy(compact_config["psf"])
    other_psf_config["aberrations"]["global_zernikes"][4] += 2.0
    supplied_psf = _psf(compact_config, psf_config=other_psf_config)

    with pytest.raises(
        ValueError,
        match="psf_data was not generated.*full_config.*psf",
    ) as error:
        FisherDetector(
            observation_baseline=observation_baseline,
            lensing_baseline=baseline,
            psf_data=supplied_psf,
            full_config=compact_config,
            fisher_config=compact_config["modeling"]["fisher"],
        )
    assert _kernel_sha256(
        pyauto_kernel_native(truth_psf.kernel)
    ) in str(error.value)
    assert _kernel_sha256(
        pyauto_kernel_native(supplied_psf.kernel)
    ) in str(error.value)


def test_fisher_delta_rejects_stale_baseline_observation(compact_config):
    """Reject baseline observations convolved with a different PSF."""
    truth_psf = _psf(compact_config)
    other_psf_config = copy.deepcopy(compact_config["psf"])
    other_psf_config["aberrations"]["global_zernikes"][4] += 2.0
    other_psf = _psf(compact_config, psf_config=other_psf_config)
    baseline_config = copy.deepcopy(compact_config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    baseline = _quiet_call(
        generate_lensing_system,
        baseline_config["lensing"],
        full_config=baseline_config,
    )
    observation_other = _quiet_call(
        generate_observation,
        baseline,
        other_psf,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )

    with pytest.raises(
        ValueError,
        match="observation_baseline was not generated",
    ):
        FisherDetector(
            observation_baseline=observation_other,
            lensing_baseline=baseline,
            psf_data=truth_psf,
            full_config=compact_config,
            fisher_config=compact_config["modeling"]["fisher"],
        )


def test_fisher_delta_rejects_stale_test_observation(compact_config):
    """Reject test observations convolved with a different PSF."""
    products = _fisher_products(compact_config)
    psf_data, baseline, test, observation_baseline, _ = products
    other_psf_config = copy.deepcopy(compact_config["psf"])
    other_psf_config["aberrations"]["global_zernikes"][4] += 2.0
    other_psf = _psf(compact_config, psf_config=other_psf_config)
    observation_test_other = _quiet_call(
        generate_observation,
        test,
        other_psf,
        observation_config=compact_config["observation"],
        full_config=compact_config,
    )
    detector = _quiet_call(
        FisherDetector,
        observation_baseline=observation_baseline,
        lensing_baseline=baseline,
        psf_data=psf_data,
        full_config=compact_config,
        fisher_config=compact_config["modeling"]["fisher"],
    )

    with pytest.raises(
        ValueError,
        match="observation_test was not generated",
    ):
        detector.compute_local(observation_test_other, test)


def test_fisher_delta_rejects_rescaled_truth_kernel(compact_config):
    """Reject identical kernel samples rewrapped at another pixel scale."""
    products = _fisher_products(compact_config)
    psf_data, baseline, _, observation_baseline, _ = products
    values = np.ascontiguousarray(
        pyauto_kernel_native(psf_data.kernel),
        dtype=np.float64,
    )
    scales = pyauto_kernel_pixel_scales(psf_data.kernel)
    rewrapped = make_pyauto_kernel(
        values=values,
        pixel_scales=float(scales[0])*2.0,
        normalize=False,
    )
    rescaled_psf = dataclasses.replace(psf_data, kernel=rewrapped)
    assert _kernel_sha256(
        pyauto_kernel_native(rescaled_psf.kernel)
    ) == _kernel_sha256(values)

    with pytest.raises(
        ValueError,
        match="pixel scales do not match",
    ):
        FisherDetector(
            observation_baseline=observation_baseline,
            lensing_baseline=baseline,
            psf_data=rescaled_psf,
            full_config=compact_config,
            fisher_config=compact_config["modeling"]["fisher"],
        )


def test_fisher_delta_rejects_swapped_preconvolved_source(compact_config):
    """Reject observations whose stored source is not the truth product."""
    products = _fisher_products(compact_config)
    psf_data, baseline, test, observation_baseline, observation_test = (
        products
    )
    swapped = dataclasses.replace(
        observation_baseline,
        noiseless_source_eps=observation_test.noiseless_source_eps,
    )

    with pytest.raises(
        ValueError,
        match="noiseless_source_eps does not reproduce",
    ):
        FisherDetector(
            observation_baseline=swapped,
            lensing_baseline=baseline,
            psf_data=psf_data,
            full_config=compact_config,
            fisher_config=compact_config["modeling"]["fisher"],
        )


def test_fisher_delta_rejects_rescaled_observation_kernel(compact_config):
    """Reject an observation PSF rewrapped at another pixel scale."""
    import autolens as al

    products = _fisher_products(compact_config)
    psf_data, baseline, _, observation_baseline, _ = products
    rescaled_kernel = make_pyauto_kernel(
        values=np.ascontiguousarray(
            pyauto_kernel_native(observation_baseline.psf),
            dtype=np.float64,
        ),
        pixel_scales=float(observation_baseline.pixel_scale)*2.0,
        normalize=False,
    )
    rescaled_imaging = al.Imaging(
        data=observation_baseline.imaging.data,
        noise_map=observation_baseline.imaging.noise_map,
        psf=make_pyauto_convolver(rescaled_kernel),
    )
    rescaled = dataclasses.replace(
        observation_baseline,
        imaging=rescaled_imaging,
    )

    with pytest.raises(
        ValueError,
        match="embedded PSF pixel scales",
    ):
        FisherDetector(
            observation_baseline=rescaled,
            lensing_baseline=baseline,
            psf_data=psf_data,
            full_config=compact_config,
            fisher_config=compact_config["modeling"]["fisher"],
        )


def test_truth_kernel_accept_digests_cover_normalization():
    """Accept both the generated bytes and their sum-normalization."""
    from hwoslaps.modeling.fisher_detector import (
        _truth_kernel_accept_digests,
    )

    non_unit = make_pyauto_kernel(
        values=np.full((5, 5), 2.0/25.0),
        pixel_scales=0.1,
        normalize=False,
    )
    digests = _truth_kernel_accept_digests(non_unit)
    assert len(digests) == 2
    assert _kernel_sha256(pyauto_kernel_native(non_unit)) in digests
    normalized = make_pyauto_kernel(
        values=np.full((5, 5), 2.0/25.0),
        pixel_scales=0.1,
        normalize=True,
    )
    assert _kernel_sha256(pyauto_kernel_native(normalized)) in digests


def test_generate_observation_preserves_shared_psf_kernel(compact_config):
    """Keep the shared PSFData kernel byte-stable across observations.

    al.Imaging sum-normalizes its kernel in place; without a private
    copy, a first observation build would rewrite the shared kernel and
    a second observation would convolve with different bytes.
    """
    baseline_config = copy.deepcopy(compact_config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    lensing = _quiet_call(
        generate_lensing_system,
        baseline_config["lensing"],
        full_config=baseline_config,
    )
    psf = _psf(compact_config)
    # Scale inside the generator's 1e-10 unit-flux tolerance, but off the
    # normalization fixed point, so an in-place al.Imaging normalization
    # would change the shared bytes.
    perturbed = dataclasses.replace(
        psf,
        kernel=make_pyauto_kernel(
            values=np.ascontiguousarray(
                pyauto_kernel_native(psf.kernel),
                dtype=np.float64,
            )*(1.0 + 5.0e-11),
            pixel_scales=pyauto_kernel_pixel_scales(psf.kernel),
            normalize=False,
        ),
    )
    before = np.array(pyauto_kernel_native(perturbed.kernel), copy=True)

    observations = [
        _quiet_call(
            generate_observation,
            lensing,
            perturbed,
            observation_config=baseline_config["observation"],
            full_config=baseline_config,
        )
        for _ in range(2)
    ]

    assert np.array_equal(
        np.asarray(pyauto_kernel_native(perturbed.kernel)),
        before,
    )
    assert np.array_equal(
        np.asarray(observations[0].noiseless_source_eps),
        np.asarray(observations[1].noiseless_source_eps),
    )


def test_fisher_delta_rejects_swapped_noise_map(compact_config):
    """Reject observations whose noise map is not the truth product."""
    import autolens as al

    products = _fisher_products(compact_config)
    psf_data, baseline, _, observation_baseline, _ = products
    doubled_noise = al.Array2D(
        values=np.asarray(observation_baseline.noise_map.native)*2.0,
        mask=observation_baseline.noise_map.mask,
    )
    tampered_imaging = al.Imaging(
        data=observation_baseline.imaging.data,
        noise_map=doubled_noise,
        psf=make_pyauto_convolver(observation_baseline.imaging.psf),
    )
    tampered = dataclasses.replace(
        observation_baseline,
        imaging=tampered_imaging,
    )

    with pytest.raises(
        ValueError,
        match="noise map does not reproduce",
    ):
        FisherDetector(
            observation_baseline=tampered,
            lensing_baseline=baseline,
            psf_data=psf_data,
            full_config=compact_config,
            fisher_config=compact_config["modeling"]["fisher"],
        )


def test_fisher_delta_rejects_altered_observation_scalars(compact_config):
    """Reject observations whose consumed scalars left the configuration."""
    products = _fisher_products(compact_config)
    psf_data, baseline, _, observation_baseline, _ = products
    tampered_metadata = dict(observation_baseline.metadata)
    tampered_metadata["exposure_time"] = (
        float(observation_baseline.exposure_time)*2.0
    )
    tampered = dataclasses.replace(
        observation_baseline,
        metadata=tampered_metadata,
    )

    with pytest.raises(ValueError, match="exposure_time"):
        FisherDetector(
            observation_baseline=tampered,
            lensing_baseline=baseline,
            psf_data=psf_data,
            full_config=compact_config,
            fisher_config=compact_config["modeling"]["fisher"],
        )


def test_fisher_bank_still_raises(compact_config):
    """Keep the nonlinear-only bank rejection intact on the Fisher path."""
    psf_data, baseline, _, observation_baseline, _ = _fisher_products(
        compact_config
    )
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"] = {
        "mode": "bank",
        "bank": {"kind": "explicit", "candidates": [
            copy.deepcopy(config["psf"]["aberrations"])
        ]},
    }

    with pytest.raises(ValueError, match="nonlinear-only"):
        FisherDetector(
            observation_baseline=observation_baseline,
            lensing_baseline=baseline,
            psf_data=psf_data,
            full_config=config,
            fisher_config=config["modeling"]["fisher"],
        )


def test_fisher_detection_transports_delta_provenance(
    compact_config,
    capsys,
):
    """Transport detector delta state into FisherDetectionData."""
    psf_data, baseline, test, observation_baseline, observation_test = (
        _fisher_products(compact_config)
    )
    result = _quiet_call(
        perform_fisher_detection,
        observation_baseline,
        observation_test,
        baseline,
        test,
        psf_data,
        detection_config=compact_config["modeling"],
        full_config=compact_config,
    )

    assert result.fit_psf_mode == "delta"
    spec = _build(compact_config)
    delta = result.fit_psf_delta
    assert delta["delta_id"] == spec.delta_id
    assert {
        "draw_aberrations",
        "orthonormal_segment",
        "orthonormal_global",
        "versions",
        "truth_kernel_sha256",
        "fit_kernel_sha256",
        "revision",
    } <= set(delta)
    assert set(delta["revision"]) == {
        "git_hash",
        "git_dirty",
        "git_dirty_paths",
        "git_diff_sha256",
    }
    assert _aberrations_from_wire(
        delta["draw_aberrations"]
    ) == spec.draw_aberrations
    assert _nested_int_map_from_wire(
        delta["orthonormal_segment"]
    ) == spec.orthonormal_segment
    assert _flat_int_map_from_wire(
        delta["orthonormal_global"]
    ) == spec.orthonormal_global
    assert delta["versions"] == spec.versions
    assert delta["truth_kernel_sha256"] == _kernel_sha256(
        pyauto_kernel_native(psf_data.kernel)
    )
    assert len(delta["fit_kernel_sha256"]) == 64
    print_fisher_summary(result)
    output = capsys.readouterr().out
    assert (
        "fit_psf mode: delta "
        f"(delta_id={result.fit_psf_delta['delta_id']}, "
        "amplitude=5.0 nm, family=combined)"
    ) in output


def test_executor_wraps_kernel_bytes_and_records_truth_binding(compact_config):
    """Preserve native kernel bytes through wrapping and record both hashes."""
    validator = _FakeValidator()
    observation = _observation(compact_config)
    result = run_psf_mismatch_case(
        validator,
        observation,
        compact_config,
        _trial(),
        fit_mode="fixed_template",
    )
    wrapped = np.ascontiguousarray(
        pyauto_kernel_native(validator.calls[0]["dataset"].psf),
        dtype=np.float64,
    )
    expected, _, _ = _quiet_call(
        generate_fit_psf,
        _build(compact_config).fit_psf_config,
        compact_config,
    )

    assert wrapped.tobytes() == expected.tobytes()
    assert result.kernel_sha256 == _kernel_sha256(wrapped)
    assert result.truth_kernel_sha256 == _kernel_sha256(
        pyauto_kernel_native(observation.psf)
    )
    assert validator.calls[0]["expected_psf_fit_sha256"] == (
        validator.calls[0]["metadata"].psf_fit_sha256
    )


def test_executor_records_revision_provenance(compact_config):
    """Record source revision and config identity with every result."""
    result = run_psf_mismatch_case(
        _FakeValidator(),
        _observation(compact_config),
        compact_config,
        _trial(),
        fit_mode="fixed_template",
    )
    revision = result.provenance["revision"]
    assert set(revision) == {
        "git_hash",
        "git_dirty",
        "git_dirty_paths",
        "git_diff_sha256",
    }
    # The suite runs from repository checkouts by project policy, so the
    # record must actually resolve the executing source revision.
    assert isinstance(revision["git_hash"], str)
    assert len(revision["git_hash"]) == 40
    assert result.provenance["config_hash"] == config_hash(compact_config)


def test_dataset_build_does_not_mutate_observation(compact_config):
    """Preserve the observation's stored source across dataset builds.

    al.Array2D zeroes masked pixels in place, so a no-copy view of
    noiseless_source_eps handed to the dataset constructor would zero
    the observation's own PSF-border pixels.
    """
    observation = _observation(compact_config)
    before = np.array(observation.noiseless_source_eps, copy=True)
    dataset_one, _ = imaging_from_observation(
        observation,
        psf_for_fit=None,
    )
    assert np.array_equal(
        np.asarray(observation.noiseless_source_eps),
        before,
    )
    dataset_two, _ = imaging_from_observation(
        observation,
        psf_for_fit=None,
    )
    assert np.array_equal(
        np.asarray(dataset_one.data.native),
        np.asarray(dataset_two.data.native),
    )


def test_dataset_metadata_digest_describes_fitted_kernel(compact_config):
    """Hash the as-fitted, autolens-normalized dataset PSF kernel."""
    observation = _observation(compact_config)
    values = np.full((7, 7), 2.0/49.0)
    wrapped = make_pyauto_convolver(
        make_pyauto_kernel(
            values=values,
            pixel_scales=observation.pixel_scale,
            normalize=False,
        )
    )
    input_digest = _kernel_sha256(pyauto_kernel_native(wrapped))
    dataset, metadata = imaging_from_observation(
        observation,
        psf_for_fit=wrapped,
        psf_fit_label="delta:test",
    )
    fitted_digest = _kernel_sha256(pyauto_kernel_native(dataset.psf))

    assert metadata.psf_fit_sha256 == fitted_digest
    assert metadata.psf_fit_sha256 != input_digest
    assert fitted_kernel_sha256(
        dataset,
        wrapped,
        observation.pixel_scale,
    ) == fitted_digest
    with pytest.raises(ValueError, match="neither the wrapped"):
        fitted_kernel_sha256(
            dataset,
            make_pyauto_kernel(
                values=values + 1.0,
                pixel_scales=observation.pixel_scale,
                normalize=False,
            ),
            observation.pixel_scale,
        )


def test_delta_rejects_erased_draw(compact_config):
    """Reject positive amplitudes fully erased by float addition."""
    config = copy.deepcopy(compact_config)
    config["psf"]["aberrations"]["global_zernikes"] = {
        4: float(2**53),
        5: float(2**53),
    }
    delta = config["modeling"]["fit_psf"]["delta"]
    delta["family"] = "global"
    delta["amplitude_rms_nm"] = 1.0e-10

    with pytest.raises(ValueError, match="completely erased"):
        _build(config)


def test_prior_digest_and_parse_share_one_read(
    compact_config,
    tmp_path,
    monkeypatch,
):
    """Hash and parse the identical prior bytes from one file read."""
    document_a = yaml.safe_dump({
        "name": "table-a",
        "segment_variance_fraction": 0.4,
        "global_weights": {4: 1.0, 5: 0.5},
        "segment_weights": {1: 1.0, 2: 0.5},
    })
    document_b = yaml.safe_dump({
        "name": "table-b",
        "segment_variance_fraction": 0.7,
        "global_weights": {4: 0.2, 5: 1.0},
        "segment_weights": {1: 0.3, 2: 1.0},
    })
    path_a = tmp_path / "prior_a.yaml"
    path_a.write_text(document_a, encoding="utf-8")
    path_b = tmp_path / "prior_b.yaml"
    path_b.write_text(document_b, encoding="utf-8")

    reference_config = copy.deepcopy(compact_config)
    reference_config["modeling"]["fit_psf"]["delta"]["prior_table"] = str(
        path_a
    )
    reference_spec = _build(reference_config)

    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"]["delta"]["prior_table"] = str(path_b)
    original_read_bytes = Path.read_bytes
    read_count = {"n": 0}

    def racing_read_bytes(self):
        if Path(self) == path_b:
            read_count["n"] += 1
            if read_count["n"] == 1:
                return document_a.encode("utf-8")
            return document_b.encode("utf-8")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    spec = _build(config)

    assert read_count["n"] == 1
    assert spec.prior_table_sha256 == hashlib.sha256(
        document_a.encode("utf-8")
    ).hexdigest()
    assert spec.prior_table_sha256 == reference_spec.prior_table_sha256
    assert spec.draw_aberrations == reference_spec.draw_aberrations


def test_bank_executor_passes_wrapped_kernel_digest(compact_config):
    """Bind each bank dataset to the wrapped candidate kernel digest."""
    config = copy.deepcopy(compact_config)
    config["modeling"]["fit_psf"] = {
        "mode": "bank",
        "bank": {
            "kind": "explicit",
            "candidates": [
                copy.deepcopy(config["psf"]["aberrations"]),
            ],
        },
    }
    bank = _quiet_call(build_psf_bank, config)
    validator = _FakeValidator()
    run_psf_bank_case(
        validator,
        _observation(config),
        config,
        _trial(),
        bank,
        fit_mode="fixed_template",
        include_anchors=False,
    )

    assert len(validator.calls) == 1
    assert validator.calls[0]["expected_psf_fit_sha256"] == (
        validator.calls[0]["metadata"].psf_fit_sha256
    )


def test_executor_rejects_kernel_pixel_scale_mismatch(compact_config):
    """Reject a fit kernel whose pixel scale differs from the observation."""
    observation = _observation(compact_config)
    observation.pixel_scale += 0.01

    with pytest.raises(ValueError, match="pixel scale"):
        run_psf_mismatch_case(
            _FakeValidator(),
            observation,
            compact_config,
            _trial(),
            fit_mode="fixed_template",
        )


def test_executor_truth_binding_rejects_different_same_scale_psf(
    compact_config,
):
    """Reject observations generated by another PSF at the same pixel scale."""
    other_psf = copy.deepcopy(compact_config["psf"])
    other_psf["aberrations"]["global_zernikes"][4] += 2.0
    observation = _observation(compact_config, psf_config=other_psf)

    with pytest.raises(
        ValueError,
        match="observation was not generated.*full_config.*psf",
    ):
        run_psf_mismatch_case(
            _FakeValidator(),
            observation,
            compact_config,
            _trial(),
            fit_mode="fixed_template",
        )


def test_explicit_executor_injects_explicit_fit_kernel(compact_config):
    """Execute nonlinear explicit mode with an explicit-prefixed label."""
    config = copy.deepcopy(compact_config)
    fit_psf = copy.deepcopy(config["psf"])
    fit_psf["aberrations"]["global_zernikes"][4] += 2.0
    config["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": fit_psf,
    }
    validator = _FakeValidator()
    result = run_psf_mismatch_case(
        validator,
        _observation(config),
        config,
        _trial(),
        fit_mode="fixed_template",
    )

    assert result.mode == "explicit"
    assert result.psf_fit_label == f"explicit:{result.delta_id}"
    assert result.requested_amplitude_rms_nm is None
    assert result.draw_aberrations_wire is None
    assert validator.calls[0]["metadata"].psf_fit_supplied is True


def test_result_json_roundtrip_preserves_typed_draw_provenance(
    compact_config,
    tmp_path,
):
    """Write slim typed JSON without the embedded nonlinear case."""
    result = run_psf_mismatch_case(
        _FakeValidator(),
        _observation(compact_config),
        compact_config,
        _trial(),
        fit_mode="fixed_template",
    )
    path = tmp_path / "mismatch.json"
    result.write_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert "case" not in payload
    assert payload == result.to_dict()
    assert _aberrations_from_wire(
        payload["draw_aberrations_wire"]
    ) == _build(compact_config).draw_aberrations


def test_result_to_dict_restores_integer_keyed_maps(compact_config):
    """Wire-encode every integer-keyed draw and orthonormal map."""
    result = run_psf_mismatch_case(
        _FakeValidator(),
        _observation(compact_config),
        compact_config,
        _trial(),
        fit_mode="fixed_template",
    )
    payload = result.to_dict()
    spec = _build(compact_config)

    assert _aberrations_from_wire(
        payload["draw_aberrations_wire"]
    ) == spec.draw_aberrations
    assert _nested_int_map_from_wire(
        payload["orthonormal_segment_wire"]
    ) == spec.orthonormal_segment
    assert _flat_int_map_from_wire(
        payload["orthonormal_global_wire"]
    ) == spec.orthonormal_global


def test_freed_detection_flag_is_none_but_fixed_template_is_computed(
    compact_config,
):
    """Null only fixed-calibration SCDD detection for freed fits."""
    observation = _observation(compact_config)
    fixed = run_psf_mismatch_case(
        _FakeValidator(),
        observation,
        compact_config,
        _trial(),
        fit_mode="fixed_template",
    )
    freed = run_psf_mismatch_case(
        _FakeValidator(),
        observation,
        compact_config,
        _trial(),
        fit_mode="freed",
        mass_context=SimpleNamespace(),
    )

    assert fixed.q_fit == 12.0
    assert fixed.detected_fit is True
    assert freed.q_fit == 12.0
    assert freed.detected_fit is None


def test_seed_reproducibility_and_independence(compact_config):
    """Isolate draws from truth contents and global NumPy RNG state."""
    first = _build(compact_config)
    np.random.seed(1)
    np.random.normal(size=100)
    second = _build(compact_config)
    changed_seed = copy.deepcopy(compact_config)
    changed_seed["modeling"]["fit_psf"]["delta"]["seed"] += 1
    third = _build(changed_seed)
    changed_truth = copy.deepcopy(compact_config)
    changed_truth["psf"]["aberrations"]["global_zernikes"][4] += 10.0
    fourth = _build(changed_truth)

    assert first.draw_aberrations == second.draw_aberrations
    assert first.draw_aberrations != third.draw_aberrations
    assert first.draw_aberrations == fourth.draw_aberrations


def test_lightweight_module_imports_and_bank_helper_reexports():
    """Keep mismatch and bank module bodies free of eager HCIPy imports."""
    for module_name, forbidden in (
        ("hwoslaps.modeling.nonlinear.psf_bank", "'hcipy'"),
        ("hwoslaps.psf.mismatch", "'hcipy', 'matplotlib'"),
    ):
        code = (
            f"import {module_name} as m; import sys; "
            f"missing = [n for n in ({forbidden},) if n in sys.modules]; "
            "assert not missing, missing"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    import hwoslaps.modeling.nonlinear.psf_bank as bank_module
    import hwoslaps.psf.mismatch as mismatch_module

    for name in (
        "_BANK_VERSION_PACKAGES",
        "_aberrations_from_wire",
        "_aberrations_to_wire",
        "_canonical_aberrations",
        "_current_versions",
        "_empty_aberrations",
        "_flat_int_map_from_wire",
        "_flat_int_map_to_wire",
        "_kernel_sha256",
        "_nested_int_map_from_wire",
        "_nested_int_map_to_wire",
        "_resolve_prior_table_path",
    ):
        assert getattr(bank_module, name) is getattr(mismatch_module, name)


def test_nonlinear_package_exposes_mismatch_symbols_lazily():
    """Expose all Item 9 nonlinear symbols through the package namespace."""
    import hwoslaps.modeling.nonlinear as nonlinear
    from hwoslaps.modeling.nonlinear.psf_mismatch import (
        PsfMismatchCaseResult,
    )
    from hwoslaps.psf.mismatch import PsfMismatchSpec

    expected = {
        "PsfMismatchSpec",
        "build_psf_mismatch_spec",
        "PsfMismatchCaseResult",
        "run_psf_mismatch_case",
    }
    assert expected <= set(dir(nonlinear))
    assert nonlinear.PsfMismatchSpec is PsfMismatchSpec
    assert nonlinear.PsfMismatchCaseResult is PsfMismatchCaseResult


def test_array_hash_includes_shape_and_dtype():
    """Distinguish equal bytes by shape and by actual dtype."""
    values = np.arange(8, dtype=np.uint8)
    assert _array_hash(values.reshape(2, 4)) != _array_hash(
        values.reshape(4, 2)
    )
    assert _array_hash(np.zeros(1, dtype=np.float32)) != _array_hash(
        np.zeros(1, dtype=np.int32)
    )


def test_analysis_key_binds_mask_and_pixel_scales():
    """Change the analysis key on each mask and pixel-scale field alone."""
    values = np.arange(36, dtype=np.float64).reshape(6, 6)
    base_mask = np.zeros((6, 6), dtype=bool)
    flipped_mask = base_mask.copy()
    flipped_mask[5, 5] = True
    metadata = {
        "dataset_kind": "asimov",
        "background_treatment": "subtract_known",
        "psf_truth_label": "truth",
        "psf_fit_label": "fit",
    }
    model_metadata = {"fit_mode": "freed", "prior_widths": {}}

    def dataset(**overrides):
        fields = {
            "data_mask": base_mask,
            "noise_mask": base_mask,
            "data_scales": (0.1, 0.1),
            "noise_scales": (0.1, 0.1),
            "psf_scales": (0.1, 0.1),
        }
        fields.update(overrides)
        return SimpleNamespace(
            data=SimpleNamespace(
                native=values,
                mask=fields["data_mask"],
                pixel_scales=fields["data_scales"],
            ),
            noise_map=SimpleNamespace(
                native=values,
                mask=fields["noise_mask"],
                pixel_scales=fields["noise_scales"],
            ),
            psf=SimpleNamespace(
                native=values,
                pixel_scales=fields["psf_scales"],
            ),
        )

    def key(**overrides):
        return analysis_key_from(
            dataset(**overrides),
            metadata,
            model_metadata,
        )

    base = key()
    variants = {
        "data_mask": key(data_mask=flipped_mask),
        "noise_mask": key(noise_mask=flipped_mask),
        "data_scales": key(data_scales=(0.2, 0.2)),
        "noise_scales": key(noise_scales=(0.2, 0.2)),
        "psf_scales": key(psf_scales=(0.2, 0.2)),
    }
    for field, variant in variants.items():
        assert variant != base, field
    assert len({base, *variants.values()}) == 6


@pytest.mark.parametrize(
    "validator,expected",
    [
        (_FakeValidator(smooth_status="failed"), {"fit_failed"}),
        (_FakeValidator(subhalo_status="failed"), {"fit_failed"}),
        (_FakeValidator(smooth_log_l=None), {"likelihood_unusable"}),
        (_FakeValidator(subhalo_log_l=float("nan")), {
            "likelihood_unusable"
        }),
        (_FakeValidator(smooth_log_l=float("inf")), {
            "likelihood_unusable"
        }),
        (_FakeValidator(smooth_logz=None), {"evidence_unusable"}),
        (_FakeValidator(subhalo_logz=float("nan")), {
            "evidence_unusable"
        }),
        (_FakeValidator(smooth_logz=float("inf")), {
            "evidence_unusable"
        }),
    ],
)
def test_executor_quality_flags(compact_config, validator, expected):
    """Flag failed fits and unusable successful likelihood/evidence values."""
    result = run_psf_mismatch_case(
        validator,
        _observation(compact_config),
        compact_config,
        _trial(),
        fit_mode="fixed_template",
    )
    assert expected <= set(result.quality_flags)
    if "fit_failed" in expected:
        assert result.q_fit is None
        assert result.delta_log_evidence is None
        assert result.detected_fit is None
        assert result.detected_evidence is None


def test_executor_cpu_path_labels_identity_and_freed_early_raise(
    compact_config,
):
    """Execute fixed-template and guard the freed mass context."""
    validator = _FakeValidator()
    result = run_psf_mismatch_case(
        validator,
        _observation(compact_config),
        compact_config,
        _trial(),
        fit_mode="fixed_template",
    )

    assert len(validator.calls) == 1
    assert result.psf_fit_label == f"delta:{result.delta_id}"
    assert validator.calls[0]["psf_case"] == result.psf_fit_label
    assert validator.calls[0]["metadata"].psf_fit_label == result.psf_fit_label
    assert validator.calls[0]["metadata"].psf_fit_supplied is True
    assert validator.calls[0]["smooth_result"] is None

    untouched = _FakeValidator()
    with pytest.raises(ValueError, match="freed mode requires mass_context"):
        run_psf_mismatch_case(
            untouched,
            _observation(compact_config),
            compact_config,
            _trial(),
        )
    assert untouched.calls == []
