"""Tests for the 2D Fisher sensitivity grid map (Item 1 / D5).

Runtime tests use the real AutoLens + HCIPy stack on a deliberately tiny
scene; layout, persistence, and plotting tests run on synthetic data.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("autolens")
pytest.importorskip("hcipy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.lensing import generate_lensing_system
from hwoslaps.modeling.fisher_detector import FisherDetector
from hwoslaps.modeling.generator_fisher import perform_fisher_detection
from hwoslaps.modeling.utils_fisher import (
    FisherDetectionData,
    FisherGridMapData,
    load_fisher_grid_map_npz,
    print_fisher_summary,
    save_fisher_grid_map_npz,
)
from hwoslaps.observation import generate_observation
from hwoslaps.pipeline import Pipeline
from hwoslaps.plotting.detection_plots import plot_fisher_detection_grid_map
from hwoslaps.provenance import config_hash
from hwoslaps.psf.generator import generate_psf_system

GRID_SPACING = 0.1
GRID_HALF_WIDTH = 0.2
SUBHALO_POSITION = (0.2, -0.1)
MISMATCH_FIELDS = {
    "mismatch_enabled",
    "amplitude_hat_mismatch",
    "q_mismatch",
    "z_mismatch",
    "amplitude_spurious",
    "q_spurious",
    "z_spurious",
}
MATCHED_NPZ_KEYS = {
    "y_coords",
    "x_coords",
    "spacing_arcsec",
    "centre_yx",
    "detection_q_threshold",
    "evaluated_mask_2d",
    "detectable_mask_2d",
    "q_asimov_2d",
    "z_asimov_2d",
    "fisher_raw_2d",
    "fisher_profiled_2d",
    "sigma_amplitude_profiled_2d",
    "degradation_2d",
    "absorbed_fraction_2d",
    "num_positions_evaluated",
    "num_detectable",
    "detectable_area_arcsec2",
    "max_z_asimov",
    "median_z_asimov",
    "subhalo_mass",
    "subhalo_model",
    "lens_einstein_radius",
    "nuisance_subset",
    "profiled_nuisance_names",
    "runtime_provenance_json",
}
MISMATCH_GRID_ARRAYS = (
    "amplitude_hat_2d",
    "q_mismatch_2d",
    "z_mismatch_2d",
    "mismatch_detectable_mask_2d",
    "amplitude_spurious_2d",
    "q_spurious_2d",
    "z_spurious_2d",
    "false_positive_mask_2d",
)


def _load_master_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "master_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_grid_config(tmp_dir: Path) -> dict:
    config = _load_master_config()
    config["run_name"] = "fisher_grid_map_test"
    config["plotting"]["enabled"] = False
    config["plotting"]["output_dir"] = str(tmp_dir)

    config["lensing"]["grid"] = {"shape": [31, 31], "pixel_scale": 0.1}
    config["lensing"]["lens_galaxy"]["mass"]["einstein_radius"] = 0.5
    config["lensing"]["lens_galaxy"]["mass"]["centre"] = [0.0, 0.0]
    config["lensing"]["source_galaxy"]["light"]["centre"] = [0.0, 0.0]
    config["lensing"]["source_galaxy"]["light"]["ell_comps"] = [0.03, -0.01]
    config["lensing"]["source_galaxy"]["light"]["intensity"] = 5.0
    config["lensing"]["source_galaxy"]["light"]["effective_radius"] = 0.2
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "model": "PointMass",
        "mass": 1.0e8,
        "position": {"type": "direct", "centre": list(SUBHALO_POSITION)},
    }

    config["psf"]["kernel"]["shape_native"] = [11, 11]
    config["psf"]["hres_psf"]["num_pix"] = 128
    config["psf"]["hres_psf"]["num_airy"] = 6
    config["psf"]["hres_psf"]["save_highres_psf_npy"] = False

    aberr = config["psf"]["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = True
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {4: 20.0}

    config["observation"]["exposure_time"] = 200.0
    config["observation"]["detector"]["sky_background"] = 0.5

    config["modeling"]["detection"] = "fisher"
    config["modeling"]["fisher"]["mode"] = "map"
    config["modeling"]["fisher"]["mask_mode"] = "all_pixels"
    config["modeling"]["fisher"]["include_psf_nuisance"] = False
    config["modeling"]["fisher"]["compute_psf_mode_scan"] = False
    config["modeling"]["fisher"]["map"] = {
        "type": "grid",
        "grid": {
            "spacing_arcsec": GRID_SPACING,
            "half_width_arcsec": GRID_HALF_WIDTH,
            "annulus": None,
        },
        "detection_q_threshold": 10.0,
        "num_workers": 1,
    }
    return config


@pytest.fixture(scope="module")
def grid_setup(tmp_path_factory):
    """Build a tiny grid-map scene and its Fisher products once."""
    tmp_dir = tmp_path_factory.mktemp("fisher-grid")
    os.environ["NUMBA_CACHE_DIR"] = str(tmp_dir / "numba-cache")
    os.environ["MPLCONFIGDIR"] = str(tmp_dir / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(tmp_dir / "xdg-cache")

    config = _build_grid_config(tmp_dir)
    config_baseline = copy.deepcopy(config)
    config_baseline["lensing"]["subhalo"]["enabled"] = False

    psf_data = generate_psf_system(config["psf"], full_config=config)
    lensing_baseline = generate_lensing_system(
        config_baseline["lensing"],
        full_config=config_baseline,
    )
    lensing_test = generate_lensing_system(config["lensing"], full_config=config)
    observation_baseline = generate_observation(
        lensing_baseline,
        psf_data,
        observation_config=config_baseline["observation"],
        full_config=config_baseline,
    )
    observation_test = generate_observation(
        lensing_test,
        psf_data,
        observation_config=config["observation"],
        full_config=config,
    )

    detector = FisherDetector(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=config,
        fisher_config=copy.deepcopy(config["modeling"]["fisher"]),
    )
    grid_map = detector.compute_grid_map()

    return {
        "config": config,
        "psf_data": psf_data,
        "lensing_baseline": lensing_baseline,
        "lensing_test": lensing_test,
        "observation_baseline": observation_baseline,
        "observation_test": observation_test,
        "detector": detector,
        "grid_map": grid_map,
    }


def _make_detector(grid_setup, fisher_map_config: dict) -> FisherDetector:
    config = copy.deepcopy(grid_setup["config"])
    config["modeling"]["fisher"]["map"] = fisher_map_config
    return FisherDetector(
        observation_baseline=grid_setup["observation_baseline"],
        lensing_baseline=grid_setup["lensing_baseline"],
        psf_data=grid_setup["psf_data"],
        full_config=config,
        fisher_config=copy.deepcopy(config["modeling"]["fisher"]),
    )


def _perfect_psf_config(config: dict) -> dict:
    psf = copy.deepcopy(config["psf"])
    aberr = psf["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}
    return psf


def _make_mismatch_detector(
    grid_setup,
    fit_psf: dict,
    fisher_map_config: dict | None = None,
) -> FisherDetector:
    config = copy.deepcopy(grid_setup["config"])
    config["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": copy.deepcopy(fit_psf),
    }
    if fisher_map_config is not None:
        config["modeling"]["fisher"]["map"] = fisher_map_config
    return FisherDetector(
        observation_baseline=grid_setup["observation_baseline"],
        lensing_baseline=grid_setup["lensing_baseline"],
        psf_data=grid_setup["psf_data"],
        full_config=config,
        fisher_config=copy.deepcopy(config["modeling"]["fisher"]),
    )


@pytest.fixture(scope="module")
def mismatch_setup(grid_setup):
    """Build one truth-aberrated, perfect-fit mismatch detector and outputs."""
    detector = _make_mismatch_detector(
        grid_setup,
        _perfect_psf_config(grid_setup["config"]),
    )
    return {
        "detector": detector,
        "grid_map": detector.compute_grid_map(),
        "local": detector.compute_local(
            observation_test=grid_setup["observation_test"],
            lensing_test=grid_setup["lensing_test"],
        ),
    }


def _layout_stub(map_config: dict, lens_centre) -> FisherDetector:
    detector = FisherDetector.__new__(FisherDetector)
    detector.map_config = map_config
    detector.map_type = str(map_config.get("type", "")).lower()
    detector._candidate_positions_cache = None
    detector._grid_layout_cache = None
    detector.full_config = {
        "lensing": {"lens_galaxy": {"mass": {"centre": list(lens_centre)}}}
    }
    return detector


def _corrupt_grid_npz(source, destination, delete=()):
    """Delete selected NPZ members from a real grid-map artifact."""
    with np.load(source, allow_pickle=False) as stored:
        payload = {
            name: np.array(stored[name], copy=True)
            for name in stored.files
            if name not in delete
        }
    with Path(destination).open("wb") as stream:
        np.savez_compressed(stream, **payload)
    return Path(destination)


def _stub_pipeline_grid_result(grid_setup, monkeypatch):
    """Replace expensive pipeline stages with one synthetic grid result."""
    import hwoslaps.modeling.generator_fisher as generator_fisher
    import hwoslaps.pipeline as pipeline_module

    grid_map = replace(
        grid_setup["grid_map"],
        config_hash=None,
        git_hash=None,
    )
    result = FisherDetectionData(
        mode="map",
        local=None,
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=1,
        n_nuisance=0,
        gram_condition_number=1.0,
        pixel_scale=0.1,
        grid_map=grid_map,
    )
    stub = object()
    monkeypatch.setattr(
        pipeline_module,
        "generate_psf_system",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_lensing_system",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_observation",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        generator_fisher,
        "perform_fisher_detection",
        lambda **kwargs: result,
    )
    return result


# ----------------------------------------------------------------------
# Layout geometry (no heavy stack beyond imports)
# ----------------------------------------------------------------------


def test_grid_layout_geometry_centred_on_lens():
    """Place grid nodes on a regular lattice centred on the lens."""
    detector = _layout_stub(
        {
            "type": "grid",
            "grid": {"spacing_arcsec": 0.05, "half_width_arcsec": 0.2, "annulus": None},
        },
        lens_centre=(0.05, -0.1),
    )
    layout = detector._grid_layout()

    assert layout.y_coords.size == 9
    assert layout.x_coords.size == 9
    np.testing.assert_allclose(np.diff(layout.y_coords), 0.05)
    np.testing.assert_allclose(np.diff(layout.x_coords), 0.05)
    np.testing.assert_allclose(layout.y_coords[4], 0.05)
    np.testing.assert_allclose(layout.x_coords[4], -0.1)
    assert layout.evaluated_mask.all()
    assert len(layout.positions_yx) == 81
    assert layout.node_indices[0] == (0, 0)
    np.testing.assert_allclose(
        layout.positions_yx[0],
        (layout.y_coords[0], layout.x_coords[0]),
    )


def test_grid_layout_annulus_restricts_nodes():
    """Evaluate only the nodes falling inside the requested annulus."""
    detector = _layout_stub(
        {
            "type": "grid",
            "grid": {
                "spacing_arcsec": 0.1,
                "half_width_arcsec": 0.2,
                "annulus": {"r_min_arcsec": 0.05, "r_max_arcsec": 0.15},
            },
        },
        lens_centre=(0.0, 0.0),
    )
    layout = detector._grid_layout()

    radii = np.hypot(
        layout.y_coords[:, None],
        layout.x_coords[None, :],
    )
    expected = (radii >= 0.05) & (radii <= 0.15)
    np.testing.assert_array_equal(layout.evaluated_mask, expected)
    assert len(layout.positions_yx) == int(np.count_nonzero(expected))
    assert len(layout.positions_yx) == 8


def test_grid_layout_rejects_empty_annulus():
    """Reject an annulus that selects no grid node at all."""
    detector = _layout_stub(
        {
            "type": "grid",
            "grid": {
                "spacing_arcsec": 0.1,
                "half_width_arcsec": 0.2,
                "annulus": {"r_min_arcsec": 0.01, "r_max_arcsec": 0.02},
            },
        },
        lens_centre=(0.0, 0.0),
    )

    with pytest.raises(ValueError, match="annulus selects no grid nodes"):
        detector._grid_layout()


# ----------------------------------------------------------------------
# Runtime grid map behavior
# ----------------------------------------------------------------------


def test_grid_map_schema_and_area(grid_setup):
    """Check grid-map array shapes, masks, and detectable area."""
    grid_map = grid_setup["grid_map"]
    assert isinstance(grid_map, FisherGridMapData)

    shape = (grid_map.y_coords.size, grid_map.x_coords.size)
    assert shape == (5, 5)
    assert grid_map.num_positions_evaluated == 25
    assert grid_map.evaluated_mask_2d.all()
    for array in (
        grid_map.q_asimov_2d,
        grid_map.z_asimov_2d,
        grid_map.fisher_raw_2d,
        grid_map.fisher_profiled_2d,
        grid_map.sigma_amplitude_profiled_2d,
        grid_map.degradation_2d,
        grid_map.absorbed_fraction_2d,
    ):
        assert array.shape == shape
    assert np.all(np.isfinite(grid_map.q_asimov_2d))

    expected_detectable = grid_map.q_asimov_2d >= grid_map.detection_q_threshold
    np.testing.assert_array_equal(grid_map.detectable_mask_2d, expected_detectable)
    assert grid_map.num_detectable == int(np.count_nonzero(expected_detectable))
    np.testing.assert_allclose(
        grid_map.detectable_area_arcsec2,
        grid_map.num_detectable * GRID_SPACING**2,
    )
    np.testing.assert_allclose(grid_map.max_z_asimov, np.max(grid_map.z_asimov_2d))
    assert grid_map.subhalo_mass == pytest.approx(1.0e8)
    assert grid_map.subhalo_model == "PointMass"
    assert grid_map.lens_einstein_radius == pytest.approx(0.5)


def test_no_fit_psf_block_leaves_mismatch_outputs_empty(grid_setup, tmp_path):
    """Preserve no-block result fields and the legacy NPZ key set."""
    local = grid_setup["detector"].compute_local(
        observation_test=grid_setup["observation_test"],
        lensing_test=grid_setup["lensing_test"],
    )
    grid_map = grid_setup["grid_map"]

    assert grid_setup["detector"].mu0_model_adu_2d is (
        grid_setup["detector"].mu0_adu_2d
    )
    assert local.mismatch_enabled is False
    for name in MISMATCH_FIELDS - {"mismatch_enabled"}:
        assert getattr(local, name) is None
    assert grid_map.mismatch_enabled is False
    for name in (
        "amplitude_hat_2d",
        "q_mismatch_2d",
        "z_mismatch_2d",
        "mismatch_detectable_mask_2d",
        "mismatch_detectable_area_arcsec2",
        "num_mismatch_detectable",
        "amplitude_spurious_2d",
        "q_spurious_2d",
        "z_spurious_2d",
        "false_positive_mask_2d",
        "false_positive_area_arcsec2",
        "num_false_positive",
        "max_z_spurious",
    ):
        assert getattr(grid_map, name) is None

    npz_path = save_fisher_grid_map_npz(grid_map, tmp_path / "matched.npz")
    with np.load(npz_path) as data:
        assert set(data.files) == MATCHED_NPZ_KEYS


def test_identical_explicit_fit_psf_recovers_matched_limit(grid_setup):
    """Recover matched local and grid outputs for an identical explicit PSF."""
    detector = _make_mismatch_detector(grid_setup, grid_setup["config"]["psf"])
    local = detector.compute_local(
        observation_test=grid_setup["observation_test"],
        lensing_test=grid_setup["lensing_test"],
    )
    matched_local = grid_setup["detector"].compute_local(
        observation_test=grid_setup["observation_test"],
        lensing_test=grid_setup["lensing_test"],
    )
    for name in matched_local.__dataclass_fields__:
        if name in MISMATCH_FIELDS:
            continue
        expected = getattr(matched_local, name)
        actual = getattr(local, name)
        if isinstance(expected, float):
            # The mismatch arm regenerates the fit PSF independently; that
            # regeneration is not guaranteed bitwise-reproducible (one-ULP
            # FFT differences from process state), so the matched limit is
            # exact only to roundoff amplified through the Fisher solves.
            np.testing.assert_allclose(actual, expected, rtol=1.0e-10)
        else:
            assert actual == expected

    assert local.mismatch_enabled is True
    np.testing.assert_allclose(local.q_mismatch, local.q_asimov_local, rtol=1.0e-10)
    assert local.amplitude_spurious == pytest.approx(0.0, abs=1.0e-12)

    grid_map = detector.compute_grid_map()
    matched_grid_map = grid_setup["grid_map"]
    for name in (
        "q_asimov_2d",
        "z_asimov_2d",
        "fisher_raw_2d",
        "fisher_profiled_2d",
        "sigma_amplitude_profiled_2d",
        "degradation_2d",
        "absorbed_fraction_2d",
    ):
        np.testing.assert_allclose(
            getattr(grid_map, name),
            getattr(matched_grid_map, name),
            rtol=1.0e-10,
        )
    np.testing.assert_array_equal(
        grid_map.detectable_mask_2d,
        matched_grid_map.detectable_mask_2d,
    )
    np.testing.assert_allclose(
        grid_map.q_mismatch_2d,
        grid_map.q_asimov_2d,
        rtol=1.0e-10,
    )
    assert grid_map.false_positive_area_arcsec2 == pytest.approx(0.0)
    assert grid_map.num_false_positive == 0


def test_local_spurious_q_scales_with_psf_offset_squared(grid_setup):
    """Scale spurious q quadratically with a small PSF offset."""
    q_values = []
    for offset_nm in (0.5, 1.0):
        fit_psf = copy.deepcopy(grid_setup["config"]["psf"])
        fit_psf["aberrations"]["global_zernikes"][4] = 20.0 + offset_nm
        detector = _make_mismatch_detector(grid_setup, fit_psf)
        local = detector.compute_local(
            observation_test=grid_setup["observation_test"],
            lensing_test=grid_setup["lensing_test"],
        )
        q_values.append(local.q_spurious)

    ratio = q_values[1] / q_values[0]
    assert ratio == pytest.approx(4.0, rel=0.15)


def test_mismatch_grid_endpoint_has_structure_and_signed_masks(mismatch_setup):
    """Produce structured spurious statistics and enforce one-sided masks."""
    grid_map = mismatch_setup["grid_map"]
    evaluated = grid_map.evaluated_mask_2d

    assert not np.all(np.isnan(grid_map.q_spurious_2d))
    assert np.nanmax(grid_map.q_spurious_2d) > np.nanmedian(grid_map.q_spurious_2d)
    expected_mismatch = (
        evaluated
        & (grid_map.amplitude_hat_2d > 0.0)
        & (grid_map.q_mismatch_2d >= grid_map.detection_q_threshold)
    )
    expected_false_positive = (
        evaluated
        & (grid_map.amplitude_spurious_2d > 0.0)
        & (grid_map.q_spurious_2d >= grid_map.detection_q_threshold)
    )
    np.testing.assert_array_equal(
        grid_map.mismatch_detectable_mask_2d,
        expected_mismatch,
    )
    np.testing.assert_array_equal(
        grid_map.false_positive_mask_2d,
        expected_false_positive,
    )
    assert grid_map.mismatch_detectable_area_arcsec2 == pytest.approx(
        np.count_nonzero(expected_mismatch) * GRID_SPACING**2
    )
    assert grid_map.false_positive_area_arcsec2 == pytest.approx(
        np.count_nonzero(expected_false_positive) * GRID_SPACING**2
    )
    assert grid_map.max_z_spurious == pytest.approx(
        np.nanmax(grid_map.z_spurious_2d)
    )


def test_grid_map_matches_explicit_bank_path(grid_setup):
    """Match grid-map values against the explicit position bank path."""
    grid_map = grid_setup["grid_map"]
    layout = grid_setup["detector"]._grid_layout()

    explicit_detector = _make_detector(
        grid_setup,
        {
            "type": "explicit",
            "explicit_positions_yx": [list(pos) for pos in layout.positions_yx],
        },
    )
    bank = explicit_detector.compute_map()

    node_idx = np.asarray(layout.node_indices, dtype=int)
    grid_q = grid_map.q_asimov_2d[node_idx[:, 0], node_idx[:, 1]]
    grid_raw = grid_map.fisher_raw_2d[node_idx[:, 0], node_idx[:, 1]]
    grid_profiled = grid_map.fisher_profiled_2d[node_idx[:, 0], node_idx[:, 1]]

    np.testing.assert_allclose(grid_q, bank.q_asimov_local_by_position, rtol=1.0e-10)
    np.testing.assert_allclose(grid_raw, bank.fisher_raw_by_position, rtol=1.0e-10)
    np.testing.assert_allclose(
        grid_profiled, bank.fisher_profiled_by_position, rtol=1.0e-10
    )


def test_grid_map_node_matches_compute_local(grid_setup):
    """Match the node at the true subhalo against compute_local."""
    grid_map = grid_setup["grid_map"]
    local = grid_setup["detector"].compute_local(
        observation_test=grid_setup["observation_test"],
        lensing_test=grid_setup["lensing_test"],
    )

    y_idx = int(np.argmin(np.abs(grid_map.y_coords - SUBHALO_POSITION[0])))
    x_idx = int(np.argmin(np.abs(grid_map.x_coords - SUBHALO_POSITION[1])))
    assert grid_map.y_coords[y_idx] == pytest.approx(SUBHALO_POSITION[0])
    assert grid_map.x_coords[x_idx] == pytest.approx(SUBHALO_POSITION[1])

    np.testing.assert_allclose(
        grid_map.q_asimov_2d[y_idx, x_idx],
        local.q_asimov_local,
        rtol=1.0e-6,
    )


def test_mismatch_grid_node_matches_compute_local(mismatch_setup):
    """Match mismatch q at the injected grid node against compute_local."""
    grid_map = mismatch_setup["grid_map"]
    local = mismatch_setup["local"]
    y_idx = int(np.argmin(np.abs(grid_map.y_coords - SUBHALO_POSITION[0])))
    x_idx = int(np.argmin(np.abs(grid_map.x_coords - SUBHALO_POSITION[1])))

    np.testing.assert_allclose(
        grid_map.q_mismatch_2d[y_idx, x_idx],
        local.q_mismatch,
        rtol=1.0e-8,
    )


def test_grid_map_annulus_marks_unevaluated_nodes(grid_setup):
    """Mark nodes outside the annulus unevaluated and leave them NaN."""
    detector = _make_detector(
        grid_setup,
        {
            "type": "grid",
            "grid": {
                "spacing_arcsec": GRID_SPACING,
                "half_width_arcsec": GRID_HALF_WIDTH,
                "annulus": {"r_min_arcsec": 0.05, "r_max_arcsec": 0.15},
            },
            "detection_q_threshold": 10.0,
            "num_workers": 1,
        },
    )
    grid_map = detector.compute_grid_map()

    assert grid_map.num_positions_evaluated == 8
    assert int(np.count_nonzero(grid_map.evaluated_mask_2d)) == 8
    outside = ~grid_map.evaluated_mask_2d
    assert np.all(np.isnan(grid_map.q_asimov_2d[outside]))
    assert not np.any(grid_map.detectable_mask_2d[outside])
    evaluated_q = grid_map.q_asimov_2d[grid_map.evaluated_mask_2d]
    assert np.all(np.isfinite(evaluated_q))
    np.testing.assert_allclose(
        grid_map.detectable_area_arcsec2,
        np.count_nonzero(evaluated_q >= grid_map.detection_q_threshold)
        * GRID_SPACING**2,
    )


def test_grid_map_parallel_matches_serial(grid_setup):
    """Produce identical grid maps with one and with two workers."""
    detector = _make_detector(
        grid_setup,
        {
            "type": "grid",
            "grid": {
                "spacing_arcsec": GRID_SPACING,
                "half_width_arcsec": GRID_HALF_WIDTH,
                "annulus": None,
            },
            "detection_q_threshold": 10.0,
            "num_workers": 2,
        },
    )
    grid_map_parallel = detector.compute_grid_map()
    grid_map_serial = grid_setup["grid_map"]

    np.testing.assert_allclose(
        grid_map_parallel.q_asimov_2d,
        grid_map_serial.q_asimov_2d,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        grid_map_parallel.fisher_raw_2d,
        grid_map_serial.fisher_raw_2d,
        rtol=1.0e-12,
    )
    assert grid_map_parallel.num_detectable == grid_map_serial.num_detectable


@pytest.mark.xtx_gpu
def test_grid_map_worker_env_override_matches_serial(grid_setup, monkeypatch):
    """Widen the pool from the environment without changing the map.

    ``num_workers`` is hashed into the run's provenance, so the runtime
    override exists to fan out a configured serial map. It must leave the
    map bit-identical and leave the configured value untouched.
    """
    monkeypatch.setenv("HWOSLAPS_FISHER_GRID_WORKERS", "2")
    detector = _make_detector(
        grid_setup,
        {
            "type": "grid",
            "grid": {
                "spacing_arcsec": GRID_SPACING,
                "half_width_arcsec": GRID_HALF_WIDTH,
                "annulus": None,
            },
            "detection_q_threshold": 10.0,
            "num_workers": 1,
        },
    )
    assert detector._grid_num_workers() == 2
    assert detector.map_config["num_workers"] == 1
    grid_map = detector.compute_grid_map()
    grid_map_serial = grid_setup["grid_map"]

    np.testing.assert_array_equal(
        grid_map.q_asimov_2d,
        grid_map_serial.q_asimov_2d,
    )
    np.testing.assert_array_equal(
        grid_map.fisher_raw_2d,
        grid_map_serial.fisher_raw_2d,
    )
    np.testing.assert_array_equal(
        grid_map.absorbed_fraction_2d,
        grid_map_serial.absorbed_fraction_2d,
    )
    assert grid_map.num_detectable == grid_map_serial.num_detectable


@pytest.mark.xtx_gpu
def test_grid_map_rejects_invalid_worker_env_override(grid_setup, monkeypatch):
    """Reject a non-positive runtime worker override rather than ignore it."""
    monkeypatch.setenv("HWOSLAPS_FISHER_GRID_WORKERS", "0")
    config = copy.deepcopy(grid_setup["config"])
    config["modeling"]["fisher"]["map"]["engine"] = "reference"
    detector = _make_detector(
        grid_setup,
        config["modeling"]["fisher"]["map"],
    )
    with pytest.raises(ValueError, match="HWOSLAPS_FISHER_GRID_WORKERS"):
        detector.compute_grid_map()


@pytest.mark.xtx_gpu
def test_jax_ignores_invalid_worker_env_override(grid_setup, monkeypatch):
    """Do not validate a reference-only override on the JAX dispatch path."""
    monkeypatch.setenv("HWOSLAPS_FISHER_GRID_WORKERS", "junk")
    map_config = copy.deepcopy(grid_setup["config"]["modeling"]["fisher"]["map"])
    map_config["engine"] = "jax"
    detector = _make_detector(grid_setup, map_config)
    assert detector._grid_num_workers() == int(map_config.get("num_workers", 1))


def test_grid_runtime_provenance_separates_requested_and_effective_workers(
    monkeypatch,
):
    """Record runtime parallelism without changing the configured map."""
    detector = FisherDetector.__new__(FisherDetector)
    detector.map_config = {"engine": "reference", "num_workers": 1}
    monkeypatch.setenv("HWOSLAPS_FISHER_GRID_WORKERS", "2")
    assert detector._grid_runtime_provenance() == {
        "fisher_grid_workers_requested": 1,
        "fisher_grid_workers_effective": 2,
        "fisher_grid_start_method": "spawn",
    }


@pytest.mark.xtx_gpu
def test_mismatch_grid_map_parallel_matches_serial(grid_setup, mismatch_setup):
    """Produce every mismatch array identically with one and two workers."""
    map_config = copy.deepcopy(grid_setup["config"]["modeling"]["fisher"]["map"])
    map_config["num_workers"] = 2
    detector = _make_mismatch_detector(
        grid_setup,
        _perfect_psf_config(grid_setup["config"]),
        map_config,
    )
    parallel = detector.compute_grid_map()
    serial = mismatch_setup["grid_map"]

    for name in MISMATCH_GRID_ARRAYS:
        np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))


@pytest.mark.xtx_gpu
def test_explicit_fit_lens_grid_map_parallel_matches_serial(grid_setup):
    """Exercise exact mismatch parity for an explicit fit lens full map."""
    config = copy.deepcopy(grid_setup["config"])
    fit_lens = {
        "mass": copy.deepcopy(config["lensing"]["lens_galaxy"]["mass"])
    }
    fit_lens["mass"]["centre"] = [0.01, -0.02]
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": fit_lens,
    }

    parallel_config = copy.deepcopy(config["modeling"]["fisher"]["map"])
    parallel_config["num_workers"] = 2
    serial_config = copy.deepcopy(parallel_config)
    serial_config["num_workers"] = 1

    parallel = FisherDetector(
        observation_baseline=grid_setup["observation_baseline"],
        lensing_baseline=grid_setup["lensing_baseline"],
        psf_data=grid_setup["psf_data"],
        full_config=config,
        fisher_config={**config["modeling"]["fisher"], "map": parallel_config},
    )
    serial = FisherDetector(
        observation_baseline=grid_setup["observation_baseline"],
        lensing_baseline=grid_setup["lensing_baseline"],
        psf_data=grid_setup["psf_data"],
        full_config=config,
        fisher_config={**config["modeling"]["fisher"], "map": serial_config},
    )
    parallel_map = parallel.compute_grid_map()
    serial_map = serial.compute_grid_map()

    for name in MISMATCH_GRID_ARRAYS:
        np.testing.assert_array_equal(
            getattr(parallel_map, name), getattr(serial_map, name)
        )


def test_generator_dispatches_grid_map(grid_setup):
    """Route a grid map config through perform_fisher_detection."""
    config = copy.deepcopy(grid_setup["config"])
    result = perform_fisher_detection(
        observation_baseline=grid_setup["observation_baseline"],
        observation_test=grid_setup["observation_test"],
        lensing_baseline=grid_setup["lensing_baseline"],
        lensing_test=grid_setup["lensing_test"],
        psf_data=grid_setup["psf_data"],
        detection_config=config["modeling"],
        full_config=config,
    )

    assert isinstance(result, FisherDetectionData)
    assert result.has_grid_map
    assert result.map is None
    assert result.grid_map.num_positions_evaluated == 25
    assert result.runtime_provenance == result.grid_map.runtime_provenance


def test_grid_map_requires_grid_type(grid_setup):
    """Reject compute_grid_map when map.type is not 'grid'."""
    detector = _make_detector(
        grid_setup,
        {
            "type": "explicit",
            "explicit_positions_yx": [[0.0, 0.0]],
        },
    )

    with pytest.raises(ValueError, match="requires modeling.fisher.map.type: 'grid'"):
        detector.compute_grid_map()


def test_mismatch_compute_map_requires_grid_type(grid_setup):
    """Reject the legacy ring bank when explicit PSF mismatch is enabled."""
    detector = _make_mismatch_detector(
        grid_setup,
        _perfect_psf_config(grid_setup["config"]),
        {
            "type": "ring",
            "ring": {"num_angles": 4, "offset_pixels": 0.0},
            "detection_q_threshold": 10.0,
            "num_workers": 1,
        },
    )

    with pytest.raises(ValueError, match="map.type: grid"):
        detector.compute_map()


# ----------------------------------------------------------------------
# Persistence and plotting
# ----------------------------------------------------------------------


def test_grid_map_npz_roundtrip(grid_setup, tmp_path):
    """Round-trip a grid map through its npz representation."""
    grid_map = grid_setup["grid_map"]
    npz_path = save_fisher_grid_map_npz(grid_map, tmp_path / "fisher_grid_map.npz")

    with np.load(npz_path) as data:
        np.testing.assert_array_equal(data["y_coords"], grid_map.y_coords)
        np.testing.assert_array_equal(data["x_coords"], grid_map.x_coords)
        np.testing.assert_array_equal(data["q_asimov_2d"], grid_map.q_asimov_2d)
        np.testing.assert_array_equal(
            data["detectable_mask_2d"], grid_map.detectable_mask_2d
        )
        np.testing.assert_array_equal(
            data["evaluated_mask_2d"], grid_map.evaluated_mask_2d
        )
        assert float(data["spacing_arcsec"]) == pytest.approx(grid_map.spacing_arcsec)
        assert float(data["detection_q_threshold"]) == pytest.approx(
            grid_map.detection_q_threshold
        )
        assert float(data["detectable_area_arcsec2"]) == pytest.approx(
            grid_map.detectable_area_arcsec2
        )
        assert int(data["num_positions_evaluated"]) == grid_map.num_positions_evaluated
        assert str(data["subhalo_model"]) == "PointMass"

    loaded = load_fisher_grid_map_npz(npz_path)
    assert loaded.source_image_asset_path is None
    assert loaded.source_image_asset_sha256_16 is None


def test_grid_map_npz_roundtrip_preserves_optional_provenance(grid_setup, tmp_path):
    """Round-trip embedded configuration and git hashes when present."""
    grid_map = replace(
        grid_setup["grid_map"],
        config_hash="0123456789abcdef",
        git_hash="f"*40,
        campaign_uuid="123e4567-e89b-12d3-a456-426614174000",
        git_dirty=True,
        worktree_diff_sha256="e"*64,
        runtime_provenance={
            "fisher_grid_workers_requested": 1,
            "fisher_grid_workers_effective": 2,
            "fisher_grid_start_method": "spawn",
        },
    )
    npz_path = save_fisher_grid_map_npz(grid_map, tmp_path / "provenance.npz")

    with np.load(npz_path, allow_pickle=False) as stored:
        assert str(stored["config_hash"]) == grid_map.config_hash
        assert str(stored["git_hash"]) == grid_map.git_hash
        assert str(stored["campaign_uuid"]) == grid_map.campaign_uuid
        assert bool(stored["git_dirty"])
        assert str(stored["worktree_diff_sha256"]) == grid_map.worktree_diff_sha256
        assert json.loads(str(stored["runtime_provenance_json"])) == (
            grid_map.runtime_provenance
        )
    loaded = load_fisher_grid_map_npz(npz_path)
    assert loaded.config_hash == grid_map.config_hash
    assert loaded.git_hash == grid_map.git_hash
    assert loaded.campaign_uuid == grid_map.campaign_uuid
    assert loaded.git_dirty is True
    assert loaded.worktree_diff_sha256 == grid_map.worktree_diff_sha256
    assert loaded.runtime_provenance == grid_map.runtime_provenance


def test_grid_map_npz_old_format_loads_missing_provenance_as_none(
    grid_setup,
    tmp_path,
):
    """Load an old-format artifact with no embedded provenance members."""
    grid_map = replace(
        grid_setup["grid_map"],
        config_hash="0123456789abcdef",
        git_hash="f"*40,
        campaign_uuid="123e4567-e89b-12d3-a456-426614174000",
    )
    current = save_fisher_grid_map_npz(grid_map, tmp_path / "current.npz")
    old_format = _corrupt_grid_npz(
        current,
        tmp_path / "old-format.npz",
        delete=(
            "config_hash",
            "git_hash",
            "git_dirty",
            "worktree_diff_sha256",
            "runtime_provenance_json",
            "campaign_uuid",
        ),
    )

    loaded = load_fisher_grid_map_npz(old_format)
    assert loaded.config_hash is None
    assert loaded.git_hash is None
    assert loaded.git_dirty is None
    assert loaded.worktree_diff_sha256 is None
    assert loaded.runtime_provenance is None
    assert loaded.campaign_uuid is None


def test_pipeline_populates_grid_map_provenance(
    grid_setup,
    tmp_path,
    monkeypatch,
):
    """Populate embedded hashes before a pipeline grid map is persisted."""
    import hwoslaps.modeling.generator_fisher as generator_fisher
    import hwoslaps.pipeline as pipeline_module

    config = copy.deepcopy(grid_setup["config"])
    config["run_name"] = "pipeline-provenance"
    config["plotting"]["enabled"] = False
    config["plotting"]["output_dir"] = str(tmp_path)
    snapshot = copy.deepcopy(config)
    run_dir = tmp_path / config["run_name"]
    run_dir.mkdir(parents=True)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(snapshot, stream, sort_keys=False)
    grid_map = replace(
        grid_setup["grid_map"],
        config_hash=None,
        git_hash=None,
    )
    result = FisherDetectionData(
        mode="map",
        local=None,
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=1,
        n_nuisance=0,
        gram_condition_number=1.0,
        pixel_scale=0.1,
        grid_map=grid_map,
    )
    stub = object()
    monkeypatch.setattr(pipeline_module, "generate_psf_system", lambda *args, **kwargs: stub)
    monkeypatch.setattr(
        pipeline_module,
        "generate_lensing_system",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_observation",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        generator_fisher,
        "perform_fisher_detection",
        lambda **kwargs: result,
    )

    monkeypatch.setenv(
        "HWOSLAPS_CAMPAIGN_UUID", "123e4567-e89b-12d3-a456-426614174000"
    )
    Pipeline(verbose=False)._run_detection_pipeline(config)

    path = tmp_path / config["run_name"] / "modeling" / "fisher_grid_map.npz"
    loaded = load_fisher_grid_map_npz(path)
    assert loaded.config_hash == config_hash(snapshot)
    assert loaded.git_hash is not None
    assert loaded.campaign_uuid == "123e4567-e89b-12d3-a456-426614174000"


def test_pipeline_omits_config_hash_without_snapshot(
    grid_setup,
    tmp_path,
    monkeypatch,
):
    """Leave direct-pipeline maps unbound when no runner snapshot exists."""
    config = copy.deepcopy(grid_setup["config"])
    config["run_name"] = "pipeline-no-snapshot"
    config["plotting"]["enabled"] = False
    config["plotting"]["output_dir"] = str(tmp_path)
    _stub_pipeline_grid_result(grid_setup, monkeypatch)

    monkeypatch.delenv("HWOSLAPS_CAMPAIGN_UUID", raising=False)
    Pipeline(verbose=False)._run_detection_pipeline(config)

    path = tmp_path / config["run_name"] / "modeling" / "fisher_grid_map.npz"
    loaded = load_fisher_grid_map_npz(path)
    assert loaded.config_hash is None
    assert loaded.git_hash is not None
    assert loaded.campaign_uuid is None


def test_pipeline_snapshot_hash_rejects_bool_int_alias(
    grid_setup,
    tmp_path,
    monkeypatch,
):
    """Reject a snapshot whose boolean only compares equal to integer one."""
    config = copy.deepcopy(grid_setup["config"])
    config["run_name"] = "pipeline-bool-int-snapshot"
    config["plotting"]["enabled"] = False
    config["plotting"]["output_dir"] = str(tmp_path)
    snapshot = copy.deepcopy(config)
    snapshot["modeling"]["fisher"]["map"]["num_workers"] = True
    run_dir = tmp_path / config["run_name"]
    run_dir.mkdir(parents=True)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(snapshot, stream, sort_keys=False)
    _stub_pipeline_grid_result(grid_setup, monkeypatch)

    with pytest.raises(ValueError, match="does not describe this run"):
        Pipeline(verbose=False)._run_detection_pipeline(config)


def test_pipeline_snapshot_hash_accepts_yaml_sequence_roundtrip(
    grid_setup,
    tmp_path,
    monkeypatch,
):
    """Accept a tuple that a YAML snapshot canonically reloads as a list."""
    config = copy.deepcopy(grid_setup["config"])
    config["run_name"] = "pipeline-sequence-snapshot"
    config["plotting"]["enabled"] = False
    config["plotting"]["output_dir"] = str(tmp_path)
    config["lensing"]["grid"]["shape"] = tuple(
        config["lensing"]["grid"]["shape"]
    )
    run_dir = tmp_path / config["run_name"]
    run_dir.mkdir(parents=True)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    _stub_pipeline_grid_result(grid_setup, monkeypatch)

    Pipeline(verbose=False)._run_detection_pipeline(config)

    path = run_dir / "modeling" / "fisher_grid_map.npz"
    loaded = load_fisher_grid_map_npz(path)
    with (run_dir / "config_used.yaml").open("r", encoding="utf-8") as stream:
        snapshot = yaml.safe_load(stream)
    assert loaded.config_hash == config_hash(snapshot)


def test_resolve_relative_output_dir_expands_and_resolves_paths():
    """Expand home paths, repo-resolve relatives, and preserve absolutes."""
    import hwoslaps.pipeline as pipeline_module

    home_config = {"plotting": {"output_dir": "~/item10-check"}}
    pipeline_module._resolve_relative_output_dir(home_config)
    home_output = Path(home_config["plotting"]["output_dir"])
    assert home_output == Path.home() / "item10-check"
    assert home_output.is_absolute()
    assert "~" not in home_output.parts

    relative_config = {"plotting": {"output_dir": "item10-relative"}}
    pipeline_module._resolve_relative_output_dir(relative_config)
    repo_root = Path(pipeline_module.__file__).resolve().parents[2]
    assert Path(relative_config["plotting"]["output_dir"]) == (
        repo_root / "item10-relative"
    )

    absolute = Path("/tmp/item10-absolute")
    absolute_config = {"plotting": {"output_dir": str(absolute)}}
    pipeline_module._resolve_relative_output_dir(absolute_config)
    assert Path(absolute_config["plotting"]["output_dir"]) == absolute


def test_pipeline_rejects_foreign_grid_map_snapshot(
    grid_setup,
    tmp_path,
    monkeypatch,
):
    """Refuse to bind a grid map to a snapshot from a different run."""
    import hwoslaps.modeling.generator_fisher as generator_fisher
    import hwoslaps.pipeline as pipeline_module

    config = copy.deepcopy(grid_setup["config"])
    config["run_name"] = "pipeline-foreign-snapshot"
    config["plotting"]["enabled"] = False
    config["plotting"]["output_dir"] = str(tmp_path)
    snapshot = copy.deepcopy(config)
    snapshot["lensing"]["subhalo"]["mass"] = 2.0e8
    run_dir = tmp_path / config["run_name"]
    run_dir.mkdir(parents=True)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(snapshot, stream, sort_keys=False)
    grid_map = replace(
        grid_setup["grid_map"],
        config_hash=None,
        git_hash=None,
    )
    result = FisherDetectionData(
        mode="map",
        local=None,
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=1,
        n_nuisance=0,
        gram_condition_number=1.0,
        pixel_scale=0.1,
        grid_map=grid_map,
    )
    stub = object()
    monkeypatch.setattr(pipeline_module, "generate_psf_system", lambda *args, **kwargs: stub)
    monkeypatch.setattr(
        pipeline_module,
        "generate_lensing_system",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_observation",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        generator_fisher,
        "perform_fisher_detection",
        lambda **kwargs: result,
    )

    with pytest.raises(ValueError, match="does not describe this run"):
        Pipeline(verbose=False)._run_detection_pipeline(config)


def test_pipeline_binds_snapshot_through_resolved_output_dir(
    grid_setup,
    tmp_path,
    monkeypatch,
):
    """Bind a raw tilde snapshot to its resolved in-memory configuration."""
    import hwoslaps.modeling.generator_fisher as generator_fisher
    import hwoslaps.pipeline as pipeline_module

    monkeypatch.setenv("HOME", str(tmp_path))
    raw_config = copy.deepcopy(grid_setup["config"])
    raw_config["run_name"] = "pipeline-resolved-snapshot"
    raw_config["plotting"]["enabled"] = False
    raw_config["plotting"]["output_dir"] = "~/outputs"
    snapshot = copy.deepcopy(raw_config)
    config = copy.deepcopy(raw_config)
    pipeline_module._resolve_relative_output_dir(config)
    assert config["plotting"]["output_dir"] == str(tmp_path / "outputs")
    run_dir = tmp_path / "outputs" / raw_config["run_name"]
    run_dir.mkdir(parents=True)
    with (run_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(snapshot, stream, sort_keys=False)
    grid_map = replace(
        grid_setup["grid_map"],
        config_hash=None,
        git_hash=None,
    )
    result = FisherDetectionData(
        mode="map",
        local=None,
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=1,
        n_nuisance=0,
        gram_condition_number=1.0,
        pixel_scale=0.1,
        grid_map=grid_map,
    )
    stub = object()
    monkeypatch.setattr(pipeline_module, "generate_psf_system", lambda *args, **kwargs: stub)
    monkeypatch.setattr(
        pipeline_module,
        "generate_lensing_system",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        pipeline_module,
        "generate_observation",
        lambda *args, **kwargs: stub,
    )
    monkeypatch.setattr(
        generator_fisher,
        "perform_fisher_detection",
        lambda **kwargs: result,
    )

    Pipeline(verbose=False)._run_detection_pipeline(config)

    loaded = load_fisher_grid_map_npz(
        run_dir / "modeling" / "fisher_grid_map.npz"
    )
    assert loaded.config_hash == config_hash(snapshot)
    assert loaded.config_hash != config_hash(config)


def test_grid_map_npz_roundtrip_preserves_source_asset_identity(grid_setup, tmp_path):
    """Round-trip optional source-image identity fields."""
    grid_map = replace(
        grid_setup["grid_map"],
        source_image_asset_path="/tmp/source-image.npz",
        source_image_asset_sha256_16="0123456789abcdef",
    )
    npz_path = save_fisher_grid_map_npz(grid_map, tmp_path / "image-source.npz")

    with np.load(npz_path) as data:
        assert str(data["source_image_asset_path"]) == "/tmp/source-image.npz"
        assert str(data["source_image_asset_sha256_16"]) == "0123456789abcdef"

    loaded = load_fisher_grid_map_npz(npz_path)
    assert loaded.source_image_asset_path == grid_map.source_image_asset_path
    assert loaded.source_image_asset_sha256_16 == grid_map.source_image_asset_sha256_16


def test_grid_map_records_profiled_nuisance_provenance(grid_setup):
    """Record which nuisance directions the grid map profiled."""
    grid_map = grid_setup["grid_map"]
    detector = grid_setup["detector"]

    assert grid_map.nuisance_subset == "all"
    assert grid_map.profiled_nuisance_names == detector.nuisance_names
    assert "lens.centre_y" in grid_map.profiled_nuisance_names


def test_grid_map_npz_roundtrip_preserves_nuisance_provenance(grid_setup, tmp_path):
    """Round-trip the resolved nuisance subset through the NPZ archive."""
    grid_map = replace(
        grid_setup["grid_map"],
        nuisance_subset="lens_only",
        profiled_nuisance_names=["lens.centre_y", "lens.centre_x"],
    )
    npz_path = save_fisher_grid_map_npz(grid_map, tmp_path / "nuisance-subset.npz")

    with np.load(npz_path, allow_pickle=False) as data:
        assert str(data["nuisance_subset"]) == "lens_only"
        np.testing.assert_array_equal(
            data["profiled_nuisance_names"],
            np.asarray(["lens.centre_y", "lens.centre_x"]),
        )

    loaded = load_fisher_grid_map_npz(npz_path)
    assert loaded.nuisance_subset == "lens_only"
    assert loaded.profiled_nuisance_names == ["lens.centre_y", "lens.centre_x"]


def test_grid_map_npz_roundtrip_preserves_empty_nuisance_provenance(
    grid_setup,
    tmp_path,
):
    """Round-trip an unprofiled grid map, whose direction list is empty."""
    grid_map = replace(
        grid_setup["grid_map"],
        nuisance_subset="none",
        profiled_nuisance_names=[],
    )
    npz_path = save_fisher_grid_map_npz(grid_map, tmp_path / "nuisance-none.npz")

    loaded = load_fisher_grid_map_npz(npz_path)
    assert loaded.nuisance_subset == "none"
    assert loaded.profiled_nuisance_names == []


def test_grid_map_npz_old_format_loads_missing_nuisance_provenance_as_none(
    grid_setup,
    tmp_path,
):
    """Load an archive written before the nuisance provenance existed."""
    current = save_fisher_grid_map_npz(
        grid_setup["grid_map"],
        tmp_path / "current-nuisance.npz",
    )
    old_format = _corrupt_grid_npz(
        current,
        tmp_path / "old-format-nuisance.npz",
        delete=("nuisance_subset", "profiled_nuisance_names"),
    )

    loaded = load_fisher_grid_map_npz(old_format)
    assert loaded.nuisance_subset is None
    assert loaded.profiled_nuisance_names is None


def test_mismatch_grid_map_npz_roundtrip(mismatch_setup, tmp_path):
    """Round-trip all optional mismatch fields through the NPZ archive."""
    grid_map = mismatch_setup["grid_map"]
    npz_path = save_fisher_grid_map_npz(grid_map, tmp_path / "mismatch.npz")

    with np.load(npz_path) as data:
        for name in (
            "amplitude_hat_2d",
            "q_mismatch_2d",
            "z_mismatch_2d",
            "mismatch_detectable_mask_2d",
            "amplitude_spurious_2d",
            "q_spurious_2d",
            "z_spurious_2d",
            "false_positive_mask_2d",
        ):
            np.testing.assert_array_equal(data[name], getattr(grid_map, name))
        assert bool(data["mismatch_enabled"])
        assert float(data["false_positive_area_arcsec2"]) == pytest.approx(
            grid_map.false_positive_area_arcsec2
        )
        assert int(data["num_false_positive"]) == grid_map.num_false_positive
        assert float(data["mismatch_detectable_area_arcsec2"]) == pytest.approx(
            grid_map.mismatch_detectable_area_arcsec2
        )
        assert int(data["num_mismatch_detectable"]) == (
            grid_map.num_mismatch_detectable
        )
        assert float(data["max_z_spurious"]) == pytest.approx(
            grid_map.max_z_spurious
        )

    loaded = load_fisher_grid_map_npz(npz_path)
    assert loaded.mismatch_enabled is True
    for name in (
        "amplitude_hat_2d",
        "q_mismatch_2d",
        "z_mismatch_2d",
        "mismatch_detectable_mask_2d",
        "amplitude_spurious_2d",
        "q_spurious_2d",
        "z_spurious_2d",
        "false_positive_mask_2d",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(grid_map, name))


def test_grid_map_plot_writes_png(grid_setup, tmp_path):
    """Write a grid-map figure to disk from FisherDetectionData."""
    detection_data = FisherDetectionData(
        mode="map",
        local=None,
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=grid_setup["detector"].pixels_unmasked,
        n_nuisance=grid_setup["detector"].n_nuisance,
        gram_condition_number=1.0,
        pixel_scale=0.1,
        grid_map=grid_setup["grid_map"],
    )

    plot_fisher_detection_grid_map(
        detection_data,
        {"output_dir": str(tmp_path)},
        run_name="grid-plot-test",
    )

    assert (tmp_path / "grid-plot-test" / "modeling" / "fisher_grid_map.png").exists()
    assert not (
        tmp_path / "grid-plot-test" / "modeling" / "fisher_grid_map_spurious.png"
    ).exists()


def test_mismatch_grid_map_plot_writes_spurious_png(mismatch_setup, tmp_path):
    """Write the additional spurious-significance heat map for mismatch."""
    grid_map = mismatch_setup["grid_map"]
    detection_data = FisherDetectionData(
        mode="map",
        local=mismatch_setup["local"],
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=mismatch_setup["detector"].pixels_unmasked,
        n_nuisance=mismatch_setup["detector"].n_nuisance,
        gram_condition_number=1.0,
        pixel_scale=0.1,
        grid_map=grid_map,
    )

    plot_fisher_detection_grid_map(
        detection_data,
        {"output_dir": str(tmp_path)},
        run_name="mismatch-plot-test",
    )

    output_dir = tmp_path / "mismatch-plot-test" / "modeling"
    assert (output_dir / "fisher_grid_map.png").exists()
    assert (output_dir / "fisher_grid_map_spurious.png").exists()


def test_mismatch_summary_prints_local_and_grid_statistics(
    mismatch_setup,
    capsys,
):
    """Print the fit mode and compact mismatch and false-positive summary."""
    detection_data = FisherDetectionData(
        mode="both",
        local=mismatch_setup["local"],
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=mismatch_setup["detector"].pixels_unmasked,
        n_nuisance=mismatch_setup["detector"].n_nuisance,
        gram_condition_number=1.0,
        pixel_scale=0.1,
        config=mismatch_setup["detector"].full_config,
        grid_map=mismatch_setup["grid_map"],
        psf_mismatch_enabled=True,
    )

    print_fisher_summary(detection_data)

    output = capsys.readouterr().out
    assert "Model mismatch:" in output
    assert "fit_psf mode: explicit" in output
    assert "q_mismatch" in output
    assert "q_spurious" in output
    assert "Mismatch-detectable area" in output
    assert "False-positive area" in output


# ----------------------------------------------------------------------
# JAX engine equivalence
# ----------------------------------------------------------------------


def _make_jax_detector(grid_setup) -> FisherDetector:
    return _make_detector(
        grid_setup,
        {
            "type": "grid",
            "grid": {
                "spacing_arcsec": GRID_SPACING,
                "half_width_arcsec": GRID_HALF_WIDTH,
                "annulus": None,
            },
            "detection_q_threshold": 10.0,
            "num_workers": 1,
            "engine": "jax",
        },
    )


def test_jax_engine_template_matches_reference(grid_setup):
    """Match the JAX signal template against the NumPy reference."""
    pytest.importorskip("jax")
    detector = _make_jax_detector(grid_setup)
    position = SUBHALO_POSITION

    reference_signal = np.asarray(
        detector._mean_adu_for_position(position) - detector.mu0_adu_2d
    )[np.asarray(detector.mask_2d, dtype=bool)]

    jax_signal = next(detector._grid_signal_iterator_jax([position]))

    scale = float(np.max(np.abs(reference_signal)))
    assert scale > 0.0
    np.testing.assert_allclose(
        jax_signal,
        reference_signal,
        rtol=1.0e-5,
        atol=1.0e-7 * scale,
    )


def test_jax_engine_grid_map_matches_reference(grid_setup):
    """Match the JAX grid map against the NumPy reference map."""
    pytest.importorskip("jax")
    detector = _make_jax_detector(grid_setup)
    jax_map = detector.compute_grid_map()
    reference_map = grid_setup["grid_map"]

    np.testing.assert_allclose(
        jax_map.q_asimov_2d,
        reference_map.q_asimov_2d,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        jax_map.fisher_raw_2d,
        reference_map.fisher_raw_2d,
        rtol=1.0e-6,
    )
    cell_area = GRID_SPACING**2
    assert (
        abs(jax_map.detectable_area_arcsec2 - reference_map.detectable_area_arcsec2)
        <= cell_area
    )


def test_jax_engine_mismatch_grid_map_matches_reference(grid_setup, mismatch_setup):
    """Match JAX and reference mismatch statistics and one-sided masks."""
    pytest.importorskip("jax")
    map_config = copy.deepcopy(grid_setup["config"]["modeling"]["fisher"]["map"])
    map_config["engine"] = "jax"
    detector = _make_mismatch_detector(
        grid_setup,
        _perfect_psf_config(grid_setup["config"]),
        map_config,
    )
    jax_map = detector.compute_grid_map()
    reference = mismatch_setup["grid_map"]

    np.testing.assert_allclose(
        jax_map.q_mismatch_2d,
        reference.q_mismatch_2d,
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        jax_map.q_spurious_2d,
        reference.q_spurious_2d,
        rtol=1.0e-6,
    )
    np.testing.assert_array_equal(
        jax_map.mismatch_detectable_mask_2d,
        reference.mismatch_detectable_mask_2d,
    )
    np.testing.assert_array_equal(
        jax_map.false_positive_mask_2d,
        reference.false_positive_mask_2d,
    )


def _source_type_detectors(grid_setup, light):
    config = copy.deepcopy(grid_setup["config"])
    config["lensing"]["source_galaxy"]["light"] = copy.deepcopy(light)
    config["modeling"]["fisher"]["map"].pop("engine", None)
    baseline_config = copy.deepcopy(config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    lensing_baseline = generate_lensing_system(
        baseline_config["lensing"], full_config=baseline_config
    )
    observation_baseline = generate_observation(
        lensing_baseline,
        grid_setup["psf_data"],
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )
    reference = FisherDetector(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=grid_setup["psf_data"],
        full_config=config,
        fisher_config=copy.deepcopy(config["modeling"]["fisher"]),
    )
    jax_config = copy.deepcopy(config)
    jax_config["modeling"]["fisher"]["map"]["engine"] = "jax"
    jax_detector = FisherDetector(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=grid_setup["psf_data"],
        full_config=jax_config,
        fisher_config=copy.deepcopy(jax_config["modeling"]["fisher"]),
    )
    return reference, jax_detector


def _write_jax_source_asset(path):
    import json

    pixel_scale = 0.1
    rows, cols = np.indices((32, 32), dtype=float)
    sb = np.exp(
        -0.5 * (((rows - 13.3) / 3.4) ** 2 + ((cols - 18.2) / 4.1) ** 2)
    )
    sb += 1.0e-4
    sb /= pixel_scale**2 * sb.sum()
    np.savez(
        path,
        sb=sb.astype(np.float64),
        pixel_scale_arcsec=np.asarray(pixel_scale, dtype=np.float64),
        metadata_json=np.asarray(
            json.dumps({"format_version": 1, "provenance": {"kind": "synthetic"}})
        ),
    )
    return path


def _write_tiny_jax_source_asset(path):
    """Write a normalized image whose support is tiny relative to the field."""
    import json

    pixel_scale = 0.0025
    sb = np.zeros((8, 8), dtype=np.float64)
    sb[3, 4] = 1.0 / pixel_scale**2
    np.savez(
        path,
        sb=sb,
        pixel_scale_arcsec=np.asarray(pixel_scale, dtype=np.float64),
        metadata_json=np.asarray(
            json.dumps({"format_version": 1, "provenance": {"kind": "synthetic"}})
        ),
    )
    return path


def test_grid_map_carries_image_asset_identity(grid_setup, tmp_path):
    """Source-image identity flows from LensingData into a grid map."""
    asset_path = _write_jax_source_asset(tmp_path / "source.npz")
    light = {
        "type": "Image",
        "asset_path": str(asset_path),
        "centre": [0.0, 0.0],
        "rotation_deg": 0.0,
        "total_flux": 1.0,
        "flux_scale": 1.0,
        "size_scale": 1.0,
    }
    reference, _ = _source_type_detectors(grid_setup, light)

    grid_map = reference.compute_grid_map()

    assert grid_map.source_image_asset_path == str(asset_path)
    assert len(grid_map.source_image_asset_sha256_16) == 16


def test_jax_engine_clumpy_q_f_matches_reference(grid_setup):
    """Match reference q_F for a Sersic host with two compact clumps."""
    pytest.importorskip("jax")
    component = {
        "centre": [0.0, 0.0],
        "ell_comps": [0.03, -0.01],
        "intensity": 5.0,
        "effective_radius": 0.2,
        "sersic_index": 1.3,
    }
    light = {
        "type": "Clumpy",
        "host": component,
        "clumps": [
            {
                **component,
                "centre": [0.06, -0.04],
                "intensity": 0.8,
                "effective_radius": 0.035,
                "sersic_index": 0.8,
            },
            {
                **component,
                "centre": [-0.07, 0.05],
                "intensity": 0.5,
                "effective_radius": 0.028,
                "sersic_index": 1.1,
            },
        ],
        "flux_scale": 1.1,
        "size_scale": 1.15,
    }
    reference, jax_detector = _source_type_detectors(grid_setup, light)

    reference_map = reference.compute_grid_map()
    jax_map = jax_detector.compute_grid_map()

    assert np.all(np.isfinite(reference_map.q_asimov_2d))
    assert np.any(reference_map.q_asimov_2d > 0.0)
    np.testing.assert_allclose(
        jax_map.q_asimov_2d,
        reference_map.q_asimov_2d,
        rtol=1.0e-6,
        atol=0.0,
    )


def test_jax_engine_image_q_f_matches_reference(grid_setup, tmp_path):
    """Match reference q_F for a synthetic bilinear image source."""
    pytest.importorskip("jax")
    asset_path = _write_jax_source_asset(tmp_path / "source.npz")
    light = {
        "type": "Image",
        "asset_path": str(asset_path),
        "centre": [0.04, -0.06],
        "rotation_deg": 19.0,
        "total_flux": 0.7,
        "flux_scale": 1.1,
        "size_scale": 0.9,
    }
    reference, jax_detector = _source_type_detectors(grid_setup, light)

    reference_map = reference.compute_grid_map()
    jax_map = jax_detector.compute_grid_map()

    assert np.all(np.isfinite(reference_map.q_asimov_2d))
    assert np.any(reference_map.q_asimov_2d > 0.0)
    np.testing.assert_allclose(
        jax_map.q_asimov_2d,
        reference_map.q_asimov_2d,
        rtol=1.0e-6,
        atol=0.0,
    )


def test_jax_engine_rejects_image_profile_convention_drift(grid_setup, tmp_path):
    """Raise when an ImageSource subclass perturbs reference evaluation."""
    pytest.importorskip("jax")
    from dataclasses import replace

    import autolens as al

    from hwoslaps.lensing.image_source import ImageSource, load_source_image_asset
    from hwoslaps.modeling.fisher_grid_jax import JaxGridTemplateEngine
    from hwoslaps.psf.utils import pyauto_kernel_native

    class BrokenImageSource(ImageSource):
        def image_2d_from(self, grid, **kwargs):
            return 1.01 * super().image_2d_from(grid=grid, **kwargs)

    asset_path = _write_jax_source_asset(tmp_path / "source.npz")
    light = {
        "type": "Image",
        "asset_path": str(asset_path),
        "centre": [0.0, 0.0],
        "rotation_deg": 19.0,
        "total_flux": 0.7,
        "flux_scale": 1.0,
        "size_scale": 1.0,
    }
    config = copy.deepcopy(grid_setup["config"])
    config["lensing"]["source_galaxy"]["light"] = light
    baseline_config = copy.deepcopy(config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    normal = generate_lensing_system(
        baseline_config["lensing"], full_config=baseline_config
    )
    asset = load_source_image_asset(asset_path)
    broken_profile = BrokenImageSource.from_asset(
        asset,
        centre=(0.0, 0.0),
        rotation_deg=19.0,
        total_flux=0.7,
        flux_scale=1.0,
        size_scale=1.0,
    )
    source = al.Galaxy(redshift=normal.source_redshift, light=broken_profile)
    tracer = al.Tracer(
        galaxies=[normal.tracer.galaxies[0], source],
        cosmology=normal.tracer.cosmology,
    )
    broken = replace(
        normal,
        tracer=tracer,
        image=tracer.image_2d_from(grid=normal.grid).native,
    )

    with pytest.raises(ValueError, match="could not reproduce image profile"):
        JaxGridTemplateEngine(
            lensing_baseline=broken,
            map_config_template=config,
            psf_kernel_native=np.asarray(
                pyauto_kernel_native(grid_setup["psf_data"].kernel), dtype=float
            ),
            mu0_adu_2d=np.zeros_like(broken.image),
            mask_2d=np.ones_like(broken.image, dtype=bool),
        )


def test_jax_source_guard_hits_tiny_image_support(grid_setup, tmp_path, monkeypatch):
    """Detect a convention error when random bbox probes all miss an image."""
    pytest.importorskip("jax")
    import autogalaxy as ag
    import autolens as al

    from hwoslaps.lensing.image_source import ImageSource
    from hwoslaps.modeling.fisher_adapter import flatten_masked_image
    from hwoslaps.modeling.fisher_grid_jax import JaxGridTemplateEngine
    from hwoslaps.psf.utils import pyauto_kernel_native

    baseline = grid_setup["lensing_baseline"]
    over_sampled = np.asarray(baseline.grid.over_sampled, dtype=float)
    deflections = np.asarray(
        baseline.tracer.deflections_yx_2d_from(grid=baseline.grid.over_sampled),
        dtype=float,
    )
    traced_macro = over_sampled - deflections
    source_centre = tuple(float(value) for value in traced_macro[0])

    asset_path = _write_tiny_jax_source_asset(tmp_path / "tiny-source.npz")
    light = {
        "type": "Image",
        "asset_path": str(asset_path),
        "centre": list(source_centre),
        "rotation_deg": 31.0,
        "total_flux": 0.7,
        "flux_scale": 1.0,
        "size_scale": 1.2,
    }
    reference, jax_detector = _source_type_detectors(grid_setup, light)
    candidate_positions = jax_detector._grid_layout().positions_yx
    engine = JaxGridTemplateEngine(
        lensing_baseline=jax_detector.lensing_baseline,
        map_config_template=copy.deepcopy(jax_detector.map_config_template),
        psf_kernel_native=np.asarray(
            pyauto_kernel_native(jax_detector.model_psf_data.kernel),
            dtype=float,
        ),
        mu0_adu_2d=jax_detector.mu0_model_adu_2d,
        mask_2d=jax_detector.mask_2d,
        candidate_positions=candidate_positions,
    )
    position = (0.0, 0.0)
    jax_signal = next(engine.signal_iterator([position]))
    reference_image = reference._mean_adu_for_position(position)
    reference_signal = flatten_masked_image(
        reference_image - reference.mu0_adu_2d,
        mask=reference.mask_2d,
    )
    np.testing.assert_allclose(jax_signal, reference_signal, rtol=1.0e-6, atol=1.0e-8)

    image_profile = next(
        profile
        for galaxy in jax_detector.lensing_baseline.tracer.galaxies
        for profile in galaxy.cls_list_from(cls=ag.LightProfile)
        if isinstance(profile, ImageSource)
    )
    params = JaxGridTemplateEngine._image_params_from_profile(image_profile)
    low = traced_macro.min(axis=0)
    high = traced_macro.max(axis=0)
    old_points = np.random.default_rng(0).uniform(
        low,
        high,
        size=(128, 2),
    )
    old_reference = np.asarray(
        image_profile.image_2d_from(
            grid=al.Grid2DIrregular(values=old_points)
        ),
        dtype=float,
    )
    original_brightness = JaxGridTemplateEngine._image_brightness_np
    old_analytic = original_brightness(params, old_points)
    assert np.all(old_reference == 0.0)
    np.testing.assert_allclose(old_analytic, old_reference, rtol=1.0e-9, atol=0.0)

    def corrupted_brightness(cls, source_params, points):
        values = original_brightness(source_params, points)
        return np.where(values != 0.0, 1.01 * values, values)

    monkeypatch.setattr(
        JaxGridTemplateEngine,
        "_image_brightness_np",
        classmethod(corrupted_brightness),
    )
    with pytest.raises(ValueError, match="could not reproduce image profile"):
        JaxGridTemplateEngine(
            lensing_baseline=jax_detector.lensing_baseline,
            map_config_template=copy.deepcopy(jax_detector.map_config_template),
            psf_kernel_native=np.asarray(
                pyauto_kernel_native(jax_detector.model_psf_data.kernel),
                dtype=float,
            ),
            mu0_adu_2d=jax_detector.mu0_model_adu_2d,
            mask_2d=jax_detector.mask_2d,
            candidate_positions=candidate_positions,
        )


def test_jax_radial_table_covers_far_grid_and_guard_rejects_truncation(grid_setup):
    """Cover candidate radii beyond the legacy table and reject truncation."""
    pytest.importorskip("jax")
    far_map = {
        "type": "grid",
        "grid": {
            "spacing_arcsec": 4.0,
            "half_width_arcsec": 12.0,
            "annulus": None,
        },
        "detection_q_threshold": 10.0,
        "num_workers": 1,
        "engine": "jax",
    }
    reference_config = copy.deepcopy(far_map)
    reference_config.pop("engine")
    reference = _make_detector(grid_setup, reference_config)
    jax_detector = _make_detector(grid_setup, far_map)
    reference_map = reference.compute_grid_map()
    jax_map = jax_detector.compute_grid_map()

    np.testing.assert_allclose(
        jax_map.q_asimov_2d,
        reference_map.q_asimov_2d,
        rtol=1.0e-6,
        atol=0.0,
    )
    over_sampled = np.asarray(
        grid_setup["lensing_baseline"].grid.over_sampled,
        dtype=float,
    )
    legacy_r_max = 4.0 * float(
        np.hypot(np.ptp(over_sampled[:, 0]), np.ptp(over_sampled[:, 1]))
    )
    assert jax_detector._jax_grid_engine._radial_r_max > legacy_r_max

    jax_detector._jax_grid_engine._radial_r_max = 0.5
    with pytest.raises(ValueError, match="radial deflection table is too small"):
        next(jax_detector._jax_grid_engine.signal_iterator([(12.0, 12.0)]))
