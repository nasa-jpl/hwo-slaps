"""Tests for the 2D Fisher sensitivity grid map (Item 1 / D5).

Runtime tests use the real AutoLens + HCIPy stack on a deliberately tiny
scene; layout, persistence, and plotting tests run on synthetic data.
"""

from __future__ import annotations

import copy
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
    save_fisher_grid_map_npz,
)
from hwoslaps.observation import generate_observation
from hwoslaps.plotting.detection_plots import plot_fisher_detection_grid_map
from hwoslaps.psf.generator import generate_psf_system

GRID_SPACING = 0.1
GRID_HALF_WIDTH = 0.2
SUBHALO_POSITION = (0.2, -0.1)


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
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}

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
