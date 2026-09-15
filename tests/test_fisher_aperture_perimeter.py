"""Focused contracts for the Plan A aperture/perimeter ladder path.

These tests deliberately keep the generic full-map reducer in the comparison
loop.  The ladder path evaluates only the closed aperture plus the original
square perimeter; interior nodes outside that union are auxiliary work and
must not be represented as fabricated values.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
TESTS_ROOT = PROJECT_ROOT / "tests"
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from hwoslaps.modeling.fisher_detector import (  # noqa: E402
    FisherDetector,
    FisherLadderGridSelection,
    FisherLadderRungData,
)
from run_ladder import _rung_metrics  # noqa: E402


def _grid_stub(*, spacing=1.0, half_width=2.0, centre=(0.5, -0.5)):
    """Build the geometry-only detector state used by selection tests."""
    detector = FisherDetector.__new__(FisherDetector)
    detector.map_type = "grid"
    detector.map_config = {
        "grid": {
            "spacing_arcsec": float(spacing),
            "half_width_arcsec": float(half_width),
            "annulus": None,
        }
    }
    detector.full_config = {
        "lensing": {"lens_galaxy": {"mass": {"centre": list(centre)}}}
    }
    detector._grid_layout_cache = None
    return detector


def _expected_union(y_coords, x_coords, aperture_centre, radius):
    yy = y_coords[:, None]
    xx = x_coords[None, :]
    inside = (yy - aperture_centre[0]) ** 2 + (xx - aperture_centre[1]) ** 2 <= radius**2
    perimeter = np.zeros(inside.shape, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True
    selected = inside | perimeter
    indices = [tuple(index) for index in np.argwhere(selected)]
    positions = [(float(y_coords[i]), float(x_coords[j])) for i, j in indices]
    return inside, perimeter, selected, indices, positions


def test_selection_preserves_off_centre_square_and_exact_closed_union():
    """Use the original square coordinates and row-major aperture union."""
    detector = _grid_stub(spacing=0.5, half_width=1.0, centre=(0.25, -0.25))
    selection = detector.prepare_ladder_grid_selection(
        centre_arcsec=(0.25, 0.25),
        radius_arcsec=0.75,
    )

    y_coords = np.asarray([-0.75, -0.25, 0.25, 0.75, 1.25])
    x_coords = np.asarray([-1.25, -0.75, -0.25, 0.25, 0.75])
    inside, perimeter, selected, indices, positions = _expected_union(
        y_coords, x_coords, (0.25, 0.25), 0.75
    )
    np.testing.assert_array_equal(selection.y_coords, y_coords)
    np.testing.assert_array_equal(selection.x_coords, x_coords)
    np.testing.assert_array_equal(selection.aperture_mask_2d, inside)
    np.testing.assert_array_equal(selection.perimeter_mask_2d, perimeter)
    np.testing.assert_array_equal(selection.selected_mask_2d, selected)
    assert selection.aperture_centre_yx == (0.25, 0.25)
    assert selection.grid_centre_yx == (0.25, -0.25)
    assert selection.positions_yx == tuple(positions)
    assert selection.node_indices == tuple(indices)
    assert selection.full_grid_node_count == 25
    assert selection.aperture_node_count == int(inside.sum())
    assert selection.perimeter_node_count == 16
    assert selection.selected_node_count == len(indices)


def test_selection_includes_exact_radius_and_edge_overlap_without_duplicates():
    """Closed radial boundaries and aperture/perimeter overlap are retained once."""
    detector = _grid_stub(spacing=1.0, half_width=2.0, centre=(0.0, 0.0))
    selection = detector.prepare_ladder_grid_selection(
        centre_arcsec=(0.0, 0.0), radius_arcsec=2.0
    )
    y = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
    x = y.copy()
    inside, perimeter, selected, indices, positions = _expected_union(
        y, x, (0.0, 0.0), 2.0
    )
    np.testing.assert_array_equal(selection.aperture_mask_2d, inside)
    np.testing.assert_array_equal(selection.perimeter_mask_2d, perimeter)
    np.testing.assert_array_equal(selection.selected_mask_2d, selected)
    assert selection.selected_node_count == len(indices)
    assert len(set(selection.node_indices)) == selection.selected_node_count
    assert selection.node_indices == tuple(indices)
    assert selection.positions_yx == tuple(positions)
    # The four axis edge nodes are in both masks, but only occur once in the union.
    assert np.count_nonzero(inside & perimeter) == 4


def test_selection_honours_nextafter_closed_radius_boundary():
    """A node exactly at the radius follows the closed ``<=`` convention."""
    detector = _grid_stub(spacing=1.0, half_width=1.0, centre=(0.0, 0.0))
    below = detector.prepare_ladder_grid_selection(
        (0.0, 0.0), np.nextafter(1.0, 0.0)
    )
    above = detector.prepare_ladder_grid_selection(
        (0.0, 0.0), np.nextafter(1.0, np.inf)
    )
    # The axis nodes are perimeter nodes in this tiny square, so inspect the
    # aperture mask itself rather than the selected union.
    assert int(below.aperture_mask_2d.sum()) == 1
    assert int(above.aperture_mask_2d.sum()) == 5


@pytest.mark.parametrize(
    ("centre", "radius"),
    [((10.0, 10.0), 0.1), ((0.1, 0.1), 0.0)],
)
def test_selection_rejects_zero_inside_geometry(centre, radius):
    """A ladder selection cannot silently change the aperture denominator."""
    detector = _grid_stub(spacing=1.0, half_width=2.0)
    with pytest.raises(ValueError, match="aperture"):
        detector.prepare_ladder_grid_selection(
            centre_arcsec=centre, radius_arcsec=radius
        )


def test_selection_rejects_annulus_because_ladder_requires_full_square():
    """A pre-cropped generic annulus cannot provide the original perimeter."""
    detector = _grid_stub()
    detector.map_config["grid"]["annulus"] = {
        "r_min_arcsec": 0.0,
        "r_max_arcsec": 1.5,
    }
    with pytest.raises(ValueError, match="annulus|full square"):
        detector.prepare_ladder_grid_selection((0.0, 0.0), 1.0)


def test_rung_metrics_keeps_perimeter_only_detection_and_skipped_maximum():
    """Only consumed aperture q values set q_max; any edge detection clips."""
    y = np.arange(-2.0, 3.0)
    x = np.arange(-2.0, 3.0)
    q = np.full((5, 5), 1.0)
    q[1, 1] = 999.0  # skipped interior reduction for a small aperture
    q[2, 2] = 7.0
    detectable = np.zeros((5, 5), dtype=bool)
    detectable[0, 0] = True  # perimeter-only detection
    metrics = _rung_metrics(
        y,
        x,
        q,
        detectable,
        spacing_arcsec=1.0,
        centre_arcsec=(0.0, 0.0),
        radius_arcsec=1.0,
    )
    assert metrics["q_max"] == 7.0
    assert metrics["detectable_area_arcsec2"] == 0.0
    assert metrics["aperture_fraction"] == 0.0
    assert metrics["perimeter_clipped"] is True


def test_rung_metrics_no_detections_and_all_inside_are_hand_checkable():
    """The all-inside square has a unit denominator and zero detected area."""
    y = np.arange(-1.0, 2.0)
    x = np.arange(-1.0, 2.0)
    q = np.arange(9.0).reshape(3, 3)
    detectable = np.zeros((3, 3), dtype=bool)
    metrics = _rung_metrics(
        y,
        x,
        q,
        detectable,
        spacing_arcsec=0.5,
        centre_arcsec=(0.0, 0.0),
        radius_arcsec=10.0,
    )
    assert metrics == {
        "q_max": 8.0,
        "detectable_area_arcsec2": 0.0,
        "aperture_fraction": 0.0,
        "perimeter_clipped": False,
    }


def test_rung_metrics_consumed_nonfinite_fails_but_skipped_nonfinite_is_irrelevant():
    """Finite validation applies to aperture values consumed by estimands."""
    y = np.arange(-2.0, 3.0)
    x = np.arange(-2.0, 3.0)
    q = np.ones((5, 5), dtype=float)
    q[1, 1] = np.nan  # skipped interior node: not an estimand value
    metrics = _rung_metrics(
        y,
        x,
        q,
        np.zeros((5, 5), dtype=bool),
        spacing_arcsec=1.0,
        centre_arcsec=(0.0, 0.0),
        radius_arcsec=1.0,
    )
    assert metrics["q_max"] == 1.0
    q[2, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _rung_metrics(
            y,
            x,
            q,
            np.zeros((5, 5), dtype=bool),
            spacing_arcsec=1.0,
            centre_arcsec=(0.0, 0.0),
            radius_arcsec=1.0,
        )


def test_rung_data_contract_exposes_compact_consumed_arrays():
    """The specialized result cannot masquerade as a complete dense map."""
    selection = FisherLadderGridSelection(
        y_coords=np.arange(3.0),
        x_coords=np.arange(3.0),
        spacing_arcsec=1.0,
        grid_centre_yx=(1.0, 1.0),
        aperture_centre_yx=(1.0, 1.0),
        aperture_radius_arcsec=1.0,
        aperture_mask_2d=np.ones((3, 3), dtype=bool),
        perimeter_mask_2d=np.zeros((3, 3), dtype=bool),
        selected_mask_2d=np.ones((3, 3), dtype=bool),
        positions_yx=tuple((float(i), float(j)) for i in range(3) for j in range(3)),
        node_indices=tuple((i, j) for i in range(3) for j in range(3)),
    )
    rung = FisherLadderRungData(
        selection=selection,
        positions_yx=np.asarray(selection.positions_yx),
        node_indices=np.asarray(selection.node_indices),
        q_asimov_by_position=np.arange(9.0),
        detectable_by_position=np.zeros(9, dtype=bool),
        detection_q_threshold=10.0,
        q_max=8.0,
        detectable_area_arcsec2=0.0,
        aperture_fraction=0.0,
        perimeter_clipped=False,
    )
    assert rung.num_positions_evaluated == 9
    assert rung.q_asimov_by_position.shape == (9,)
    assert rung.detectable_by_position.shape == (9,)
    assert rung.node_indices.shape == (9, 2)


def test_specialized_summary_matches_dense_reduction_on_consumed_nodes(monkeypatch):
    """A compact summary agrees with the old dense reducer at every consumed node."""
    detector = _grid_stub(spacing=1.0, half_width=2.0, centre=(0.0, 0.0))
    detector.map_config["grid"]["engine"] = "reference"
    detector.map_config["detection_q_threshold"] = 10.0
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 1.0)

    dense_q = np.ones((5, 5), dtype=float)
    dense_q[2, 2] = 7.0
    dense_q[1, 1] = 999.0  # maximum deliberately lies in skipped interior
    dense_q[0, 2] = 12.0  # perimeter-only detection
    dense_detectable = dense_q >= 10.0

    def evaluate(positions):
        values = [dense_q[i, j] for i, j in selection.node_indices]
        return [SimpleNamespace(q_asimov_local=np.asarray(values, dtype=float))]

    monkeypatch.setattr(detector, "_evaluate_grid_positions", evaluate)
    detector.mismatch_enabled = False
    result = detector.compute_ladder_summary(selection)

    dense_metrics = _rung_metrics(
        selection.y_coords,
        selection.x_coords,
        dense_q,
        dense_detectable,
        selection.spacing_arcsec,
        selection.aperture_centre_yx,
        selection.aperture_radius_arcsec,
    )
    expected_q = np.asarray([dense_q[i, j] for i, j in selection.node_indices])
    expected_detectable = np.asarray(
        [dense_detectable[i, j] for i, j in selection.node_indices], dtype=bool
    )
    np.testing.assert_array_equal(result.node_indices, np.asarray(selection.node_indices))
    np.testing.assert_allclose(result.q_asimov_by_position, expected_q)
    np.testing.assert_array_equal(result.detectable_by_position, expected_detectable)
    assert result.q_max == dense_metrics["q_max"] == 7.0
    assert result.detectable_area_arcsec2 == dense_metrics["detectable_area_arcsec2"] == 0.0
    assert result.aperture_fraction == dense_metrics["aperture_fraction"] == 0.0
    assert result.perimeter_clipped is dense_metrics["perimeter_clipped"] is True


def test_specialized_summary_rejects_consumed_nonfinite_and_unsupported_mismatch(
    monkeypatch,
):
    """Fail closed on consumed q values and preserve mismatch-map semantics."""
    detector = _grid_stub()
    detector.map_config["grid"]["engine"] = "reference"
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 1.0)
    values = np.ones(selection.selected_node_count, dtype=float)
    values[0] = np.nan
    monkeypatch.setattr(
        detector,
        "_evaluate_grid_positions",
        lambda positions: [SimpleNamespace(q_asimov_local=values)],
    )
    detector.mismatch_enabled = False
    with pytest.raises(ValueError, match="non-finite"):
        detector.compute_ladder_summary(selection)

    detector.mismatch_enabled = True
    with pytest.raises(ValueError, match="PSF-mismatch"):
        detector.compute_ladder_summary(selection)


def test_specialized_summary_rejects_selection_with_dropped_perimeter_node(
    monkeypatch,
):
    """Replacing a prepared selection cannot discard a square edge node."""
    detector = _grid_stub()
    detector.map_config["grid"]["engine"] = "reference"
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 1.0)
    dropped = list(selection.node_indices)
    dropped.pop(0)
    dropped_positions = tuple(
        (float(selection.y_coords[i]), float(selection.x_coords[j]))
        for i, j in dropped
    )
    bad = replace(
        selection,
        positions_yx=dropped_positions,
        node_indices=tuple(dropped),
        selected_mask_2d=np.asarray(selection.selected_mask_2d).copy(),
    )
    monkeypatch.setattr(
        detector,
        "_evaluate_grid_positions",
        lambda positions: [
            SimpleNamespace(q_asimov_local=np.ones(len(positions), dtype=float))
        ],
    )
    detector.mismatch_enabled = False
    with pytest.raises(ValueError, match="perimeter|selection"):
        detector.compute_ladder_summary(bad)


def test_prepared_selection_arrays_are_immutable_and_annulus_is_rejected(
    monkeypatch,
):
    """A prepared selection is read-only and cannot be reused for annulus maps."""
    detector = _grid_stub()
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 1.0)
    for array in (
        selection.y_coords,
        selection.x_coords,
        selection.aperture_mask_2d,
        selection.perimeter_mask_2d,
        selection.selected_mask_2d,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)

    detector.map_config["grid"]["annulus"] = {
        "r_min_arcsec": 0.0,
        "r_max_arcsec": 2.0,
    }
    detector.mismatch_enabled = False
    monkeypatch.setattr(
        detector,
        "_evaluate_grid_positions",
        lambda positions: [
            SimpleNamespace(q_asimov_local=np.ones(len(positions), dtype=float))
        ],
    )
    with pytest.raises(ValueError, match="annulus|full square"):
        detector.compute_ladder_summary(selection)


@pytest.mark.parametrize("nonfinite_index", [0, 10])
def test_specialized_summary_rejects_nonfinite_edge_or_aperture_value(
    monkeypatch, nonfinite_index
):
    """Every consumed perimeter and aperture value is finite-checked."""
    detector = _grid_stub()
    detector.map_config["grid"]["engine"] = "reference"
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 1.0)
    values = np.ones(selection.selected_node_count, dtype=float)
    values[nonfinite_index] = np.inf
    monkeypatch.setattr(
        detector,
        "_evaluate_grid_positions",
        lambda positions: [SimpleNamespace(q_asimov_local=values)],
    )
    detector.mismatch_enabled = False
    with pytest.raises(ValueError, match="non-finite"):
        detector.compute_ladder_summary(selection)


@pytest.mark.parametrize(
    ("aperture_q", "perimeter_q", "expected_area", "expected_fraction", "clipped"),
    [
        (1.0, 1.0, 0.0, 0.0, False),  # no detections
        (11.0, 1.0, 5.0, 1.0, False),  # all five aperture nodes detected
    ],
)
def test_specialized_summary_handles_no_detection_and_all_inside(
    monkeypatch, aperture_q, perimeter_q, expected_area, expected_fraction, clipped
):
    """The compact reducer keeps the aperture denominator and edge flag separate."""
    detector = _grid_stub(centre=(0.0, 0.0))
    detector.map_config["grid"]["engine"] = "reference"
    detector.map_config["detection_q_threshold"] = 10.0
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 1.0)
    values = np.full(selection.selected_node_count, perimeter_q, dtype=float)
    node_idx = np.asarray(selection.node_indices)
    values[selection.aperture_mask_2d[node_idx[:, 0], node_idx[:, 1]]] = aperture_q
    monkeypatch.setattr(
        detector,
        "_evaluate_grid_positions",
        lambda positions: [SimpleNamespace(q_asimov_local=values)],
    )
    detector.mismatch_enabled = False
    summary = detector.compute_ladder_summary(selection)
    assert summary.detectable_area_arcsec2 == expected_area
    assert summary.aperture_fraction == expected_fraction
    assert summary.perimeter_clipped is clipped


def test_specialized_summary_handles_aperture_containing_every_lattice_node(
    monkeypatch,
):
    """When every node is inside, the aperture denominator is the full square."""
    detector = _grid_stub(centre=(0.0, 0.0))
    detector.map_config["grid"]["engine"] = "reference"
    detector.map_config["detection_q_threshold"] = 10.0
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 10.0)
    values = np.full(selection.selected_node_count, 11.0, dtype=float)
    monkeypatch.setattr(
        detector,
        "_evaluate_grid_positions",
        lambda positions: [SimpleNamespace(q_asimov_local=values)],
    )
    detector.mismatch_enabled = False
    summary = detector.compute_ladder_summary(selection)
    assert summary.aperture_fraction == 1.0
    assert summary.detectable_area_arcsec2 == 25.0
    assert summary.perimeter_clipped is True


@pytest.fixture(scope="module")
def jax_ladder_fixture(tmp_path_factory):
    """Build the existing tiny 5x5 JAX grid fixture when its stack is usable."""
    pytest.importorskip("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
    from test_fisher_grid_map import _build_grid_config
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.modeling.fisher_detector import FisherDetector
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf.generator import generate_psf_system

    tmp_dir = tmp_path_factory.mktemp("fisher-ladder-jax")
    config = _build_grid_config(tmp_dir)
    config["modeling"]["fisher"]["map"]["engine"] = "jax"
    config_baseline = copy.deepcopy(config)
    config_baseline["lensing"]["subhalo"]["enabled"] = False
    psf_data = generate_psf_system(config["psf"], full_config=config)
    lensing_baseline = generate_lensing_system(
        config_baseline["lensing"], full_config=config_baseline
    )
    observation_baseline = generate_observation(
        lensing_baseline,
        psf_data,
        observation_config=config_baseline["observation"],
        full_config=config_baseline,
    )
    detector = FisherDetector(
        observation_baseline=observation_baseline,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=config,
        fisher_config=copy.deepcopy(config["modeling"]["fisher"]),
    )
    dense = detector.compute_grid_map()
    return detector, dense


def test_jax_dense_and_ladder_match_each_consumed_q_and_radial_table(
    jax_ladder_fixture,
):
    """The specialized CPU JAX path equals dense q/masks at each consumed node."""
    detector, dense = jax_ladder_fixture
    selection = detector.prepare_ladder_grid_selection((0.0, 0.0), 0.1)
    assert selection.selected_node_count < selection.full_grid_node_count
    dense_engine = detector._jax_grid_engine
    radii_before = np.asarray(dense_engine._radii).copy()
    alpha_before = np.asarray(dense_engine._alpha_radial).copy()
    radial_r_max_before = dense_engine._radial_r_max

    # Force the specialized path to construct its own engine.  Its constructor
    # must still receive the complete original square, not just the union.
    detector._jax_grid_engine = None
    summary = detector.compute_ladder_summary(selection)
    node_idx = np.asarray(selection.node_indices, dtype=int)
    dense_q = dense.q_asimov_2d[node_idx[:, 0], node_idx[:, 1]]
    dense_detectable = dense.detectable_mask_2d[node_idx[:, 0], node_idx[:, 1]]
    np.testing.assert_allclose(summary.q_asimov_by_position, dense_q, rtol=1.0e-9, atol=0.0)
    np.testing.assert_array_equal(summary.detectable_by_position, dense_detectable)

    dense_metrics = _rung_metrics(
        dense.y_coords,
        dense.x_coords,
        dense.q_asimov_2d,
        dense.detectable_mask_2d,
        dense.spacing_arcsec,
        selection.aperture_centre_yx,
        selection.aperture_radius_arcsec,
    )
    assert summary.q_max == pytest.approx(dense_metrics["q_max"], rel=1e-9, abs=0.0)
    assert summary.detectable_area_arcsec2 == dense_metrics["detectable_area_arcsec2"]
    assert summary.aperture_fraction == dense_metrics["aperture_fraction"]
    assert summary.perimeter_clipped is dense_metrics["perimeter_clipped"]

    # Both paths must retain the exact radial interpolation table geometry,
    # including bounds and sample count, rather than rebuilding from the union.
    np.testing.assert_array_equal(detector._jax_grid_engine._radii, radii_before)
    np.testing.assert_array_equal(detector._jax_grid_engine._alpha_radial, alpha_before)
    assert detector._jax_grid_engine._radii.size == radii_before.size
    assert detector._jax_grid_engine._radial_r_max == radial_r_max_before
