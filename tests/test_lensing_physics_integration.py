"""Integration tests for lensing physics paths requiring `autolens`."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes
import numpy as np
import pytest


pytest.importorskip("autolens")


TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from _lensing_physics_helpers import (
    bootstrap_hwoslaps_namespace,
    load_lensing_anchor_fixture,
    load_master_config,
    load_module,
)


def _load_lensing_generator_module():
    bootstrap_hwoslaps_namespace()
    load_module("constants.py", "hwoslaps.constants")
    load_module("lensing/mass_models.py", "hwoslaps.lensing.mass_models")
    load_module("lensing/utils.py", "hwoslaps.lensing.utils")
    return load_module("lensing/generator.py", "hwoslaps.lensing.generator")


def _load_generator_chernoff_module():
    bootstrap_hwoslaps_namespace()
    load_module("lensing/utils.py", "hwoslaps.lensing.utils")
    load_module("observation/utils.py", "hwoslaps.observation.utils")
    load_module("modeling/chernoff_detector.py", "hwoslaps.modeling.chernoff_detector")
    return load_module("modeling/generator_chernoff.py", "hwoslaps.modeling.generator_chernoff")


def _load_detection_plot_modules():
    bootstrap_hwoslaps_namespace()
    load_module("lensing/utils.py", "hwoslaps.lensing.utils")
    load_module("observation/utils.py", "hwoslaps.observation.utils")
    chi_square_module = load_module(
        "modeling/chi_square_detector.py",
        "hwoslaps.modeling.chi_square_detector",
    )
    modeling_utils_module = load_module("modeling/utils.py", "hwoslaps.modeling.utils")
    chernoff_module = load_module("modeling/chernoff_detector.py", "hwoslaps.modeling.chernoff_detector")
    load_module("plotting/registry.py", "hwoslaps.plotting.registry")
    detection_plots_module = load_module("plotting/detection_plots.py", "hwoslaps.plotting.detection_plots")
    return detection_plots_module, modeling_utils_module, chi_square_module, chernoff_module


def _build_lensing_config_for_model(model_name: str):
    config = load_master_config()
    cfg = copy.deepcopy(config)
    cfg["run_name"] = f"physics-{model_name.lower()}"
    cfg["global_seed"] = 11
    cfg["lensing"]["grid"]["shape"] = [64, 64]
    cfg["lensing"]["subhalo"]["enabled"] = True
    cfg["lensing"]["subhalo"]["model"] = model_name
    cfg["lensing"]["subhalo"]["position"] = {
        "type": "direct",
        "centre": [0.08, -0.05],
    }
    if model_name != "NFW":
        cfg["lensing"]["subhalo"].pop("concentration", None)
    return cfg


class _Array:
    """Minimal array wrapper exposing `.native`."""

    def __init__(self, native: np.ndarray):
        self.native = native


class _DummyObservation:
    """Minimal observation object required by plotting functions."""

    def __init__(self, data: np.ndarray, pixel_scale: float):
        self.data = _Array(data)
        self.pixel_scale = pixel_scale


def _capture_scatter_calls(monkeypatch):
    """Capture Matplotlib scatter coordinates."""
    calls = []

    def _scatter_spy(_self, x_value, y_value, *args, **kwargs):
        x_float = float(np.asarray(x_value).reshape(-1)[0])
        y_float = float(np.asarray(y_value).reshape(-1)[0])
        calls.append((x_float, y_float, kwargs.get("marker")))
        return None

    monkeypatch.setattr(matplotlib.axes._axes.Axes, "scatter", _scatter_spy)
    return calls


@pytest.mark.parametrize("model_name", ["PointMass", "SIS", "NFW"])
def test_env_02_smoke_generate_lensing_system_by_model(model_name: str):
    generator_module = _load_lensing_generator_module()
    cfg = _build_lensing_config_for_model(model_name)

    lensing_data = generator_module.generate_lensing_system(cfg["lensing"], full_config=cfg)

    assert lensing_data.image.shape == tuple(cfg["lensing"]["grid"]["shape"])
    assert lensing_data.tracer is not None
    assert lensing_data.image.size > 0
    assert np.isfinite(lensing_data.total_flux)
    if model_name == "NFW":
        assert lensing_data.subhalo_einstein_radius is None
    else:
        assert lensing_data.subhalo_einstein_radius is not None
        assert lensing_data.subhalo_einstein_radius > 0.0


def test_if_03_chernoff_handoff_preserves_canonical_yx(monkeypatch):
    generator_chernoff = _load_generator_chernoff_module()
    captured = {}

    class _MockChernoffDetector:
        def __init__(
            self,
            observation_data_no_subhalo,
            observation_data_with_subhalo_ref,
            lensing_test,
            snr_threshold,
            use_template,
        ):
            captured["snr_threshold"] = float(snr_threshold)
            captured["use_template"] = bool(use_template)

        def detect_at_position(self, observation_with_subhalo, subhalo_position, compute_asimov):
            captured["subhalo_position"] = subhalo_position
            captured["compute_asimov"] = bool(compute_asimov)
            return {"position": subhalo_position}

    class _DummyLensing:
        subhalo_position = (0.3, -0.2)

    monkeypatch.setattr(generator_chernoff, "ChernoffSubhaloDetector", _MockChernoffDetector)

    result = generator_chernoff.perform_chernoff_detection(
        observation_baseline=object(),
        observation_ref_with_subhalo=object(),
        observation_test=object(),
        lensing_test=_DummyLensing(),
        detection_config={"snr_threshold": 3.0},
    )

    assert captured["snr_threshold"] == pytest.approx(3.0)
    assert captured["use_template"] is True
    assert captured["compute_asimov"] is True
    assert captured["subhalo_position"] == (0.3, -0.2)
    assert result["position"] == (0.3, -0.2)


def test_if_04_gof_plot_marker_coordinates_match_truth(monkeypatch, tmp_path):
    detection_plots, modeling_utils, chi_square, _ = _load_detection_plot_modules()
    scatter_calls = _capture_scatter_calls(monkeypatch)

    image_shape = (10, 10)
    pixel_scale = 0.1
    position_yx = (0.2, -0.3)
    expected_x = position_yx[1] / pixel_scale + image_shape[1] / 2.0
    expected_y = position_yx[0] / pixel_scale + image_shape[0] / 2.0

    flat_size = image_shape[0] * image_shape[1]
    snr_mask = np.ones(flat_size, dtype=bool)
    detection_result = chi_square.DetectionResult(
        chi2_value=1.0,
        threshold=0.5,
        detected=True,
        significance_level=1.0e-3,
        dof=flat_size,
        position=position_yx,
        snr_mask=snr_mask,
        residual=np.zeros(flat_size, dtype=float),
        num_regions=1,
        max_region_snr=2.0,
    )
    detection_data = modeling_utils.DetectionData(
        detection_results={1.0e-3: detection_result},
        chi2_value=1.0,
        degrees_of_freedom=flat_size,
        snr_threshold=0.5,
        significance_levels=[1.0e-3],
        pixels_unmasked=flat_size,
        num_regions=1,
        max_region_snr=2.0,
        snr_mask=snr_mask,
        snr_array=np.ones(flat_size, dtype=float),
        labeled_regions=np.ones(flat_size, dtype=int),
        residual_map=np.zeros(flat_size, dtype=float),
        image_shape=image_shape,
        true_subhalo_position=position_yx,
        true_subhalo_mass=1.0e8,
        true_subhalo_model="PointMass",
        pixel_scale=pixel_scale,
        config={"run_name": "marker-test"},
    )

    obs_baseline = _DummyObservation(np.zeros(image_shape, dtype=float), pixel_scale=pixel_scale)
    obs_test = _DummyObservation(np.ones(image_shape, dtype=float), pixel_scale=pixel_scale)

    detection_plots.plot_detection_comparison(
        detection_data=detection_data,
        plot_config={"output_dir": str(tmp_path), "run_name": "marker-test"},
        obs_baseline=obs_baseline,
        obs_test=obs_test,
    )

    assert any(
        marker == "x"
        and np.isclose(x_value, expected_x)
        and np.isclose(y_value, expected_y)
        for x_value, y_value, marker in scatter_calls
    )


def test_if_04_chernoff_plot_marker_coordinates_match_truth(monkeypatch, tmp_path):
    detection_plots, _, _, chernoff_module = _load_detection_plot_modules()
    scatter_calls = _capture_scatter_calls(monkeypatch)

    image_shape = (10, 10)
    pixel_scale = 0.1
    position_yx = (0.2, -0.3)
    expected_x = position_yx[1] / pixel_scale + image_shape[1] / 2.0
    expected_y = position_yx[0] / pixel_scale + image_shape[0] / 2.0

    flat_size = image_shape[0] * image_shape[1]
    snr_mask = np.ones(flat_size, dtype=bool)
    result = chernoff_module.ChernoffDetectionResult(
        delta_chi2=1.0,
        p_value=0.2,
        sigma=1.0,
        position=position_yx,
        alpha_hat=1.0,
        used_template=True,
        snr_mask=snr_mask,
        pixels_unmasked=flat_size,
        asimov_delta_chi2=1.0,
    )
    detection_data = chernoff_module.ChernoffDetectionData(
        result=result,
        snr_threshold=0.5,
        max_region_snr=2.0,
        num_regions=1,
        snr_array=np.ones(flat_size, dtype=float),
        labeled_regions=np.ones(flat_size, dtype=int),
        variance_2d=np.ones(image_shape, dtype=float),
        config={"run_name": "marker-test"},
    )

    obs_baseline = _DummyObservation(np.zeros(image_shape, dtype=float), pixel_scale=pixel_scale)
    obs_test = _DummyObservation(np.ones(image_shape, dtype=float), pixel_scale=pixel_scale)

    detection_plots.plot_chernoff_detection_comparison(
        detection_data=detection_data,
        plot_config={"output_dir": str(tmp_path), "run_name": "marker-test"},
        obs_baseline=obs_baseline,
        obs_test=obs_test,
    )

    assert any(
        marker == "x"
        and np.isclose(x_value, expected_x)
        and np.isclose(y_value, expected_y)
        for x_value, y_value, marker in scatter_calls
    )


@pytest.mark.parametrize("model_name", ["PointMass", "SIS", "NFW"])
def test_reg_04_optional_image_summary_anchors(model_name: str):
    anchors = load_lensing_anchor_fixture()
    summary_anchors = anchors["integration_image_summary"]
    model_key = model_name.lower()
    expected_summary = summary_anchors.get(model_key)
    if expected_summary is None:
        pytest.skip("No integration image summary anchor set for this model.")

    generator_module = _load_lensing_generator_module()
    cfg = _build_lensing_config_for_model(model_name)
    lensing_data = generator_module.generate_lensing_system(cfg["lensing"], full_config=cfg)
    observed = {
        "shape": list(lensing_data.image.shape),
        "total_flux": float(np.sum(lensing_data.image)),
        "peak": float(np.max(lensing_data.image)),
    }

    assert observed["shape"] == expected_summary["shape"]
    assert observed["total_flux"] == pytest.approx(expected_summary["total_flux"], rel=1.0e-10)
    assert observed["peak"] == pytest.approx(expected_summary["peak"], rel=1.0e-10)
