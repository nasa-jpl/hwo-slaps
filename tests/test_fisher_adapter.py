"""Tests for the image-space adapters around the Fisher core."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = PROJECT_ROOT / "src" / "hwoslaps" / "modeling" / "fisher_core.py"
ADAPTER_PATH = PROJECT_ROOT / "src" / "hwoslaps" / "modeling" / "fisher_adapter.py"
TEST_PACKAGE = "hwoslaps_fisher_adapter_testpkg"

# Load core first so relative imports in the adapter resolve cleanly when
# the module is imported by path.
import sys
import types

# Build lightweight package placeholders in a private namespace so relative
# imports resolve without polluting the real `hwoslaps` package for other
# tests.
test_pkg = sys.modules.setdefault(TEST_PACKAGE, types.ModuleType(TEST_PACKAGE))
test_pkg.__path__ = []
modeling_pkg_name = f"{TEST_PACKAGE}.modeling"
modeling_pkg = sys.modules.setdefault(modeling_pkg_name, types.ModuleType(modeling_pkg_name))
modeling_pkg.__path__ = []

core_spec = importlib.util.spec_from_file_location(
    f"{modeling_pkg_name}.fisher_core",
    CORE_PATH,
)
core_module = importlib.util.module_from_spec(core_spec)
sys.modules[core_spec.name] = core_module
core_spec.loader.exec_module(core_module)

adapter_spec = importlib.util.spec_from_file_location(
    f"{modeling_pkg_name}.fisher_adapter",
    ADAPTER_PATH,
)
adapter_module = importlib.util.module_from_spec(adapter_spec)
sys.modules[adapter_spec.name] = adapter_module
adapter_spec.loader.exec_module(adapter_module)

flatten_masked_image = adapter_module.flatten_masked_image
stack_masked_images = adapter_module.stack_masked_images
compute_asimov_from_images = adapter_module.compute_asimov_from_images
evaluate_signal_bank_from_images = adapter_module.evaluate_signal_bank_from_images
compute_spurious_from_images = adapter_module.compute_spurious_from_images
scan_systematic_modes_from_images = adapter_module.scan_systematic_modes_from_images


def test_flatten_masked_image_respects_mask():
    """Flatten only the masked-in pixels in row-major order."""
    image = np.arange(9, dtype=float).reshape(3, 3)
    mask = np.array(
        [
            [True, False, True],
            [False, True, False],
            [True, False, False],
        ]
    )
    flat = flatten_masked_image(image, mask=mask)
    np.testing.assert_allclose(flat, np.array([0.0, 2.0, 4.0, 6.0]))


def test_stack_masked_images_builds_design_matrix():
    """Stack masked images into a column-per-image design matrix."""
    img1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    img2 = np.array([[10.0, 20.0], [30.0, 40.0]])
    mask = np.array([[True, False], [True, True]])
    design = stack_masked_images([img1, img2], mask=mask)
    expected = np.array(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    np.testing.assert_allclose(design, expected)


def test_compute_asimov_from_images_matches_manual_vector_call():
    """Match the image adapter against a manual vector call."""
    smooth = np.array([[1.0, 1.0], [1.0, 1.0]])
    subhalo = np.array([[2.0, 1.0], [1.0, 1.0]])
    sigma = np.ones_like(smooth)
    nuisance = [np.array([[0.0, 1.0], [0.0, 0.0]])]
    mask = np.array([[True, True], [False, True]])

    image_result = compute_asimov_from_images(
        smooth_mean_image=smooth,
        subhalo_mean_image=subhalo,
        sigma_image=sigma,
        nuisance_images=nuisance,
        mask=mask,
    )

    signal = flatten_masked_image(subhalo - smooth, mask=mask)
    nuisance_vec = stack_masked_images(nuisance, mask=mask)
    manual_result = core_module.compute_asimov_detectability(
        signal=signal,
        nuisance_jacobian=nuisance_vec,
        sigma=np.ones(signal.size),
    )

    assert image_result.fisher_profiled == pytest.approx(manual_result.fisher_profiled)
    assert image_result.z_asimov_local == pytest.approx(manual_result.z_asimov_local)


def test_signal_bank_from_images_is_vectorized_over_templates():
    """Evaluate a bank of image templates in one vectorized call."""
    smooth = np.zeros((2, 2))
    templates = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0]]),
    ]
    bank = evaluate_signal_bank_from_images(
        smooth_mean_image=smooth,
        subhalo_mean_images=templates,
        sigma_image=np.ones((2, 2)),
    )
    np.testing.assert_allclose(bank.fisher_profiled, np.array([1.0, 1.0]))
    np.testing.assert_allclose(bank.z_asimov_local, np.array([1.0, 1.0]))


def test_signal_bank_rejects_broadcastable_wrong_shape_template():
    """Reject a template whose shape only broadcasts to the scene."""
    smooth = np.zeros((2, 2))

    with pytest.raises(ValueError, match="subhalo_mean_image"):
        evaluate_signal_bank_from_images(
            smooth_mean_image=smooth,
            subhalo_mean_images=[np.zeros((1, 2))],
            sigma_image=np.ones((2, 2)),
        )


def test_compute_asimov_rejects_wrong_shape_sigma_image():
    """Reject a sigma image that does not match the scene shape."""
    smooth = np.zeros((2, 2))
    subhalo = np.ones((2, 2))

    with pytest.raises(ValueError, match="sigma_image"):
        compute_asimov_from_images(
            smooth_mean_image=smooth,
            subhalo_mean_image=subhalo,
            sigma_image=np.ones((1, 2)),
        )


def test_compute_asimov_rejects_wrong_shape_nuisance_image_without_mask():
    """Reject a nuisance image of the wrong shape with no mask given."""
    smooth = np.zeros((2, 2))
    subhalo = np.ones((2, 2))

    with pytest.raises(ValueError, match="images\\[0\\]"):
        compute_asimov_from_images(
            smooth_mean_image=smooth,
            subhalo_mean_image=subhalo,
            sigma_image=np.ones((2, 2)),
            nuisance_images=[np.ones((1, 2))],
        )


def test_spurious_adapter_rejects_wrong_shape_bias_image():
    """Reject a bias image that does not match the scene shape."""
    smooth = np.zeros((2, 2))
    subhalo = np.ones((2, 2))

    with pytest.raises(ValueError, match="bias_image"):
        compute_spurious_from_images(
            smooth_mean_image=smooth,
            subhalo_mean_image=subhalo,
            bias_image=np.ones((1, 2)),
            sigma_image=np.ones((2, 2)),
        )


def test_scan_systematic_modes_rejects_wrong_shape_mode_image_without_mask():
    """Reject a systematic mode image of the wrong shape."""
    smooth = np.zeros((2, 2))
    subhalo = np.ones((2, 2))

    with pytest.raises(ValueError, match="images\\[0\\]"):
        scan_systematic_modes_from_images(
            smooth_mean_image=smooth,
            subhalo_mean_image=subhalo,
            systematic_mode_images=[np.ones((1, 2))],
            sigma_image=np.ones((2, 2)),
        )


def test_scan_systematic_modes_from_images_returns_named_modes():
    """Return scan couplings labelled with the supplied mode names."""
    smooth = np.zeros((2, 2))
    subhalo = np.array([[1.0, 0.0], [0.0, 0.0]])
    systematic_modes = [
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 1.0], [0.0, 0.0]]),
    ]
    result = scan_systematic_modes_from_images(
        smooth_mean_image=smooth,
        subhalo_mean_image=subhalo,
        systematic_mode_images=systematic_modes,
        sigma_image=np.ones((2, 2)),
        mode_names=["psf_focus", "psf_coma"],
        mode_sigmas=[0.1, 0.2],
    )

    assert [c.mode_name for c in result.couplings] == ["psf_focus", "psf_coma"]
    assert result.couplings[0].z_per_unit == pytest.approx(1.0)
    assert result.couplings[0].one_sigma_z == pytest.approx(0.1)
    assert result.couplings[1].z_per_unit == pytest.approx(0.0)
