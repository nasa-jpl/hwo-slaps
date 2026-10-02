"""Tests for Fisher nuisance-subset selection and the fixed-annulus mask.

Both features are pure configuration logic, so the detector module is loaded
with light-weight stubs for AutoLens / HCIPy while the real statistical core
and adapter are used for the profiling identity.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"
TEST_PACKAGE = "hwoslaps_fisher_subset_testpkg"

PIXEL_SCALE = 0.1
GRID_SHAPE = (11, 11)

def _load_real_submodule(module_name: str, relative_path: str) -> types.ModuleType:
    """Load one real package module under the stub test package."""
    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_detector_stubs() -> None:
    """Stub the AutoLens-dependent imports of the detector module."""

    def ensure_module(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    pkg = ensure_module(TEST_PACKAGE)
    pkg.__path__ = []
    for sub in ("modeling", "lensing", "observation", "psf"):
        ensure_module(f"{TEST_PACKAGE}.{sub}").__path__ = []
    sys.modules[f"{TEST_PACKAGE}.modeling"].__path__ = [str(SRC_ROOT / "modeling")]

    fake_al = types.ModuleType("autolens")

    class _Mask2D:
        @staticmethod
        def all_false(*args, **kwargs):
            return None

    class _Array2D:
        def __init__(self, *args, **kwargs):
            pass

    fake_al.Mask2D = _Mask2D
    fake_al.Array2D = _Array2D
    sys.modules["autolens"] = fake_al

    lensing_mod = ensure_module(f"{TEST_PACKAGE}.lensing")
    lensing_mod.generate_lensing_system = lambda *args, **kwargs: None

    lensing_utils = ensure_module(f"{TEST_PACKAGE}.lensing.utils")
    lensing_utils.LensingData = object
    lensing_utils.get_einstein_ring_position = lambda *args, **kwargs: (0.0, 0.0)

    observation_utils = ensure_module(f"{TEST_PACKAGE}.observation.utils")
    observation_utils.ObservationData = object

    psf_generator = ensure_module(f"{TEST_PACKAGE}.psf.generator")
    psf_generator.generate_psf_system = lambda *args, **kwargs: None

    psf_utils = ensure_module(f"{TEST_PACKAGE}.psf.utils")
    psf_utils.PSFData = object
    psf_utils.make_pyauto_convolver = lambda kernel: kernel
    psf_utils.make_pyauto_kernel = lambda *args, **kwargs: None
    psf_utils.pyauto_kernel_native = lambda kernel: kernel.native
    psf_utils.pyauto_kernel_pixel_scales = lambda kernel: kernel.pixel_scales

    # The statistical core, adapter and result containers are pure NumPy, so
    # the real modules are used rather than stubs.
    _load_real_submodule(f"{TEST_PACKAGE}.modeling.fisher_core", "modeling/fisher_core.py")
    _load_real_submodule(
        f"{TEST_PACKAGE}.modeling.fisher_adapter",
        "modeling/fisher_adapter.py",
    )
    _load_real_submodule(
        f"{TEST_PACKAGE}.modeling.utils_fisher",
        "modeling/utils_fisher.py",
    )


def _load_detector_module() -> types.ModuleType:
    module_name = f"{TEST_PACKAGE}.modeling.fisher_detector"
    if module_name in sys.modules:
        return sys.modules[module_name]
    original_autolens = sys.modules.get("autolens")
    _install_detector_stubs()
    try:
        module = _load_real_submodule(module_name, "modeling/fisher_detector.py")
    finally:
        if original_autolens is None:
            sys.modules.pop("autolens", None)
        else:
            sys.modules["autolens"] = original_autolens
    return module


def _light_config(light_type: str) -> dict:
    if light_type == "Image":
        return {
            "type": "Image",
            "asset_path": "source.npz",
            "centre": [0.0, 0.0],
            "flux_scale": 1.0,
            "size_scale": 1.0,
        }
    return {
        "type": "Exponential",
        "centre": [0.0, 0.0],
        "ell_comps": [0.0, 0.0],
        "intensity": 1.0,
        "effective_radius": 0.2,
    }


def _scene_config(light_type: str, lens_centre=(0.0, 0.0)) -> dict:
    return {
        "lensing": {
            "lens_galaxy": {
                "mass": {
                    "type": "Isothermal",
                    "centre": [float(lens_centre[0]), float(lens_centre[1])],
                    "ell_comps": [0.0, 0.0],
                    "einstein_radius": 0.5,
                }
            },
            "source_galaxy": {"light": _light_config(light_type)},
        }
    }


def _mask_detector(
    *,
    mask_mode: str,
    mask_annulus=None,
    lens_centre=(0.0, 0.0),
    shape=GRID_SHAPE,
):
    """Return a stub detector carrying a synthetic image grid."""
    module = _load_detector_module()
    detector = module.FisherDetector.__new__(module.FisherDetector)
    rows, cols = shape
    y_arcsec = ((rows - 1) / 2.0 - np.arange(rows)) * PIXEL_SCALE
    x_arcsec = (np.arange(cols) - (cols - 1) / 2.0) * PIXEL_SCALE
    grid_native = np.stack(
        np.meshgrid(y_arcsec, x_arcsec, indexing="ij"),
        axis=-1,
    )
    detector.mu0_adu_2d = np.ones(shape, dtype=float)
    detector.source_adu_2d = np.ones(shape, dtype=float)
    detector.sigma_adu_2d = np.ones(shape, dtype=float)
    detector.snr_threshold = 0.5
    detector.mask_mode = mask_mode
    detector.mask_annulus = mask_annulus
    detector.fit_full_config = _scene_config("Exponential", lens_centre=lens_centre)
    detector.lensing_baseline = SimpleNamespace(
        grid=SimpleNamespace(native=grid_native)
    )
    return detector, grid_native


# ----------------------------------------------------------------------
# Nuisance-subset selection
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Fixed-annulus mask
# ----------------------------------------------------------------------


def test_fixed_annulus_selects_the_expected_pixel_count():
    """Select exactly the pixels whose radius lies in the closed annulus."""
    detector, grid_native = _mask_detector(
        mask_mode="fixed_annulus",
        mask_annulus={
            "inner_arcsec": 0.15,
            "outer_arcsec": 0.35,
            "centre": "grid",
        },
    )

    mask = detector._build_mask()

    expected = np.zeros(GRID_SHAPE, dtype=bool)
    for i in range(GRID_SHAPE[0]):
        for j in range(GRID_SHAPE[1]):
            radius = float(np.hypot(grid_native[i, j, 0], grid_native[i, j, 1]))
            expected[i, j] = 0.15 <= radius <= 0.35
    np.testing.assert_array_equal(mask, expected)
    # Offsets (dy, dx) in pixels with 1.5 <= hypot <= 3.5: four axial pairs at
    # r=2 and r=3, eight (2,1)-type, eight (3,1)-type and four (2,2) diagonals.
    assert int(np.count_nonzero(mask)) == 28


def test_fixed_annulus_includes_its_closed_boundaries():
    """Keep pixels sitting exactly on the inner and outer radii."""
    detector, _ = _mask_detector(
        mask_mode="fixed_annulus",
        mask_annulus={
            "inner_arcsec": PIXEL_SCALE,
            "outer_arcsec": 2.0 * PIXEL_SCALE,
            "centre": "grid",
        },
    )

    mask = detector._build_mask()

    centre = (GRID_SHAPE[0] // 2, GRID_SHAPE[1] // 2)
    assert mask[centre[0], centre[1] + 1]
    assert mask[centre[0], centre[1] + 2]
    assert not mask[centre[0], centre[1]]
    assert not mask[centre[0], centre[1] + 3]


def test_fixed_annulus_lens_centre_offsets_the_aperture():
    """Centre the aperture on the analysis lens centre by default."""
    detector, grid_native = _mask_detector(
        mask_mode="fixed_annulus",
        mask_annulus={"inner_arcsec": 0.0, "outer_arcsec": 0.15},
        lens_centre=(0.2, -0.3),
    )

    mask = detector._build_mask()

    radius = np.hypot(
        grid_native[..., 0] - 0.2,
        grid_native[..., 1] + 0.3,
    )
    np.testing.assert_array_equal(mask, radius <= 0.15)


def test_fixed_annulus_grid_centre_ignores_the_lens_centre():
    """Centre the aperture on the grid when 'grid' is requested."""
    detector, grid_native = _mask_detector(
        mask_mode="fixed_annulus",
        mask_annulus={
            "inner_arcsec": 0.0,
            "outer_arcsec": 0.15,
            "centre": "grid",
        },
        lens_centre=(0.2, -0.3),
    )

    mask = detector._build_mask()

    radius = np.hypot(grid_native[..., 0], grid_native[..., 1])
    np.testing.assert_array_equal(mask, radius <= 0.15)


def test_fixed_annulus_rejects_an_empty_aperture():
    """Fail loudly when the declared annulus holds no pixel."""
    detector, _ = _mask_detector(
        mask_mode="fixed_annulus",
        mask_annulus={
            "inner_arcsec": 5.0,
            "outer_arcsec": 6.0,
            "centre": "grid",
        },
    )

    with pytest.raises(ValueError, match="Degenerate Fisher mask"):
        detector._build_mask()


def test_fixed_annulus_requires_its_block():
    """Reject the fixed-annulus mask with no annulus declared."""
    detector, _ = _mask_detector(mask_mode="fixed_annulus")

    with pytest.raises(ValueError, match="mask_annulus is required"):
        detector._build_mask()


@pytest.mark.parametrize(
    "annulus, match",
    [
        ({"inner_arcsec": -0.1, "outer_arcsec": 0.4}, "must be non-negative"),
        ({"inner_arcsec": 0.4, "outer_arcsec": 0.4}, "must be greater than"),
        ({"inner_arcsec": 0.5, "outer_arcsec": 0.4}, "must be greater than"),
        (
            {"inner_arcsec": float("nan"), "outer_arcsec": 0.4},
            "must be finite",
        ),
        ({"inner_arcsec": "0.1", "outer_arcsec": 0.4}, "must be numeric"),
        ({"inner_arcsec": True, "outer_arcsec": 0.4}, "must be numeric"),
        ({"outer_arcsec": 0.4}, "inner_arcsec is required"),
        ({"inner_arcsec": 0.1}, "outer_arcsec is required"),
        (
            {"inner_arcsec": 0.1, "outer_arcsec": 0.4, "centre": "source"},
            "must be 'lens' or 'grid'",
        ),
        (
            {"inner_arcsec": 0.1, "outer_arcsec": 0.4, "radius": 1.0},
            "unsupported keys: radius",
        ),
    ],
)
def test_fixed_annulus_rejects_invalid_blocks(annulus, match):
    """Reject malformed annulus declarations before any mask is built."""
    detector, _ = _mask_detector(mask_mode="fixed_annulus", mask_annulus=annulus)

    with pytest.raises(ValueError, match=match):
        detector._build_mask()


# ----------------------------------------------------------------------
# PSF-border mask
# ----------------------------------------------------------------------


def _psf_border_detector(kernel_shape):
    detector, _ = _mask_detector(mask_mode="psf_border")
    detector.fit_full_config["psf"] = {
        "kernel": {"shape_native": list(kernel_shape)}
    }
    return detector


def test_psf_border_mask_matches_the_nonlinear_dataset_support():
    """Reproduce dataset_builder._exclude_psf_edge_pixels exactly."""
    pytest.importorskip("autolens")
    from hwoslaps.modeling.nonlinear.dataset_builder import (
        _exclude_psf_edge_pixels,
    )

    for kernel_shape in ((5, 5), (3, 7), (1, 1)):
        detector = _psf_border_detector(kernel_shape)
        np.testing.assert_array_equal(
            detector._build_mask(),
            _exclude_psf_edge_pixels(
                np.ones(GRID_SHAPE, dtype=bool),
                psf_shape=kernel_shape,
            ),
        )


def test_psf_border_mask_removes_half_kernel_borders():
    """Keep exactly the interior rectangle inside the half-kernel border."""
    detector = _psf_border_detector((5, 3))

    mask = detector._build_mask()

    expected = np.zeros(GRID_SHAPE, dtype=bool)
    expected[2:-2, 1:-1] = True
    np.testing.assert_array_equal(mask, expected)
    assert int(np.count_nonzero(mask)) == (11 - 4) * (11 - 2)


def test_psf_border_mask_rejects_an_annulus_block():
    """Reject an annulus block the PSF-border mode never reads."""
    detector = _psf_border_detector((5, 5))
    detector.mask_annulus = {"inner_arcsec": 0.1, "outer_arcsec": 0.4}

    with pytest.raises(ValueError, match="only accepted when"):
        detector._build_mask()


@pytest.mark.parametrize("mask_mode", ["source_snr", "all_pixels"])
def test_annulus_block_is_rejected_for_other_mask_modes(mask_mode):
    """Reject an annulus block that the configured mask mode never reads."""
    detector, _ = _mask_detector(
        mask_mode=mask_mode,
        mask_annulus={"inner_arcsec": 0.1, "outer_arcsec": 0.4},
    )

    with pytest.raises(ValueError, match="only accepted when"):
        detector._build_mask()


def test_default_mask_modes_are_unchanged():
    """Keep the source-S/N and all-pixel masks exactly as they were."""
    all_pixels, _ = _mask_detector(mask_mode="all_pixels")
    np.testing.assert_array_equal(
        all_pixels._build_mask(),
        np.ones(GRID_SHAPE, dtype=bool),
    )

    source_snr, _ = _mask_detector(mask_mode="source_snr")
    source_snr.source_adu_2d = np.zeros(GRID_SHAPE, dtype=float)
    source_snr.source_adu_2d[3, 4] = 10.0
    expected = np.zeros(GRID_SHAPE, dtype=bool)
    expected[3, 4] = True
    np.testing.assert_array_equal(source_snr._build_mask(), expected)


def test_unknown_mask_mode_is_rejected():
    """Reject a mask mode outside the supported vocabulary."""
    detector, _ = _mask_detector(mask_mode="everything")

    with pytest.raises(ValueError, match="mask_mode must be"):
        detector._build_mask()


def test_pixel_coordinates_must_match_the_mean_image_shape():
    """Fail loudly when the lensing grid does not describe the mean image."""
    detector, _ = _mask_detector(
        mask_mode="fixed_annulus",
        mask_annulus={"inner_arcsec": 0.0, "outer_arcsec": 1.0},
    )
    detector.mu0_adu_2d = np.ones((7, 7), dtype=float)

    with pytest.raises(ValueError, match="does not match the mean-image shape"):
        detector._build_mask()
