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

EXPONENTIAL_NUISANCE_NAMES = [
    "lens.centre_y",
    "lens.centre_x",
    "lens.einstein_radius",
    "lens.ell_comp_1",
    "lens.ell_comp_2",
    "source.centre_y",
    "source.centre_x",
    "source.ell_comp_1",
    "source.ell_comp_2",
    "source.intensity",
    "source.effective_radius",
    "observation.background_offset_adu",
]

IMAGE_NUISANCE_NAMES = [
    name
    for name in EXPONENTIAL_NUISANCE_NAMES
    if name not in {"source.ell_comp_1", "source.ell_comp_2"}
]


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


def _subset_detector(light_type: str, subset) -> tuple:
    """Return the module and a stub detector ready for subset selection."""
    module = _load_detector_module()
    detector = module.FisherDetector.__new__(module.FisherDetector)
    detector.fit_full_config = _scene_config(light_type)
    detector.full_config = copy.deepcopy(detector.fit_full_config)
    detector.prior_sigmas = {}
    detector.include_background_offset = True
    detector.nuisance_subset = subset
    return module, detector


def _selected_names(light_type: str, subset) -> tuple:
    module, detector = _subset_detector(light_type, subset)
    specs, label = detector._select_nuisance_subset(
        detector._build_scalar_nuisance_specs()
    )
    return [spec.name for spec in specs], label


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


def test_exponential_scene_has_twelve_scalar_directions():
    """Build the documented scalar direction set for an exponential source."""
    names, label = _selected_names("Exponential", None)

    assert names == EXPONENTIAL_NUISANCE_NAMES
    assert len(names) == 12
    assert label == "all"


def test_image_scene_has_ten_scalar_directions():
    """Drop the two source ellipticity directions for an image source."""
    names, label = _selected_names("Image", None)

    assert names == IMAGE_NUISANCE_NAMES
    assert len(names) == 10
    assert label == "all"


@pytest.mark.parametrize("light_type", ["Exponential", "Image"])
@pytest.mark.parametrize("selector", [None, "all", "ALL"])
def test_nuisance_subset_all_matches_the_unfiltered_directions(light_type, selector):
    """Keep every scalar direction for the default and explicit 'all'."""
    expected = (
        EXPONENTIAL_NUISANCE_NAMES if light_type == "Exponential" else IMAGE_NUISANCE_NAMES
    )
    names, label = _selected_names(light_type, selector)

    assert names == expected
    assert label == "all"


@pytest.mark.parametrize("light_type", ["Exponential", "Image"])
def test_nuisance_subset_none_selects_nothing(light_type):
    """Profile no scalar direction at all for the reserved word 'none'."""
    names, label = _selected_names(light_type, "none")

    assert names == []
    assert label == "none"


@pytest.mark.parametrize("light_type", ["Exponential", "Image"])
def test_nuisance_subset_lens_only_selects_lens_directions(light_type):
    """Select every lens direction and no source or background direction."""
    names, label = _selected_names(light_type, "lens_only")

    assert names == [
        "lens.centre_y",
        "lens.centre_x",
        "lens.einstein_radius",
        "lens.ell_comp_1",
        "lens.ell_comp_2",
    ]
    assert label == "lens_only"


def test_nuisance_subset_source_only_selects_source_directions():
    """Select every source direction of an exponential-source scene."""
    names, label = _selected_names("Exponential", "source_only")

    assert names == [
        "source.centre_y",
        "source.centre_x",
        "source.ell_comp_1",
        "source.ell_comp_2",
        "source.intensity",
        "source.effective_radius",
    ]
    assert label == "source_only"


def test_nuisance_subset_source_only_drops_ellipticity_for_image_source():
    """Select the four source directions an image-source scene defines."""
    names, label = _selected_names("Image", "source_only")

    assert names == [
        "source.centre_y",
        "source.centre_x",
        "source.intensity",
        "source.effective_radius",
    ]
    assert label == "source_only"


@pytest.mark.parametrize(
    "light_type, expected_count",
    [("Exponential", 11), ("Image", 9)],
)
def test_nuisance_subset_lens_and_source_excludes_background(light_type, expected_count):
    """Select lens and source directions but never the background offset."""
    names, label = _selected_names(light_type, "lens_and_source")

    assert len(names) == expected_count
    assert "observation.background_offset_adu" not in names
    assert all(name.startswith(("lens.", "source.")) for name in names)
    assert label == "lens_and_source"


def test_nuisance_subset_explicit_list_keeps_canonical_order():
    """Select the named directions in their canonical construction order."""
    names, label = _selected_names(
        "Exponential",
        ["source.intensity", "lens.centre_y"],
    )

    assert names == ["lens.centre_y", "source.intensity"]
    assert label == "explicit"


def test_nuisance_subset_explicit_list_may_name_the_background_direction():
    """Allow the background direction to be named when it exists."""
    names, label = _selected_names(
        "Exponential",
        ["observation.background_offset_adu"],
    )

    assert names == ["observation.background_offset_adu"]
    assert label == "explicit"


def test_nuisance_subset_rejects_direction_absent_from_the_scene():
    """Reject a source ellipticity direction an image scene does not define."""
    with pytest.raises(ValueError, match="unknown direction 'source.ell_comp_1'"):
        _selected_names("Image", ["source.ell_comp_1"])


def test_nuisance_subset_unknown_name_error_lists_the_valid_names():
    """Name every valid direction when rejecting an unknown one."""
    with pytest.raises(ValueError) as excinfo:
        _selected_names("Exponential", ["lens.centre_z"])

    message = str(excinfo.value)
    for name in EXPONENTIAL_NUISANCE_NAMES:
        assert name in message


def test_nuisance_subset_rejects_background_direction_when_flag_is_off():
    """Reject the background direction when the flag leaves it undefined."""
    module, detector = _subset_detector(
        "Exponential",
        ["observation.background_offset_adu"],
    )
    detector.include_background_offset = False

    with pytest.raises(ValueError, match="unknown direction"):
        detector._select_nuisance_subset(detector._build_scalar_nuisance_specs())


def test_nuisance_subset_rejects_psf_mode_names():
    """Reject PSF modes, which the PSF selectors alone govern."""
    with pytest.raises(ValueError, match="must not name PSF modes"):
        _selected_names("Exponential", ["psf.global_zernikes[4]"])


def test_nuisance_subset_rejects_unknown_reserved_word():
    """Reject a reserved word outside the documented vocabulary."""
    with pytest.raises(ValueError, match="nuisance_subset must be one of"):
        _selected_names("Exponential", "lens_and_background")


def test_nuisance_subset_rejects_duplicate_names():
    """Reject a list that names the same direction twice."""
    with pytest.raises(ValueError, match="duplicate direction"):
        _selected_names("Exponential", ["lens.centre_y", "lens.centre_y"])


def test_nuisance_subset_rejects_empty_list():
    """Reject an empty list and point at the reserved word 'none'."""
    with pytest.raises(ValueError, match="must be non-empty"):
        _selected_names("Exponential", [])


@pytest.mark.parametrize("selector", [12, True, {"lens": True}])
def test_nuisance_subset_rejects_non_name_selectors(selector):
    """Reject selectors that are neither a reserved word nor a name list."""
    with pytest.raises(ValueError, match="nuisance_subset must be one of"):
        _selected_names("Exponential", selector)


def test_nuisance_subset_rejects_non_string_list_entries():
    """Reject a list entry that is not a direction name."""
    with pytest.raises(ValueError, match="entries must be nuisance"):
        _selected_names("Exponential", ["lens.centre_y", 3])


def test_no_nuisance_directions_make_profiled_information_equal_raw():
    """Recover the raw information when 'none' leaves nothing to profile."""
    module = _load_detector_module()
    adapter = sys.modules[f"{TEST_PACKAGE}.modeling.fisher_adapter"]

    names, label = _selected_names("Exponential", "none")
    assert (names, label) == ([], "none")

    rng = np.random.default_rng(20260819)
    smooth = 100.0 + rng.normal(size=GRID_SHAPE)
    subhalo = smooth + 0.5 * rng.normal(size=GRID_SHAPE)
    sigma = np.full(GRID_SHAPE, 2.0)
    mask = np.ones(GRID_SHAPE, dtype=bool)

    result = adapter.compute_asimov_from_images(
        smooth_mean_image=smooth,
        subhalo_mean_image=subhalo,
        sigma_image=sigma,
        # The detector passes None when the profiled nuisance list is empty.
        nuisance_images=None,
        prior_precision=module.FisherDetector._build_prior_precision_matrix([]),
        mask=mask,
        amplitude_true=1.0,
        nuisance_names=[],
    )

    assert result.fisher_profiled == result.fisher_raw
    assert result.degradation == 1.0


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
