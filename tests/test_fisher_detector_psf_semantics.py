"""Tests for PSF-basis semantics in the Fisher detector.

These tests target the three design requirements called out in review:

1. Derivatives must be anchored to the true science fiducial, not placeholder YAML
   coefficients in disabled PSF families.
2. The instrument basis must be defined explicitly and therefore independent of
   which coefficients happen to be present/nonzero in the science config.
3. The PSF nuisance-fit basis and PSF scan basis must be disjoint.

The full detector depends on AutoLens / HCIPy, which are not imported here.
Instead we load the module with light-weight stubs and exercise only the helper
logic that defines PSF semantics.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "hwoslaps"
TEST_PACKAGE = "hwoslaps_fisher_detector_testpkg"


def _install_detector_stubs():
    def ensure_module(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    pkg = ensure_module(TEST_PACKAGE)
    pkg.__path__ = []
    modeling_pkg = ensure_module(f"{TEST_PACKAGE}.modeling")
    modeling_pkg.__path__ = []
    lensing_pkg = ensure_module(f"{TEST_PACKAGE}.lensing")
    lensing_pkg.__path__ = []
    observation_pkg = ensure_module(f"{TEST_PACKAGE}.observation")
    observation_pkg.__path__ = []
    psf_pkg = ensure_module(f"{TEST_PACKAGE}.psf")
    psf_pkg.__path__ = []

    fake_al = types.ModuleType("autolens")
    sys.modules["autolens"] = fake_al

    class _Kernel2D:
        @staticmethod
        def no_mask(*args, **kwargs):
            return None

    class _Mask2D:
        @staticmethod
        def all_false(*args, **kwargs):
            return None

    class _Array2D:
        def __init__(self, *args, **kwargs):
            pass

    class _SimulatorImaging:
        def __init__(self, *args, **kwargs):
            pass

        def via_image_from(self, image):
            return SimpleNamespace(data=SimpleNamespace(native=None))

    fake_al.Kernel2D = _Kernel2D
    fake_al.Mask2D = _Mask2D
    fake_al.Array2D = _Array2D
    fake_al.SimulatorImaging = _SimulatorImaging

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

    adapter = ensure_module(f"{TEST_PACKAGE}.modeling.fisher_adapter")
    adapter.compute_asimov_from_images = lambda *args, **kwargs: None
    adapter.evaluate_signal_bank_from_images = lambda *args, **kwargs: None
    adapter.extract_masked_covariance = lambda *args, **kwargs: None
    adapter.flatten_masked_image = lambda *args, **kwargs: None
    adapter.scan_systematic_modes_from_images = lambda *args, **kwargs: None
    adapter.stack_masked_images = lambda *args, **kwargs: None

    core = ensure_module(f"{TEST_PACKAGE}.modeling.fisher_core")

    class _Whitener:
        @classmethod
        def from_covariance(cls, covariance):
            return cls()

        @classmethod
        def from_sigma(cls, sigma):
            return cls()

        def apply(self, arr):
            return arr

    class _Workspace:
        def __init__(self, *args, **kwargs):
            self.nuisance_condition_number = 1.0

    core.ProfileLikelihoodWorkspace = _Workspace
    core.Whitener = _Whitener

    utils = ensure_module(f"{TEST_PACKAGE}.modeling.utils_fisher")
    for name in (
        "FisherLocalData",
        "FisherMapData",
        "FisherModeCouplingData",
        "FisherModeScanData",
    ):
        setattr(utils, name, type(name, (), {}))


def _load_detector_module():
    module_name = f"{TEST_PACKAGE}.modeling.fisher_detector"
    if module_name in sys.modules:
        return sys.modules[module_name]
    original_autolens = sys.modules.get("autolens")
    _install_detector_stubs()
    module_path = SRC_ROOT / "modeling" / "fisher_detector.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if original_autolens is None:
            sys.modules.pop("autolens", None)
        else:
            sys.modules["autolens"] = original_autolens
    return module


def _make_detector_stub():
    module = _load_detector_module()
    detector = module.FisherDetector.__new__(module.FisherDetector)
    detector.psf_mode_steps = {}
    detector.psf_mode_prior_sigmas = {}
    detector.psf_data = SimpleNamespace(num_segments=6)
    return module, detector


def test_science_fiducial_ignores_placeholder_coefficients_in_disabled_family():
    module, detector = _make_detector_stub()
    detector.full_config = {
        "psf": {
            "aberrations": {
                "enable_segment_pistons": False,
                "enable_segment_tiptilts": False,
                "enable_segment_hexikes": False,
                "enable_global_zernikes": False,
                "segment_pistons": {0: 5.0},
                "segment_tiptilts": {0: [1.0, -1.0]},
                "segment_hexikes": {0: {1: 7.5}},
                "global_zernikes": {4: 3.0},
            }
        }
    }

    detector.science_psf_config_template = detector._build_science_psf_config_template()
    aberr = detector.science_psf_config_template["psf"]["aberrations"]
    assert aberr["segment_pistons"] == {}
    assert aberr["segment_tiptilts"] == {}
    assert aberr["segment_hexikes"] == {}
    assert aberr["global_zernikes"] == {}

    spec = module._PsfModeSpec(
        name="psf.segment_hexikes[0][1]",
        family="segment_hexikes",
        path=("psf", "aberrations", "segment_hexikes", 0, 1),
        enable_flag_path=("psf", "aberrations", "enable_segment_hexikes"),
        step=1.0,
        prior_sigma=None,
    )
    assert detector._science_psf_base_value(spec) == 0.0


def test_segment_hexike_derivative_assignment_preserves_dict_shape_for_perfect_psf():
    module, detector = _make_detector_stub()
    config = {
        "psf": {
            "aberrations": {
                "enable_segment_hexikes": False,
                "segment_hexikes": {},
            }
        }
    }
    spec = module._PsfModeSpec(
        name="psf.segment_hexikes[0][2]",
        family="segment_hexikes",
        path=("psf", "aberrations", "segment_hexikes", 0, 2),
        enable_flag_path=("psf", "aberrations", "enable_segment_hexikes"),
        step=1.0,
        prior_sigma=None,
    )

    detector._ensure_psf_derivative_container(config, spec)
    detector._set_path_value_create(config, spec.path, 1.0)

    assert config["psf"]["aberrations"]["segment_hexikes"] == {0: {2: 1.0}}


def test_explicit_psf_basis_is_independent_of_yaml_occupancy():
    module, detector = _make_detector_stub()
    detector.full_config = {
        "psf": {
            "aberrations": {
                "enable_segment_pistons": False,
                "enable_segment_tiptilts": False,
                "enable_segment_hexikes": False,
                "enable_global_zernikes": False,
                "segment_hexikes": {},
            }
        }
    }

    specs = detector._build_psf_mode_specs_from_selection(
        {
            "segment_hexikes": {
                "segments": [0, 2],
                "mode_nolls": [1, 2],
            },
            "global_zernikes": {"mode_nolls": [4]},
        },
        context="test.psf_basis",
    )
    names = {spec.name for spec in specs}
    assert names == {
        "psf.segment_hexikes[0][1]",
        "psf.segment_hexikes[0][2]",
        "psf.segment_hexikes[2][1]",
        "psf.segment_hexikes[2][2]",
        "psf.global_zernikes[4]",
    }


def test_fit_and_scan_psf_bases_must_be_disjoint():
    module, detector = _make_detector_stub()
    detector.full_config = {
        "psf": {
            "aberrations": {
                "enable_segment_pistons": True,
                "enable_segment_tiptilts": False,
                "enable_segment_hexikes": False,
                "enable_global_zernikes": False,
                "segment_pistons": {0: 0.0, 1: 0.0},
            }
        }
    }
    detector.psf_basis_config = {"segment_pistons": {"segments": [0, 1]}}
    detector.instrument_psf_mode_specs = detector._build_psf_mode_specs_from_selection(
        detector.psf_basis_config,
        context="test.psf_basis",
    )
    detector._instrument_psf_mode_name_set = {spec.name for spec in detector.instrument_psf_mode_specs}

    detector.fit_psf_mode_specs = detector._build_psf_mode_specs_from_selection(
        {"segment_pistons": {"segments": [0]}},
        context="test.fit_psf_mode_selection",
    )
    detector.scan_psf_mode_specs = detector._build_psf_mode_specs_from_selection(
        {"segment_pistons": {"segments": [0, 1]}},
        context="test.scan_psf_mode_selection",
    )

    with pytest.raises(ValueError, match="must be disjoint"):
        detector._validate_psf_mode_spec_sets()
