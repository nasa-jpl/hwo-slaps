"""Tests for fit-side macro-lens Fisher mismatch support."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path

import autolens as al
import matplotlib.axes
import numpy as np
import pytest
import yaml

from hwoslaps.lensing import generate_lensing_system
from hwoslaps.modeling.fisher_adapter import flatten_masked_image
from hwoslaps.modeling.fisher_detector import (
    FisherDetector,
    _mean_adu_images_from_lensing_arrays,
)
from hwoslaps.modeling.fisher_grid_jax import JaxGridTemplateEngine
from hwoslaps.modeling.generator_fisher import perform_fisher_detection
from hwoslaps.modeling.nonlinear.autolens_model_builder import (
    smooth_model_spec_from_config,
)
from hwoslaps.modeling.utils_fisher import (
    FisherDetectionData,
    load_fisher_grid_map_npz,
    print_fisher_summary,
    save_fisher_grid_map_npz,
)
from hwoslaps.observation import generate_observation
from hwoslaps.pipeline import Pipeline
from hwoslaps.psf.generator import generate_psf_system
from hwoslaps.psf.utils import pyauto_kernel_native
from hwoslaps.plotting.detection_plots import plot_fisher_detection_grid_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _nuisance_detector(mass: dict, shear=None) -> FisherDetector:
    """Return a detector stub sufficient for nuisance-spec construction."""
    detector = FisherDetector.__new__(FisherDetector)
    lens_galaxy = {"mass": deepcopy(mass)}
    if shear is not None:
        lens_galaxy["shear"] = list(shear)
    detector.fit_full_config = {
        "lensing": {
            "lens_galaxy": lens_galaxy,
            "source_galaxy": {
                "light": {
                    "type": "Exponential",
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.0, 0.0],
                    "intensity": 1.0,
                    "effective_radius": 0.2,
                }
            },
        }
    }
    detector.full_config = deepcopy(detector.fit_full_config)
    detector.prior_sigmas = {}
    detector.include_background_offset = False
    return detector


def test_power_law_nuisance_names_follow_analysis_model_order():
    """Place slope, m3/m4 pairs, and shear after legacy lens directions."""
    mass = {
        "type": "PowerLaw",
        "centre": [0.0, 0.0],
        "ell_comps": [0.1, 0.0],
        "einstein_radius": 1.0,
        "slope": 2.0,
        "multipoles": {"m4": [0.02, 0.0], "m3": [0.0, 0.01]},
    }
    detector = _nuisance_detector(mass, shear=(0.02, -0.01))
    names = [spec.name for spec in detector._build_scalar_nuisance_specs()]
    assert names == [
        "lens.centre_y",
        "lens.centre_x",
        "lens.einstein_radius",
        "lens.ell_comp_1",
        "lens.ell_comp_2",
        "lens.slope",
        "lens.multipole_m3_1",
        "lens.multipole_m3_2",
        "lens.multipole_m4_1",
        "lens.multipole_m4_2",
        "lens.shear_1",
        "lens.shear_2",
        "source.centre_y",
        "source.centre_x",
        "source.ell_comp_1",
        "source.ell_comp_2",
        "source.intensity",
        "source.effective_radius",
    ]


def test_isothermal_nuisance_names_remain_unchanged():
    """Keep the legacy Isothermal scalar nuisance list unchanged."""
    mass = {
        "type": "Isothermal",
        "centre": [0.0, 0.0],
        "ell_comps": [0.1, 0.0],
        "einstein_radius": 1.0,
    }
    detector = _nuisance_detector(mass)
    names = [spec.name for spec in detector._build_scalar_nuisance_specs()]
    assert names == [
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
    ]


def _runtime_config(tmp_dir: Path) -> dict:
    """Return a small flexible scene for complete Fisher-path tests."""
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)
    config["run_name"] = "item5-fisher-test"
    config["plotting"] = {"enabled": False, "output_dir": str(tmp_dir)}
    config["lensing"]["grid"] = {"shape": [15, 15], "pixel_scale": 0.1}
    config["lensing"]["lens_galaxy"]["mass"] = {
        "type": "PowerLaw",
        "centre": [0.0, 0.0],
        "ell_comps": [0.1, 0.0],
        "einstein_radius": 0.5,
        "slope": 2.05,
        "multipoles": {"m3": [0.0, 0.006], "m4": [0.012, 0.0]},
    }
    config["lensing"]["lens_galaxy"]["shear"] = [0.015, -0.008]
    light = config["lensing"]["source_galaxy"]["light"]
    light.update(
        {
            "centre": [0.02, 0.03],
            "ell_comps": [0.03, -0.01],
            "intensity": 4.0,
            "effective_radius": 0.16,
        }
    )
    config["lensing"]["subhalo"] = {
        "enabled": True,
        "model": "PointMass",
        "mass": 1.0e8,
        "position": {"type": "direct", "centre": [0.1, 0.0]},
    }
    config["psf"]["kernel"]["shape_native"] = [7, 7]
    config["psf"]["hres_psf"].update(
        {"num_pix": 64, "num_airy": 4, "save_highres_psf_npy": False}
    )
    aberrations = config["psf"]["aberrations"]
    aberrations.update(
        {
            "enable_segment_pistons": False,
            "enable_segment_tiptilts": False,
            "enable_segment_hexikes": False,
            "enable_global_zernikes": True,
            "segment_pistons": {},
            "segment_tiptilts": {},
            "segment_hexikes": {},
            "global_zernikes": {4: 10.0},
        }
    )
    config["observation"]["exposure_time"] = 50.0
    config["modeling"]["fisher"].update(
        {
            "mode": "both",
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
                "detection_q_threshold": 1.0,
                "num_workers": 1,
                "engine": "reference",
            },
        }
    )
    return config


def _scene_products(config: dict, psf_data=None) -> dict:
    """Generate truth baseline/test lensing and observations."""
    baseline_config = deepcopy(config)
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    if psf_data is None:
        psf_data = generate_psf_system(config["psf"], full_config=config)
    baseline = generate_lensing_system(
        baseline_config["lensing"], full_config=baseline_config
    )
    test = generate_lensing_system(config["lensing"], full_config=config)
    observation_baseline = generate_observation(
        baseline,
        psf_data,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )
    observation_test = generate_observation(
        test,
        psf_data,
        observation_config=config["observation"],
        full_config=config,
    )
    return {
        "psf": psf_data,
        "baseline": baseline,
        "test": test,
        "observation_baseline": observation_baseline,
        "observation_test": observation_test,
    }


def _detector(config: dict, products: dict) -> FisherDetector:
    """Construct a detector for one truth scene and analysis config."""
    return FisherDetector(
        observation_baseline=products["observation_baseline"],
        lensing_baseline=products["baseline"],
        psf_data=products["psf"],
        full_config=config,
        fisher_config=deepcopy(config["modeling"]["fisher"]),
    )


def _perform_detection(config: dict, products: dict) -> FisherDetectionData:
    """Run the production Fisher orchestration on prepared truth products."""
    return perform_fisher_detection(
        observation_baseline=products["observation_baseline"],
        observation_test=products["observation_test"],
        lensing_baseline=products["baseline"],
        lensing_test=products["test"],
        psf_data=products["psf"],
        detection_config=config["modeling"],
        full_config=config,
    )


def _fit_lens_from_truth(config: dict) -> dict:
    """Return the fit-lens schema for the configured truth macro model."""
    truth = config["lensing"]["lens_galaxy"]
    result = {"mass": deepcopy(truth["mass"])}
    if "shear" in truth:
        result["shear"] = deepcopy(truth["shear"])
    return result


def _isothermal_fit(config: dict) -> dict:
    """Return a legacy Isothermal fit model sharing primary parameters."""
    truth_mass = config["lensing"]["lens_galaxy"]["mass"]
    return {
        "mass": {
            "type": "Isothermal",
            "centre": deepcopy(truth_mass["centre"]),
            "ell_comps": deepcopy(truth_mass["ell_comps"]),
            "einstein_radius": truth_mass["einstein_radius"],
        }
    }


def _direct_tracer(
    config: dict,
    cosmology,
    *,
    subhalo_position_yx=None,
    subhalo_einstein_radius=None,
):
    """Build the runtime test tracer directly from AutoLens classes."""
    lens_config = config["lensing"]["lens_galaxy"]
    mass = lens_config["mass"]
    if mass["type"] == "PowerLaw":
        profiles = {
            "mass": al.mp.PowerLaw(
                centre=tuple(mass["centre"]),
                ell_comps=tuple(mass["ell_comps"]),
                einstein_radius=mass["einstein_radius"],
                slope=mass["slope"],
            )
        }
        for order_name, components in sorted(
            mass.get("multipoles", {}).items()
        ):
            profiles[f"multipole_{order_name}"] = al.mp.PowerLawMultipole(
                m=int(order_name[1:]),
                centre=tuple(mass["centre"]),
                einstein_radius=mass["einstein_radius"],
                slope=mass["slope"],
                multipole_comps=tuple(components),
            )
    else:
        profiles = {
            "mass": al.mp.Isothermal(
                centre=tuple(mass["centre"]),
                ell_comps=tuple(mass["ell_comps"]),
                einstein_radius=mass["einstein_radius"],
            )
        }
    if "shear" in lens_config:
        profiles["shear"] = al.mp.ExternalShear(
            gamma_1=lens_config["shear"][0],
            gamma_2=lens_config["shear"][1],
        )
    if subhalo_position_yx is not None:
        profiles["subhalo"] = al.mp.PointMass(
            centre=tuple(subhalo_position_yx),
            einstein_radius=subhalo_einstein_radius,
        )
    lens = al.Galaxy(
        redshift=lens_config["redshift"],
        **profiles,
    )

    source_config = config["lensing"]["source_galaxy"]
    light = source_config["light"]
    assert light["type"] == "Exponential"
    source = al.Galaxy(
        redshift=source_config["redshift"],
        light=al.lp.Exponential(
            centre=tuple(light["centre"]),
            ell_comps=tuple(light["ell_comps"]),
            intensity=light["intensity"],
            effective_radius=light["effective_radius"],
        ),
    )
    return al.Tracer(galaxies=[lens, source], cosmology=cosmology)


def _direct_adu_image(
    config: dict,
    reference_lensing,
    kernel,
    *,
    subhalo_position_yx=None,
    subhalo_einstein_radius=None,
):
    """Render one direct tracer and apply the shared observation transform."""
    tracer = _direct_tracer(
        config,
        reference_lensing.tracer.cosmology,
        subhalo_position_yx=subhalo_position_yx,
        subhalo_einstein_radius=subhalo_einstein_radius,
    )
    image = tracer.image_2d_from(grid=reference_lensing.grid).native
    direct_lensing = replace(
        reference_lensing,
        tracer=tracer,
        image=image,
    )
    return _mean_adu_images_from_lensing_arrays(
        direct_lensing,
        config["observation"],
        (kernel,),
    )[0]


@pytest.fixture(scope="module")
def flexible_setup(tmp_path_factory):
    """Build one flexible truth scene and matched detector."""
    tmp_dir = tmp_path_factory.mktemp("item5-fisher")
    os.environ["NUMBA_CACHE_DIR"] = str(tmp_dir / "numba-cache")
    config = _runtime_config(tmp_dir)
    products = _scene_products(config)
    detector = _detector(config, products)
    return {
        "config": config,
        "products": products,
        "detector": detector,
        "grid": detector.compute_grid_map(),
        "local": detector.compute_local(
            products["observation_test"], products["test"]
        ),
    }


@pytest.fixture(scope="module")
def lens_mismatch_setup(flexible_setup):
    """Build the flexible-truth versus Isothermal-fit mismatch arm."""
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }
    detector = _detector(config, flexible_setup["products"])
    return {
        "config": config,
        "products": flexible_setup["products"],
        "detector": detector,
        "grid": detector.compute_grid_map(),
        "local": detector.compute_local(
            flexible_setup["products"]["observation_test"],
            flexible_setup["products"]["test"],
        ),
    }


@pytest.fixture(scope="module")
def transported_detection_results(lens_mismatch_setup):
    """Run lens-only and combined mismatch through the payload boundary."""
    lens_config = deepcopy(lens_mismatch_setup["config"])
    lens_config["modeling"]["fisher"]["mode"] = "map"
    lens_result = _perform_detection(
        lens_config,
        lens_mismatch_setup["products"],
    )

    combined_config = deepcopy(lens_config)
    fit_psf = deepcopy(combined_config["psf"])
    fit_psf["aberrations"]["global_zernikes"][4] = 13.0
    combined_config["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": fit_psf,
    }
    combined_result = _perform_detection(
        combined_config,
        lens_mismatch_setup["products"],
    )
    return {"lens": lens_result, "combined": combined_result}


def test_detection_payload_transports_lens_mismatch_without_config(
    transported_detection_results,
    capsys,
):
    """Keep lens mismatch metadata after provenance config is removed."""
    result = transported_detection_results["lens"]
    assert result.lens_mismatch_enabled is True
    assert result.psf_mismatch_enabled is False

    result.config = None
    print_fisher_summary(result)
    output = capsys.readouterr().out
    assert "fit_lens mode: explicit" in output
    assert "fit_psf mode: explicit" not in output


def test_detection_payload_transports_combined_mismatch(
    transported_detection_results,
    capsys,
):
    """Carry and print both explicit mismatch-channel flags."""
    result = transported_detection_results["combined"]
    assert result.lens_mismatch_enabled is True
    assert result.psf_mismatch_enabled is True

    print_fisher_summary(result)
    output = capsys.readouterr().out
    assert "fit_lens mode: explicit" in output
    assert "fit_psf mode: explicit" in output


def test_flexible_scalar_derivatives_match_independent_central_difference(
    flexible_setup,
):
    """Match every new lens direction to a nonzero independent derivative."""
    detector = flexible_setup["detector"]
    kernel = detector._ensure_odd_kernel(detector.model_psf_data.kernel)
    finite_diff = flexible_setup["config"]["modeling"]["fisher"]["finite_diff"]
    cases = (
        (
            "lens.slope",
            ("lensing", "lens_galaxy", "mass", "slope"),
            finite_diff["slope"],
        ),
        (
            "lens.multipole_m3_1",
            (
                "lensing",
                "lens_galaxy",
                "mass",
                "multipoles",
                "m3",
                0,
            ),
            finite_diff["multipole_comp"],
        ),
        (
            "lens.multipole_m3_2",
            (
                "lensing",
                "lens_galaxy",
                "mass",
                "multipoles",
                "m3",
                1,
            ),
            finite_diff["multipole_comp"],
        ),
        (
            "lens.multipole_m4_1",
            (
                "lensing",
                "lens_galaxy",
                "mass",
                "multipoles",
                "m4",
                0,
            ),
            finite_diff["multipole_comp"],
        ),
        (
            "lens.multipole_m4_2",
            (
                "lensing",
                "lens_galaxy",
                "mass",
                "multipoles",
                "m4",
                1,
            ),
            finite_diff["multipole_comp"],
        ),
        (
            "lens.shear_1",
            ("lensing", "lens_galaxy", "shear", 0),
            finite_diff["shear_comp"],
        ),
        (
            "lens.shear_2",
            ("lensing", "lens_galaxy", "shear", 1),
            finite_diff["shear_comp"],
        ),
    )
    for name, path, step in cases:
        index = detector.nuisance_names.index(name)
        plus = deepcopy(detector.baseline_config_template)
        minus = deepcopy(detector.baseline_config_template)
        plus_value = plus
        minus_value = minus
        for key in path[:-1]:
            plus_value = plus_value[key]
            minus_value = minus_value[key]
        plus_value[path[-1]] += step
        minus_value[path[-1]] -= step
        plus_scene = generate_lensing_system(plus["lensing"], full_config=plus)
        minus_scene = generate_lensing_system(minus["lensing"], full_config=minus)
        plus_image = _mean_adu_images_from_lensing_arrays(
            plus_scene, plus["observation"], (kernel,)
        )[0]
        minus_image = _mean_adu_images_from_lensing_arrays(
            minus_scene, minus["observation"], (kernel,)
        )[0]
        independent = (plus_image - minus_image) / (2.0 * step)
        production = detector.scalar_nuisance_images[index]
        assert np.linalg.norm(production) > 1.0e-12
        np.testing.assert_allclose(
            production,
            independent,
            rtol=1.0e-10,
            atol=1.0e-10,
        )


def test_slope_finite_difference_rejects_domain_crossing(flexible_setup):
    """Reject a central slope step whose lower arm leaves the domain."""
    config = deepcopy(flexible_setup["config"])
    config["lensing"]["lens_galaxy"]["mass"]["slope"] = 1.0005
    products = _scene_products(
        config,
        psf_data=flexible_setup["products"]["psf"],
    )

    with pytest.raises(ValueError) as error:
        _detector(config, products)

    message = str(error.value)
    assert "slope=1.0005" in message
    assert "step=0.001" in message
    assert "bounds (1, 3)" in message


def test_slope_finite_difference_accepts_interior_step(flexible_setup):
    """Keep the central slope derivative available at slope two."""
    config = deepcopy(flexible_setup["config"])
    config["lensing"]["lens_galaxy"]["mass"]["slope"] = 2.0
    products = _scene_products(
        config,
        psf_data=flexible_setup["products"]["psf"],
    )
    detector = _detector(config, products)

    assert "lens.slope" in detector.nuisance_names


def test_identical_explicit_fit_lens_recovers_reference_matched_limit(
    flexible_setup,
):
    """Recover q_F and negligible spurious fields for identical fit lens."""
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _fit_lens_from_truth(config),
    }
    detector = _detector(config, flexible_setup["products"])
    grid = detector.compute_grid_map()
    np.testing.assert_allclose(
        grid.q_asimov_2d,
        flexible_setup["grid"].q_asimov_2d,
        rtol=1.0e-10,
    )
    assert np.nanmax(np.abs(grid.z_spurious_2d)) < 1.0e-4
    assert np.nanmax(grid.q_spurious_2d) < 1.0e-8


def test_flexible_grid_node_matches_local(flexible_setup):
    """Match a PowerLaw-plus-shear grid node to local evaluation."""
    grid = flexible_setup["grid"]
    y_index = int(np.argmin(np.abs(grid.y_coords - 0.1)))
    x_index = int(np.argmin(np.abs(grid.x_coords)))
    np.testing.assert_allclose(
        grid.q_asimov_2d[y_index, x_index],
        flexible_setup["local"].q_asimov_local,
        rtol=1.0e-6,
    )


def test_flexible_parallel_grid_matches_serial(flexible_setup):
    """Match two-worker and serial PowerLaw reference grid maps."""
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fisher"]["map"]["num_workers"] = 2
    parallel = _detector(config, flexible_setup["products"]).compute_grid_map()
    np.testing.assert_allclose(
        parallel.q_asimov_2d,
        flexible_setup["grid"].q_asimov_2d,
        rtol=1.0e-10,
    )


def test_lens_mismatch_has_structured_spurious_statistics(lens_mismatch_setup):
    """Produce finite positive non-constant spurious macro-mismatch fields."""
    q_spurious = lens_mismatch_setup["grid"].q_spurious_2d
    assert np.all(np.isfinite(q_spurious))
    assert np.max(q_spurious) > 1.0e-12
    assert np.ptp(q_spurious) > 1.0e-12


def test_lens_mismatch_grid_node_matches_local_fields(lens_mismatch_setup):
    """Match local and grid mismatch statistics at the injected position."""
    grid = lens_mismatch_setup["grid"]
    local = lens_mismatch_setup["local"]
    y_index = int(np.argmin(np.abs(grid.y_coords - 0.1)))
    x_index = int(np.argmin(np.abs(grid.x_coords)))
    for local_name, grid_name in (
        ("q_asimov_local", "q_asimov_2d"),
        ("z_mismatch", "z_mismatch_2d"),
        ("q_mismatch", "q_mismatch_2d"),
        ("amplitude_spurious", "amplitude_spurious_2d"),
        ("q_spurious", "q_spurious_2d"),
    ):
        np.testing.assert_allclose(
            getattr(local, local_name),
            getattr(grid, grid_name)[y_index, x_index],
            rtol=1.0e-8,
            atol=1.0e-10,
        )


def test_lens_mismatch_parallel_grid_matches_serial(lens_mismatch_setup):
    """Match spawn-pool and serial dual-macro mismatch fields."""
    config = deepcopy(lens_mismatch_setup["config"])
    config["modeling"]["fisher"]["map"]["num_workers"] = 2
    parallel = _detector(
        config, lens_mismatch_setup["products"]
    ).compute_grid_map()
    serial = lens_mismatch_setup["grid"]
    for name in (
        "q_asimov_2d",
        "z_mismatch_2d",
        "q_mismatch_2d",
        "amplitude_spurious_2d",
        "q_spurious_2d",
    ):
        np.testing.assert_allclose(
            getattr(parallel, name),
            getattr(serial, name),
            rtol=1.0e-10,
            atol=1.0e-12,
        )


def test_lens_mismatch_npz_roundtrip(lens_mismatch_setup, tmp_path):
    """Round-trip every generic Item 2 mismatch field for fit_lens."""
    path = save_fisher_grid_map_npz(
        lens_mismatch_setup["grid"], tmp_path / "lens-mismatch.npz"
    )
    loaded = load_fisher_grid_map_npz(path)
    for name in (
        "amplitude_hat_2d",
        "q_mismatch_2d",
        "z_mismatch_2d",
        "amplitude_spurious_2d",
        "q_spurious_2d",
        "z_spurious_2d",
        "mismatch_detectable_mask_2d",
        "false_positive_mask_2d",
    ):
        np.testing.assert_array_equal(
            getattr(loaded, name), getattr(lens_mismatch_setup["grid"], name)
        )
    original = lens_mismatch_setup["grid"]
    for name in (
        "mismatch_detectable_area_arcsec2",
        "false_positive_area_arcsec2",
        "max_z_spurious",
    ):
        assert getattr(loaded, name) == pytest.approx(getattr(original, name))
    for name in (
        "mismatch_enabled",
        "num_mismatch_detectable",
        "num_false_positive",
    ):
        assert getattr(loaded, name) == getattr(original, name)


def test_lens_mismatch_legacy_ring_map_raises(flexible_setup):
    """Reject fit_lens mismatch for the legacy ring-bank map."""
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }
    config["modeling"]["fisher"]["map"] = {
        "type": "ring",
        "ring": {"num_angles": 4, "offset_pixels": 0.0},
    }
    detector = _detector(config, flexible_setup["products"])
    with pytest.raises(ValueError, match="fit_psf and fit_lens"):
        detector.compute_map()


def test_lens_mismatch_legacy_explicit_bank_map_raises(flexible_setup):
    """Reject fit_lens mismatch for the legacy explicit-position bank."""
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }
    config["modeling"]["fisher"]["map"] = {
        "type": "explicit",
        "explicit_positions_yx": [[0.0, 0.0]],
    }
    detector = _detector(config, flexible_setup["products"])

    with pytest.raises(ValueError, match="fit_psf and fit_lens"):
        detector.compute_map()


def test_lens_mismatch_truth_residual_uses_model_baseline(lens_mismatch_setup):
    """Subtract mu0_model, not mu0_truth, in truth-side node residuals."""
    detector = lens_mismatch_setup["detector"]
    position = detector._grid_layout().positions_yx[0]
    pair = next(detector._grid_signal_iterator([position], num_workers=1))
    truth_kernel = detector._ensure_odd_kernel(detector.psf_data.kernel)
    truth_image = _direct_adu_image(
        detector.full_config,
        detector.lensing_baseline,
        truth_kernel,
        subhalo_position_yx=position,
        subhalo_einstein_radius=(
            lens_mismatch_setup["products"]["test"].subhalo_einstein_radius
        ),
    )
    expected = flatten_masked_image(
        truth_image - detector.mu0_model_adu_2d,
        mask=detector.mask_2d,
    )
    wrong = flatten_masked_image(
        truth_image - detector.mu0_adu_2d,
        mask=detector.mask_2d,
    )
    np.testing.assert_allclose(pair[1], expected, rtol=1.0e-10, atol=1.0e-10)
    assert not np.allclose(pair[1], wrong)


def test_jax_matched_flexible_lens_matches_reference(flexible_setup):
    """Match JAX and reference q_F for the full flexible macro model."""
    pytest.importorskip("jax")
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fisher"]["map"]["engine"] = "jax"
    jax_grid = _detector(config, flexible_setup["products"]).compute_grid_map()
    np.testing.assert_allclose(
        jax_grid.q_asimov_2d,
        flexible_setup["grid"].q_asimov_2d,
        rtol=1.0e-6,
    )
    np.testing.assert_array_equal(
        jax_grid.detectable_mask_2d,
        flexible_setup["grid"].detectable_mask_2d,
    )


def test_jax_lens_mismatch_fields_match_reference(lens_mismatch_setup):
    """Match all signed and squared lens-mismatch fields in JAX."""
    pytest.importorskip("jax")
    config = deepcopy(lens_mismatch_setup["config"])
    config["modeling"]["fisher"]["map"]["engine"] = "jax"
    jax_grid = _detector(
        config, lens_mismatch_setup["products"]
    ).compute_grid_map()
    reference = lens_mismatch_setup["grid"]
    has_nonzero_field = False
    for name in (
        "q_asimov_2d",
        "z_mismatch_2d",
        "q_mismatch_2d",
        "amplitude_spurious_2d",
        "q_spurious_2d",
    ):
        actual = getattr(jax_grid, name)
        expected = getattr(reference, name)
        floor = (np.abs(expected) + np.abs(actual)) > 1.0e-12
        assert np.any(floor), f"{name} comparison subset is empty"
        has_nonzero_field = has_nonzero_field or bool(
            np.nanmax(np.abs(expected)) > 1.0e-12
        )
        np.testing.assert_allclose(
            actual[floor], expected[floor], rtol=1.0e-6, atol=1.0e-12
        )
    assert has_nonzero_field
    np.testing.assert_array_equal(
        jax_grid.detectable_mask_2d, reference.detectable_mask_2d
    )
    np.testing.assert_array_equal(
        jax_grid.mismatch_detectable_mask_2d,
        reference.mismatch_detectable_mask_2d,
    )
    np.testing.assert_array_equal(
        jax_grid.false_positive_mask_2d,
        reference.false_positive_mask_2d,
    )


def test_multipole_mismatch_spurious_q_scales_quadratically(flexible_setup):
    """Pin the physical m4 residual and its quadratic spurious scaling."""
    q_values = []
    for epsilon in (0.001, 0.002):
        config = deepcopy(flexible_setup["config"])
        mass = config["lensing"]["lens_galaxy"]["mass"]
        mass["slope"] = 2.0
        mass["multipoles"] = {"m4": [epsilon, 0.0]}
        config["lensing"]["lens_galaxy"].pop("shear", None)
        config["modeling"]["fit_lens"] = {
            "mode": "explicit",
            "lens_galaxy": _isothermal_fit(config),
        }
        products = _scene_products(
            config,
            psf_data=flexible_setup["products"]["psf"],
        )
        detector = _detector(config, products)
        q_values.append(detector.compute_grid_map().q_spurious_2d)
        if epsilon == 0.001:
            kernel = detector._ensure_odd_kernel(detector.psf_data.kernel)
            direct_truth = _direct_adu_image(
                detector.full_config,
                detector.lensing_baseline,
                kernel,
            )
            direct_fit = _direct_adu_image(
                detector.fit_full_config,
                detector.lensing_baseline,
                kernel,
            )
            independent_residual = direct_truth - direct_fit
            production_residual = (
                detector.mu0_adu_2d - detector.mu0_model_adu_2d
            )
            dominant = np.abs(independent_residual) > (
                np.max(np.abs(independent_residual)) * 1.0e-6
            )
            assert np.any(dominant)
            np.testing.assert_allclose(
                production_residual[dominant],
                independent_residual[dominant],
                rtol=1.0e-6,
                atol=1.0e-10,
            )
    floor = q_values[0] > max(np.max(q_values[0]) * 1.0e-6, 1.0e-12)
    assert np.any(floor)
    ratios = q_values[1][floor] / q_values[0][floor]
    np.testing.assert_allclose(ratios, 4.0, rtol=0.15, atol=0.0)


def test_combined_mismatch_equal_and_different_models(
    flexible_setup,
    lens_mismatch_setup,
):
    """Recover the limit and prove both combined mismatch channels act."""
    equal = deepcopy(flexible_setup["config"])
    equal["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _fit_lens_from_truth(equal),
    }
    equal["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": deepcopy(equal["psf"]),
    }
    equal_grid = _detector(equal, flexible_setup["products"]).compute_grid_map()
    np.testing.assert_allclose(
        equal_grid.q_asimov_2d,
        flexible_setup["grid"].q_asimov_2d,
        rtol=1.0e-10,
    )
    assert np.nanmax(np.abs(equal_grid.z_spurious_2d)) < 1.0e-4
    assert np.nanmax(equal_grid.q_spurious_2d) < 1.0e-8

    different = deepcopy(flexible_setup["config"])
    different["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(different),
    }
    fit_psf = deepcopy(different["psf"])
    fit_psf["aberrations"]["global_zernikes"][4] = 13.0
    different["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": fit_psf,
    }
    different_grid = _detector(
        different, flexible_setup["products"]
    ).compute_grid_map()
    assert np.all(np.isfinite(different_grid.q_mismatch_2d))
    assert np.all(np.isfinite(different_grid.q_spurious_2d))

    psf_only = deepcopy(flexible_setup["config"])
    psf_only["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": deepcopy(fit_psf),
    }
    psf_only_grid = _detector(
        psf_only,
        flexible_setup["products"],
    ).compute_grid_map()
    evaluated = different_grid.evaluated_mask_2d
    for name in ("z_mismatch_2d", "q_spurious_2d"):
        combined_values = getattr(different_grid, name)[evaluated]
        lens_values = getattr(lens_mismatch_setup["grid"], name)[evaluated]
        psf_values = getattr(psf_only_grid, name)[evaluated]
        assert not np.allclose(combined_values, lens_values)
        assert not np.allclose(combined_values, psf_values)


def test_combined_mismatch_fit_psf_derivative_uses_fit_lens(flexible_setup):
    """Differentiate the complete fit lens plus fit PSF model."""
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }
    config["modeling"]["fit_psf"] = {
        "mode": "explicit",
        "psf": deepcopy(config["psf"]),
    }
    fisher = config["modeling"]["fisher"]
    fisher["include_psf_nuisance"] = True
    fisher["fit_psf_mode_selection"] = {
        "global_zernikes": {"mode_nolls": [4]}
    }
    detector = _detector(config, flexible_setup["products"])
    spec = detector.fit_psf_mode_specs[0]
    plus = deepcopy(detector.fit_psf_config_template)
    minus = deepcopy(detector.fit_psf_config_template)
    detector._set_path_value_create(plus, spec.path, 10.0 + spec.step)
    detector._set_path_value_create(minus, spec.path, 10.0 - spec.step)
    psf_plus = detector._quiet_generate_psf_system(plus)
    psf_minus = detector._quiet_generate_psf_system(minus)
    assert detector.lensing_baseline_fit is not None
    complete_plus = _mean_adu_images_from_lensing_arrays(
        detector.lensing_baseline_fit,
        config["observation"],
        (detector._ensure_odd_kernel(psf_plus.kernel),),
    )[0]
    complete_minus = _mean_adu_images_from_lensing_arrays(
        detector.lensing_baseline_fit,
        config["observation"],
        (detector._ensure_odd_kernel(psf_minus.kernel),),
    )[0]
    independent = (complete_plus - complete_minus) / (2.0 * spec.step)
    np.testing.assert_allclose(
        detector.fit_psf_mode_images[0], independent, rtol=1.0e-8, atol=1.0e-8
    )
    truth_plus = _mean_adu_images_from_lensing_arrays(
        detector.lensing_baseline,
        config["observation"],
        (detector._ensure_odd_kernel(psf_plus.kernel),),
    )[0]
    truth_minus = _mean_adu_images_from_lensing_arrays(
        detector.lensing_baseline,
        config["observation"],
        (detector._ensure_odd_kernel(psf_minus.kernel),),
    )[0]
    truth_construction = (truth_plus - truth_minus) / (2.0 * spec.step)
    assert not np.allclose(detector.fit_psf_mode_images[0], truth_construction)


def test_truth_centre_controls_reference_and_jax_lattices(flexible_setup):
    """Centre both engine lattices on truth when the fit centre differs."""
    config = deepcopy(flexible_setup["config"])
    config["lensing"]["lens_galaxy"]["mass"]["centre"] = [0.07, -0.04]
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }
    config["modeling"]["fit_lens"]["lens_galaxy"]["mass"]["centre"] = [0.0, 0.0]
    products = _scene_products(config, psf_data=flexible_setup["products"]["psf"])
    for engine in ("reference", "jax"):
        engine_config = deepcopy(config)
        engine_config["modeling"]["fisher"]["map"]["engine"] = engine
        grid = _detector(engine_config, products).compute_grid_map()
        assert grid.centre_yx == pytest.approx((0.07, -0.04))
        assert grid.y_coords[1] == pytest.approx(0.07)
        assert grid.x_coords[1] == pytest.approx(-0.04)


def test_jax_combined_mismatch_fields_match_reference(flexible_setup):
    """Match all required fields for combined PSF and lens mismatch."""
    config = deepcopy(flexible_setup["config"])
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }
    fit_psf = deepcopy(config["psf"])
    fit_psf["aberrations"]["global_zernikes"][4] = 13.0
    config["modeling"]["fit_psf"] = {"mode": "explicit", "psf": fit_psf}
    reference = _detector(config, flexible_setup["products"]).compute_grid_map()
    config["modeling"]["fisher"]["map"]["engine"] = "jax"
    actual = _detector(config, flexible_setup["products"]).compute_grid_map()
    has_nonzero_field = False
    for name in (
        "q_asimov_2d",
        "z_mismatch_2d",
        "q_mismatch_2d",
        "amplitude_spurious_2d",
        "q_spurious_2d",
    ):
        actual_values = getattr(actual, name)
        expected_values = getattr(reference, name)
        floor = (np.abs(expected_values) + np.abs(actual_values)) > 1.0e-12
        assert np.any(floor), f"{name} comparison subset is empty"
        has_nonzero_field = has_nonzero_field or bool(
            np.nanmax(np.abs(expected_values)) > 1.0e-12
        )
        np.testing.assert_allclose(
            actual_values[floor],
            expected_values[floor],
            rtol=1.0e-6,
            atol=1.0e-12,
        )
    assert has_nonzero_field
    for name in (
        "detectable_mask_2d",
        "mismatch_detectable_mask_2d",
        "false_positive_mask_2d",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))


def _jax_engine_inputs(detector: FisherDetector) -> dict:
    """Return common direct JAX-engine construction arguments."""
    kernel = detector._ensure_odd_kernel(detector.model_psf_data.kernel)
    truth_kernel = detector._ensure_odd_kernel(detector.psf_data.kernel)
    return {
        "lensing_baseline": detector.lensing_baseline,
        "map_config_template": deepcopy(detector.map_config_template),
        "psf_kernel_native": np.asarray(pyauto_kernel_native(kernel), dtype=float),
        "truth_psf_kernel_native": np.asarray(
            pyauto_kernel_native(truth_kernel), dtype=float
        ),
        "mu0_adu_2d": detector.mu0_model_adu_2d,
        "mask_2d": detector.mask_2d,
        "candidate_positions": detector._grid_layout().positions_yx,
        "truth_lens_centre_yx": detector._grid_layout().centre_yx,
    }


def test_jax_build_rejects_different_fit_source(lens_mismatch_setup):
    """Reject a fit baseline whose source is not identical to truth."""
    detector = lens_mismatch_setup["detector"]
    fit_config = deepcopy(detector.baseline_config_template)
    fit_config["lensing"]["source_galaxy"]["light"]["centre"][0] += 0.01
    fit_baseline = generate_lensing_system(
        fit_config["lensing"], full_config=fit_config
    )
    with pytest.raises(ValueError, match="identical source profiles"):
        JaxGridTemplateEngine(
            lensing_baseline_fit=fit_baseline,
            **_jax_engine_inputs(detector),
        )


def test_jax_build_rejects_different_fit_source_redshift(
    lens_mismatch_setup,
):
    """Reject a fit tracer differing only in source-plane redshift."""
    detector = lens_mismatch_setup["detector"]
    fit_config = deepcopy(detector.baseline_config_template)
    fit_config["lensing"]["source_galaxy"]["redshift"] = 0.7
    fit_baseline = generate_lensing_system(
        fit_config["lensing"],
        full_config=fit_config,
    )

    with pytest.raises(ValueError) as error:
        JaxGridTemplateEngine(
            lensing_baseline_fit=fit_baseline,
            **_jax_engine_inputs(detector),
        )

    message = str(error.value)
    assert "plane redshifts" in message
    assert "truth=(0.2, 0.6)" in message
    assert "fit=(0.2, 0.7)" in message


@pytest.mark.parametrize("geometry", ["shape", "pixel_scale", "over_sampling"])
def test_jax_build_rejects_fit_grid_geometry(lens_mismatch_setup, geometry):
    """Reject native-shape, pixel-scale, and over-sampling disagreements."""
    detector = lens_mismatch_setup["detector"]
    fit_baseline = detector.lensing_baseline_fit
    assert fit_baseline is not None
    if geometry == "shape":
        fit_baseline = replace(
            fit_baseline, image=fit_baseline.image[:-1, :]
        )
        message = "native grid shape"
    elif geometry == "pixel_scale":
        fit_baseline = replace(
            fit_baseline, pixel_scale=fit_baseline.pixel_scale + 0.01
        )
        message = "pixel scale"
    else:
        grid = al.Grid2D.uniform(
            shape_native=fit_baseline.image.shape,
            pixel_scales=fit_baseline.pixel_scale,
            over_sample_size=2,
        )
        fit_baseline = replace(fit_baseline, grid=grid)
        message = "over-sampling"
    with pytest.raises(ValueError, match=message):
        JaxGridTemplateEngine(
            lensing_baseline_fit=fit_baseline,
            **_jax_engine_inputs(detector),
        )


def _detection_data(config: dict, detector: FisherDetector, grid) -> FisherDetectionData:
    """Wrap one detector grid for summary and plotting smoke tests."""
    fisher = config["modeling"]["fisher"]
    return FisherDetectionData(
        mode="map",
        local=None,
        map=None,
        grid_map=grid,
        snr_threshold=fisher["snr_threshold"],
        include_background_offset=fisher["include_background_offset"],
        finite_diff=deepcopy(fisher["finite_diff"]),
        map_config=deepcopy(fisher["map"]),
        pixels_unmasked=detector.pixels_unmasked,
        n_nuisance=detector.n_nuisance,
        gram_condition_number=detector.gram_condition_number,
        pixel_scale=detector.lensing_baseline.pixel_scale,
        config=config,
        psf_mismatch_enabled=detector.psf_mismatch_enabled,
        lens_mismatch_enabled=detector.lens_mismatch_enabled,
    )


def test_power_law_mismatch_plotting_writes_both_maps(
    lens_mismatch_setup,
    tmp_path,
    monkeypatch,
):
    """Write both maps with a model-generalized spurious title."""
    config = deepcopy(lens_mismatch_setup["config"])
    config["run_name"] = "item5-plot-smoke"
    data = _detection_data(
        config,
        lens_mismatch_setup["detector"],
        lens_mismatch_setup["grid"],
    )
    titles = []
    original_set_title = matplotlib.axes.Axes.set_title

    def capture_title(axis, label, *args, **kwargs):
        titles.append(label)
        return original_set_title(axis, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", capture_title)
    plot_fisher_detection_grid_map(
        data,
        {"output_dir": str(tmp_path)},
        run_name=config["run_name"],
    )
    output = tmp_path / config["run_name"] / "modeling"
    assert (output / "fisher_grid_map.png").is_file()
    assert (output / "fisher_grid_map_spurious.png").is_file()
    assert any("Model-Mismatch" in title for title in titles)
    assert all("PSF-Mismatch" not in title for title in titles)


def test_plotting_enabled_pipeline_writes_both_grid_maps(tmp_path):
    """Route a flexible mismatch grid run through the plotting registry."""
    config = _runtime_config(tmp_path)
    config["run_name"] = "item5-pipeline-plot-smoke"
    config["plotting"] = {"enabled": True, "output_dir": str(tmp_path)}
    config["modeling"]["fisher"]["mode"] = "map"
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }

    result = Pipeline(verbose=False).run(config)

    assert result.has_grid_map
    output = tmp_path / config["run_name"] / "modeling"
    for filename in (
        "fisher_grid_map.png",
        "fisher_grid_map_spurious.png",
    ):
        path = output / filename
        assert path.is_file()
        assert path.stat().st_size > 0


def test_fisher_summary_names_active_mismatch_blocks(
    lens_mismatch_setup, capsys
):
    """Name fit_lens without claiming every mismatch is PSF-specific."""
    data = _detection_data(
        lens_mismatch_setup["config"],
        lens_mismatch_setup["detector"],
        lens_mismatch_setup["grid"],
    )
    print_fisher_summary(data)
    output = capsys.readouterr().out
    assert "Model mismatch" in output
    assert "fit_lens mode: explicit" in output
    assert "~2x per-node ray-tracing work" in output
    assert "fit_psf mode: explicit" not in output


def test_nonlinear_builder_supports_power_law_truth(flexible_setup):
    """Build PowerLaw truth after the Item 7 guard removal."""
    spec = smooth_model_spec_from_config(flexible_setup["config"])
    assert spec.galaxies["lens"].components["mass"].class_name == "PowerLaw"


def test_nonlinear_builder_supports_explicit_fit_lens(flexible_setup):
    """Build an explicit fit-side SIE macro after Item 7."""
    config = deepcopy(flexible_setup["config"])
    config["lensing"]["lens_galaxy"]["mass"] = _isothermal_fit(config)["mass"]
    config["lensing"]["lens_galaxy"].pop("shear", None)
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": _isothermal_fit(config),
    }
    spec = smooth_model_spec_from_config(config)
    assert spec.galaxies["lens"].components["mass"].class_name == "Isothermal"


def test_nonlinear_builder_power_law_contains_linked_multipoles(flexible_setup):
    """Build fit-side multipoles linked to the PowerLaw macro."""
    spec = smooth_model_spec_from_config(flexible_setup["config"])
    lens = spec.galaxies["lens"].components
    assert lens["multipole_m3"].parameters["slope"].kind == "linked"


def test_nonlinear_builder_explicit_fit_lens_has_no_truth_shear(flexible_setup):
    """Omit truth-only shear from an explicit SIE fit model."""
    config = deepcopy(flexible_setup["config"])
    isothermal = _isothermal_fit(config)
    config["lensing"]["lens_galaxy"]["mass"] = deepcopy(
        isothermal["mass"]
    )
    config["lensing"]["lens_galaxy"].pop("shear", None)
    config["modeling"]["fit_lens"] = {
        "mode": "explicit",
        "lens_galaxy": isothermal,
    }

    spec = smooth_model_spec_from_config(config)
    assert set(spec.galaxies["lens"].components) == {"mass"}
