from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("autolens")
import autolens as al

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.lensing.utils import LensingData
from hwoslaps.observation.utils import ObservationData
from hwoslaps.modeling.fisher_detector import FisherDetector


def _make_imaging(data_adu: np.ndarray, noise_map_adu: np.ndarray, pixel_scale: float) -> al.Imaging:
    mask = al.Mask2D.all_false(shape_native=data_adu.shape, pixel_scales=pixel_scale)
    data = al.Array2D(values=data_adu, mask=mask)
    noise = al.Array2D(values=noise_map_adu, mask=mask)
    kernel = al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=pixel_scale, normalize=False)
    return al.Imaging(data=data, noise_map=noise, psf=kernel)


def _build_observation(
    noiseless_source_eps: np.ndarray,
    exposure_time: float,
    detector: dict,
    pixel_scale: float,
    run_name: str,
) -> ObservationData:
    gain = detector["gain"]
    read_noise = detector["read_noise"]
    dark_current = detector["dark_current"]
    sky_background = detector["sky_background"]

    source_e = noiseless_source_eps * exposure_time
    sky_e = sky_background * exposure_time
    dark_e = dark_current * exposure_time
    expected_e = source_e + sky_e + dark_e
    data_adu = expected_e / gain
    noise_map_adu = np.sqrt(expected_e + read_noise ** 2) / gain

    imaging = _make_imaging(data_adu=data_adu, noise_map_adu=noise_map_adu, pixel_scale=pixel_scale)
    metadata = {
        "generated": "unit-test",
        "exposure_time": exposure_time,
        "detector": detector.copy(),
        "noise_seed": None,
        "pixel_scale": pixel_scale,
        "field_of_view": (
            noiseless_source_eps.shape[0] * pixel_scale,
            noiseless_source_eps.shape[1] * pixel_scale,
        ),
        "run_name": run_name,
    }
    return ObservationData(
        imaging=imaging,
        noiseless_source_eps=noiseless_source_eps,
        noise_components={},
        config={"exposure_time": exposure_time, "detector": detector.copy()},
        metadata=metadata,
    )


def _make_lensing(shape=(6, 6), pixel_scale=0.1, run_name: str = "unit") -> LensingData:
    image = np.zeros(shape, dtype=float)
    grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale, over_sample_size=1)
    tracer = al.Tracer(galaxies=[])
    return LensingData(
        image=image,
        grid=grid,
        tracer=tracer,
        pixel_scale=pixel_scale,
        lens_redshift=0.2,
        source_redshift=2.0,
        lens_einstein_radius=1.3,
        cosmology_name="Planck15",
        config={"run_name": run_name},
    )


def _make_full_config() -> dict:
    return {
        "run_name": "unit",
        "global_seed": 1,
        "lensing": {
            "grid": {"shape": [6, 6], "pixel_scale": 0.1},
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "einstein_radius": 1.3,
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.05, -0.02],
                },
            },
            "source_galaxy": {
                "redshift": 2.0,
                "light": {
                    "type": "Exponential",
                    "centre": [0.01, -0.02],
                    "ell_comps": [0.03, -0.01],
                    "intensity": 1.8,
                    "effective_radius": 0.14,
                },
            },
            "subhalo": {
                "enabled": True,
                "model": "PointMass",
                "mass": 1.0e8,
                "position": {"type": "direct", "centre": [0.2, -0.1]},
            },
            "cosmology": "Planck15",
        },
        "observation": {
            "exposure_time": 1000.0,
            "detector": {
                "gain": 1.0,
                "read_noise": 3.0,
                "dark_current": 0.01,
                "sky_background": 0.2,
            },
        },
    }


def _make_fisher_config(
    mode: str = "both",
    snr_threshold: float = 0.1,
    include_background_offset: bool = True,
    num_angles: int = 8,
    explicit_positions_yx=None,
) -> dict:
    return {
        "mode": mode,
        "snr_threshold": snr_threshold,
        "include_background_offset": include_background_offset,
        "finite_diff": {
            "centre_arcsec": 1.0e-3,
            "einstein_radius_arcsec": 1.0e-3,
            "ell_comp": 1.0e-3,
            "source_intensity_frac": 1.0e-2,
            "source_reff_frac": 1.0e-2,
        },
        "map": {
            "num_angles": num_angles,
            "offset_pixels": 0.0,
            "explicit_positions_yx": explicit_positions_yx,
        },
    }


def _full_rank_jacobian(pixels_unmasked: int, n_nuisance: int) -> np.ndarray:
    jacobian = np.zeros((pixels_unmasked, n_nuisance), dtype=float)
    for idx in range(n_nuisance):
        jacobian[idx % pixels_unmasked, idx] = 1.0
    if pixels_unmasked > n_nuisance:
        jacobian[n_nuisance:, :] = 0.05
    return jacobian


def _build_detector(
    monkeypatch,
    baseline_obs: ObservationData,
    lensing_baseline: LensingData,
    fisher_config: dict,
    patch_map_mean: bool = False,
) -> FisherDetector:
    monkeypatch.setattr(
        FisherDetector,
        "_build_nuisance_jacobian",
        lambda self: _full_rank_jacobian(self.pixels_unmasked, self.n_nuisance),
    )

    if patch_map_mean:
        def _fake_mean_for_position(self, position_yx):
            amplitude = 1.0e-3 * (position_yx[0] + 2.0 * position_yx[1])
            return self.mu0_adu_2d + amplitude

        monkeypatch.setattr(FisherDetector, "_mean_adu_for_position", _fake_mean_for_position)

    psf_data = SimpleNamespace(
        kernel=al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=baseline_obs.pixel_scale, normalize=False)
    )

    return FisherDetector(
        observation_baseline=baseline_obs,
        lensing_baseline=lensing_baseline,
        psf_data=psf_data,
        full_config=_make_full_config(),
        fisher_config=fisher_config,
    )


def test_local_zero_signal_gives_zero_or_near_zero_snr(monkeypatch):
    shape = (6, 6)
    source = np.full(shape, 80.0, dtype=float)
    detector_cfg = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}

    obs0 = _build_observation(source, 1000.0, detector_cfg, pixel_scale=0.1, run_name="h0")
    obs1 = _build_observation(source.copy(), 1000.0, detector_cfg, pixel_scale=0.1, run_name="h1")

    lens0 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens0")
    lens1 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens1")
    lens1.subhalo_position = (0.1, -0.2)
    lens1.subhalo_mass = 1.0e8
    lens1.subhalo_model = "PointMass"

    fisher_cfg = _make_fisher_config(mode="local", include_background_offset=True, snr_threshold=0.1)
    detector = _build_detector(monkeypatch, obs0, lens0, fisher_cfg)
    local = detector.compute_local(observation_test=obs1, lensing_test=lens1)

    assert local.snr_asimov <= 1.0e-8
    assert local.delta_chi2_raw <= 1.0e-8
    assert local.delta_chi2_profiled <= 1.0e-8


def test_profiled_delta_chi2_not_greater_than_raw(monkeypatch):
    shape = (6, 6)
    source0 = np.full(shape, 90.0, dtype=float)
    source1 = source0.copy()
    source1[2:4, 2:4] += 2.0
    detector_cfg = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}

    obs0 = _build_observation(source0, 800.0, detector_cfg, pixel_scale=0.1, run_name="h0")
    obs1 = _build_observation(source1, 800.0, detector_cfg, pixel_scale=0.1, run_name="h1")

    lens0 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens0")
    lens1 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens1")
    lens1.subhalo_position = (0.0, 0.0)
    lens1.subhalo_mass = 1.0e8
    lens1.subhalo_model = "PointMass"

    fisher_cfg = _make_fisher_config(mode="local", include_background_offset=True, snr_threshold=0.1)
    detector = _build_detector(monkeypatch, obs0, lens0, fisher_cfg)
    local = detector.compute_local(observation_test=obs1, lensing_test=lens1)

    assert local.delta_chi2_profiled >= 0.0
    assert local.delta_chi2_raw >= 0.0
    assert local.delta_chi2_profiled <= local.delta_chi2_raw + 1.0e-8


def test_mask_failfast_when_too_few_pixels():
    shape = (3, 4)
    source = np.full(shape, 50.0, dtype=float)
    detector_cfg = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}
    obs0 = _build_observation(source, 500.0, detector_cfg, pixel_scale=0.1, run_name="h0")
    lens0 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens0")
    fisher_cfg = _make_fisher_config(mode="local", include_background_offset=False, snr_threshold=0.1)

    psf_data = SimpleNamespace(
        kernel=al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=obs0.pixel_scale, normalize=False)
    )
    with pytest.raises(ValueError, match="unmasked pixels must exceed nuisance directions \\+ 2"):
        FisherDetector(
            observation_baseline=obs0,
            lensing_baseline=lens0,
            psf_data=psf_data,
            full_config=_make_full_config(),
            fisher_config=fisher_cfg,
        )


def test_gram_failfast_on_ill_conditioning(monkeypatch):
    shape = (6, 6)
    source = np.full(shape, 70.0, dtype=float)
    detector_cfg = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}
    obs0 = _build_observation(source, 600.0, detector_cfg, pixel_scale=0.1, run_name="h0")
    lens0 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens0")
    fisher_cfg = _make_fisher_config(mode="local", include_background_offset=False, snr_threshold=0.1)

    monkeypatch.setattr(
        FisherDetector,
        "_build_nuisance_jacobian",
        lambda self: np.ones((self.pixels_unmasked, self.n_nuisance), dtype=float),
    )

    psf_data = SimpleNamespace(
        kernel=al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=obs0.pixel_scale, normalize=False)
    )
    with pytest.raises(ValueError, match="Ill-conditioned nuisance Gram matrix"):
        FisherDetector(
            observation_baseline=obs0,
            lensing_baseline=lens0,
            psf_data=psf_data,
            full_config=_make_full_config(),
            fisher_config=fisher_cfg,
        )


def test_map_returns_expected_number_of_positions_default_angles(monkeypatch):
    shape = (6, 6)
    source = np.full(shape, 75.0, dtype=float)
    detector_cfg = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}
    obs0 = _build_observation(source, 700.0, detector_cfg, pixel_scale=0.1, run_name="h0")
    lens0 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens0")
    fisher_cfg = _make_fisher_config(mode="map", include_background_offset=True, snr_threshold=0.1, num_angles=10)

    detector = _build_detector(monkeypatch, obs0, lens0, fisher_cfg, patch_map_mean=True)
    fmap = detector.compute_map()

    assert fmap.num_positions == 10
    assert fmap.positions_yx.shape[0] == 10
    assert fmap.snr_asimov_by_position.shape[0] == 10


def test_map_explicit_positions_override_angles(monkeypatch):
    shape = (6, 6)
    source = np.full(shape, 75.0, dtype=float)
    detector_cfg = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}
    obs0 = _build_observation(source, 700.0, detector_cfg, pixel_scale=0.1, run_name="h0")
    lens0 = _make_lensing(shape=shape, pixel_scale=0.1, run_name="lens0")
    explicit_positions = [[0.1, 0.2], [-0.2, 0.3], [0.0, -0.4]]
    fisher_cfg = _make_fisher_config(
        mode="map",
        include_background_offset=True,
        snr_threshold=0.1,
        num_angles=24,
        explicit_positions_yx=explicit_positions,
    )

    detector = _build_detector(monkeypatch, obs0, lens0, fisher_cfg, patch_map_mean=True)
    fmap = detector.compute_map()

    assert fmap.num_positions == len(explicit_positions)
    np.testing.assert_allclose(fmap.positions_yx, np.array(explicit_positions, dtype=float))
