import numpy as np
import autolens as al

from hwoslaps.observation.utils import ObservationData
from hwoslaps.modeling.mejiro_detector import MejiroDetector, MejiroConfig


def _make_imaging(data_adu: np.ndarray, noise_map_adu: np.ndarray, pixel_scale: float) -> al.Imaging:
    mask = al.Mask2D.all_false(shape_native=data_adu.shape, pixel_scales=pixel_scale)
    data = al.Array2D(values=data_adu, mask=mask)
    noise = al.Array2D(values=noise_map_adu, mask=mask)
    kernel = al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=pixel_scale, normalize=False)
    return al.Imaging(data=data, noise_map=noise, psf=kernel)


def _build_observation_from_expected(E_adu: np.ndarray, S_eps: np.ndarray, exposure_time: float, detector: dict, pixel_scale: float, run_name: str) -> ObservationData:
    gain = detector["gain"]
    read_noise = detector["read_noise"]
    # Build noise map consistent with baseline source counts
    source_e = S_eps * exposure_time
    var_e = source_e + detector["dark_current"] * exposure_time + detector["sky_background"] * exposure_time + read_noise ** 2
    noise_map_adu = np.sqrt(var_e) / gain
    imaging = _make_imaging(E_adu, noise_map_adu, pixel_scale)
    metadata = {
        "generated": "unit-test",
        "exposure_time": exposure_time,
        "detector": detector.copy(),
        "noise_seed": None,
        "pixel_scale": pixel_scale,
        "field_of_view": (E_adu.shape[0] * pixel_scale, E_adu.shape[1] * pixel_scale),
        "run_name": run_name,
    }
    return ObservationData(
        imaging=imaging,
        noiseless_source_eps=S_eps,
        noise_components={},
        config={"exposure_time": exposure_time, "detector": detector.copy()},
        metadata=metadata,
    )


def test_identity_A_equals_B_yields_zero_chi2():
    shape = (16, 16)
    pixel_scale = 0.1
    exposure_time = 1000.0
    detector = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}

    S_eps = np.full(shape, 50.0)
    sky_e = detector["sky_background"] * exposure_time
    dark_e = detector["dark_current"] * exposure_time
    E0_adu = (S_eps * exposure_time + sky_e + dark_e) / detector["gain"]

    obs0 = _build_observation_from_expected(E0_adu, S_eps, exposure_time, detector, pixel_scale, run_name="H0")
    obsA = _build_observation_from_expected(E0_adu.copy(), S_eps, exposure_time, detector, pixel_scale, run_name="A")

    det = MejiroDetector(baseline=obs0, config=MejiroConfig(snr_threshold=1.0))
    chi2_val, dof, _ = det.compute_chi2(observation_with_subhalo=obsA)

    assert chi2_val == 0.0
    assert dof == max(1, int(np.sum(det.snr_mask)) - 3)


def test_single_pixel_sanity():
    shape = (8, 8)
    pixel_scale = 0.1
    exposure_time = 100.0
    detector = {"gain": 1.0, "read_noise": 1e-6, "dark_current": 0.0, "sky_background": 0.0}

    S_eps = np.zeros(shape)
    S_eps[4, 4] = 10.0
    E0_adu = (S_eps * exposure_time) / detector["gain"]
    obs0 = _build_observation_from_expected(E0_adu, S_eps, exposure_time, detector, pixel_scale, run_name="H0")

    A_adu = E0_adu.copy()
    A_adu[4, 4] += 5.0
    obsA = _build_observation_from_expected(A_adu, S_eps, exposure_time, detector, pixel_scale, run_name="A")

    det = MejiroDetector(baseline=obs0, config=MejiroConfig(snr_threshold=0.1, eps_counts_floor=1e-12))
    chi2_val, dof, _ = det.compute_chi2(observation_with_subhalo=obsA)

    mask = det.snr_mask.reshape(shape)
    assert mask[4, 4]
    # chi2 contribution = (A-B)^2 / B at the single pixel
    expected = (5.0 ** 2) / max(E0_adu[4, 4], 1e-12)
    assert np.isclose(chi2_val, expected, rtol=1e-10, atol=1e-12)


def test_mask_gating_effect():
    shape = (16, 16)
    pixel_scale = 0.1
    exposure_time = 1000.0
    detector = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}

    S_eps = np.full(shape, 5.0)
    sky_e = detector["sky_background"] * exposure_time
    dark_e = detector["dark_current"] * exposure_time
    E0_adu = (S_eps * exposure_time + sky_e + dark_e) / detector["gain"]

    # A has a uniform bump
    A_adu = E0_adu + 0.1

    obs0 = _build_observation_from_expected(E0_adu, S_eps, exposure_time, detector, pixel_scale, run_name="H0")
    obsA = _build_observation_from_expected(A_adu, S_eps, exposure_time, detector, pixel_scale, run_name="A")

    det_lo = MejiroDetector(baseline=obs0, config=MejiroConfig(snr_threshold=0.5))
    chi2_lo, _, _ = det_lo.compute_chi2(observation_with_subhalo=obsA)

    det_hi = MejiroDetector(baseline=obs0, config=MejiroConfig(snr_threshold=2.0))
    chi2_hi, _, _ = det_hi.compute_chi2(observation_with_subhalo=obsA)

    assert det_hi.pixels_unmasked <= det_lo.pixels_unmasked
    assert chi2_hi <= chi2_lo


def test_monotonic_bump_magnitude():
    shape = (16, 16)
    pixel_scale = 0.1
    exposure_time = 1000.0
    detector = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.01, "sky_background": 0.2}

    S_eps = np.full(shape, 10.0)
    sky_e = detector["sky_background"] * exposure_time
    dark_e = detector["dark_current"] * exposure_time
    E0_adu = (S_eps * exposure_time + sky_e + dark_e) / detector["gain"]

    obs0 = _build_observation_from_expected(E0_adu, S_eps, exposure_time, detector, pixel_scale, run_name="H0")

    bumps = [0.0, 0.05, 0.1, 0.2]
    chi2_vals = []
    for b in bumps:
        A_adu = E0_adu + b
        obsA = _build_observation_from_expected(A_adu, S_eps, exposure_time, detector, pixel_scale, run_name=f"A_{b}")
        det = MejiroDetector(baseline=obs0, config=MejiroConfig(snr_threshold=1.0))
        chi2_val, _, _ = det.compute_chi2(observation_with_subhalo=obsA)
        chi2_vals.append(chi2_val)

    chi2_vals = np.array(chi2_vals)
    assert np.all(np.diff(chi2_vals) >= -1e-10)


