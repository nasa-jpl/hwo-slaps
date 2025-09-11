import numpy as np
import autolens as al
from scipy.stats import chi2 as chi2_dist

from hwoslaps.lensing.utils import LensingData
from hwoslaps.observation.utils import ObservationData
from hwoslaps.modeling.chernoff_detector import ChernoffSubhaloDetector


def _make_imaging(data_adu: np.ndarray, noise_map_adu: np.ndarray, pixel_scale: float) -> al.Imaging:
    """Construct a minimal autolens Imaging from raw ADU arrays."""
    mask = al.Mask2D.all_false(shape_native=data_adu.shape, pixel_scales=pixel_scale)
    data = al.Array2D(values=data_adu, mask=mask)
    noise = al.Array2D(values=noise_map_adu, mask=mask)
    kernel = al.Kernel2D.no_mask(values=np.array([[1.0]]), pixel_scales=pixel_scale, normalize=False)
    return al.Imaging(data=data, noise_map=noise, psf=kernel)


def _make_lensing_data(shape=(32, 32), pixel_scale=0.1, run_name: str = "unit") -> LensingData:
    image = np.zeros(shape, dtype=float)
    grid = al.Grid2D.uniform(shape_native=shape, pixel_scales=pixel_scale, over_sample_size=1)
    tracer = al.Tracer(galaxies=[])
    return LensingData(
        image=image,
        grid=grid,
        tracer=tracer,
        pixel_scale=pixel_scale,
        lens_redshift=0.5,
        source_redshift=1.0,
        lens_einstein_radius=1.5,
        cosmology_name="Planck15",
        config={"run_name": run_name},
    )


def _build_observation(noiseless_source_eps: np.ndarray,
                       exposure_time: float,
                       detector: dict,
                       pixel_scale: float,
                       run_name: str,
                       data_mode: str = "E0") -> ObservationData:
    """Create ObservationData with exact control of E0 and noise map.

    data_mode:
      - "E0": imaging.data is set to the H0 expectation E0 (no noise)
      - "custom": caller will overwrite imaging.data later
    """
    gain = detector["gain"]
    read_noise = detector["read_noise"]
    dark_current = detector["dark_current"]
    sky_background = detector["sky_background"]

    source_e = noiseless_source_eps * exposure_time
    dark_e = dark_current * exposure_time
    sky_e = sky_background * exposure_time

    expected_e = source_e + dark_e + sky_e
    E0_adu = expected_e / gain
    noise_map_adu = np.sqrt(expected_e + read_noise ** 2) / gain

    imaging = _make_imaging(
        data_adu=E0_adu if data_mode == "E0" else np.zeros_like(E0_adu),
        noise_map_adu=noise_map_adu,
        pixel_scale=pixel_scale,
    )

    metadata = {
        "generated": "unit-test",
        "exposure_time": exposure_time,
        "detector": detector.copy(),
        "noise_seed": None,
        "pixel_scale": pixel_scale,
        "field_of_view": (noiseless_source_eps.shape[0] * pixel_scale,
                           noiseless_source_eps.shape[1] * pixel_scale),
        "run_name": run_name,
    }

    return ObservationData(
        imaging=imaging,
        noiseless_source_eps=noiseless_source_eps,
        noise_components={"expected_e": expected_e},
        config={"exposure_time": exposure_time, "detector": detector.copy()},
        metadata=metadata,
    )


def _external_expectations_and_mask(base_eps: np.ndarray,
                                    ref_eps: np.ndarray,
                                    exposure_time: float,
                                    detector: dict,
                                    snr_threshold: float) -> dict:
    gain = detector["gain"]
    read_noise = detector["read_noise"]
    dark_current = detector["dark_current"]
    sky_background = detector["sky_background"]

    source_e0 = base_eps * exposure_time
    source_e1 = ref_eps * exposure_time
    dark_e = dark_current * exposure_time
    sky_e = sky_background * exposure_time

    E0_adu = (source_e0 + sky_e + dark_e) / gain
    E1_ref_adu = (source_e1 + sky_e + dark_e) / gain
    T_adu = (E1_ref_adu - E0_adu)

    noise_map_adu = np.sqrt(source_e0 + sky_e + dark_e + read_noise ** 2) / gain
    var_adu = noise_map_adu ** 2

    source_adu = source_e0 / gain
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(noise_map_adu > 0, source_adu / noise_map_adu, 0.0)
    mask = (snr > snr_threshold).flatten()

    return {
        "E0_adu": E0_adu,
        "E1_ref_adu": E1_ref_adu,
        "T_adu": T_adu,
        "var_adu": var_adu,
        "mask_flat": mask,
        "snr_2d": snr,
    }


def _matched_filter_ext(O_adu: np.ndarray,
                        E0_adu: np.ndarray,
                        T_adu: np.ndarray,
                        var_adu: np.ndarray,
                        mask_flat: np.ndarray) -> dict:
    O = O_adu.flatten()[mask_flat]
    E0 = E0_adu.flatten()[mask_flat]
    T = T_adu.flatten()[mask_flat]
    V = var_adu.flatten()[mask_flat] + 1e-10
    w = 1.0 / V
    r = O - E0
    N = float(np.sum(T * r * w))
    D = float(np.sum(T * T * w))
    if D <= 0.0:
        alpha_hat = 0.0
        delta = 0.0
    else:
        alpha_unc = N / D
        alpha_hat = max(0.0, alpha_unc)
        delta = 0.0 if alpha_hat <= 0.0 else (N * N) / D
    return {"N": N, "D": D, "alpha_hat": alpha_hat, "delta": delta}


def test_chernoff_asimov_identities_and_pvalue():
    # Configuration
    shape = (32, 32)
    pixel_scale = 0.1
    exposure_time = 1000.0
    detector = {
        "gain": 1.0,
        "read_noise": 3.0,
        "dark_current": 0.01,
        "sky_background": 0.2,
    }
    m_ref = 2.0

    # Baseline noiseless source (e-/s)
    base_eps = np.full(shape, 100.0, dtype=float)

    # Template in e-/s (localized structure)
    template_eps = np.zeros(shape, dtype=float)
    cy, cx, rad = 16, 16, 5
    for y in range(shape[0]):
        for x in range(shape[1]):
            if (y - cy) ** 2 + (x - cx) ** 2 <= rad ** 2:
                template_eps[y, x] = 0.5

    # Reference H1 noiseless = base + m_ref * template
    ref_eps = base_eps + m_ref * template_eps

    # External expectations and mask (spec-side truth)
    ext = _external_expectations_and_mask(base_eps, ref_eps, exposure_time, detector, snr_threshold=1.0)
    E0_adu_ext = ext["E0_adu"]
    E1_ref_adu_ext = ext["E1_ref_adu"]
    T_adu_ext = ext["T_adu"] / m_ref  # ensure per-unit-mass template
    var_adu_ext = ext["var_adu"]
    mask_ext = ext["mask_flat"]

    # Construct baseline and reference ObservationData
    obs0 = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name="H0", data_mode="E0")
    obs1_ref = _build_observation(ref_eps, exposure_time, detector, pixel_scale, run_name="H1_ref", data_mode="E0")

    # Build a test observation equal to the exact H1 expectation (Asimov data)
    obs_test_asimov = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name="Asimov", data_mode="custom")
    obs_test_asimov.imaging.data = al.Array2D(
        values=E1_ref_adu_ext,
        mask=al.Mask2D.all_false(shape_native=shape, pixel_scales=pixel_scale),
    )

    # Minimal lensing record carrying subhalo mass and position
    lensing = _make_lensing_data(shape=shape, pixel_scale=pixel_scale, run_name="lens")
    lensing.subhalo_mass = m_ref
    lensing.subhalo_position = (0.0, 0.0)

    # Detector
    det = ChernoffSubhaloDetector(
        observation_data_no_subhalo=obs0,
        observation_data_with_subhalo_ref=obs1_ref,
        lensing_test=lensing,
        snr_threshold=1.0,
        use_template=True,
    )

    out = det.detect_at_position(observation_with_subhalo=obs_test_asimov, subhalo_position=lensing.subhalo_position, compute_asimov=True)
    r = out.result

    # Compute independent N, D using only external arrays
    mf = _matched_filter_ext(E1_ref_adu_ext, E0_adu_ext, T_adu_ext, var_adu_ext, mask_ext)
    N, D = mf["N"], mf["D"]

    print("\n[Chernoff Asimov] Key quantities:")
    print(f"pixels_unmasked_ext={int(np.sum(mask_ext))}, D_ext={D:.6f}, N_ext={N:.6f}")
    print(f"alpha_hat(det)={r.alpha_hat:.6f}, alpha_hat(ext)={mf['alpha_hat']:.6f} (expected {m_ref:.6f})")
    print(f"delta_chi2(det)={r.delta_chi2:.6f}, delta_chi2(ext)={mf['delta']:.6f}")

    # Asserts: exact identities in noiseless Asimov case
    assert np.isclose(r.alpha_hat, m_ref, rtol=1e-8, atol=1e-10)
    assert np.isclose(r.alpha_hat, mf["alpha_hat"], rtol=1e-8, atol=1e-10)
    assert np.isclose(r.delta_chi2, (N * N) / D if D > 0 else 0.0, rtol=1e-8, atol=1e-10)
    assert np.isclose(r.delta_chi2, mf["delta"], rtol=1e-8, atol=1e-10)

    # P-value mapping: p = 0.5 * sf_chi2_1(delta)
    p_theory = 0.5 * chi2_dist(df=1).sf(r.delta_chi2)
    print(f"p_value={r.p_value:.6e}, p_theory={p_theory:.6e}")
    assert np.isclose(r.p_value, p_theory, rtol=1e-10, atol=1e-14)


def test_chernoff_null_mixture_statistics():
    # Configuration
    shape = (24, 24)
    pixel_scale = 0.1
    exposure_time = 1000.0
    detector = {
        "gain": 1.0,
        "read_noise": 3.0,
        "dark_current": 0.01,
        "sky_background": 0.2,
    }
    m_ref = 1.5

    # Baseline and reference noiseless sources (reference differs only by template, scaling m_ref)
    base_eps = np.full(shape, 50.0, dtype=float)
    template_eps = np.zeros(shape, dtype=float)
    template_eps[8:16, 10:18] = 0.25
    ref_eps = base_eps + m_ref * template_eps

    # Build ObservationData instances
    obs0 = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name="H0", data_mode="E0")
    obs1_ref = _build_observation(ref_eps, exposure_time, detector, pixel_scale, run_name="H1_ref", data_mode="E0")

    # External expectations and mask
    ext = _external_expectations_and_mask(base_eps, ref_eps, exposure_time, detector, snr_threshold=1.0)
    E0_adu_ext = ext["E0_adu"]
    T_adu_ext = ext["T_adu"] / m_ref
    var_adu_ext = ext["var_adu"]
    mask_ext = ext["mask_flat"]
    sqrtV = np.sqrt(var_adu_ext.flatten()[mask_ext])

    # Chernoff detector
    lensing = _make_lensing_data(shape=shape, pixel_scale=pixel_scale, run_name="lens")
    lensing.subhalo_mass = m_ref
    lensing.subhalo_position = (0.0, 0.0)
    det = ChernoffSubhaloDetector(
        observation_data_no_subhalo=obs0,
        observation_data_with_subhalo_ref=obs1_ref,
        lensing_test=lensing,
        snr_threshold=1.0,
        use_template=True,
    )

    rng = np.random.default_rng(12345)
    num_trials = 200
    alphas_ext = []
    deltas_ext = []
    deltas_det = []

    for i in range(num_trials):
        # Draw masked-pixel noise and form O under H0
        noise_masked = rng.normal(loc=0.0, scale=sqrtV)
        O_full = E0_adu_ext.flatten().copy()
        O_full[mask_ext] = O_full[mask_ext] + noise_masked
        O_2d = O_full.reshape(E0_adu_ext.shape)

        # External matched-filter calculation
        mf = _matched_filter_ext(O_2d, E0_adu_ext, T_adu_ext, var_adu_ext, mask_ext)
        alphas_ext.append(mf["alpha_hat"])
        deltas_ext.append(mf["delta"])

        # Detector output
        obs_test = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name=f"trial_{i}", data_mode="custom")
        obs_test.imaging.data = al.Array2D(
            values=O_2d,
            mask=al.Mask2D.all_false(shape_native=O_2d.shape, pixel_scales=pixel_scale),
        )
        r = det.detect_at_position(obs_test, subhalo_position=lensing.subhalo_position, compute_asimov=False).result
        deltas_det.append(r.delta_chi2)

    alphas_ext = np.array(alphas_ext)
    deltas_ext = np.array(deltas_ext)
    deltas_det = np.array(deltas_det)

    # Per-trial agreement between detector and external truth
    np.testing.assert_allclose(deltas_det, deltas_ext, rtol=1e-6, atol=1e-12)

    frac_zero = float(np.mean(alphas_ext == 0.0))
    mean_delta = float(np.mean(deltas_ext))
    positive_deltas = deltas_ext[deltas_ext > 0]
    median_positive = float(np.median(positive_deltas)) if positive_deltas.size > 0 else 0.0
    theo_median_chi2_1 = float(chi2_dist(df=1).ppf(0.5))

    print("\n[Chernoff Null] Empirical mixture diagnostics:")
    print(f"Trials={num_trials}, fraction(alpha_hat==0)={frac_zero:.3f} (theory ~0.5)")
    print(f"E[Δχ²]={mean_delta:.3f} (theory ~0.5)")
    print(f"Median(Δχ² | Δχ²>0)={median_positive:.3f} vs median(χ1²)={theo_median_chi2_1:.3f}")

    # Statistical properties under H0 Chernoff mixture (½ δ0 + ½ χ1²)
    assert 0.35 <= frac_zero <= 0.65
    assert 0.35 <= mean_delta <= 0.65
    if positive_deltas.size > 10:
        assert abs(median_positive - theo_median_chi2_1) < 0.2


def test_chernoff_boundary_H0_asimov():
    shape = (20, 20)
    pixel_scale = 0.1
    exposure_time = 500.0
    detector = {"gain": 2.0, "read_noise": 2.5, "dark_current": 0.02, "sky_background": 0.1}
    m_ref = 1.0

    base_eps = np.full(shape, 40.0, dtype=float)
    ref_eps = base_eps + m_ref * np.zeros(shape)

    ext = _external_expectations_and_mask(base_eps, ref_eps, exposure_time, detector, snr_threshold=1.0)
    E0_adu = ext["E0_adu"]
    var_adu = ext["var_adu"]
    mask = ext["mask_flat"]

    obs0 = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name="H0", data_mode="E0")
    obs1_ref = _build_observation(ref_eps, exposure_time, detector, pixel_scale, run_name="H1_ref", data_mode="E0")
    lensing = _make_lensing_data(shape=shape, pixel_scale=pixel_scale, run_name="lens")
    lensing.subhalo_mass = m_ref
    lensing.subhalo_position = (0.0, 0.0)
    det = ChernoffSubhaloDetector(obs0, obs1_ref, lensing, snr_threshold=1.0, use_template=True)

    # O = E0 (pure H0 Asimov)
    obs_test = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name="Asimov_H0", data_mode="custom")
    obs_test.imaging.data = al.Array2D(
        values=E0_adu,
        mask=al.Mask2D.all_false(shape_native=shape, pixel_scales=pixel_scale),
    )

    r = det.detect_at_position(obs_test, subhalo_position=lensing.subhalo_position, compute_asimov=False).result
    mf = _matched_filter_ext(E0_adu, E0_adu, np.zeros_like(E0_adu), var_adu, mask)

    print("\n[Chernoff H0 Asimov] alpha_hat, delta_chi2, p_value:")
    print(f"alpha_hat={r.alpha_hat:.6e}, delta={r.delta_chi2:.6e}, p={r.p_value:.6f}")

    assert r.alpha_hat == 0.0
    assert r.delta_chi2 == 0.0
    assert np.isclose(r.p_value, 0.5, rtol=0, atol=0)
    assert np.isclose(mf["alpha_hat"], 0.0)
    assert np.isclose(mf["delta"], 0.0)


def test_chernoff_gain_invariance_asimov():
    # Same physical scene, change gain scaling ⇒ Δχ² and p unchanged
    shape = (24, 24)
    pixel_scale = 0.1
    exposure_time = 800.0
    m_ref = 1.7

    base_eps = np.full(shape, 60.0, dtype=float)
    template_eps = np.zeros(shape, dtype=float)
    template_eps[7:14, 9:16] = 0.3
    ref_eps = base_eps + m_ref * template_eps

    detA = {"gain": 1.0, "read_noise": 3.0, "dark_current": 0.02, "sky_background": 0.15}
    detB = {"gain": 2.0, "read_noise": 3.0, "dark_current": 0.02, "sky_background": 0.15}

    # Build for gain A
    extA = _external_expectations_and_mask(base_eps, ref_eps, exposure_time, detA, snr_threshold=1.0)
    obs0A = _build_observation(base_eps, exposure_time, detA, pixel_scale, run_name="H0_A", data_mode="E0")
    obs1A = _build_observation(ref_eps, exposure_time, detA, pixel_scale, run_name="H1_A", data_mode="E0")
    lensA = _make_lensing_data(shape=shape, pixel_scale=pixel_scale, run_name="lensA")
    lensA.subhalo_mass = m_ref
    lensA.subhalo_position = (0.0, 0.0)
    detA_obj = ChernoffSubhaloDetector(obs0A, obs1A, lensA, snr_threshold=1.0, use_template=True)

    obsAsimovA = _build_observation(base_eps, exposure_time, detA, pixel_scale, run_name="Asimov_A", data_mode="custom")
    obsAsimovA.imaging.data = al.Array2D(values=extA["E1_ref_adu"], mask=al.Mask2D.all_false(shape_native=shape, pixel_scales=pixel_scale))
    rA = detA_obj.detect_at_position(obsAsimovA, subhalo_position=lensA.subhalo_position, compute_asimov=False).result

    # Build for gain B (same electrons, different ADU scaling)
    extB = _external_expectations_and_mask(base_eps, ref_eps, exposure_time, detB, snr_threshold=1.0)
    obs0B = _build_observation(base_eps, exposure_time, detB, pixel_scale, run_name="H0_B", data_mode="E0")
    obs1B = _build_observation(ref_eps, exposure_time, detB, pixel_scale, run_name="H1_B", data_mode="E0")
    lensB = _make_lensing_data(shape=shape, pixel_scale=pixel_scale, run_name="lensB")
    lensB.subhalo_mass = m_ref
    lensB.subhalo_position = (0.0, 0.0)
    detB_obj = ChernoffSubhaloDetector(obs0B, obs1B, lensB, snr_threshold=1.0, use_template=True)

    obsAsimovB = _build_observation(base_eps, exposure_time, detB, pixel_scale, run_name="Asimov_B", data_mode="custom")
    obsAsimovB.imaging.data = al.Array2D(values=extB["E1_ref_adu"], mask=al.Mask2D.all_false(shape_native=shape, pixel_scales=pixel_scale))
    rB = detB_obj.detect_at_position(obsAsimovB, subhalo_position=lensB.subhalo_position, compute_asimov=False).result

    print("\n[Chernoff Gain Invariance] Δχ² and p-values under different gain:")
    print(f"gain=1: delta={rA.delta_chi2:.6f}, p={rA.p_value:.3e}")
    print(f"gain=2: delta={rB.delta_chi2:.6f}, p={rB.p_value:.3e}")

    assert np.isclose(rA.delta_chi2, rB.delta_chi2, rtol=1e-8, atol=0)
    assert np.isclose(rA.p_value, rB.p_value, rtol=1e-10, atol=0)


def test_chernoff_monotonicity_with_mass_asimov():
    shape = (24, 24)
    pixel_scale = 0.1
    exposure_time = 600.0
    m_ref = 1.0

    base_eps = np.full(shape, 30.0, dtype=float)
    template_eps = np.zeros(shape, dtype=float)
    template_eps[9:15, 9:15] = 0.4
    detector = {"gain": 1.0, "read_noise": 2.0, "dark_current": 0.02, "sky_background": 0.12}

    # Reference for building template
    ref_eps = base_eps + m_ref * template_eps
    obs0 = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name="H0", data_mode="E0")
    obs1_ref = _build_observation(ref_eps, exposure_time, detector, pixel_scale, run_name="H1_ref", data_mode="E0")
    lens = _make_lensing_data(shape=shape, pixel_scale=pixel_scale, run_name="lens")
    lens.subhalo_mass = m_ref
    lens.subhalo_position = (0.0, 0.0)
    det = ChernoffSubhaloDetector(obs0, obs1_ref, lens, snr_threshold=1.0, use_template=True)

    ext = _external_expectations_and_mask(base_eps, ref_eps, exposure_time, detector, snr_threshold=1.0)
    E0_adu = ext["E0_adu"]
    T_adu = ext["T_adu"] / m_ref
    var_adu = ext["var_adu"]
    mask = ext["mask_flat"]

    masses = [0.0, 0.5, 1.0, 1.5, 2.0]
    deltas_ext = []
    deltas_det = []
    for m in masses:
        O_adu = E0_adu + m * T_adu
        mf = _matched_filter_ext(O_adu, E0_adu, T_adu, var_adu, mask)
        deltas_ext.append(mf["delta"])

        obs_test = _build_observation(base_eps, exposure_time, detector, pixel_scale, run_name=f"Asimov_m{m}", data_mode="custom")
        obs_test.imaging.data = al.Array2D(values=O_adu, mask=al.Mask2D.all_false(shape_native=shape, pixel_scales=pixel_scale))
        r = det.detect_at_position(obs_test, subhalo_position=lens.subhalo_position, compute_asimov=False).result
        deltas_det.append(r.delta_chi2)

    deltas_ext = np.array(deltas_ext)
    deltas_det = np.array(deltas_det)

    print("\n[Chernoff Monotonicity] Δχ² vs mass:")
    print("masses:", masses)
    print("delta_ext:", deltas_ext)
    print("delta_det:", deltas_det)

    # Non-decreasing sequence
    assert np.all(np.diff(deltas_ext) >= -1e-10)
    assert np.all(np.diff(deltas_det) >= -1e-10)
    # Agreement between detector and external calcs
    np.testing.assert_allclose(deltas_det, deltas_ext, rtol=1e-8, atol=1e-10)



