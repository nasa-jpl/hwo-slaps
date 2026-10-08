"""The orthonormal aperture bases of the prior tables and their raw-coefficient transform."""

import numpy as np
import pytest

from hwoslaps.optics.aperture_basis import ApertureBasisTransform, positive_diagonal_qr
from hwoslaps.optics.knowledge_error import WavefrontDrawSpec, draw_wavefront
from hwoslaps.optics.mode_priors import ModeWeightPriorSpec, draw_global_orthonormal, load_prior
from hwoslaps.optics.pupils import build_pupil, parse_pupil
from hwoslaps.optics.wavefront import WavefrontBasis


def test_positive_diagonal_qr():
    values = np.random.default_rng(2).normal(size=(40, 6))
    q_matrix, r_matrix = positive_diagonal_qr(values)
    np.testing.assert_allclose(q_matrix @ r_matrix, values, rtol=0.0, atol=1e-14)
    assert np.all(np.diag(r_matrix) > 0.0) and np.allclose(np.tril(r_matrix, -1), 0.0)
    orthonormal = np.linalg.qr(values)[0] * np.sign(np.diag(np.linalg.qr(values)[1]))[np.newaxis, :]
    _, scaled = positive_diagonal_qr(orthonormal * np.sqrt(40))
    np.testing.assert_allclose(scaled, np.sqrt(40) * np.eye(6), rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="rank deficient"):
        positive_diagonal_qr(np.column_stack([values[:, 0], 2.0 * values[:, 0]]))
    with pytest.raises(ValueError, match="cannot carry"):
        positive_diagonal_qr(values[:5])


@pytest.mark.backend
@pytest.mark.parametrize("mode_source", ["small", "packaged_drift"])
def test_transform_realizes_orthonormal_coefficients(p1_pupil, mode_source):
    pupil = build_pupil(parse_pupil(p1_pupil, "pupil"))
    basis = WavefrontBasis(pupil, reference_wavelength_m=5e-7)
    global_nolls, segment_nolls = (4, 5, 6, 7), (1, 2, 3)
    if mode_source == "packaged_drift":
        prior, _ = load_prior(ModeWeightPriorSpec("packaged", "jwst_wss_drift_v1", None, None))
        global_nolls, segment_nolls = tuple(prior.global_weights), tuple(prior.segment_weights)
    transform = ApertureBasisTransform(basis, global_nolls=global_nolls, segment_nolls=segment_nolls)
    mask = pupil.illuminated_mask
    segment_mask = mask & (np.asarray(pupil.segments[3]) > 0.5)
    for index in range(len(global_nolls)):
        unit = {noll: float(j == index) for j, noll in enumerate(global_nolls)}
        raw = transform.to_raw(global_=unit)
        opd = basis.opd_nm(raw).ravel()[mask]
        assert np.mean(opd ** 2) == pytest.approx(1.0, rel=1e-10)
        if index == 0:
            assert [value != 0.0 for _, value in raw.zernikes()] == [True] + [False] * (len(global_nolls) - 1)
    for index in range(len(segment_nolls)):
        unit = {noll: float(j == index) for j, noll in enumerate(segment_nolls)}
        raw = transform.to_raw(segment={3: unit})
        assert {segment for segment, _, _ in raw.segment_hexikes()} == {3}
        opd = basis.opd_nm(raw).ravel()[segment_mask]
        assert np.mean(opd ** 2) == pytest.approx(1.0, rel=1e-10)
        if index == 0:
            assert [value != 0.0 for _, _, value in raw.segment_hexikes()] == [True] + [False] * (len(segment_nolls) - 1)
    with pytest.raises(ValueError, match="exactly the transform modes"):
        transform.to_raw(global_={4: 1.0})
    if mode_source == "packaged_drift":
        # Independent sign-fixed QR and projection on every real packaged global mode.
        values = basis.zernike_samples(global_nolls, mask)
        q_matrix, r_matrix = np.linalg.qr(values, mode="reduced")
        signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
        orthonormal = np.sqrt(values.shape[0]) * q_matrix * signs[np.newaxis, :]
        draw = draw_global_orthonormal(np.random.default_rng(20260806), prior, 1.0)
        expected = np.array([draw[noll] for noll in global_nolls])
        raw = transform.to_raw(global_=draw)
        realized = basis.opd_nm(raw).ravel()[mask]
        np.testing.assert_allclose(realized, orthonormal @ expected, rtol=1e-10, atol=1e-11)
        recovered = np.linalg.lstsq(orthonormal, realized, rcond=None)[0]
        np.testing.assert_allclose(recovered, expected, rtol=1e-10, atol=1e-11)
        # The public draw must use that transform before its physical RMS normalization.
        runtime = draw_wavefront(basis, WavefrontDrawSpec(
            ModeWeightPriorSpec("packaged", "jwst_wss_drift_v1", None, None), 1.0, 20260806, "global"))
        direction = np.array([runtime.orthonormal_global[noll] for noll in global_nolls])
        expected_opd = orthonormal @ direction
        expected_scale = 1.0 / np.std(expected_opd)
        physical = basis.opd_nm(runtime.coefficients).ravel()[mask]
        np.testing.assert_allclose(physical, expected_opd * expected_scale, rtol=1e-10, atol=1e-11)
        projected = np.linalg.lstsq(orthonormal, physical, rcond=None)[0]
        np.testing.assert_allclose(projected, direction * expected_scale, rtol=1e-10, atol=1e-11)
