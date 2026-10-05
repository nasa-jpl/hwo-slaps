"""The orthonormal aperture bases of the prior tables and their raw-coefficient transform."""

import numpy as np
import pytest

from hwoslaps.optics.aperture_basis import ApertureBasisTransform, positive_diagonal_qr
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
def test_transform_realizes_orthonormal_coefficients(p1_pupil):
    pupil = build_pupil(parse_pupil(p1_pupil, "pupil"))
    basis = WavefrontBasis(pupil, reference_wavelength_m=5e-7)
    global_nolls, segment_nolls = (4, 5, 6, 7), (1, 2, 3)
    transform = ApertureBasisTransform(basis, global_nolls=global_nolls, segment_nolls=segment_nolls)
    mask = pupil.illuminated_mask
    segment_mask = mask & (np.asarray(pupil.segments[3]) > 0.5)
    for index in range(len(global_nolls)):
        unit = {noll: float(j == index) for j, noll in enumerate(global_nolls)}
        raw = transform.to_raw(global_=unit)
        opd = basis.opd_nm(raw).ravel()[mask]
        assert np.mean(opd ** 2) == pytest.approx(1.0, rel=1e-10)
        if index == 0:
            assert [value != 0.0 for _, value in raw.zernikes()] == [True, False, False, False]
    for index in range(len(segment_nolls)):
        unit = {noll: float(j == index) for j, noll in enumerate(segment_nolls)}
        raw = transform.to_raw(segment={3: unit})
        assert {segment for segment, _, _ in raw.segment_hexikes()} == {3}
        opd = basis.opd_nm(raw).ravel()[segment_mask]
        assert np.mean(opd ** 2) == pytest.approx(1.0, rel=1e-10)
        if index == 0:
            assert [value != 0.0 for _, _, value in raw.segment_hexikes()] == [True, False, False]
    with pytest.raises(ValueError, match="exactly the transform modes"):
        transform.to_raw(global_={4: 1.0})
