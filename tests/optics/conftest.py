"""Fixtures of the optics lane: small pupil and optics mappings.

Mappings are returned fresh by each fixture, so a test may edit its copy.
"""

import pytest


@pytest.fixture
def p1_pupil():
    """The parity P1 pupil: two rings of 1.65 m segments with 6 mm gaps, 128 pixels over 7.225765 m."""
    return {"kind": "hex_segmented", "diameter_m": 7.225765, "pixels": 128, "supersampling": 2, "rings": 2,
            "segment_point_to_point_m": 1.65, "gap_m": 0.006}


@pytest.fixture
def paper_pupil(p1_pupil):
    """The paper's HWO pupil sampling: the P1 geometry at 512 pixels, supersampling 4."""
    return {**p1_pupil, "pixels": 512, "supersampling": 4}


@pytest.fixture
def p1_truth(p1_pupil):
    """The parity P1 optical truth at 500 nm (17 x 17 kernel at 0.03 arcsec, oversampling 11)."""
    return {"kind": "optical", "pupil": p1_pupil, "focal_length_m": 144.0, "wavelength_nm": 500.0,
            "detector_oversampling": 11, "kernel_shape": [17, 17],
            "wavefront": {"segment_hexikes": {0: {4: 10.0}, 3: {5: 8.0}, 7: {6: 12.0}},
                          "zernikes": {4: 5.0, 5: 5.0, 8: 5.0}}}


@pytest.fixture
def circular_pupil():
    """A 1 m circular pupil sampled by 256 pixels, supersampling 4."""
    return {"kind": "circular", "diameter_m": 1.0, "pixels": 256, "supersampling": 4}
