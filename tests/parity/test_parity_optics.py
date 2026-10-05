"""K1: the paper's HWO reference PSF, bitwise against the submitted code (41621de).

The fixture ``k1_paper_kernel.json`` is written by ``tests/scripts/generate_paper_parity.py
--kernel-anchor`` from the 41621de tree, never by this engine: the paper-format digests of the
``science_hwo35`` truth kernel at the Fisher-ladder support (999 x 999) and the nonlinear-fit
support (51 x 51), and the state's wavefront coefficients. Here the same optics are configured in
the final schema, with the truth wavefront drawn from the packaged static prior as the paper drew
it. The test runs with 16 BLAS threads, so it also proves that kernel bytes do not depend on the
caller's thread count.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.optics.providers import build_psf_provider, parse_psf

pytestmark = pytest.mark.backend

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity" / "k1_paper_kernel.json"
THREADS = 16


def _paper_digest(values):
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(f"{array.shape[0]}x{array.shape[1]}:".encode("ascii") + array.tobytes()).hexdigest()


def _paper_truth(shape):
    return {"truth": {
        "kind": "optical",
        "pupil": {"kind": "hex_segmented", "diameter_m": 7.225765, "pixels": 512, "supersampling": 4, "rings": 2,
                  "segment_point_to_point_m": 1.65, "gap_m": 0.006},
        "focal_length_m": 144.0, "wavelength_nm": 500.0, "detector_oversampling": 3, "kernel_shape": [shape, shape],
        "draw": {"prior": {"packaged": "jwst_wss_static_v1"}, "amplitude_rms_nm": 35.0, "seed": 20260835,
                 "family": "combined"}}}


def _blas_threads():
    from threadpoolctl import threadpool_info

    return [library["num_threads"] for library in threadpool_info() if library["user_api"] == "blas"]


def test_paper_optics_reproduce_the_submitted_code():
    import scipy.linalg  # noqa: F401  (HCIPy's propagation runs on SciPy's BLAS: load it before raising threads)
    from threadpoolctl import threadpool_limits

    anchor = json.loads(FIXTURE.read_text(encoding="utf-8"))
    aberrations = anchor["psf"]["aberrations"]
    expected_wavefront = {
        "segment_hexikes": {int(s): {int(n): v for n, v in modes.items()}
                            for s, modes in aberrations["segment_hexikes"].items()},
        "zernikes": {int(n): v for n, v in aberrations["global_zernikes"].items()}}
    with threadpool_limits(limits=THREADS):
        assert set(_blas_threads()) == {THREADS}, _blas_threads()
        for shape in (999, 51):
            psf = build_psf_provider(parse_psf(_paper_truth(shape)).truth,
                                     pixel_scale_arcsec=anchor["pixel_scale_arcsec"])
            assert psf.coefficients.to_mapping() == expected_wavefront
            kernel = psf.kernel().kernel
            assert kernel.shape == tuple(anchor["kernels"][f"{shape}x{shape}"]["shape"])
            assert _paper_digest(kernel) == anchor["kernels"][f"{shape}x{shape}"]["sha256"]
        assert set(_blas_threads()) == {THREADS}, _blas_threads()
