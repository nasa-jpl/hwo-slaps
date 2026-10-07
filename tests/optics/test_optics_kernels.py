"""Detector kernel values, kernel files, kernel sharing, the convolution bridge and BLAS pinning (optics.kernels)."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.signal

from hwoslaps.identity import array_digest
from hwoslaps.optics.kernels import DetectorPSF, KernelBinding, convolve_real_space, make_convolver

SCALE = 0.03


def _kernel(seed, shape=(5, 5)):
    values = np.random.default_rng(seed).random(shape)
    return values / values.sum()


@pytest.mark.parametrize("values, keywords, message", [
    (np.full((4, 5), 0.05), {}, "odd sides"),
    (np.full((5, 4), 0.05), {}, "odd sides"),
    (np.ones(5), {}, "two-dimensional"),
    (np.full((3, 3, 3), 1 / 27), {}, "two-dimensional"),
    (np.array([[0.0, np.nan, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), {}, "finite"),
    (np.pad([[1.0 + 2.0j]], 1), {}, "must be real"),
    (np.pad([[complex(1.0, np.nan)]], 1), {}, "must be real"),
    (np.array([[0.0, -0.1, 0.0], [0.0, 1.1, 0.0], [0.0, 0.0, 0.0]]), {}, "non-negative"),
    (np.zeros((3, 3)), {}, "positive finite flux"),
    (np.pad([[1.0 + 1e-9]], 1), {"normalize": False}, "sum to one"),
    (np.pad([[1.0]], 1), {"pixel_scale_arcsec": 0.0}, "pixel_scale_arcsec"),
    (np.pad([[1.0]], 1), {"pixel_scale_arcsec": -1.0}, "pixel_scale_arcsec"),
    (np.pad([[1.0]], 1), {"pixel_scale_arcsec": np.nan}, "pixel_scale_arcsec"),
    (np.pad([[1.0]], 1), {"pixel_scale_arcsec": np.inf}, "pixel_scale_arcsec"),
    (np.pad([[1.0]], 1), {"pixel_scale_arcsec": True}, "pixel_scale_arcsec"),
    (np.pad([[1.0]], 1), {"pixel_scale_arcsec": "0.1"}, "pixel_scale_arcsec"),
    (np.pad([[1.0]], 1), {"source": {"kind": "array"}}, "must record 'captured_power_fraction'"),
    (np.pad([[1.0]], 1), {"source": {"kind": "optical", "captured_power_fraction": 1.5}},
     r"None or a number in \(0, 1\]"),
], ids=["even-rows", "even-columns", "1-d", "3-d", "nan", "complex-nonzero", "complex-nonfinite", "negative", "zero-flux", "unnormalized", "zero-scale",
        "negative-scale", "nan-scale", "inf-scale", "bool-scale", "text-scale", "no-captured-fraction", "captured-fraction-above-one"])
def test_detector_psf_validation(values, keywords, message):
    arguments = {"pixel_scale_arcsec": SCALE, "normalize": True, **keywords}
    with pytest.raises(ValueError, match=message):
        DetectorPSF.from_array(values, arguments.pop("pixel_scale_arcsec"), **arguments)


def test_detector_psf_normalizes_once_into_a_private_read_only_copy():
    values = np.random.default_rng(3).random((7, 5)) * 4.0
    psf = DetectorPSF.from_array(values, SCALE, normalize=True)
    assert psf.kernel.tobytes() == (values / float(np.sum(values))).tobytes()
    assert psf.kernel.dtype == np.float64 and psf.kernel.flags.c_contiguous and psf.shape == (7, 5)
    values[3, 2] = 100.0
    assert psf.kernel[3, 2] != 100.0
    with pytest.raises(ValueError):
        psf.kernel[3, 2] = 0.0
    assert dict(psf.source) == {"kind": "array", "captured_power_fraction": None}

    source = {"kind": "array", "captured_power_fraction": None,
              "calibration": {"label": "test-calibration"}}
    explicit = DetectorPSF.from_array(psf.kernel, SCALE, normalize=False, source=source)
    source["kind"] = "changed"
    source["calibration"]["label"] = "changed"
    assert dict(explicit.source) == {"kind": "array", "captured_power_fraction": None,
                                     "calibration": {"label": "test-calibration"}}
    with pytest.raises(TypeError):
        explicit.source["kind"] = "changed"
    identity = psf.kernel_identity()
    assert (identity.sha256, identity.shape, identity.pixel_scale_arcsec) == (array_digest(psf.kernel), (7, 5), SCALE)


def test_kernel_files_load_with_integrity_checks(tmp_path):
    values = np.random.default_rng(4).random((9, 7))
    npy = tmp_path / "kernel.npy"
    np.save(npy, values)
    npz = tmp_path / "kernels.npz"
    np.savez(npz, wide=values, other=np.ones((3, 3)))
    sha = hashlib.sha256(npy.read_bytes()).hexdigest()

    from_npy = DetectorPSF.from_file(npy, pixel_scale_arcsec=SCALE, array_key=None, normalize=True, file_sha256=sha)
    from_npz = DetectorPSF.from_file(npz, pixel_scale_arcsec=SCALE, array_key="wide", normalize=True,
                                     file_sha256=None)
    expected = (values / values.sum()).tobytes()
    assert from_npy.kernel.tobytes() == expected and from_npz.kernel.tobytes() == expected
    assert dict(from_npy.source) == {"kind": "file", "path": str(npy), "array_key": None, "file_sha256": sha,
                                     "normalized": True, "captured_power_fraction": None}

    with pytest.raises(ValueError, match="array_key must be None"):
        DetectorPSF.from_file(npy, pixel_scale_arcsec=SCALE, array_key="kernel", normalize=True, file_sha256=None)
    with pytest.raises(ValueError, match="needs the member name"):
        DetectorPSF.from_file(npz, pixel_scale_arcsec=SCALE, array_key=None, normalize=True, file_sha256=None)
    with pytest.raises(ValueError, match="no member 'kernel'"):
        DetectorPSF.from_file(npz, pixel_scale_arcsec=SCALE, array_key="kernel", normalize=True, file_sha256=None)
    garbage = tmp_path / "garbage.npy"
    garbage.write_bytes(b"not an array")
    with pytest.raises(ValueError, match="file SHA-256 is"):
        DetectorPSF.from_file(garbage, pixel_scale_arcsec=SCALE, array_key=None, normalize=True, file_sha256=sha)
    pickled = tmp_path / "pickled.npy"
    np.save(pickled, np.array([{"kernel": values}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match="pickle"):
        DetectorPSF.from_file(pickled, pixel_scale_arcsec=SCALE, array_key=None, normalize=True, file_sha256=None)


@pytest.mark.parametrize("imaginary", [2.0, np.nan], ids=["nonzero", "nonfinite"])
def test_kernel_file_refuses_complex_intensities_before_casting(tmp_path, imaginary):
    path = tmp_path / "complex.npy"
    np.save(path, np.pad([[complex(1.0, imaginary)]], 1))
    with pytest.raises(ValueError, match="must be real"):
        DetectorPSF.from_file(path, pixel_scale_arcsec=SCALE, array_key=None, normalize=False, file_sha256=None)


@pytest.mark.backend
def test_kernel_bytes_survive_backend_use():
    import autoarray as aa

    psf = DetectorPSF.from_array(_kernel(5, (7, 7)) * 3.0, SCALE, normalize=True)
    assert float(np.sum(psf.kernel)) != 1.0, "a dataset's normalization must change these bytes"
    kernel_bytes = psf.kernel.tobytes()
    cached = psf.convolver()
    cached_bytes = np.asarray(cached.kernel.native).tobytes()
    image = aa.Array2D(values=np.random.default_rng(6).random((15, 15)),
                       mask=aa.Mask2D.all_false(shape_native=(15, 15), pixel_scales=SCALE))
    cached.convolved_image_via_real_space_from(image=image, blurring_image=None)

    dataset_psf = make_convolver(psf.kernel, SCALE)
    aa.Imaging(data=aa.Array2D.no_mask(values=np.ones((15, 15)), pixel_scales=SCALE),
               noise_map=aa.Array2D.no_mask(values=np.ones((15, 15)), pixel_scales=SCALE), psf=dataset_psf)

    assert np.asarray(dataset_psf.kernel.native).tobytes() != kernel_bytes
    assert psf.kernel.tobytes() == kernel_bytes
    assert psf.convolver() is cached and np.asarray(cached.kernel.native).tobytes() == cached_bytes
    with pytest.raises(ValueError):
        psf.kernel[3, 3] = 0.0


def test_kernel_binding_shares_kernels_by_digest():
    first = DetectorPSF.from_array(_kernel(7), SCALE, normalize=False)
    same_values = DetectorPSF.from_array(_kernel(7), SCALE, normalize=False)
    other = DetectorPSF.from_array(_kernel(8), SCALE, normalize=False)
    binding = KernelBinding.from_groups({"source:disk": other, "lens": first, "source": same_values})

    assert binding.kernels == (other, first)
    assert dict(binding.group_index) == {"source:disk": 0, "lens": 1, "source": 1}
    assert binding.for_group("source") is first
    assert binding.groups_of(1) == ("lens", "source")
    assert binding.to_mapping() == {
        "kernels": [{"identity": kernel.kernel_identity().to_mapping(), "source": dict(kernel.source)}
                    for kernel in (other, first)],
        "groups": {"source:disk": 0, "lens": 1, "source": 1}}
    with pytest.raises(ValueError, match="2 distinct kernels"):
        binding.single
    assert KernelBinding.uniform(first, ["lens", "source"]).single is first
    with pytest.raises(KeyError, match="no kernel bound"):
        binding.for_group("perturbers_0")
    coarse = DetectorPSF.from_array(_kernel(9), 2 * SCALE, normalize=False)
    with pytest.raises(ValueError, match="one pixel scale"):
        KernelBinding.from_groups({"lens": first, "source": coarse})
    with pytest.raises(ValueError, match="must be distinct"):
        KernelBinding((first, same_values), {"lens": 0, "source": 1})


@pytest.mark.backend
def test_real_space_convolution_is_a_true_convolution_with_signed_kernels():
    rng = np.random.default_rng(10)
    image = rng.random((21, 25))
    kernel = rng.normal(size=(5, 7))
    kernel -= kernel.mean()
    expected = scipy.signal.convolve2d(image, kernel, mode="same")
    np.testing.assert_allclose(convolve_real_space(image, kernel, SCALE), expected, rtol=0.0, atol=1e-13)

    point = np.zeros((21, 25))
    point[10, 12] = 1.0
    np.testing.assert_allclose(convolve_real_space(point, kernel, SCALE)[8:13, 9:16], kernel, rtol=0.0, atol=1e-14)
    with pytest.raises(ValueError, match="odd sides"):
        convolve_real_space(image, kernel[:4], SCALE)


CHILD = """
import json, sys
from threadpoolctl import threadpool_info
from hwoslaps.optics.kernels import deterministic_blas
assert "scipy.linalg" not in sys.modules

def blas():
    return [library["num_threads"] for library in threadpool_info() if library["user_api"] == "blas"]

with deterministic_blas():
    import hcipy  # loads SciPy's BLAS, which HCIPy's matrix Fourier transform calls
    inside = blas()
print(json.dumps({"inside": inside, "after": blas()}))
"""


@pytest.mark.backend
def test_deterministic_blas_pins_libraries_loaded_inside_the_block():
    import hwoslaps

    environment = {**os.environ, "PYTHONPATH": str(Path(hwoslaps.__file__).resolve().parents[1]),
                   **{name: "4" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    completed = subprocess.run([sys.executable, "-c", CHILD], env=environment, capture_output=True, text=True,
                               check=True)
    threads = json.loads(completed.stdout.strip().splitlines()[-1])
    assert len(threads["after"]) == len(threads["inside"]) >= 1
    assert max(threads["after"]) > 1, "the libraries must default to more than one thread for the test to bite"
    assert set(threads["inside"]) == {1}, threads
