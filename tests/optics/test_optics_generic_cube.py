"""Cube transport, exact node semantics and coupled byte identities at the provider boundary."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from hwoslaps.identity import validate_loaded_file
from hwoslaps.optics import providers
from hwoslaps.optics.providers import KernelCubePSF, KernelCubeSpec, build_psf_provider, parse_psf
from hwoslaps.optics.wavefront import WavefrontCoefficients


@pytest.fixture
def cube_input(tmp_path):
    values = np.zeros((2, 3, 5))
    values[0, 1, 2] = 1.0
    values[1, 1, 1:4] = [0.25, 0.5, 0.25]
    waves = np.array([4.0e-7, 6.0e-7])
    path = tmp_path / "cube.npz"
    np.savez(path, kernels=values, wavelengths_m=waves)
    spec = parse_psf({"truth": {"kind": "kernel_cube", "path": str(path),
                               "pixel_scale_arcsec": 0.03, "normalize": False}}).truth
    return spec, values, waves


def test_cube_keeps_declared_nodes_unit_bytes_source_and_support(cube_input):
    spec, values, waves = cube_input
    content = spec.path.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    provider = build_psf_provider(spec, pixel_scale_arcsec=0.03, bandpass_support_m=(3.1e-7, 6.9e-7))
    assert isinstance(provider, KernelCubePSF)
    assert provider.shape == (3, 5) and provider.pixel_scale_arcsec == 0.03
    assert provider.wavelengths_m == tuple(waves)
    assert provider.support_m == pytest.approx((3.0e-7, 7.0e-7), rel=1e-15)
    assert provider.file_digests == {str(spec.path): actual}
    with pytest.raises(TypeError):
        provider.file_digests[str(spec.path)] = "0" * 64
    nodes = provider.kernels()
    for index, (wavelength, node) in enumerate(zip(waves, nodes)):
        assert provider.kernel(wavelength) is node
        np.testing.assert_array_equal(node.kernel, values[index])
        assert node.source["kind"] == "cube" and node.source["captured_power_fraction"] is None
        assert node.source["wavelength_m"] == wavelength
        assert node.source["file_sha256"] == actual and node.source["array_index"] == index
    assert provider.kernels(waves) == nodes
    assert (provider.basis, provider.coefficients, provider.collecting_area_m2) == (None, None, None)
    for wavelength in (None, 3.0e-7, 5.0e-7, 7.0e-7, np.nextafter(waves[0], np.inf)):
        with pytest.raises(ValueError, match="tabulated"):
            provider.kernel(wavelength)
    with pytest.raises(ValueError, match="wavefront basis"):
        provider.kernel(waves[0], coefficients=WavefrontCoefficients.empty())
    with pytest.raises(ValueError, match="coverage"):
        build_psf_provider(spec, pixel_scale_arcsec=0.03, bandpass_support_m=(2.9e-7, 6.9e-7))
    with pytest.raises(ValueError, match="never resampled"):
        build_psf_provider(spec, pixel_scale_arcsec=0.031)
    with pytest.raises(ValueError, match="cannot replace"):
        build_psf_provider(spec, pixel_scale_arcsec=0.03, wavelengths_m=waves)


@pytest.mark.parametrize("defect", ["missing-member", "shape", "node-count", "duplicate", "descending",
                                    "nonfinite", "nonpositive", "negative-kernel", "even-kernel", "bad-sum",
                                    "complex-nonzero", "complex-nonfinite"])
def test_invalid_cube_data_is_refused_at_the_real_builder(tmp_path, defect):
    values = np.zeros((2, 3, 3))
    values[:, 1, 1] = 1.0
    waves = np.array([4.0e-7, 6.0e-7])
    if defect == "shape":
        values = values[0]
    elif defect == "node-count":
        waves = waves[:1]
    elif defect == "duplicate":
        waves[1] = waves[0]
    elif defect == "descending":
        waves = waves[::-1]
    elif defect == "nonfinite":
        waves[1] = np.nan
    elif defect == "nonpositive":
        waves[0] = 0.0
    elif defect == "negative-kernel":
        values[0, 0, 0] = -0.01
    elif defect == "even-kernel":
        values = values[:, :, :2]
    elif defect == "bad-sum":
        values *= 2.0
    elif defect.startswith("complex"):
        values = values.astype(complex)
        values[0, 1, 1] = complex(1.0, 2.0 if defect == "complex-nonzero" else np.nan)
    path = tmp_path / "invalid_cube.npz"
    if defect == "missing-member":
        np.savez(path, kernels=values)
    else:
        np.savez(path, kernels=values, wavelengths_m=waves)
    with pytest.raises(ValueError, match="must be real" if defect.startswith("complex") else None):
        build_psf_provider(KernelCubeSpec(path, 0.03, False, None), pixel_scale_arcsec=0.03)


def test_cube_reads_arrays_labels_and_digest_from_the_same_snapshot(cube_input, monkeypatch):
    spec, expected_values, expected_waves = cube_input
    expected_content = spec.path.read_bytes()
    expected_digest = hashlib.sha256(expected_content).hexdigest()
    reader = providers.read_file_snapshot

    def read_then_publish_another_valid_cube(path):
        content, digest = reader(path)
        changed = expected_values[:, :, ::-1].copy()
        changed[0, 1] = [0.0, 1.0, 0.0, 0.0, 0.0]
        np.savez(path, kernels=changed, wavelengths_m=[4.5e-7, 6.5e-7])
        return content, digest

    monkeypatch.setattr(providers, "read_file_snapshot", read_then_publish_another_valid_cube)
    provider = build_psf_provider(replace(spec, file_sha256=expected_digest), pixel_scale_arcsec=0.03)
    assert provider.wavelengths_m == tuple(expected_waves)
    assert provider.file_digests == {str(spec.path): expected_digest}
    np.testing.assert_array_equal(np.stack([node.kernel for node in provider.kernels()]), expected_values)
    changed_digest = hashlib.sha256(spec.path.read_bytes()).hexdigest()
    assert changed_digest != expected_digest
    with pytest.raises(ValueError, match="changed while being loaded"):
        validate_loaded_file(spec.path, provider.file_digests[str(spec.path)], {str(spec.path): changed_digest})


def test_cube_declared_digest_is_checked_before_decoding(cube_input):
    spec, _, _ = cube_input
    spec.path.write_bytes(b"a different invalid archive")
    with pytest.raises(ValueError, match="file SHA-256"):
        build_psf_provider(replace(spec, file_sha256="0" * 64), pixel_scale_arcsec=0.03)


def test_single_node_cube_is_mono_without_finite_band_coverage(tmp_path):
    path = tmp_path / "single.npz"
    np.savez(path, kernels=[[[1.0]]], wavelengths_m=[5.0e-7])
    spec = KernelCubeSpec(path, 0.03, False, None)
    provider = build_psf_provider(spec, pixel_scale_arcsec=0.03)
    assert provider.kernel() is provider.kernel(5.0e-7)
    assert provider.kernels() == (provider.kernel(),)
    assert provider.support_m == (5.0e-7, 5.0e-7)
    with pytest.raises(ValueError, match="no finite-width"):
        build_psf_provider(spec, pixel_scale_arcsec=0.03, bandpass_support_m=(4.9e-7, 5.1e-7))


def test_midpoint_cube_coverage_accepts_only_boundary_rounding(tmp_path):
    from hwoslaps.spectra.bandpass import bandpass_nodes

    support = (450. / 1e9, 550. / 1e9)
    nodes = bandpass_nodes(support, 11)
    path = tmp_path / "midpoint_cube.npz"
    np.savez(path, kernels=np.ones((11, 1, 1)), wavelengths_m=nodes)
    spec = KernelCubeSpec(path, 0.03, False, None)
    provider = build_psf_provider(spec, pixel_scale_arcsec=0.03, bandpass_support_m=support)
    assert provider.support_m == pytest.approx(support, rel=4 * np.finfo(float).eps, abs=0.0)
    outside = (support[0] * (1 - 32 * np.finfo(float).eps), support[1])
    with pytest.raises(ValueError, match="outside cube coverage"):
        build_psf_provider(spec, pixel_scale_arcsec=0.03, bandpass_support_m=outside)
    with pytest.raises(ValueError, match="not a tabulated"):
        provider.kernel(np.nextafter(nodes[0], np.inf))


@pytest.mark.backend
def test_optical_node_stack_saved_as_cube_round_trips_bitwise(circular_pupil, tmp_path):
    spec = parse_psf({"truth": {"kind": "optical", "pupil": circular_pupil, "focal_length_m": 10.0,
                               "wavelength_samples": 2, "detector_oversampling": 3, "kernel_shape": [11, 11]}}).truth
    waves = (4.0e-7, 6.0e-7)
    optical = build_psf_provider(spec, pixel_scale_arcsec=0.03, wavelengths_m=waves)
    nodes = optical.kernels()
    path = tmp_path / "optical_cube.npz"
    np.savez(path, kernels=np.stack([node.kernel for node in nodes]), wavelengths_m=waves)
    cube = build_psf_provider(KernelCubeSpec(path, 0.03, False, None), pixel_scale_arcsec=0.03)
    for wavelength, node in zip(waves, nodes):
        loaded = cube.kernel(wavelength)
        np.testing.assert_array_equal(loaded.kernel, node.kernel)
        assert loaded.source["captured_power_fraction"] is None
