"""Validation tests for the PSF high-resolution and detector branches."""

from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path

import numpy as np
import pytest
import yaml

from hwoslaps.psf.generator import generate_psf_system


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def compact_no_aberration_config() -> dict:
    with (PROJECT_ROOT / "configs" / "master_config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = copy.deepcopy(cfg)
    cfg["plotting"]["enabled"] = False
    cfg["psf"]["hres_psf"]["num_pix"] = 64
    cfg["psf"]["hres_psf"]["num_airy"] = 4
    cfg["psf"]["hres_psf"]["save_highres_psf_npy"] = False
    cfg["psf"]["kernel"]["shape_native"] = [9, 9]

    aberr = cfg["psf"]["aberrations"]
    aberr["enable_segment_pistons"] = False
    aberr["enable_segment_tiptilts"] = False
    aberr["enable_segment_hexikes"] = False
    aberr["enable_global_zernikes"] = False
    aberr["segment_pistons"] = {}
    aberr["segment_tiptilts"] = {}
    aberr["segment_hexikes"] = {}
    aberr["global_zernikes"] = {}
    return cfg


def _quiet_generate(config: dict):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return generate_psf_system(config["psf"], full_config=config)


def _center_crop(array_2d: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    y0 = (array_2d.shape[0] - shape[0]) // 2
    x0 = (array_2d.shape[1] - shape[1]) // 2
    return array_2d[y0:y0 + shape[0], x0:x0 + shape[1]]


def test_detector_kernel_matches_binned_highres_psf(compact_no_aberration_config: dict):
    """The detector branch should match flux-conserving binning of the high-res branch."""
    psf_data = _quiet_generate(compact_no_aberration_config)

    subsampling = psf_data.integer_subsampling_factor
    kernel = np.asarray(psf_data.kernel.native)
    highres_power = np.asarray(psf_data.psf.power.shaped)

    assert subsampling >= 1
    assert psf_data.pixel_scale_arcsec * subsampling == pytest.approx(
        psf_data.kernel_pixel_scale,
        rel=1e-12,
        abs=1e-15,
    )

    crop_shape = (
        kernel.shape[0] * subsampling,
        kernel.shape[1] * subsampling,
    )
    highres_crop = _center_crop(highres_power, crop_shape)
    binned_highres = highres_crop.reshape(
        kernel.shape[0],
        subsampling,
        kernel.shape[1],
        subsampling,
    ).sum(axis=(1, 3))
    binned_highres = binned_highres / np.sum(binned_highres)

    assert np.sum(kernel) == pytest.approx(1.0, rel=1e-12, abs=1e-15)
    assert np.sum(binned_highres) == pytest.approx(1.0, rel=1e-12, abs=1e-15)
    assert np.allclose(kernel, binned_highres, rtol=1e-12, atol=1e-14)
