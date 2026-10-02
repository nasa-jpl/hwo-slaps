"""Absolute image-source flux contracts using synthetic morphology."""

import json

import numpy as np
import pytest

pytest.importorskip('autolens')

from hwoslaps.lensing import generate_lensing_system


def test_synthetic_morphology_is_magnified_once_by_circular_isothermal_lens(tmp_path):
    """An extended source follows the independent SIS mu=2 theta_E/beta law.

    This catches missing or duplicated magnification, and missing pixel-area
    conversion. No bank asset, telescope reference, or engine-computed expected
    image participates in the oracle.
    """
    asset_scale = 0.01
    axis = (np.arange(31) - 15) * asset_scale
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    sb = np.exp(-(xx**2 + yy**2) / (2 * 0.05**2))
    sb /= sb.sum() * asset_scale**2
    asset_path = tmp_path / 'synthetic.npz'
    np.savez(
        asset_path, sb=sb, pixel_scale_arcsec=np.asarray(asset_scale, dtype=np.float64),
        metadata_json=np.asarray(json.dumps({'format_version': 1, 'provenance': {}})),
    )
    intrinsic_flux = 7.4
    theta_e = 2.0
    beta = np.hypot(yy, 1.0 + xx)
    expected_flux = intrinsic_flux * np.sum(sb * asset_scale**2 * (2 * theta_e / beta))
    lensing = {
        'grid': {'shape': [400, 400], 'pixel_scale': 0.02},
        'lens_galaxy': {
            'redshift': 0.2,
            'mass': {'type': 'Isothermal', 'centre': [0.0, 0.0],
                     'ell_comps': [0.0, 0.0], 'einstein_radius': theta_e},
        },
        'source_galaxy': {
            'redshift': 0.6,
            'light': {'type': 'Image', 'asset_path': str(asset_path),
                      'centre': [0.0, 1.0], 'rotation_deg': 0.0,
                      'total_flux': intrinsic_flux, 'flux_scale': 1.0, 'size_scale': 1.0},
        },
        'subhalo': {'enabled': False},
        'cosmology': 'Planck15',
    }
    result = generate_lensing_system(lensing, seed=5, run_name='synthetic-magnification')
    image = np.asarray(result.image)
    assert np.all(image[[0, -1], :] == 0.0)
    assert np.all(image[:, [0, -1]] == 0.0)
    assert image.sum() * 0.02**2 == pytest.approx(expected_flux, rel=0.01)
    lensing['source_galaxy']['light']['total_flux'] *= 2
    doubled = generate_lensing_system(lensing, seed=5, run_name='double-flux')
    np.testing.assert_array_equal(np.asarray(doubled.image), 2 * image)
