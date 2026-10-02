"""Physical scaling and matched limits of configurable PSF errors."""

from copy import deepcopy

import numpy as np
import pytest
import yaml

from hwoslaps.psf.mismatch import build_psf_mismatch_spec
from test_psf_mismatch import compact_config, prior_table  # noqa: F401
from test_fisher_grid_map import grid_setup  # noqa: F401


def _flatten_draw(draw):
    return np.asarray([
        *[float(draw['segment_hexikes'][segment][mode])
          for segment in sorted(draw['segment_hexikes'])
          for mode in sorted(draw['segment_hexikes'][segment])],
        *[float(draw['global_zernikes'][mode]) for mode in sorted(draw['global_zernikes'])],
    ])


def test_paired_direction_draws_scale_linearly(compact_config):
    low_config = deepcopy(compact_config)
    high_config = deepcopy(compact_config)
    low_config['modeling']['fit_psf']['delta']['amplitude_rms_nm'] = 2.0
    high_config['modeling']['fit_psf']['delta']['amplitude_rms_nm'] = 10.0
    low = build_psf_mismatch_spec(low_config)
    high = build_psf_mismatch_spec(high_config)
    np.testing.assert_allclose(_flatten_draw(high.draw_aberrations), 5 * _flatten_draw(low.draw_aberrations), rtol=1e-12, atol=1e-12)
    assert low.measured_draw_rms_nm == pytest.approx(2.0, rel=1e-9)
    assert high.measured_draw_rms_nm == pytest.approx(10.0, rel=1e-9)


def test_delta_zero_grid_map_is_the_matched_limit(grid_setup, tmp_path):
    from hwoslaps.modeling.fisher_detector import FisherDetector

    prior_path = tmp_path / 'prior.yaml'
    prior_path.write_text(yaml.safe_dump({
        'name': 'synthetic', 'segment_variance_fraction': 0.4,
        'global_weights': {4: 1.0, 5: 0.5}, 'segment_weights': {1: 1.0, 2: 0.5},
    }))
    config = deepcopy(grid_setup['config'])
    config['modeling']['fit_psf'] = {
        'mode': 'delta', 'delta': {'prior_table': str(prior_path), 'seed': 43,
                                 'family': 'combined', 'amplitude_rms_nm': 0.0},
    }
    detector = FisherDetector(
        observation_baseline=grid_setup['observation_baseline'],
        lensing_baseline=grid_setup['lensing_baseline'], psf_data=grid_setup['psf_data'],
        full_config=config, fisher_config=deepcopy(config['modeling']['fisher']),
    )
    delta_map = detector.compute_grid_map()
    matched_map = grid_setup['grid_map']
    np.testing.assert_array_equal(delta_map.detectable_mask_2d, matched_map.detectable_mask_2d)
    np.testing.assert_array_equal(delta_map.mismatch_detectable_mask_2d, delta_map.detectable_mask_2d)
    np.testing.assert_allclose(delta_map.q_asimov_2d, matched_map.q_asimov_2d, rtol=1e-10)
    np.testing.assert_allclose(delta_map.q_mismatch_2d, matched_map.q_asimov_2d, rtol=1e-10)
    assert delta_map.num_false_positive == 0
