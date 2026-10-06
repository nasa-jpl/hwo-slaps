"""Small real configurations shared only by the batch owner tests."""
from copy import deepcopy

import pytest

from hwoslaps.batch import parse_batch
from hwoslaps.config.loading import merge_configs


@pytest.fixture
def tiny_batch_spec(tmp_path, minimal_mapping):
    def build(**sections):
        base = deepcopy(minimal_mapping)
        base['run_name'] = 'tiny'
        base['scene']['grid']['shape'] = [40, 40]
        base['scene']['lens']['mass']['mass']['einstein_radius'] = .6
        base['scene']['injection'] = {'mass_msun': 1e8, 'position': {'kind': 'direct', 'centre': [.1, .2]}}
        base['forecast']['positions'] = sections.pop('forecast_positions', {
            'kind': 'grid', 'spacing_arcsec': .2, 'half_width_arcsec': .4})
        mapping = {'name': 'tiny', 'seed': 7, 'config': base,
                   'population': {'seed': 3, 'count': 2, 'variables': {
                       'radius': {'kind': 'uniform', 'low': .55, 'high': .65}},
                       'bind': {'scene.lens.mass.mass.einstein_radius': 'radius'}},
                   'forecast': {'masses_msun': [1e8]},
                   'execution': {'devices': 'cpu', 'workers_per_device': 2}}
        for key, value in sections.items():
            mapping[key] = merge_configs(mapping[key], value) if key in ('population', 'execution') and value is not None else value
        return parse_batch(mapping, base_dir=tmp_path)
    return build


@pytest.fixture
def optical_batch_spec(tiny_batch_spec):
    def build(**sections):
        spec = tiny_batch_spec(**sections)
        mapping = spec.to_mapping()
        mapping['config']['scene']['grid']['shape'] = [64, 64]
        optical = {'kind': 'optical', 'pupil': {'kind': 'circular', 'diameter_m': 1., 'pixels': 32,
                   'supersampling': 1}, 'focal_length_m': 20., 'wavelength_nm': 500.,
                   'detector_oversampling': 1, 'kernel_shape': [11, 11], 'wavefront': {}}
        mapping['config']['psf'] = {'truth': optical, 'model': {'kind': 'matched'}}
        return parse_batch(mapping, base_dir=spec.base_dir)
    return build
