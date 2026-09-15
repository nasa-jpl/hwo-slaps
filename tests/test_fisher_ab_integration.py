"""Cross-path numerical and runner contracts for the combined A/B candidate."""
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_fisher_aperture_perimeter import jax_ladder_fixture, _grid_stub
from test_run_ladder import _extraction, _staged_config
import run_ladder as runner


@pytest.mark.parametrize('batch_size', [7, 8])
def test_compact_affine_matches_dense_general_interpolation(
    jax_ladder_fixture, monkeypatch, batch_size,
):
    """Compare consumed q values across BOTH changes, including full/tail batches."""
    import hwoslaps.modeling.fisher_grid_jax as grid
    detector, _ = jax_ladder_fixture
    selection = detector.prepare_ladder_grid_selection((0., 0.), .1)
    init = grid.JaxGridTemplateEngine.__init__
    validator = grid._affine_log_grid_parameters
    constructions = []

    def build(engine, **kwargs):
        constructions.append(np.asarray(kwargs['candidate_positions']).copy())
        kwargs['batch_size'] = batch_size
        init(engine, **kwargs)

    monkeypatch.setattr(grid.JaxGridTemplateEngine, '__init__', build)
    monkeypatch.setattr(grid, '_affine_log_grid_parameters', lambda knots: None)
    detector._jax_grid_engine = None
    dense = detector.compute_grid_map()
    dense_engine = detector._jax_grid_engine
    assert dense_engine._log_grid_origin is None
    radii = np.asarray(dense_engine._radii).copy()
    alpha = np.asarray(dense_engine._alpha_radial).copy()
    monkeypatch.setattr(grid, '_affine_log_grid_parameters', validator)
    detector._jax_grid_engine = None
    compact = detector.compute_ladder_summary(selection)
    fast = detector._jax_grid_engine
    assert fast._log_grid_origin is not None
    np.testing.assert_array_equal(constructions[0], constructions[1])
    assert len(constructions[1]) == selection.full_grid_node_count
    np.testing.assert_array_equal(fast._radii, radii)
    np.testing.assert_array_equal(np.asarray(fast._alpha_radial), alpha)
    assert fast._radial_r_max == dense_engine._radial_r_max
    ij = np.asarray(selection.node_indices)
    reference = dense.q_asimov_2d[ij[:, 0], ij[:, 1]]
    np.testing.assert_allclose(compact.q_asimov_by_position, reference, rtol=1e-9, atol=0)
    np.testing.assert_array_equal(compact.q_asimov_by_position[reference == 0], 0.)
    np.testing.assert_array_equal(
        compact.detectable_by_position, dense.detectable_mask_2d[ij[:, 0], ij[:, 1]],
    )
    batches = list(fast._position_batches(selection.positions_yx))
    sizes = [len(batch) for batch in batches]
    assert sum(sizes) == selection.selected_node_count
    assert sizes[0] == batch_size
    # 21 consumed nodes: 7 has full batches; 8 has a final batch of 5.
    assert sizes[-1] == (selection.selected_node_count % batch_size or batch_size)


@pytest.mark.parametrize('tier,explicit', [('selected', False), ('full_pool', True)])
def test_runner_writes_readable_artifacts_and_calls_observer(tmp_path, monkeypatch, tier, explicit):
    """Exercise the real loop/writers with controlled detector values and real tier gate."""
    import hwoslaps.config.validation as validation
    import hwoslaps.psf as psf
    import hwoslaps.psf.utils as psf_utils
    extraction = _extraction()
    config = _staged_config(extraction, tier=tier)
    config['plotting']['output_dir'] = str(tmp_path)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(runner, '_enable_float64', lambda: None)
    monkeypatch.setattr(runner, '_enable_jax_compilation_cache', lambda: None)
    monkeypatch.setattr(runner, '_verify_code_revision', lambda c: c['stage0']['code_revision'])
    monkeypatch.setattr(runner, '_verify_source_asset', lambda c: c['stage0']['source_asset_sha256'])
    monkeypatch.setattr(runner, '_extract_theta_e_eff', lambda c: extraction)
    monkeypatch.setattr(validation, 'validate_or_raise', lambda c: None)
    monkeypatch.setattr(psf, 'generate_psf_system', lambda *a, **k: SimpleNamespace(kernel=None))
    monkeypatch.setattr(psf_utils, 'pyauto_kernel_native', lambda k: np.ones((999, 999)))
    monkeypatch.setattr(runner, '_verify_psf_rms', lambda p: 35.)
    selection = _grid_stub().prepare_ladder_grid_selection((0., 0.), 1.)
    detector = SimpleNamespace(
        prepare_ladder_grid_selection=lambda *a: selection,
        compute_ladder_summary=lambda s: SimpleNamespace(
            q_max=1., detectable_area_arcsec2=0., aperture_fraction=0., perimeter_clipped=False),
    )
    monkeypatch.setattr(runner, '_build_detector', lambda *a: detector)
    visited = []
    monkeypatch.setattr(runner, '_point_detector_at_rung', lambda d, m: visited.append(m))
    events = []
    kwargs = {'observer': lambda event, **data: events.append((event, data))}
    if explicit:
        kwargs['allowed_tiers'] = (*runner.TIERS, 'full_pool')
    runner.main([str(path)], **kwargs)
    output = runner._output_dir(config)
    with np.load(output/runner.ARTIFACT_NAME, allow_pickle=False) as artifact:
        np.testing.assert_array_equal(artifact['rung_logm'], visited)
        assert artifact['rung_logm'].size == 15
    assert events == [('initialization', {})] + [('rung', {'logm': m}) for m in visited]
    sidecar = yaml.safe_load((output/runner.EXECUTION_PROVENANCE_NAME).read_text())
    assert sidecar['actual_evaluated_node_count_per_rung'] == selection.selected_node_count
    assert sidecar['full_grid_node_count'] == selection.full_grid_node_count
    assert sidecar['skipped_auxiliary_node_count'] == selection.full_grid_node_count-selection.selected_node_count
    assert yaml.safe_load((output/'provenance.yaml').read_text())
    assert not list(output.glob('*.tmp'))


def test_default_main_rejects_full_pool_before_initializing(tmp_path):
    config = _staged_config(tier='full_pool')
    config['plotting']['output_dir'] = str(tmp_path)
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    assert inspect.signature(runner.main).parameters['allowed_tiers'].default == runner.TIERS
    with pytest.raises(ValueError, match='not one of'):
        runner.main([str(path)])
