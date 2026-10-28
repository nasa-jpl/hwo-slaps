"""Exact native/oversampled NFW and TNFW values across accelerated engines."""
import numpy as np
import pytest

pytestmark = pytest.mark.backend


@pytest.mark.parametrize('kind', ['NFW','TNFW'])
@pytest.mark.parametrize('sampling', [1,2])
def test_removable_halo_centres_match_reference(minimal_mapping, kind, sampling):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
    model = {'type':kind,'concentration':{'kind':'fixed','value':20.}}
    if kind == 'TNFW':
        model['truncation'] = {'kind':'tau','tau':3.}
    minimal_mapping['scene']['subhalo'] = model
    minimal_mapping['scene']['grid']['over_sample_size'] = sampling
    with prepare_forecast(minimal_mapping) as reference, prepare_forecast(
            minimal_mapping,execution=Execution(engine='jax')) as accelerated:
        coordinates = np.asarray(reference.scene.grid.over_sampled)
        centre = coordinates[np.argmin(np.sum((coordinates-[0.,.8])**2,axis=1))]
        positions = np.stack([centre,centre+[0.,1e-12],centre+[1e-9,0.]])
        expected = forecast(reference,masses_msun=[1e8],positions=positions)
        actual = forecast(accelerated,masses_msun=[1e8],positions=positions)
        assert np.all(np.isfinite(expected.q_asimov))
        np.testing.assert_allclose(actual.q_asimov,expected.q_asimov,rtol=2e-11,atol=1e-11)
        assert accelerated.engine.describe()['kind'] == 'jax'


@pytest.mark.xtx_gpu
@pytest.mark.parametrize('kind', ['NFW','TNFW'])
@pytest.mark.parametrize('sampling', [1,2])
def test_cuda_removable_halo_centres_match_reference(minimal_mapping,kind,sampling):
    import jax
    assert jax.devices()[0].platform == 'gpu'
    test_removable_halo_centres_match_reference(minimal_mapping,kind,sampling)
