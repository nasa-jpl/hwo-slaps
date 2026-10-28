"""The singular-family operator is shared by smooth controls and halo templates."""
import numpy as np
import pytest

pytestmark = pytest.mark.backend


@pytest.mark.parametrize('kind', ['PointMass', 'SIS'])
def test_singular_hypothesis_signal_vanishes_with_mass_and_reports_cpu_route(minimal_mapping, kind):
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast
    minimal_mapping['scene']['subhalo'] = {'type':kind}
    minimal_mapping['scene']['grid'] = {'shape':[21,21], 'pixel_scale_arcsec':.1, 'over_sample_size':1}
    minimal_mapping['psf']['truth']['pixel_scale_arcsec'] = .1
    minimal_mapping['forecast']['nuisances'] = {'fixed':['*'], 'background_offset':False}
    with prepare_forecast(minimal_mapping, execution=Execution(engine='jax')) as prepared:
        actual = forecast(prepared, masses_msun=[1.e4, 1.e2, 1.e-2], positions=[[0.,.8]])
        q = actual.q_asimov[:,0]
        assert np.all(np.isfinite(q))
        assert q[-1] < 1.e-7
        assert q[1] < q[0]
        assert q[-1] < q[1]
        execution = prepared.engine.describe()
        assert execution['kind'] == 'reference'
        assert execution['requested_kind'] == 'jax'
        assert execution['jax_accelerated'] is False
        assert execution['device'] == 'cpu'
        assert execution['native_binning'] == 'adaptive-pixel-integral'
        assert execution['integration_rtol'] == 2.e-6
        assert execution['integration_max_cells'] == 1024
