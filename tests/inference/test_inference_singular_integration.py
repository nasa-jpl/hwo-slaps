"""Public fixed-template likelihoods share the truth's integrated native image."""
import numpy as np
import pytest

pytestmark = pytest.mark.backend


@pytest.mark.parametrize('kind', ['PointMass', 'SIS'])
@pytest.mark.parametrize('sampling', [1, 2])
def test_numpy_fixed_template_truth_uses_shared_integrated_operator(minimal_mapping, kind, sampling):
    from hwoslaps.fisher.api import Execution, prepare_forecast
    from hwoslaps.inference.api import prepare_case
    from hwoslaps.inference.settings import FitSpec
    from hwoslaps.scene.singular_integration import SingularIntegrationCapabilityError
    from hwoslaps.simulation import simulate
    minimal_mapping['scene']['subhalo'] = {'type':kind}
    minimal_mapping['scene']['grid']['over_sample_size'] = sampling
    with prepare_forecast(minimal_mapping, execution=Execution(engine='reference')) as prepared:
        coordinate = np.asarray(prepared.scene.grid.over_sampled)
        position = tuple(coordinate[np.argmin(np.sum((coordinate-[0., .8])**2, axis=1))])
        halo = prepared.hypothesis(1.e8, position)
        observation = simulate(prepared, subhalo=halo, noise_seed=None)
        case = prepare_case(prepared, halo, observation, fit=FitSpec(mode='fixed_template'), use_jax=False)
        fit = case.fit_at('subhalo', case.truth_vector('subhalo'))
        assert np.isfinite(fit.log_likelihood)
        assert fit.chi_squared < 1.e-14
        with pytest.raises(SingularIntegrationCapabilityError, match='JAX nonlinear'):
            prepare_case(prepared, halo, observation, fit=FitSpec(mode='fixed_template'), use_jax=True)
