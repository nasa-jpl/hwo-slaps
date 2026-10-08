"""Inverse-CDF laws against independent distribution CDFs and strict domain oracles."""

import numpy as np
import pytest
from scipy import stats

from hwoslaps.population import PopulationError, PopulationSpec, sample_population


def spec(variable):
    return {"variables":{"x":variable},"bind":{"scene.lens.redshift":"x"}}


@pytest.mark.parametrize("law,cdf",[
    ({"kind":"uniform","low":-2.,"high":3.},stats.uniform(loc=-2,scale=5).cdf),
    ({"kind":"log_uniform","low":.1,"high":100.},stats.loguniform(.1,100).cdf),
    ({"kind":"normal","mean":3.,"std":2.},stats.norm(loc=3,scale=2).cdf),
    ({"kind":"truncated_normal","mean":.5,"std":.2,"low":.2,"high":1.},stats.truncnorm(-1.5,2.5,loc=.5,scale=.2).cdf),
    ({"kind":"truncated_normal","mean":0.,"std":1.,"low":8.,"high":9.},stats.truncnorm(8,9).cdf),
    ({"kind":"lognormal","median":2.,"sigma_ln":.4},stats.lognorm(.4,scale=2).cdf),
    ({"kind":"truncated_lognormal","median":2.,"sigma_ln":.4,"low":.5,"high":4.},
     lambda x:stats.truncnorm(np.log(.5/2)/.4,np.log(4/2)/.4).cdf(np.log(x/2)/.4)),
])
def test_draws_follow_their_distribution(law,cdf):
    values=np.array([row["x"] for row in sample_population(spec(law),4000,seed=20261006)])
    assert np.all(np.isfinite(values))
    if "low" in law:assert np.all(values>=law["low"]) and np.all(values<=law["high"])
    assert stats.kstest(cdf(values),"uniform").pvalue>1e-3
    if law.get("low")==8.:assert len(np.unique(values))>3900


def test_weighted_choice_frequencies_follow_declared_weights():
    rows=sample_population(spec({"kind":"choice","values":["a","b","c"],"weights":[1,2,7]}),4000,seed=20261006)
    counts=[sum(row["x"]==label for row in rows) for label in ("a","b","c")]
    assert stats.chisquare(counts,np.array([.1,.2,.7])*4000).pvalue>1e-3


@pytest.mark.parametrize("law",[
    {"kind":"unknown"},{"kind":"uniform","low":0.,"high":1.,"typo":True},{"kind":"normal","mean":0.},
    {"kind":"uniform","low":1.,"high":1.},{"kind":"normal","mean":0.,"std":0.},
    {"kind":"choice","values":[1,2],"weights":[1,-1]},
    {"kind":"normal","mean":"1e7","std":1.},{"kind":"uniform","low":False,"high":1.},
    {"kind":"log_uniform","low":0.,"high":1.},
])
def test_invalid_specs_raise_before_a_draw(law):
    with pytest.raises(PopulationError,match="x"):
        PopulationSpec.from_mapping(spec(law))


@pytest.mark.parametrize("law",[
    {"kind":"lognormal","median":1e308,"sigma_ln":10.},
    {"kind":"uniform","low":-1e308,"high":1e308},
])
def test_unrepresentable_draws_raise_with_member_and_variable(law):
    with pytest.raises(PopulationError,match=r"member \d+, variable x: distribution result:.*got .*\binf\b"):
        sample_population(spec(law),20,seed=1)


def test_reference_parameters_get_the_same_domain_check_at_draw_time():
    mapping={"variables":{"bad":{"kind":"constant","value":-1.},
                          "x":{"kind":"normal","mean":0.,"std":{"var":"bad"}}},
             "bind":{"scene.lens.redshift":"x"}}
    with pytest.raises(PopulationError,match="member0|member 0"):
        sample_population(mapping,1,seed=3)


def test_open_uniforms_match_the_documented_integer_lattice():
    from hwoslaps.population.distributions import open_uniforms
    a=np.random.default_rng(11);b=np.random.default_rng(11)
    actual=open_uniforms(a,4000)
    expected=(2*b.integers(0,2**52,size=4000,dtype=np.int64)+1)*2.**-53
    np.testing.assert_array_equal(actual,expected)
    assert np.all((actual>=2.**-53)&(actual<=1-2.**-53))
