"""Named independent streams, copula algebra and actual conversion/function consumers."""

from copy import deepcopy
import math

import numpy as np
import pytest
from scipy import stats

from hwoslaps.population import PopulationError, PopulationSpec, sample_population


def mapping(variables):return {"variables":variables,"bind":{"scene.lens.redshift":next(iter(variables))}}


def test_member_draws_are_invariant_to_partition_count_and_unrelated_variables():
    variables={"u1":{"kind":"uniform","low":0.,"high":1.},"u2":{"kind":"uniform","low":0.,"high":1.}}
    original=mapping(variables)
    whole=sample_population(original,4000,seed=31)
    chunked=sample_population(original,1253,seed=31)+sample_population(original,2747,seed=31,start=1253)
    assert whole==chunked
    assert whole[:7]==sample_population(original,7,seed=31)
    assert whole==sample_population(mapping(dict(reversed(list(variables.items())))),4000,seed=31)
    extended={**variables,"extra":{"kind":"normal","mean":0.,"std":1.}}
    assert [{name:row[name] for name in variables} for row in sample_population(mapping(extended),4000,seed=31)]==whole
    assert whole!=sample_population(original,4000,seed=32)
    u1,u2=np.array([[row["u1"],row["u2"]] for row in whole]).T
    assert np.all(u1!=u2)
    assert abs(np.corrcoef(u1,u2)[0,1])<5/math.sqrt(4000-2)


def test_conditional_bounds_give_the_conditional_distribution(tmp_path,monkeypatch):
    hook=tmp_path/"population_test_hooks.py"
    hook.write_text("def lower(z_lens):\n    return max(1.,z_lens+.3)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    variables={"zl":{"kind":"uniform","low":.2,"high":1.5},
               "floor":{"kind":"function","function":"population_test_hooks:lower","inputs":{"z_lens":{"var":"zl"}}},
               "zs":{"kind":"truncated_normal","mean":2.,"std":.7,"low":{"var":"floor"},"high":4.}}
    rows=sample_population(mapping(variables),4000,seed=32)
    zl,floor,zs=np.array([[row[name] for name in ("zl","floor","zs")] for row in rows]).T
    np.testing.assert_array_equal(floor,np.maximum(1.,zl+.3))
    assert np.all((zs>=floor)&(zs<=4.))
    probabilities=stats.truncnorm.cdf(zs,(floor-2.)/.7,(4.-2.)/.7,loc=2.,scale=.7)
    assert stats.kstest(probabilities,"uniform").pvalue>1e-3


@pytest.mark.parametrize("normal",[True,False])
def test_gaussian_copula_reproduces_declared_dependence(normal):
    variables=({"x":{"kind":"normal","mean":2.,"std":3.},"y":{"kind":"normal","mean":-1.,"std":2.}}
               if normal else {"x":{"kind":"lognormal","median":2.,"sigma_ln":.5},"y":{"kind":"uniform","low":0.,"high":1.}})
    spec=mapping(variables);rho=-.35
    spec["copulas"]={"pair":{"variables":["x","y"],"correlation":[[1.,rho],[rho,1.]]}}
    rows=sample_population(spec,20000,seed=11)
    values=np.array([[row["x"],row["y"]] for row in rows])
    if normal:
        expected=np.array([[9.,6*rho],[6*rho,4.]])
        standard=np.sqrt((expected**2+np.outer(np.diag(expected),np.diag(expected)))/(len(rows)-1))
        assert np.all(np.abs(np.cov(values.T)-expected)<5*standard)
    else:
        expected=6/np.pi*np.arcsin(rho/2)
        assert abs(stats.spearmanr(values[:,0],values[:,1]).statistic-expected)<5/math.sqrt(len(rows)-2)


@pytest.mark.parametrize("variable,expected",[
    ({"kind":"vector","of":[.1,.2,.3]},(.1,.2,.3)),
    ({"kind":"polar_offset","radius":1.,"angle_deg":90.},(1.,0.)),
    ({"kind":"ell_comps","axis_ratio":.5,"angle_deg":45.},(1/3,0.)),
    ({"kind":"shear_components","magnitude":.1,"angle_deg":22.5},(.1/math.sqrt(2),.1/math.sqrt(2))),
    ({"kind":"multipole_components","strength":.02,"angle_deg":30.,"order":3},(.02,0.)),
])
def test_conversion_derivations_have_hand_oracles(variable,expected):
    value=sample_population(mapping({"x":variable}),1,seed=2)[0]["x"]
    np.testing.assert_allclose(value,expected,rtol=1e-15,atol=1e-15)


def test_function_hook_calls_actual_keywords_and_validates_output(tmp_path,monkeypatch):
    module=tmp_path/"population_vector_hooks.py"
    module.write_text("def pair(a,b):\n    return (a+b,a-b)\ndef bad(a):\n    return [float('nan'), float('inf'), [1., 2.], ((1., 2.),), True, '1', None][int(a)]\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    values={"u":{"kind":"constant","value":[2.,3.]},
            "x":{"kind":"function","function":"population_vector_hooks:pair","inputs":{"a":{"var":"u[0]"},"b":{"var":"u[1]"}}}}
    assert sample_population(mapping(values),1,seed=2)[0]["x"]==(5.,-1.)
    for index in range(7):
        values["x"] = {"kind": "function", "function": "population_vector_hooks:bad", "inputs": {"a": index}}
        with pytest.raises(PopulationError, match="variable x.*output"):
            sample_population(mapping(values), 1, seed=2)


@pytest.mark.parametrize("variables",[
    {"x":{"kind":"normal","mean":{"var":"y"},"std":1.},"y":{"kind":"constant","value":1.}},
    {"v":{"kind":"vector","of":[1.,2.]},"x":{"kind":"normal","mean":{"var":"v"},"std":1.}},
    {"v":{"kind":"vector","of":[1.,2.]},"x":{"kind":"normal","mean":{"var":"v[2]"},"std":1.}},
    {"x":{"kind":"maximum","of":[1.,2.]}},
])
def test_reference_order_shape_and_removed_recipes_are_strict(variables):
    with pytest.raises(PopulationError):PopulationSpec.from_mapping(mapping(variables))


def test_sampling_leaves_global_state_and_inputs_untouched():
    values={"x":{"kind":"constant","value":{"items":[1]}},"y":{"kind":"choice","values":[{"id":"a"}],"weights":[1]}}
    spec=mapping(values);before=deepcopy(spec)
    state=np.random.get_state()
    rows=sample_population(spec,3,seed=4)
    after=np.random.get_state()
    assert state[0]==after[0] and np.array_equal(state[1],after[1]) and state[2:]==after[2:]
    rows[0]["x"]["items"].append(2);rows[0]["y"]["id"]="changed"
    assert rows[1]=={"x":{"items":[1]},"y":{"id":"a"}} and spec==before
