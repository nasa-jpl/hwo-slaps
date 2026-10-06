"""Actual effective engine configuration bindings, deterministic identities and redraws."""

from copy import deepcopy
import hashlib
import math

import numpy as np
import pytest

from hwoslaps.config.schema import parse_config
from hwoslaps.population import PopulationError, PopulationSpec, iter_population_members


def spec(variable,bind):return {"variables":{"x":variable},"bind":bind}


def test_bind_writes_effective_scalar_vector_block_and_list_paths(minimal_mapping):
    base=parse_config(minimal_mapping)
    examples=[
        ({"kind":"constant","value":.3},{"scene.lens.redshift":"x"},lambda c:c.scene.lens.redshift,.3),
        ({"kind":"vector","of":[.01,.02]},{"scene.lens.mass.mass.centre":"x"},lambda c:c.scene.lens.mass[0].values["centre"],(.01,.02)),
        ({"kind":"vector","of":[.01,.02]},{"scene.lens.mass.mass.centre.0":"x[1]"},lambda c:c.scene.lens.mass[0].values["centre"][0],.02),
        ({"kind":"constant","value":{"rate_e_per_s":2.}},{"observation.sky":"x"},lambda c:c.observation.sky["rate_e_per_s"],2.),
        ({"kind":"constant","value":.2},{"forecast.positions.positions_yx.0.1":"x"},lambda c:c.forecast.positions.positions_yx[0][1],.2),
    ]
    for variable,bind,read,expected in examples:
        member=next(iter_population_members(base,spec(variable,bind),1,seed=2))
        assert read(member.config)==expected
    assert base.to_mapping()==parse_config(minimal_mapping).to_mapping()


@pytest.mark.parametrize("bind",[
    {"scene.lens.typo":"x"},{"seed":"x"},{"run_name":"x"},
    {"scene.lens.mass.mass":"x","scene.lens.mass.mass.centre":"x"},
    {"forecast.positions.positions_yx.99":"x"},
])
def test_binding_refuses_typo_reserved_overlap_and_index(bind,minimal_mapping):
    with pytest.raises(PopulationError,match="bind"):
        list(iter_population_members(parse_config(minimal_mapping),spec({"kind":"constant","value":.3},bind),1,seed=2))


def test_member_identity_is_cantor_and_base_is_unchanged(minimal_mapping):
    base=parse_config(minimal_mapping);before=deepcopy(base.to_mapping())
    mapping=spec({"kind":"uniform","low":.2,"high":.4},{"scene.lens.redshift":"x"})
    members=list(iter_population_members(base,mapping,3,seed=5,start=7))
    for member in members:
        expected=(5+member.index)*(5+member.index+1)//2+member.index
        assert member.run_name==f"system_{member.index:06d}"
        assert member.seed==member.config.seed==expected
        assert member.attempt==0 and member.to_mapping()["config_digest"]==member.config.digest()
    assert members[0].to_mapping()==next(iter_population_members(base,mapping,1,seed=5,start=7)).to_mapping()
    assert base.to_mapping()==before


def independent_e(seed,index,attempt):
    digest=hashlib.sha256(b"population/e").digest()
    words=tuple(int.from_bytes(digest[i:i+4],"little") for i in range(0,32,4))
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed,spawn_key=(index,attempt,*words))))
    u=(2*int(rng.integers(0,2**52,dtype=np.int64))+1)*2.**-53
    return 1.2*u


def test_rejection_redraws_invalid_members_and_reports_original_error(minimal_mapping):
    base=parse_config(minimal_mapping).replace({"scene":{"lens":{"mass":{"mass":{"ell_comps":[0.,0.]}}}}})
    mapping={"variables":{"e":{"kind":"uniform","low":0.,"high":1.2}},
             "bind":{"scene.lens.mass.mass.ell_comps.0":"e"},"max_attempts":1}
    invalid=next(i for i in range(100) if independent_e(3,i,0)>=.999)
    with pytest.raises(PopulationError,match=f"member {invalid}.*last error.*ell_comps"):
        list(iter_population_members(base,mapping,1,seed=3,start=invalid))
    mapping["max_attempts"]=50
    members=list(iter_population_members(base,mapping,100,seed=3))
    for member in members:
        expected=next(a for a in range(50) if independent_e(3,member.index,a)<.999)
        assert member.attempt==expected
        assert member.values["e"]==independent_e(3,member.index,expected)
        assert member.config.scene.lens.mass[0].values["ell_comps"][0]==member.values["e"]
