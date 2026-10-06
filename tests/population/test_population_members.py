"""Actual effective engine configuration bindings, deterministic identities and redraws."""

from copy import deepcopy
import hashlib
import math

import numpy as np
import pytest

from hwoslaps.config.schema import parse_config
from hwoslaps.population import PopulationError, PopulationSpec, iter_population_members, sample_population


def spec(variable,bind):return {"variables":{"x":variable},"bind":bind}


def test_bind_writes_effective_scalar_vector_block_and_list_paths(minimal_mapping):
    base=parse_config(minimal_mapping)
    examples=[
        ({"kind":"constant","value":.3},{"scene.lens.redshift":"x"},lambda c:c.scene.lens.redshift,.3),
        ({"kind":"vector","of":[.01,.02]},{"scene.lens.mass.mass.centre":"x"},lambda c:c.scene.lens.mass[0].values["centre"],(.01,.02)),
        ({"kind":"vector","of":[.01,.02]},{"scene.lens.mass.mass.centre.0":"x[1]"},lambda c:c.scene.lens.mass[0].values["centre"][0],.02),
        ({"kind":"constant","value":{"rate_e_per_s":2.}},{"observation.sky":"x"},lambda c:c.observation.sky.rate_e_per_s,2.),
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
        assert member.catalog_sha256 is None and member.to_mapping()["catalog_sha256"] is None
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


def test_vector_choice_references_reach_scalar_laws_and_config_binding(minimal_mapping):
    variables = {
        "v": {"kind": "choice", "values": [[.2, .4], [.3, .5]]},
        "x": {"kind": "normal", "mean": {"var": "v[0]"}, "std": .01},
    }
    mapping = {"variables": variables, "bind": {"scene.lens.redshift": "v[0]"}}
    members = list(iter_population_members(parse_config(minimal_mapping), mapping, 40, seed=2))
    centred = deepcopy(mapping)
    centred["variables"]["x"]["mean"] = 0.
    centred_rows = sample_population(centred, 40, seed=2)
    assert {tuple(member.values["v"]) for member in members} == {(.2, .4), (.3, .5)}
    for member, row in zip(members, centred_rows, strict=True):
        assert member.config.scene.lens.redshift == member.values["v"][0]
        assert member.values["x"] == member.values["v"][0] + row["x"]


@pytest.mark.parametrize("values,reference", [
    ([.2, .3], "v[0]"),
    ([[.2, .4], [.3, .5]], "v[2]"),
    ([[.2, .4], [.3, .5]], "v"),
])
def test_choice_known_incompatible_scalar_reference_is_refused(values, reference):
    mapping = {
        "variables": {
            "v": {"kind": "choice", "values": values},
            "x": {"kind": "normal", "mean": {"var": reference}, "std": .01},
        },
        "bind": {"scene.lens.redshift": "x"},
    }
    with pytest.raises(PopulationError, match="reference|indexed"):
        PopulationSpec.from_mapping(mapping)


@pytest.mark.parametrize("paths,first_value", [
    (("scene.lens.mass.mass.centre.0", "scene.lens.mass.mass.centre.00"), .1),
    (("forecast.positions.positions_yx.00", "forecast.positions.positions_yx.0.1"), [.1, .3]),
])
def test_binding_aliases_refuse_equal_or_prefix_targets_before_draw(paths, first_value, minimal_mapping):
    base = parse_config(minimal_mapping)
    before = deepcopy(base.to_mapping())
    mapping = {
        "variables": {
            "a": {"kind": "constant", "value": first_value},
            "b": {"kind": "constant", "value": .2},
        },
        "bind": dict(zip(paths, ("a", "b"), strict=True)),
    }
    spec = PopulationSpec.from_mapping(mapping)
    reversed_mapping = deepcopy(mapping)
    reversed_mapping["bind"] = dict(reversed(list(mapping["bind"].items())))
    reverse_spec = PopulationSpec.from_mapping(reversed_mapping)
    assert spec.digest() == reverse_spec.digest()
    for candidate in (spec, reverse_spec):
        # Even an empty request resolves the base and refuses ambiguous bindings.
        with pytest.raises(PopulationError, match="overlapping resolved targets"):
            list(iter_population_members(base, candidate, 0, seed=2))
        with pytest.raises(PopulationError, match="overlapping resolved targets"):
            list(iter_population_members(base, candidate, 1, seed=2))
    assert base.to_mapping() == before


def test_member_identity_records_catalog_bytes_consumed_during_publication(minimal_mapping, tmp_path, monkeypatch):
    from hwoslaps.population import sampling

    path = tmp_path / "published.csv"
    first = b"redshift\n0.2\n"
    second = b"redshift\n0.3\n"
    path.write_bytes(first)
    mapping = {
        "variables": {},
        "catalog": {"path": str(path), "columns": {"zl": "redshift"}},
        "bind": {"scene.lens.redshift": "zl"},
    }
    spec = PopulationSpec.from_mapping(mapping)
    planned = spec.digest()
    original_load = sampling.load_catalog

    def publish_during_load(catalog_spec):
        path.write_bytes(second)
        try:
            return original_load(catalog_spec)
        finally:
            path.write_bytes(first)

    monkeypatch.setattr(sampling, "load_catalog", publish_during_load)
    member = next(iter_population_members(parse_config(minimal_mapping), spec, 1, seed=2))
    assert member.values["zl"] == member.config.scene.lens.redshift == .3
    assert path.read_bytes() == first and spec.digest() == planned
    expected_sha = hashlib.sha256(second).hexdigest()
    assert member.catalog_sha256 == member.to_mapping()["catalog_sha256"] == expected_sha
    captured = spec.captured_digest(catalog_sha256=member.catalog_sha256)
    assert captured != planned
    path.unlink()
    assert spec.captured_digest(catalog_sha256=expected_sha) == captured
    with pytest.raises(PopulationError, match="catalog_sha256"):
        spec.captured_digest(catalog_sha256=None)

    no_catalog = PopulationSpec.from_mapping({
        "variables": {"x": {"kind": "constant", "value": .2}},
        "bind": {"scene.lens.redshift": "x"},
    })
    assert no_catalog.captured_digest(catalog_sha256=None) == no_catalog.digest()
    with pytest.raises(PopulationError, match="without a catalog"):
        no_catalog.captured_digest(catalog_sha256=expected_sha)
