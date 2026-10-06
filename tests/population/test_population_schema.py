"""The actual owning section table exposes nested keys to DOCS/BATCH consumers."""

import pytest

from hwoslaps.config.checks import render_reference
from hwoslaps.population import PopulationError, PopulationSpec
from hwoslaps.population.sampling import POPULATION_TABLE


def test_owning_population_schema_renders_nested_accepted_fields():
    rendered=render_reference([("population",POPULATION_TABLE)])
    for name in ("variables","copulas","catalog","columns","text_columns","bind","max_attempts",
                 "low","high","mean","std","median","sigma_ln","weights","of","axis_ratio",
                 "angle_deg","radius","centre_y","centre_x","magnitude","strength","order","function","inputs","var","correlation"):
        assert name in rendered
    mapping={"variables":{"x":{"kind":"uniform","low":0.,"high":1.}},"bind":{"scene.lens.redshift":"x"}}
    assert POPULATION_TABLE.read(mapping,"population")["variables"]["x"]["high"]==1.
    spec=PopulationSpec.from_mapping(mapping)
    assert PopulationSpec.from_mapping(spec.to_mapping()).to_mapping()==spec.to_mapping()
    assert PopulationSpec.from_mapping(spec.to_mapping()).digest()==spec.digest()


@pytest.mark.parametrize("edit",[
    {"typo":1},{"max_attempts":False},{"max_attempts":0},{"bind":{}},
    {"copulas":{"bad":{"variables":["x","x"],"correlation":[[1.,0.],[0.,1.]]}}},
    {"copulas":{"bad":{"variables":["x","y"],"correlation":[[True,0.],[0.,True]]}}},
    {"copulas":{"bad":{"variables":["x","y"],"correlation":[[1.,.1],[.2,1.]]}}},
    {"copulas":{"bad":{"variables":["x","y"],"correlation":[[1.,1.],[1.,1.]]}}},
])
def test_population_section_schema_and_copula_domains_are_strict(edit):
    mapping={"variables":{"x":{"kind":"uniform","low":0.,"high":1.},
                          "y":{"kind":"uniform","low":0.,"high":1.}},"bind":{"scene.lens.redshift":"x"},**edit}
    with pytest.raises(PopulationError):PopulationSpec.from_mapping(mapping)
