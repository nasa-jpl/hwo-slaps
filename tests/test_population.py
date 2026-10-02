"""Generic population reproducibility and configuration contracts."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hwoslaps.population import iter_population_configs, sample_population


PARAMETERS = {
    "lens.mass": {"distribution": "log_uniform", "low": 1e6, "high": 1e9},
    "source.template": {"distribution": "choice", "values": ["clumpy", "smooth"]},
    "source.z": {"distribution": "truncated_normal", "mean": 0.7, "std": 0.2, "low": 0.4, "high": 1.5},
}


def test_chunking_parameter_order_and_extension_preserve_samples():
    whole = sample_population(PARAMETERS, 20, seed=31)
    chunks = sample_population(PARAMETERS, 7, seed=31) + sample_population(PARAMETERS, 13, seed=31, start=7)
    assert whole == chunks
    assert whole == sample_population(dict(reversed(list(PARAMETERS.items()))), 20, seed=31)
    extended = {**PARAMETERS, "exposure": {"distribution": "uniform", "low": 100, "high": 2000}}
    assert [{k: row[k] for k in PARAMETERS} for row in sample_population(extended, 20, seed=31)] == whole
    assert whole[:5] == sample_population(PARAMETERS, 5, seed=31)
    assert whole != sample_population(PARAMETERS, 20, seed=32)


def test_declared_support_and_global_rng_isolation():
    np.random.seed(41)
    state = np.random.get_state()
    rows = sample_population(PARAMETERS, 100, seed=11)
    assert all(1e6 <= row["lens.mass"] <= 1e9 for row in rows)
    assert all(0.4 <= row["source.z"] <= 1.5 for row in rows)
    assert all(row["source.template"] in {"clumpy", "smooth"} for row in rows)
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def test_config_population_is_independent_and_rejects_typos():
    base = {"run_name": "base", "global_seed": 3, "source": {"asset": "a", "size": 0.1}}
    before = deepcopy(base)
    fields = {"source.size": {"distribution": "uniform", "low": 0.1, "high": 0.5}}
    configs = list(iter_population_configs(base, fields, 4, seed=2, validate=False))
    assert len({c["global_seed"] for c in configs}) == 4
    assert configs[0]["run_name"] == "system_000000"
    assert configs[0] == next(iter_population_configs(base, fields, 1, seed=2, validate=False))
    configs[0]["source"]["asset"] = "changed"
    assert configs[1]["source"]["asset"] == "a" and base == before
    with pytest.raises(ValueError, match="existing"):
        list(iter_population_configs(
            base, {"source.typo": {"distribution": "constant", "value": 1}},
            1, seed=2, validate=False,
        ))


def test_choices_and_constants_are_copied_not_shared():
    fields = {"template": {"distribution": "choice", "values": [{"id": "a"}], "weights": [1]},
              "fixed": {"distribution": "constant", "value": {"x": [1]}}}
    rows = sample_population(fields, 2, seed=4)
    rows[0]["template"]["id"] = "b"
    rows[0]["fixed"]["x"].append(2)
    assert rows[1] == {"template": {"id": "a"}, "fixed": {"x": [1]}}
    assert fields["template"]["values"] == [{"id": "a"}]


@pytest.mark.parametrize("spec", [
    {"distribution": "uniform", "low": 1, "high": 1},
    {"distribution": "log_uniform", "low": 0, "high": 1},
    {"distribution": "normal", "mean": 0, "std": -1},
    {"distribution": "choice", "values": [1, 2], "weights": [1, -1]},
    {"distribution": "normal", "mean": 0, "std": 1, "typo": True},
    {"distribution": "missing"},
])
def test_invalid_distributions_fail_before_drawing(spec):
    with pytest.raises(ValueError):
        sample_population({"field": spec}, 1, seed=3)


def test_reserved_and_overlapping_config_paths_are_rejected():
    base = {"global_seed": 1, "source": {"size": 1}}
    constant = {"distribution": "constant", "value": 1}
    for fields in ({"global_seed": constant}, {"source": constant, "source.size": constant}):
        with pytest.raises(ValueError):
            list(iter_population_configs(base, fields, 1, seed=3, validate=False))


def test_large_population_member_seeds_do_not_collide():
    base = {"global_seed": 0, "run_name": "base", "source": {"size": 1}}
    fields = {"source.size": {"distribution": "constant", "value": 1}}
    # These members collided under the original 32-bit draw for population seed
    # 2.
    first = next(iter_population_configs(base, fields, 1, seed=2, start=110170, validate=False))
    second = next(iter_population_configs(base, fields, 1, seed=2, start=111187, validate=False))
    assert first["global_seed"] != second["global_seed"]
    seeds = {
        next(iter_population_configs(base, fields, 1, seed=s, start=i, validate=False))['global_seed']
        for s in range(3) for i in (110170, 111187)
    }
    assert len(seeds) == 6


@pytest.mark.parametrize("spec", [
    {"distribution": "lognormal", "median": 1e308, "sigma_ln": 10},
    {"distribution": "uniform", "low": -1e308, "high": 1e308},
])
def test_out_of_float_support_draws_fail_with_parameter_context(spec):
    with pytest.raises(ValueError, match="extreme.*(finite|non-finite)"):
        sample_population({"extreme": spec}, 10, seed=1)
