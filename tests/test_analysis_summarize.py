"""Tests for completeness summaries, Wilson intervals, and aggregation contracts."""

import math

import pytest

from hwoslaps.analysis.aggregate import RESULTS_CSV_COLUMNS, config_hash
from hwoslaps.analysis.summarize import completeness_summary, wilson_interval


def test_wilson_interval_basic_properties():
    fraction, lower, upper = wilson_interval(8, 10)
    assert fraction == pytest.approx(0.8)
    assert 0.0 <= lower < fraction < upper <= 1.0


def test_wilson_interval_edge_cases():
    fraction, lower, upper = wilson_interval(0, 20)
    assert fraction == 0.0
    assert lower == 0.0
    assert upper > 0.0
    fraction, lower, upper = wilson_interval(20, 20)
    assert fraction == 1.0
    assert upper == pytest.approx(1.0)
    assert lower < 1.0
    assert all(math.isnan(val) for val in wilson_interval(0, 0))


def test_completeness_summary_groups_and_counts():
    rows = [
        {"mass": 1.0e7, "detected": True, "q": 12.0},
        {"mass": 1.0e7, "detected": False, "q": 8.0},
        {"mass": 1.0e7, "detected": True, "q": 15.0},
        {"mass": 1.0e8, "detected": True, "q": 40.0},
    ]
    summary = completeness_summary(
        rows, group_cols=["mass"], detection_cols=["detected"], value_cols=["q"]
    )
    assert len(summary) == 2
    low_mass, high_mass = summary
    assert low_mass["mass"] == 1.0e7
    assert low_mass["n"] == 3
    assert low_mass["detected_count"] == 2
    assert low_mass["detected_fraction"] == pytest.approx(2/3)
    assert low_mass["median_q"] == pytest.approx(12.0)
    assert high_mass["detected_fraction"] == 1.0


def test_completeness_summary_wilson_bounds_match_direct_call():
    rows = [{"group": "a", "detected": idx < 7} for idx in range(10)]
    (record,) = completeness_summary(rows, group_cols=["group"], detection_cols=["detected"])
    fraction, lower, upper = wilson_interval(7, 10)
    assert record["detected_fraction"] == pytest.approx(fraction)
    assert record["detected_wilson_lo_1sigma"] == pytest.approx(lower)
    assert record["detected_wilson_hi_1sigma"] == pytest.approx(upper)


def test_config_hash_is_stable_and_order_independent():
    config_a = {"psf": {"wavelength": 5.0e-7}, "run_name": "x"}
    config_b = {"run_name": "x", "psf": {"wavelength": 5.0e-7}}
    assert config_hash(config_a) == config_hash(config_b)
    assert len(config_hash(config_a)) == 16
    assert config_hash({"run_name": "y"}) != config_hash(config_a)


def test_results_schema_contains_required_metric_fields():
    required = {
        "run_name",
        "config_hash",
        "git_hash",
        "global_seed",
        "q_f",
        "z_f",
        "delta_log_l_f_equiv",
        "detected_scdd",
        "map_detectable_ring_fraction",
    }
    assert required.issubset(set(RESULTS_CSV_COLUMNS))
