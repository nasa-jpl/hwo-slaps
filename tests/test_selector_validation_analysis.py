"""Unit tests for the validation-sample three-curve selector analysis."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]/"scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import selector_validation_analysis as sva  # noqa: E402

NAN = float("nan")


def _rows(m_best: dict) -> dict:
    return {
        system_id: {"system_id": system_id, "m_best": value, "m10": value, "m50": value}
        for system_id, value in m_best.items()
    }


def test_restrict_ranking_preserves_order_and_fails_on_a_missing_member():
    ranking = ["sys0003", "sys0001", "sys0002", "sys0000"]
    assert sva.restrict_ranking(ranking, {"sys0000", "sys0002"}) == ["sys0002", "sys0000"]
    with pytest.raises(ValueError, match="absent"):
        sva.restrict_ranking(ranking, {"sys0000", "sys0999"})


def test_censored_last_values_tie_every_censored_member_above_the_finite_ones():
    mapped = sva.censored_last_values([7.5, NAN, 8.0, NAN])
    assert mapped[0] == 7.5 and mapped[2] == 8.0
    assert mapped[1] == mapped[3] == 9.0
    with pytest.raises(ValueError):
        sva.censored_last_values([NAN, NAN])


def test_spearman_by_convention_reports_both_conventions():
    positions = [0, 1, 2, 3]
    values = [7.0, 7.5, NAN, 8.0]
    out = sva.spearman_by_convention(positions, values)
    assert out["finite_only"]["n"] == 3
    assert out["finite_only"]["spearman"] == pytest.approx(1.0)
    assert out["censored_last"]["n"] == 4 and out["censored_last"]["n_censored"] == 1
    assert out["censored_last"]["spearman"] == pytest.approx(0.8)


def test_oracle_ranking_puts_censored_members_last_by_id():
    m_lim = {"sys0002": 7.9, "sys0001": NAN, "sys0000": 7.1, "sys0003": NAN}
    assert sva.oracle_ranking(list(m_lim), m_lim) == ["sys0000", "sys0002", "sys0001", "sys0003"]


def test_three_curve_comparison_is_hand_calculable():
    sample = [f"sys{i:04d}" for i in range(6)]
    rankings = {
        "s_only": ["sys0005", "sys0004", "sys0003", "sys0002", "sys0001", "sys0000", "sys0099"],
        "s_plus_c": ["sys0000", "sys0001", "sys0002", "sys0003", "sys0004", "sys0005", "sys0099"],
    }
    rows = _rows({"sys0000": 7.0, "sys0001": 7.2, "sys0002": 7.4, "sys0003": 7.6, "sys0004": 7.8, "sys0005": NAN})
    template_of = {system_id: ("a" if int(system_id[3:]) % 2 else "b") for system_id in sample}
    report = sva.three_curve_comparison(sample, rankings, rows, template_of, k=2)
    assert report["n_censored"] == 1
    assert [row["system_id"] for row in report["oracle"]["top_k"]] == ["sys0000", "sys0001"]
    good = report["curves"]["s_plus_c"]
    bad = report["curves"]["s_only"]
    assert good["oracle_recovered_fraction"] == 1.0 and bad["oracle_recovered_fraction"] == 0.0
    assert good["spearman_position_vs_estimand"]["m_best"]["censored_last"]["spearman"] == pytest.approx(1.0)
    assert bad["spearman_position_vs_estimand"]["m_best"]["censored_last"]["spearman"] == pytest.approx(-1.0)
    assert good["spearman_position_vs_estimand"]["m_best"]["finite_only"]["n"] == 5
    assert report["operational_top_k_jaccard"] == 0.0
    assert set(good["spearman_position_vs_m_lim_per_template"]) == {"a", "b"}
    assert good["top_k_m_lim_median_finite"] == pytest.approx(7.1)


def test_three_curve_comparison_refuses_too_few_finite_members():
    sample = ["sys0000", "sys0001", "sys0002"]
    rankings = {"s_only": list(sample), "s_plus_c": list(sample)}
    rows = _rows({"sys0000": 7.0, "sys0001": NAN, "sys0002": NAN})
    with pytest.raises(ValueError, match="fewer than the tier"):
        sva.three_curve_comparison(sample, rankings, rows, {s: "a" for s in sample}, k=2)


def test_frozen_top_k_outcomes_fails_closed_on_an_unmeasured_member():
    rankings = {"s_only": ["sys0000", "sys0001", "sys0009"], "s_plus_c": ["sys0000", "sys0002", "sys0001"]}
    rows = _rows({"sys0000": 7.0, "sys0001": 7.5, "sys0002": NAN})
    template_of = {s: "a" for s in ("sys0000", "sys0001", "sys0002", "sys0009")}
    tier_of = {"sys0000": "selected", "sys0001": "parent", "sys0002": "validation"}
    out = sva.frozen_top_k_outcomes(rankings, rows, template_of, tier_of, k=2)
    assert out["shared_members"] == ["sys0000"]
    assert out["top_k_jaccard"] == pytest.approx(1/3)
    assert out["sets"]["s_plus_c"]["n_censored"]["m_best"] == 1
    assert out["sets"]["s_only"]["medians_finite_only"]["m_best"] == pytest.approx(7.25)
    with pytest.raises(ValueError, match="unmeasured"):
        sva.frozen_top_k_outcomes(rankings, rows, template_of, tier_of, k=3)


def test_pool_context_and_censored_median():
    sample_rows = list(_rows({"sys0000": 7.0, "sys0001": 8.0, "sys0002": 9.0, "sys0003": NAN}).values())
    selected_rows = list(_rows({"sys0100": 7.5}).values())
    out = sva.pool_context(sample_rows, selected_rows)
    assert out["sample_n_censored"] == 1
    assert out["sample_median_censoring_aware"] == pytest.approx(8.5)
    assert out["selected"][0]["sample_fraction_at_or_below"] == pytest.approx(0.25)
    assert sva._censored_median([7.0, NAN, NAN]) is None
    assert math.isclose(sva._censored_median([7.0, 8.0, NAN]), 8.0)
