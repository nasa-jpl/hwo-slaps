"""Hand-counted areas, archived Table 5 tolerances and fixed keyed-cohort rules."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hwoslaps.analysis.knowledge_error import (
    ToleranceCriterion, knowledge_error_areas, knowledge_error_tolerance,
)
from hwoslaps.fisher.positions import GridIndex, PositionSet, explicit_positions, grid_positions
from hwoslaps.fisher.result import ForecastResult


def _provenance(relation):
    # Two genuine truth groups ensure the full binding, including group assignment, matters.
    return {"psf_relation": relation, "comparison_digest": "same-science",
            "mask": {"digest": "same-mask"}, "nuisance_names": ["source.light.light.intensity"],
            "truth_kernels": {"kernels": [
                {"identity": {"sha256": "a" * 64, "shape": [3, 3], "pixel_scale_arcsec": 0.5}, "source": {}},
                {"identity": {"sha256": "b" * 64, "shape": [3, 3], "pixel_scale_arcsec": 0.5}, "source": {}}],
                "groups": {"source.light.light": 0, "lens.light.light": 1}}}


def _result(layout, information, *, amplitude_hat=None, amplitude_spurious=None):
    values = np.asarray(information, dtype=float)
    relation = "matched" if amplitude_hat is None else "knowledge_error"
    return ForecastResult(np.geomspace(1e7, 1e8, values.shape[0]), layout, values + 1, values,
                          amplitude_hat, amplitude_spurious, relation, {}, _provenance(relation))


def _pair():
    coords = np.arange(4) * 0.5 - 0.75
    indices = np.indices((4, 4)).reshape(2, -1).T
    layout = PositionSet("grid", np.column_stack((coords[indices[:, 0]], coords[indices[:, 1]])),
                         (0, 0), 2.0, np.full(16, 0.25), None, GridIndex(coords, coords, indices, 0.5))
    information = np.zeros((1, 16))
    information[0, :4] = 4
    reference = _result(layout, information)
    fitted = np.zeros((1, 16))
    fitted[0, [0, 2, 4, 5]] = 1
    fitted[0, 1] = -10
    spurious = np.zeros((1, 16))
    spurious[0, [1, 6, 8, 9, 10]] = 1
    spurious[0, 11], spurious[0, 12] = -10, 0.0
    return reference, _result(layout, np.full((1, 16), 4), amplitude_hat=fitted, amplitude_spurious=spurious)


def test_areas_on_hand_built_results():
    reference, mismatch = _pair()
    selected = np.arange(16) < 8
    areas = knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=4, selection=selected)
    assert areas.q_threshold == 4 and areas.min_reference_count == 4
    np.testing.assert_array_equal(areas.masses_msun, reference.masses_msun)
    expected = {"reference_count": 4, "reference_area_arcsec2": 1,
                "mismatch_count": 4, "mismatch_area_arcsec2": 1,
                "retained_count": 2, "retained_area_arcsec2": 0.5,
                "spurious_count": 5, "spurious_area_arcsec2": 1.25,
                "spurious_in_selection_count": 2, "spurious_in_selection_area_arcsec2": 0.5,
                "retention": 0.5, "detected_area_ratio": 1, "spurious_ratio": 0.5}
    for name, value in expected.items():
        np.testing.assert_array_equal(getattr(areas, name), [value], err_msg=name)
    full = knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=1)
    assert full.spurious_in_selection_count[0] == full.spurious_count[0] == 5
    assert full.spurious_ratio[0] == 1.25
    empty = knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=1,
                                  selection=np.zeros(16, dtype=bool))
    assert empty.reference_count[0] == 0 and empty.spurious_count[0] == 5 and np.isnan(empty.retention[0])


def test_ratios_are_nan_below_the_reference_floor():
    layout = grid_positions((0, 0), spacing_arcsec=0.05, half_width_arcsec=0.25, annulus=None)
    information = np.zeros((3, 121))
    for row, count in enumerate((20, 33, 100)):
        information[row, :count] = 4
    reference = _result(layout, information)
    mismatch = _result(layout, information, amplitude_hat=np.where(information > 0.0, 1.0, np.nan),
                       amplitude_spurious=np.where(information > 0.0, 0.0, np.nan))
    areas = knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=33)
    np.testing.assert_array_equal(areas.reference_count, [20, 33, 100])
    assert areas.reference_area_arcsec2.tobytes() == np.array([n * 0.05 ** 2 for n in (20, 33, 100)]).tobytes()
    np.testing.assert_array_equal(areas.retention, [np.nan, 1, 1])
    np.testing.assert_array_equal(areas.detected_area_ratio, [np.nan, 1, 1])
    np.testing.assert_array_equal(areas.spurious_ratio, [np.nan, 0, 0])


@pytest.mark.parametrize("change,match", [
    ("mass", "masses_msun"), ("positions", "positions_yx"), ("area", "cell_areas_arcsec2"),
    ("no_area", "cell_areas_arcsec2"), ("reference_relation", "reference psf_relation"),
    ("missing_statistics", "mismatch statistics"), ("comparison_digest", "comparison_digest"),
    ("mask", "mask digest"), ("nuisance_names", "nuisance_names"),
    ("second_truth", "truth_kernels"), ("truth_groups", "truth_kernels"),
    ("missing_identity", "comparison_digest"),
])
def test_area_inputs_must_share_geometry_and_relations(change, match):
    reference, mismatch = _pair()
    if change == "mass":
        mismatch = replace(mismatch, masses_msun=np.array([2e7]))
    elif change == "positions":
        layout = mismatch.positions
        coords = layout.grid.y_coords + 0.01
        grid = replace(layout.grid, y_coords=coords)
        points = layout.positions_yx.copy()
        points[:, 0] += 0.01
        mismatch = replace(mismatch, positions=replace(layout, positions_yx=points, grid=grid))
    elif change == "area":
        layout = mismatch.positions
        # The same sampled positions on a finer sparse lattice represent smaller cells.
        grid = GridIndex(np.linspace(layout.grid.y_coords[0], layout.grid.y_coords[-1], 7),
                         np.linspace(layout.grid.x_coords[0], layout.grid.x_coords[-1], 7),
                         layout.grid.indices * 2, layout.grid.spacing_arcsec / 2)
        mismatch = replace(mismatch, positions=replace(layout, cell_areas_arcsec2=np.full(16, grid.spacing_arcsec**2),
                                                       boundary=np.ones(16, dtype=bool), grid=grid))
    elif change == "no_area":
        mismatch = replace(mismatch, positions=explicit_positions(mismatch.positions_yx, (0, 0)))
    elif change == "reference_relation":
        provenance = deepcopy(reference.provenance)
        provenance["psf_relation"] = "kernel"
        reference = replace(reference, provenance=provenance)
    elif change == "missing_statistics":
        mismatch = replace(mismatch, psf_relation="matched", amplitude_hat=None, amplitude_spurious=None)
    else:
        provenance = deepcopy(mismatch.provenance)
        if change == "second_truth":
            provenance["truth_kernels"]["kernels"][1]["identity"]["sha256"] = "c" * 64
        elif change == "truth_groups":
            provenance["truth_kernels"]["groups"]["lens.light.light"] = 0
        elif change == "mask":
            provenance["mask"]["digest"] = "another-mask"
        elif change == "missing_identity":
            del provenance["comparison_digest"]
        else:
            provenance[change] = "different"
        mismatch = replace(mismatch, provenance=provenance)
    with pytest.raises(ValueError, match=match):
        knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=1)


def test_area_inputs_must_share_the_comparison_digest(minimal_mapping):
    from hwoslaps.config.schema import resolve_config

    reference, mismatch = _pair()
    config = resolve_config(minimal_mapping)
    model = config.replace({"psf": {"model": minimal_mapping["psf"]["truth"]}})
    assert config.digest() != model.digest() and config.comparison_digest() == model.comparison_digest()
    provenance = deepcopy(reference.provenance)
    provenance["comparison_digest"] = config.comparison_digest()
    reference = replace(reference, provenance=provenance)
    provenance = deepcopy(mismatch.provenance)
    provenance["comparison_digest"] = model.comparison_digest()
    mismatch = replace(mismatch, provenance=provenance)
    assert knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=1).retention[0] == 0.5
    exposed = model.replace({"observation": {"exposure_time_s": 1800}})
    provenance["comparison_digest"] = exposed.comparison_digest()
    with pytest.raises(ValueError, match="comparison_digest"):
        knowledge_error_areas(reference, replace(mismatch, provenance=provenance), q_threshold=4, min_reference_count=1)


@pytest.mark.parametrize("floor,selection", [(0, None), (True, None), (1.5, None),
                                            (1, np.ones(16)), (1, np.ones(15, dtype=bool))])
def test_area_refuses_invalid_floor_or_selection(floor, selection):
    reference, mismatch = _pair()
    with pytest.raises(ValueError):
        knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=floor, selection=selection)


def test_tolerance_inclusive_limits_on_the_declared_gate_table():
    # Exact 41621de:tests/test_psf_knowledge.py:287 table. Endpoint35 is omitted by caller.
    keys = {(0, direction) for direction in (1, 2, 3)}
    retention = {1.0: dict.fromkeys(keys, 0.9), 5.0: dict.fromkeys(keys, 0.8)}
    spurious = {a: dict.fromkeys(keys, 0.03) for a in retention}
    for lower, upper, expected, passing, failure in ((0.9, 0.1, 1.0, (1.0,), 5.0),
                                                    (0.8, 0.2, 5.0, (1.0, 5.0), None),
                                                    (0.99, 0.01, None, (), 1.0)):
        criterion = ToleranceCriterion(0.1, lower, 0.9, upper)
        result = knowledge_error_tolerance(retention, spurious, eligible=keys, criterion=criterion)
        assert (result.amplitude, result.passing, result.first_failing) == (expected, passing, failure)
        assert result.criterion == criterion and result.eligible == 3 and result.excluded == 0


def test_tolerance_uses_one_eligible_cohort():
    keys = {(member, 1) for member in range(10)}
    retention = {a: dict.fromkeys(keys, 0.95) for a in (1.0, 5.0)}
    retention[1.0][(9, 1)], retention[5.0][(9, 1)] = 0.91, np.nan
    spurious = {a: dict.fromkeys(keys, 0.01) for a in retention}
    criterion = ToleranceCriterion(0.1, 0.9, 0.9, 0.1)
    with pytest.raises(ValueError, match=r"ratio.*5.0.*\(9, 1\)"):
        knowledge_error_tolerance(retention, spurious, eligible=keys, criterion=criterion)
    result = knowledge_error_tolerance(retention, spurious, eligible=keys - {(9, 1)}, criterion=criterion)
    assert result.amplitude == 5 and result.eligible == 9 and result.excluded == 1
    del spurious[5.0][(9, 1)]
    with pytest.raises(ValueError, match=r"spurious.*5.0.*\(9, 1\)"):
        knowledge_error_tolerance(retention, spurious, eligible=keys - {(9, 1)}, criterion=criterion)


def test_tolerance_linear_quantiles_and_nonmonotonic_passes():
    keys = [(1, 1), (1, 2)]
    criterion = ToleranceCriterion(0.25, 0.8125, 0.75, 0.1875)
    retention = {1.0: dict(zip(keys, (0.75, 1.0))), 5.0: dict(zip(keys, (0.7, 0.7))),
                 10.0: dict(zip(keys, (0.9, 0.9)))}
    spurious = {a: dict(zip(keys, (0.0, 0.25))) for a in retention}
    result = knowledge_error_tolerance(retention, spurious, eligible=keys, criterion=criterion)
    assert result.passing == (1.0, 10.0) and result.amplitude == 10 and result.first_failing == 5


@pytest.mark.parametrize("change", ["amplitudes", "missing_retention", "extra_spurious", "empty", "outside", "infinite"])
def test_tolerance_refuses_invalid_cohorts(change):
    retention = {1.0: {(0, 1): 0.95, (0, 2): 0.95}, 5.0: {(0, 1): 0.95, (0, 2): 0.95}}
    spurious = {a: dict.fromkeys([(0, 1), (0, 2)], 0.01) for a in retention}
    eligible = [(0, 1), (0, 2)]
    if change == "amplitudes":
        del spurious[5.0]
    elif change == "missing_retention":
        del retention[5.0][(0, 2)]
    elif change == "extra_spurious":
        spurious[5.0][(1, 1)] = 0.01
    elif change == "empty":
        eligible = []
    elif change == "outside":
        eligible = [(2, 1)]
    else:
        spurious[5.0][(0, 1)] = np.inf
    with pytest.raises(ValueError):
        knowledge_error_tolerance(retention, spurious, eligible=eligible, criterion=ToleranceCriterion(.1, .9, .9, .1))


@pytest.mark.parametrize("field,value", [("retention_quantile", -.1), ("retention_quantile", 1.1),
                                         ("spurious_quantile", np.nan), ("retention_min", np.inf),
                                         ("spurious_max", True)])
def test_criterion_refuses_invalid_limits(field, value):
    fields = dict(retention_quantile=.1, retention_min=.9, spurious_quantile=.9, spurious_max=.1)
    fields[field] = value
    with pytest.raises(ValueError, match=field):
        ToleranceCriterion(**fields)


@pytest.mark.parametrize("reference_count,retained_count,spurious_count", [(50, 45, 0), (530, 530, 53)])
def test_count_ratios_preserve_inclusive_tolerance_gates(reference_count, retained_count, spurious_count):
    # Uniform 0.05-arcsec cells must cancel before testing the inclusive 0.9/0.1 limits.
    layout = grid_positions((0, 0), spacing_arcsec=0.05, half_width_arcsec=0.6, annulus=None)
    information = np.zeros((1, len(layout)))
    information[0, :reference_count] = 4
    reference = _result(layout, information)
    fitted = np.zeros_like(information)
    fitted[0, :retained_count] = 1
    spurious = np.zeros_like(information)
    spurious[0, :spurious_count] = 1
    mismatch = _result(layout, np.full_like(information, 4), amplitude_hat=fitted,
                       amplitude_spurious=spurious)
    areas = knowledge_error_areas(reference, mismatch, q_threshold=4, min_reference_count=33)
    assert areas.retention[0] == retained_count / reference_count
    assert areas.detected_area_ratio[0] == retained_count / reference_count
    assert areas.spurious_ratio[0] == spurious_count / reference_count
    result = knowledge_error_tolerance(ratio={2.0: {1: areas.retention[0]}},
                                      spurious={2.0: {1: areas.spurious_ratio[0]}}, eligible={1},
                                      criterion=ToleranceCriterion(0.1, 0.9, 0.9, 0.1))
    assert result.amplitude == 2.0 and result.passing == (2.0,) and result.first_failing is None


_TABLE5 = json.loads((Path(__file__).parents[1] / "fixtures/analysis/table5_knowledge_error.json").read_text())


def test_table5_fixture_covers_both_rung_classes_of_the_twelve_systems():
    cases = _TABLE5["cases"]
    systems = {case["system_id"] for case in cases}
    assert len(systems) == 12 and len(cases) == 24
    assert {(case["system_id"], case["rung_class"]) for case in cases} == {
        (system, rung) for system in systems for rung in ("m10", "m50")}


@pytest.mark.parametrize("case", _TABLE5["cases"],
                         ids=lambda case: f"{case['system_id']}-{case['rung_class']}")
def test_tolerance_reproduces_archived_table5_retained_locations(case):
    # Real integer aperture-mask counts; the golden tolerances come from the original figure reducer.
    # No artificial directions or copies of already-reduced quantiles are supplied.
    reference_count = case["reference_count"]
    directions = _TABLE5["direction_order"]
    retention, total_area, spurious = {}, {}, {}
    for amplitude, rows in case["directions_by_amplitude_nm"].items():
        amplitude = float(amplitude)
        retention[amplitude] = {direction: counts[0] / reference_count
                                for direction, counts in zip(directions, rows, strict=True)}
        total_area[amplitude] = {direction: counts[1] / reference_count
                                 for direction, counts in zip(directions, rows, strict=True)}
        spurious[amplitude] = {direction: counts[2] / reference_count
                               for direction, counts in zip(directions, rows, strict=True)}
    criterion = ToleranceCriterion(**_TABLE5["criterion"])
    result = knowledge_error_tolerance(ratio=retention, spurious=spurious, eligible=directions,
                                      criterion=criterion)
    assert result.amplitude == case["expected_tolerance_K_nm"]
    assert result.eligible == 8 and result.excluded == 0
    # The same raw inputs also preserve the distinct pre-submission total-area statistic.
    total = knowledge_error_tolerance(ratio=total_area, spurious=spurious, eligible=directions,
                                     criterion=criterion)
    assert total.amplitude == case["expected_tolerance_total_area_nm"]


@pytest.mark.backend
def test_zero_amplitude_knowledge_error_gives_unit_retention(minimal_mapping):
    from hwoslaps.fisher.api import forecast, prepare_forecast

    minimal_mapping["psf"] = {"truth": {
        "kind": "optical", "pupil": {"kind": "hex_segmented", "diameter_m": 7.225765,
        "pixels": 64, "supersampling": 2, "rings": 2, "segment_point_to_point_m": 1.65, "gap_m": .006},
        "focal_length_m": 144, "wavelength_nm": 500, "detector_oversampling": 11, "kernel_shape": [7, 7],
        "wavefront": {"zernikes": {4: 5.0}}}}
    minimal_mapping["forecast"]["positions"] = {"kind": "grid", "spacing_arcsec": .4, "half_width_arcsec": .4}
    with prepare_forecast(minimal_mapping) as prepared:
        reference = forecast(prepared, masses_msun=[1e9])
    minimal_mapping["psf"]["model"] = {"kind": "knowledge_error", "draw": {
        "prior": {"packaged": "jwst_wss_drift_v1"}, "amplitude_rms_nm": 0.0, "seed": 7, "family": "combined"}}
    with prepare_forecast(minimal_mapping) as prepared:
        mismatch = forecast(prepared, masses_msun=[1e9])
    assert reference.provenance["comparison_digest"] == mismatch.provenance["comparison_digest"]
    assert reference.provenance["config_digest"] != mismatch.provenance["config_digest"]
    assert reference.provenance["truth_kernels"] == mismatch.provenance["truth_kernels"]
    threshold = float(np.min(reference.q_asimov)) / 2
    assert threshold > 0
    areas = knowledge_error_areas(reference, mismatch, q_threshold=threshold, min_reference_count=1)
    assert areas.reference_count[0] == 9
    np.testing.assert_array_equal(areas.retention, [1])
    np.testing.assert_array_equal(areas.detected_area_ratio, [1])
    np.testing.assert_array_equal(areas.spurious_ratio, [0])
    np.testing.assert_array_equal(areas.spurious_area_arcsec2, [0])


def _validity_pair():
    layout = grid_positions((0.0, 0.0), spacing_arcsec=1.0, half_width_arcsec=1.0, annulus=None)
    information = np.full((1, 9), 4.0)
    reference = _result(layout, information)
    mismatch = _result(layout, information, amplitude_hat=np.ones_like(information),
                       amplitude_spurious=np.zeros_like(information))
    return reference, mismatch


@pytest.mark.parametrize("outcome", ["finite_nondetection", "zero_information", "failed_numeric"])
def test_failed_statistics_are_distinct_from_true_nondetections_and_zero_information(outcome):
    reference, mismatch = _validity_pair()
    information = mismatch.fisher_profiled.copy()
    fitted, spurious = mismatch.amplitude_hat.copy(), mismatch.amplitude_spurious.copy()
    if outcome == "zero_information":
        information[0, 0], fitted[0, 0], spurious[0, 0] = 0.0, np.nan, np.nan
    elif outcome == "finite_nondetection":
        fitted[0, 0] = 0.0
    else:
        fitted[0, 0] = np.nan
    mismatch = _result(reference.positions, information, amplitude_hat=fitted, amplitude_spurious=spurious)
    if outcome == "failed_numeric":
        with pytest.raises(ValueError, match="failed_numeric"):
            knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1)
    areas = knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                                  invalid_policy="report")
    assert areas.reference_count[0] == 9
    assert areas.known_counts.retained[0] == 8
    assert areas.reference_cohort.shape == reference.q_asimov.shape
    np.testing.assert_array_equal(areas.reference_cohort, True)
    assert areas.mismatch_validity.evaluated_count == 9
    assert areas.mismatch_validity.failed_numeric_count[0] == (outcome == "failed_numeric")
    assert areas.mismatch_validity.zero_information_count[0] == (outcome == "zero_information")
    if outcome == "failed_numeric":
        assert np.isnan(areas.retention[0]) and np.isnan(areas.spurious_ratio[0])
        assert np.isnan(areas.spurious_count[0]) and np.isnan(areas.mismatch_count[0])
        assert not areas.complete[0]
    else:
        assert areas.retention[0] == 8 / 9
        assert areas.spurious_ratio[0] == 0.0
        assert areas.mismatch_count[0] == 8 and areas.complete[0]


def test_failed_nodes_do_not_shrink_reference_cohort_across_mismatch_amplitudes():
    reference, healthy = _validity_pair()
    reports = []
    for failed_positions in ([], [0], [0, 1, 2], list(range(9))):
        fitted = healthy.amplitude_hat.copy()
        fitted[0, failed_positions] = np.nan
        mismatch = _result(reference.positions, healthy.fisher_profiled, amplitude_hat=fitted,
                           amplitude_spurious=healthy.amplitude_spurious)
        reports.append(knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                                              invalid_policy="report"))
    for report, failed_count in zip(reports, (0, 1, 3, 9), strict=True):
        np.testing.assert_array_equal(report.reference_cohort, reports[0].reference_cohort)
        assert report.reference_count[0] == 9
        assert report.mismatch_validity.failed_numeric_count[0] == failed_count
        assert report.known_counts.retained[0] == 9 - failed_count
        if failed_count:
            assert np.isnan(report.retention[0]) and np.isnan(report.spurious_ratio[0])
        else:
            assert report.retention[0] == 1.0 and report.spurious_ratio[0] == 0.0


def test_common_selection_gives_complete_ratios_without_hiding_full_domain_failures():
    reference, healthy = _validity_pair()
    # One declared spatial cohort is shared by every error amplitude.
    selected = np.arange(9) >= 2
    cohort = None
    for failed_position in (0, 1):
        fitted = healthy.amplitude_hat.copy()
        fitted[0, failed_position] = np.nan
        mismatch = _result(reference.positions, healthy.fisher_profiled, amplitude_hat=fitted,
                           amplitude_spurious=healthy.amplitude_spurious)
        with pytest.raises(ValueError, match="mismatched domain"):
            knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                                  selection=selected)
        report = knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                                       selection=selected, invalid_policy="report")
        assert report.selection_complete[0] and not report.complete[0]
        assert report.reference_count[0] == 7
        assert report.retention[0] == 1.0 and report.spurious_ratio[0] == 0.0
        assert np.isnan(report.spurious_area_arcsec2[0])
        assert report.mismatch_validity.failed_numeric_count[0] == 0
        assert report.domain_validity.failed_numeric_count[0] == 1
        if cohort is not None:
            np.testing.assert_array_equal(report.reference_cohort, cohort)
        cohort = report.reference_cohort


def test_failed_reference_leaves_reference_counts_and_comparison_ratios_unavailable():
    reference, mismatch = _validity_pair()
    information = reference.fisher_profiled.copy()
    information[0, 0] = np.nan
    reference = _result(reference.positions, information)
    with pytest.raises(ValueError, match="reference selection"):
        knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1)
    report = knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                                   invalid_policy="report")
    assert report.reference_validity.failed_numeric_count[0] == 1
    assert report.known_counts.reference[0] == 8
    assert np.isnan(report.reference_count[0]) and np.isnan(report.reference_area_arcsec2[0])
    assert np.isnan(report.retention[0]) and np.isnan(report.spurious_ratio[0])
    assert report.mismatch_count[0] == 9 and report.spurious_count[0] == 0


def test_finite_provisional_failure_is_not_treated_as_a_usable_detection():
    reference, mismatch = _validity_pair()
    status = mismatch.template_status.copy()
    status[0, 0] = "failed_numeric"
    mismatch = replace(mismatch, template_status=status)
    with pytest.raises(ValueError, match="failed_numeric"):
        knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1)
    report = knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                                   invalid_policy="report")
    assert report.known_counts.mismatch[0] == 8
    assert np.isnan(report.mismatch_count[0]) and np.isnan(report.retention[0])


@pytest.mark.parametrize("policy", ["exclude", "ignore", None, True])
def test_area_requires_a_declared_validity_policy(policy):
    reference, mismatch = _validity_pair()
    with pytest.raises(ValueError, match="invalid_policy"):
        knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                              invalid_policy=policy)


def test_area_report_json_preserves_null_ratios_and_fixed_cohort(tmp_path):
    from hwoslaps.artifacts import write_json
    reference, mismatch = _validity_pair()
    fitted = mismatch.amplitude_hat.copy()
    fitted[0, 0] = np.nan
    mismatch = _result(reference.positions, mismatch.fisher_profiled, amplitude_hat=fitted,
                       amplitude_spurious=mismatch.amplitude_spurious)
    report = knowledge_error_areas(reference, mismatch, q_threshold=4.0, min_reference_count=1,
                                   invalid_policy="report")
    mapping = report.to_mapping()
    assert mapping["retention"] == [None] and mapping["spurious_ratio"] == [None]
    assert mapping["reference_count"] == [9]
    assert mapping["reference_cohort"] == [[True] * 9]
    assert mapping["mismatch_validity"]["failed_numeric_count"] == [1]
    assert mapping["known_counts"]["retained"] == [8]
    assert mapping["complete"] == [False] and mapping["selection_complete"] == [False]
    path = write_json(tmp_path / "areas.json", mapping)
    assert json.loads(path.read_text()) == mapping
    assert "NaN" not in path.read_text() and "Infinity" not in path.read_text()
