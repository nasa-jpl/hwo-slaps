"""Contracts for the adaptive mass ladder and its per-job runner.

The walk and the estimands are pure, so the whole adaptive policy is
pinned here against hand-computed values on synthetic curves: every stop
case of the coarse ascent, every way the curve closes downward on
measured zeros, where the refinement rungs land, the three estimand
interpolations, and the nulls a ladder that never crosses must produce.

What is pinned of the runner is the plumbing between a staged
configuration and its artifact: the fail-closed refusals it inherits from
the Stage 0 runner and the ones the ladder block adds, the rendering
declarations it applies, the reduction of one grid map to a rung, the
identity members the campaign layer validates, and the gate that fails if
the runner ever reaches for a random stream. The rendering path itself is
exercised by the campaign smoke rather than by a unit test.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT/"scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_ladder as runner  # noqa: E402
from hwoslaps.campaign import ladder_walk as walk  # noqa: E402


FREEZE_PATH = PROJECT_ROOT/"configs"/"design"/"design_freeze_v1.yaml"


def _policy(**overrides) -> walk.LadderPolicy:
    """Return the frozen walk policy, with any field overridden."""
    fields = {
        "coarse_low": 6.0,
        "coarse_high": 9.5,
        "coarse_step": 0.25,
        "refine_step": 0.1,
        "extend_down_zero_rungs": 2,
        "saturation_fraction": 0.99,
        "q_threshold": 10.0,
    }
    fields.update(overrides)
    return walk.LadderPolicy(**fields)


def _row(logm, q_max=0.0, area=0.0, fraction=0.0) -> dict:
    """Return one per-rung table row."""
    return {
        "logm": logm,
        "q_max": q_max,
        "detectable_area_arcsec2": area,
        "aperture_fraction": fraction,
    }


def _ascending(low, high, step=0.25):
    """Return the coarse lattice rungs from ``low`` to ``high``."""
    count = int(round((high - low)/step)) + 1
    return [round(low + step*index, 2) for index in range(count)]


# ---------------------------------------------------------------------------
# The frozen policy this walk is executed under
# ---------------------------------------------------------------------------


def _freeze():
    """Return the committed design freeze."""
    from hwoslaps.campaign.design_freeze import load_design_freeze

    return load_design_freeze(FREEZE_PATH)


def test_the_policy_is_the_frozen_mass_ladder():
    """The walk policy is the freeze's mass ladder, read as staged."""
    mass_ladder = dict(_freeze()["mass_ladder"])
    mass_ladder["saturation_fraction"] = 0.99
    policy = walk.policy_from_mass_ladder(mass_ladder)
    assert policy.coarse_low == 6.0
    assert policy.coarse_high == 9.5
    assert policy.coarse_step == 0.25
    assert policy.refine_step == 0.1
    assert policy.extend_down_zero_rungs == 2
    assert policy.saturation_fraction == 0.99
    assert policy.q_threshold == walk.Q_THRESHOLD


def test_a_threshold_the_checkout_does_not_implement_fails_closed():
    """A ladder frozen on another detection threshold refuses to walk."""
    mass_ladder = dict(_freeze()["mass_ladder"])
    mass_ladder["saturation_fraction"] = 0.99
    mass_ladder["threshold"] = "q_F >= 25"
    with pytest.raises(ValueError, match="this checkout implements"):
        walk.policy_from_mass_ladder(mass_ladder)


def test_the_runner_constants_match_the_committed_freeze():
    """Every rendering constant the runner pins is a frozen value."""
    freeze = _freeze()
    assert runner.APERTURE_THETA_E_FACTOR == freeze["aperture"][
        "theta_e_factor"
    ]
    assert walk.THRESHOLD_DECLARATION == freeze["mass_ladder"]["threshold"]
    ruling = freeze["ratifications"]
    spacing = [item for item in ruling if item["id"] == "spacing_systematic"]
    assert len(spacing) == 1
    assert f"{runner.NODE_SPACING_ARCSEC} arcsec node spacing" in spacing[0][
        "ruling"
    ]
    assert "spatial_sampling_qmax" in freeze["declared_systematics"]
    assert runner.ARTIFACT_NAME == "ladder_result.npz"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"coarse_step": 0.0}, "must be positive"),
        ({"coarse_high": 5.0}, "must lie above"),
        ({"coarse_high": 9.6}, "whole number"),
        ({"refine_step": 0.25}, "must be finer"),
        ({"extend_down_zero_rungs": 0}, "at least one"),
        ({"saturation_fraction": 1.5}, "must lie in"),
    ],
)
def test_an_unwalkable_policy_fails_closed(overrides, expected):
    """A policy the walk cannot execute is rejected at construction."""
    with pytest.raises(ValueError, match=expected):
        _policy(**overrides)


# ---------------------------------------------------------------------------
# Coarse ascent
# ---------------------------------------------------------------------------


def test_an_empty_ladder_starts_at_the_coarse_floor():
    """The walk opens at ``coarse.low``."""
    step = walk.next_rung([], _policy())
    assert step.logm == 6.0
    assert step.phase == walk.PHASE_COARSE
    assert step.stop_reason is None


def test_the_ascent_climbs_one_coarse_step_at_a_time():
    """Each undetected rung asks for the next rung up."""
    table = [_row(6.0), _row(6.25)]
    step = walk.next_rung(table, _policy())
    assert step.logm == 6.5
    assert step.phase == walk.PHASE_COARSE
    assert walk.ascent_stop(table, _policy()) is None


def test_the_ascent_stops_when_m50_is_bracketed():
    """Half the aperture detected ends the climb with M50 bracketed."""
    table = [_row(logm, fraction=0.0, area=0.0) for logm in _ascending(6.0, 7.5)]
    table.append(_row(7.75, q_max=40.0, area=1.0, fraction=0.60))
    assert walk.ascent_stop(table, _policy()) == walk.STOP_M50
    assert walk.next_rung(table, _policy()).phase != walk.PHASE_COARSE


def test_the_ascent_stops_when_the_aperture_saturates():
    """A saturated aperture is reported ahead of the M50 level it passed."""
    table = [_row(logm, fraction=0.0, area=0.0) for logm in _ascending(6.0, 7.5)]
    table.append(_row(7.75, q_max=400.0, area=1.0, fraction=0.995))
    assert walk.ascent_stop(table, _policy()) == walk.STOP_SATURATED


def test_the_ascent_stops_at_the_coarse_ceiling():
    """A curve that never reaches M50 stops at ``coarse.high``."""
    table = [_row(logm) for logm in _ascending(6.0, 9.5)]
    assert walk.ascent_stop(table, _policy()) == walk.STOP_CEILING
    assert walk.next_rung(table, _policy()).logm is None


def test_a_gap_in_the_ascent_is_refilled_rather_than_skipped():
    """The ascent frontier is the contiguous chain from ``coarse.low``."""
    table = [_row(6.0), _row(6.5)]
    assert walk.next_rung(table, _policy()).logm == 6.25


# ---------------------------------------------------------------------------
# extend_down: closing the curve on measured zeros
# ---------------------------------------------------------------------------


def _stopped_at_ceiling(rows):
    """Return a table whose ascent has stopped, carrying ``rows``."""
    table = {row["logm"]: row for row in rows}
    for logm in _ascending(6.0, 9.5):
        table.setdefault(logm, _row(logm))
    return list(table.values())


def test_a_curve_already_closed_needs_no_downward_rung():
    """Two measured zeros below the lowest detection close the curve."""
    table = _stopped_at_ceiling([
        _row(6.5, area=0.4, fraction=0.2),
        _row(6.75, area=0.8, fraction=0.4),
    ])
    assert walk.extend_down_rung(table, _policy()) is None


def test_a_curve_with_one_zero_below_needs_one_more_rung():
    """One zero below the lowest detection is not closure."""
    table = _stopped_at_ceiling([
        _row(6.25, area=0.4, fraction=0.2),
        _row(6.5, area=0.8, fraction=0.4),
    ])
    assert walk.extend_down_rung(table, _policy()) == 5.75
    table.append(_row(5.75))
    assert walk.extend_down_rung(table, _policy()) is None


def test_a_curve_detecting_at_the_floor_forces_sub_floor_rungs():
    """Detection at ``coarse.low`` extends the lattice below it."""
    table = _stopped_at_ceiling([_row(6.0, area=0.4, fraction=0.2)])
    added = []
    while True:
        logm = walk.extend_down_rung(table, _policy())
        if logm is None:
            break
        added.append(logm)
        table.append(_row(logm))
    assert added == [5.75, 5.5]


def test_a_curve_detecting_below_the_floor_keeps_extending():
    """The extension continues while the lowest rung still detects."""
    table = _stopped_at_ceiling([_row(6.0, area=0.4, fraction=0.2)])
    table.append(_row(5.75, area=0.1, fraction=0.05))
    added = []
    while True:
        logm = walk.extend_down_rung(table, _policy())
        if logm is None:
            break
        added.append(logm)
        table.append(_row(logm))
    assert added == [5.5, 5.25]


def test_a_descent_that_never_closes_fails_loudly():
    """Area persisting two dex below the ladder is an error, not a walk."""
    table = _stopped_at_ceiling(
        [_row(logm, q_max=20.0, area=1.0, fraction=0.2)
         for logm in _ascending(4.25, 6.0)]
    )
    assert walk.extend_down_rung(table, _policy()) == 4.0
    table.append(_row(4.0, q_max=20.0, area=1.0, fraction=0.2))
    with pytest.raises(ValueError, match="physics failure"):
        walk.extend_down_rung(table, _policy())


def test_a_curve_with_no_detection_anywhere_has_nothing_to_close():
    """An all-zero curve is already closed on measured zeros."""
    table = _stopped_at_ceiling([])
    assert walk.extend_down_rung(table, _policy()) is None
    assert walk.next_rung(table, _policy()).logm is None


def test_the_walk_closes_the_curve_before_it_refines():
    """extend_down runs ahead of refinement."""
    table = _stopped_at_ceiling([
        _row(6.0, q_max=20.0, area=0.4, fraction=0.2),
    ])
    step = walk.next_rung(table, _policy())
    assert step.phase == walk.PHASE_EXTEND_DOWN
    assert step.logm == 5.75
    assert step.stop_reason == walk.STOP_CEILING


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


def _bracketing_table():
    """Return a closed coarse ladder bracketing the threshold in 8.0-8.25."""
    rows = [_row(logm) for logm in _ascending(6.0, 7.75)]
    rows.append(_row(8.0, q_max=1.0, area=0.2, fraction=0.10))
    rows.append(_row(8.25, q_max=100.0, area=1.2, fraction=0.60))
    return rows


def test_the_refinement_fills_the_bracketing_coarse_pair():
    """The interior of the bracketing pair is filled at the refine step."""
    assert walk.refinement_rungs(_bracketing_table(), _policy()) == (8.1, 8.2)


def test_a_ladder_that_never_crosses_is_not_refined():
    """No crossing means no refinement rung."""
    table = _stopped_at_ceiling([])
    assert walk.refinement_rungs(table, _policy()) == ()


def test_the_walk_emits_the_refinement_rungs_in_order():
    """The refinement rungs are asked for ascending, then the walk ends."""
    table = _bracketing_table()
    first = walk.next_rung(table, _policy())
    assert (first.logm, first.phase) == (8.1, walk.PHASE_REFINE)
    table.append(_row(8.1, q_max=3.0, area=0.5, fraction=0.2))
    second = walk.next_rung(table, _policy())
    assert (second.logm, second.phase) == (8.2, walk.PHASE_REFINE)
    table.append(_row(8.2, q_max=8.0, area=0.9, fraction=0.4))
    done = walk.next_rung(table, _policy())
    assert done.logm is None
    assert done.phase == walk.PHASE_COMPLETE
    assert done.stop_reason == walk.STOP_M50


def test_the_walk_terminates_and_measures_each_rung_once():
    """A whole walk over a monotone curve halts on unique rungs."""
    policy = _policy()
    table = []
    for _ in range(200):
        step = walk.next_rung(table, policy)
        if step.logm is None:
            break
        fraction = min(1.0, max(0.0, (step.logm - 7.5)/2.0))
        table.append(_row(
            step.logm,
            q_max=10.0**((step.logm - 8.0)*2.0),
            area=fraction,
            fraction=fraction,
        ))
    else:
        pytest.fail("the walk did not terminate")
    rungs = [row["logm"] for row in table]
    assert len(rungs) == len(set(rungs))
    assert min(rungs) == 6.0


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------


def _estimand_table():
    """Return a ladder whose three crossings are hand-computable.

    ``q_max`` steps 1 to 100 across 8.00-8.25, so the log-linear
    crossing of 10 sits exactly half way at 8.125. The aperture fraction
    steps 0.00 to 0.20 across 7.00-7.25 and 0.40 to 0.60 across
    8.00-8.25, putting the linear 0.10 and 0.50 crossings a quarter and
    a half of the way in.
    """
    fractions = {
        6.0: 0.0, 6.25: 0.0, 6.5: 0.0, 6.75: 0.0, 7.0: 0.0,
        7.25: 0.20, 7.5: 0.25, 7.75: 0.30, 8.0: 0.40, 8.25: 0.60,
    }
    quality = {
        6.0: 0.0, 6.25: 0.0, 6.5: 0.01, 6.75: 0.02, 7.0: 0.05,
        7.25: 0.1, 7.5: 0.3, 7.75: 0.6, 8.0: 1.0, 8.25: 100.0,
    }
    return [
        _row(logm, q_max=quality[logm], area=fractions[logm]*2.0,
             fraction=fractions[logm])
        for logm in sorted(fractions)
    ]


def test_m_best_is_the_log_linear_crossing_of_the_threshold():
    """M_best interpolates q_max through 10 in the log of q."""
    crossing = walk.log_linear_crossing(_estimand_table(), _policy())
    assert crossing.lower_logm == 8.0
    assert crossing.upper_logm == 8.25
    assert crossing.lower_value == 1.0
    assert crossing.upper_value == 100.0
    assert crossing.logm == pytest.approx(8.125, abs=1.0e-12)


def test_m10_is_the_linear_crossing_of_a_tenth_of_the_aperture():
    """M10 interpolates the aperture fraction through 0.10 in log-mass."""
    crossing = walk.aperture_fraction_crossing(
        _estimand_table(), _policy(), walk.M10_LEVEL
    )
    assert (crossing.lower_logm, crossing.upper_logm) == (7.0, 7.25)
    assert crossing.logm == pytest.approx(7.125, abs=1.0e-12)


def test_m50_is_the_linear_crossing_of_half_the_aperture():
    """M50 interpolates the aperture fraction through 0.50 in log-mass."""
    crossing = walk.aperture_fraction_crossing(
        _estimand_table(), _policy(), walk.M50_LEVEL
    )
    assert (crossing.lower_logm, crossing.upper_logm) == (8.0, 8.25)
    assert crossing.logm == pytest.approx(8.125, abs=1.0e-12)


def test_m_best_reads_the_refined_ladder_and_m10_the_coarse_one():
    """The refinement tightens M_best without moving the coarse estimands."""
    policy = _policy()
    table = _estimand_table()
    table.append(_row(8.1, q_max=10.0, area=0.9, fraction=0.45))
    crossing = walk.log_linear_crossing(table, policy)
    assert (crossing.lower_logm, crossing.upper_logm) == (8.0, 8.1)
    assert crossing.logm == pytest.approx(8.1, abs=1.0e-12)
    coarse = walk.aperture_fraction_crossing(table, policy, walk.M50_LEVEL)
    assert (coarse.lower_logm, coarse.upper_logm) == (8.0, 8.25)


def test_the_first_upward_crossing_is_the_one_taken():
    """A later re-crossing does not displace the first upward one."""
    policy = _policy()
    table = [
        _row(6.0, fraction=0.0),
        _row(6.25, fraction=0.2),
        _row(6.5, fraction=0.05),
        _row(6.75, fraction=0.4),
    ]
    crossing = walk.aperture_fraction_crossing(table, policy, walk.M10_LEVEL)
    assert (crossing.lower_logm, crossing.upper_logm) == (6.0, 6.25)


def test_a_ladder_that_never_crosses_yields_nothing():
    """An unbracketed crossing is a finding, never an extrapolation."""
    policy = _policy()
    table = [
        _row(logm, q_max=0.5, area=0.01, fraction=0.02)
        for logm in _ascending(6.0, 9.5)
    ]
    assert walk.log_linear_crossing(table, policy) is None
    assert walk.aperture_fraction_crossing(
        table, policy, walk.M10_LEVEL
    ) is None
    assert walk.aperture_fraction_crossing(
        table, policy, walk.M50_LEVEL
    ) is None


def test_a_threshold_reached_from_exactly_zero_fails_closed():
    """A zero lower bracket has no logarithm, so the crossing refuses."""
    table = [_row(6.0, q_max=0.0), _row(6.25, q_max=50.0)]
    with pytest.raises(ValueError, match="no logarithm"):
        walk.log_linear_crossing(table, _policy())


def test_a_rung_measured_twice_fails_closed():
    """One rung carries one measurement."""
    with pytest.raises(ValueError, match="more than once"):
        walk.measurements([_row(6.0), _row(6.0, q_max=1.0)])


@pytest.mark.parametrize(
    "row, expected",
    [
        (_row(6.0, q_max=-1.0), "non-negative finite"),
        (_row(6.0, area=float("nan")), "non-negative finite"),
        (_row(6.0, fraction=1.5), "above the whole aperture"),
        (_row(float("inf")), "must be finite"),
    ],
)
def test_an_unmeasurable_rung_fails_closed(row, expected):
    """A rung carrying an impossible measurement is rejected."""
    with pytest.raises(ValueError, match=expected):
        walk.measurement(row)


# ---------------------------------------------------------------------------
# Runner plumbing: the staged configuration
# ---------------------------------------------------------------------------


def _aberrations() -> dict:
    """Return aberrations shaped like the science35 truth state."""
    return {
        "enable_segment_pistons": False,
        "enable_segment_tiptilts": False,
        "enable_segment_hexikes": True,
        "enable_global_zernikes": True,
        "segment_hexikes": {0: {1: 2.5, 2: -1.5}},
        "global_zernikes": {4: 3.0, 5: -2.0},
    }


def _extraction(theta_e_eff=1.0, factor=2.0, margin=0.1):
    """Return one realized extraction carrying a closed square contour."""
    from hwoslaps.lensing import critical_curve as cc

    return cc.ThetaEExtraction(
        contour_arcsec=np.array(
            [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]], dtype=float
        ),
        area_arcsec2=float(np.pi*theta_e_eff**2),
        theta_e_eff_arcsec=theta_e_eff,
        aperture=cc.ApertureDefinition(
            centre_arcsec=(0.0, 0.0),
            theta_e_eff_arcsec=theta_e_eff,
            theta_e_factor=factor,
            computational_margin_fraction=margin,
        ),
        grid=cc.CriticalCurveGrid(
            requested_half_width_arcsec=4.0,
            pixel_scale_arcsec=0.01,
        ),
        lens_centre_arcsec=(0.0, 0.0),
        curve_counts={"extracted": 1, "closed": 1, "enclosing": 1},
    )


def _ladder_block(extraction, **overrides) -> dict:
    """Return the staged ladder block of one job."""
    aperture = extraction.aperture
    block = {
        "tier": "selected",
        "golden": True,
        "parent_overlap": False,
        "psf_state": "science35",
        "kernel": "k999",
        "engine": "jax",
        "mask_mode": "all_pixels",
        "node_spacing_arcsec": 0.05,
        "threshold": "q_F >= 10",
        "aperture": {
            "theta_e_factor": aperture.theta_e_factor,
            "theta_e_eff_arcsec": aperture.theta_e_eff_arcsec,
            "radius_arcsec": aperture.radius_arcsec,
            "required_map_half_width_arcsec": (
                aperture.required_map_half_width_arcsec
            ),
            "stage0_contour_sha256": extraction.contour_sha256,
            "stage0_aperture_sha256": aperture.sha256,
            "perimeter_cap_flag": False,
        },
        "mass_ladder": {
            "coarse": {"step_dex": 0.25, "low": 6.0, "high": 9.5},
            "refine": {"step_dex": 0.1, "where": "around every crossing"},
            "extend_down": {"zero_rungs": 2, "rule": "two measured zeros"},
            "extend_up": {"stop": "the first of M50 or aperture saturation"},
            "threshold": "q_F >= 10",
            "saturation_fraction": 0.99,
        },
        "estimand_conventions": "the validated panel conventions",
    }
    block.update(overrides)
    return block


def _staged_config(extraction=None, **ladder_overrides) -> dict:
    """Return a staged ladder job configuration."""
    extraction = extraction or _extraction()
    return {
        "run_name": "ladder_selected_sys0007",
        "global_seed": 12345,
        "plotting": {"enabled": False, "output_dir": "outputs"},
        "stage0": {
            "system_id": "sys0007",
            "source_asset_path": "configs/source_assets/cosmos_48849_hlr011.npz",
            "source_asset_sha256": "a"*64,
            "code_revision": {
                "git_hash": "b"*40,
                "git_dirty": False,
                "sha256": "c"*64,
            },
        },
        "lensing": {
            "grid": {"shape": [500, 500], "pixel_scale": 0.00716},
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "einstein_radius": 1.0,
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.1, 0.0],
                },
            },
            "source_galaxy": {"redshift": 0.6, "light": {"type": "Image"}},
            "subhalo": {"enabled": False, "mass": 1.0e7, "model": "NFW"},
            "cosmology": "Planck15",
        },
        "psf": {
            "kernel": {"shape_native": [51, 51]},
            "aberrations": _aberrations(),
        },
        "observation": {"exposure_time": 2000.0},
        "modeling": {
            "enabled": False,
            "fisher": {
                "mode": "both",
                "mask_mode": "source_snr",
                "map": {
                    "type": "grid",
                    "engine": "reference",
                    "detection_q_threshold": 10.0,
                    "grid": {
                        "spacing_arcsec": 0.05,
                        "half_width_arcsec": 1.5,
                        "annulus": None,
                    },
                },
            },
        },
        "ladder": _ladder_block(extraction, **ladder_overrides),
    }


def test_a_conforming_ladder_block_is_accepted():
    """The declarations this checkout implements pass unchanged."""
    config = _staged_config()
    assert runner._verify_ladder_block(config) is config["ladder"]


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"psf_state": "perfect"}, "ladder psf_state"),
        ({"kernel": "k51"}, "ladder kernel"),
        ({"engine": "reference"}, "ladder engine"),
        ({"mask_mode": "source_snr"}, "ladder mask_mode"),
        ({"threshold": "q_F >= 25"}, "ladder threshold"),
        ({"node_spacing_arcsec": 0.02}, "node spacing"),
        ({"tier": "golden"}, "ladder tier"),
    ],
)
def test_a_ladder_declaration_this_checkout_does_not_implement_refuses(
    overrides, expected
):
    """A job declared under other conditions refuses before it renders."""
    with pytest.raises(ValueError, match=expected):
        runner._verify_ladder_block(_staged_config(**overrides))


def test_an_unmatched_fit_psf_refuses():
    """The ladder fits the truth PSF, so any other fit PSF refuses."""
    config = _staged_config()
    config["modeling"]["fit_psf"] = {"mode": "delta"}
    with pytest.raises(ValueError, match="matched fit PSF"):
        runner._verify_ladder_block(config)


def test_the_science35_aberrations_are_accepted():
    """Combined global and segment content passes the state check."""
    runner._verify_psf_state(_staged_config())


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"enable_segment_pistons": True}, "enable_segment_pistons"),
        ({"enable_segment_hexikes": False}, "enable_segment_hexikes"),
        ({"enable_global_zernikes": False}, "enable_global_zernikes"),
        ({"segment_hexikes": {}}, "no segment_hexikes coefficients"),
        ({"global_zernikes": {}}, "no global_zernikes coefficients"),
    ],
)
def test_a_psf_state_that_is_not_science35_refuses(mutation, expected):
    """A staged state that is not science35 refuses to render."""
    config = _staged_config()
    config["psf"]["aberrations"].update(mutation)
    with pytest.raises(ValueError, match=expected):
        runner._verify_psf_state(config)


class _StubPsfData:
    """Stand-in carrying only the measured RMS the runner verifies."""

    def __init__(self, total_rms_nm):
        self.total_rms_nm = total_rms_nm


def test_the_declared_psf_rms_is_verified_and_recorded():
    """The science35 state is defined at its measured RMS."""
    assert runner._verify_psf_rms(_StubPsfData(35.0)) == 35.0
    with pytest.raises(ValueError, match="is not science35"):
        runner._verify_psf_rms(_StubPsfData(65.0))


def test_the_bound_aperture_is_accepted():
    """The ladder aperture that matches the realized extraction passes."""
    extraction = _extraction()
    config = _staged_config(extraction)
    assert runner._verify_aperture(config, extraction) is config["ladder"][
        "aperture"
    ]


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("theta_e_factor", 1.5, "D-F7 ruling pins"),
        ("theta_e_eff_arcsec", 1.2, "theta_e_eff_arcsec"),
        ("radius_arcsec", 2.5, "radius_arcsec"),
        ("required_map_half_width_arcsec", 3.0, "required_map_half_width"),
        ("stage0_contour_sha256", "d"*64, "contour hashes to"),
        ("stage0_aperture_sha256", "e"*64, "aperture hashes to"),
    ],
)
def test_an_aperture_the_ladder_is_not_bound_to_refuses(
    field, value, expected
):
    """A ladder pointed at another aperture than the selection refuses."""
    extraction = _extraction()
    config = _staged_config(extraction)
    config["ladder"]["aperture"][field] = value
    with pytest.raises(ValueError, match=expected):
        runner._verify_aperture(config, extraction)


def test_the_rendering_declarations_are_applied_once():
    """The ladder's rendering conditions replace the Stage 0 ones."""
    extraction = _extraction()
    config = _staged_config(extraction)
    rung_config = runner._rung_config(
        config, config["ladder"], config["ladder"]["aperture"]
    )
    assert rung_config["psf"]["kernel"]["shape_native"] == [999, 999]
    assert rung_config["lensing"]["subhalo"]["enabled"] is True
    assert rung_config["modeling"]["enabled"] is True
    fisher = rung_config["modeling"]["fisher"]
    assert fisher["mode"] == "map"
    assert fisher["mask_mode"] == "all_pixels"
    assert fisher["map"]["type"] == "grid"
    assert fisher["map"]["engine"] == "jax"
    assert fisher["map"]["detection_q_threshold"] == 10.0
    assert fisher["map"]["grid"] == {
        "spacing_arcsec": 0.05,
        "half_width_arcsec": extraction.aperture.required_map_half_width_arcsec,
        "annulus": None,
    }
    assert config["psf"]["kernel"]["shape_native"] == [51, 51]
    assert config["modeling"]["enabled"] is False


# ---------------------------------------------------------------------------
# Runner plumbing: reducing one grid map to one rung
# ---------------------------------------------------------------------------


def _grid_map_arrays():
    """Return a five-by-five map whose aperture holds nine nodes."""
    coords = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    q_values = np.full((5, 5), 1.0)
    detectable = np.zeros((5, 5), dtype=bool)
    return coords, q_values, detectable


def test_a_rung_is_measured_inside_the_aperture_only():
    """The aperture bounds q_max, the detected area and the fraction."""
    coords, q_values, detectable = _grid_map_arrays()
    q_values[2, 2] = 42.0
    q_values[0, 0] = 999.0
    detectable[2, 2] = True
    detectable[2, 3] = True
    detectable[3, 2] = True
    metrics = runner._rung_metrics(
        coords, coords, q_values, detectable, 0.05, (0.0, 0.0), 0.075
    )
    assert metrics["q_max"] == 42.0
    assert metrics["detectable_area_arcsec2"] == pytest.approx(3*0.05**2)
    assert metrics["aperture_fraction"] == pytest.approx(3.0/9.0)
    assert metrics["perimeter_clipped"] is False


def test_a_detection_on_the_map_perimeter_is_flagged():
    """A detected region reaching the map edge is a lower bound."""
    coords, q_values, detectable = _grid_map_arrays()
    detectable[0, 2] = True
    metrics = runner._rung_metrics(
        coords, coords, q_values, detectable, 0.05, (0.0, 0.0), 0.075
    )
    assert metrics["perimeter_clipped"] is True
    assert metrics["detectable_area_arcsec2"] == 0.0


def test_a_non_finite_aperture_node_fails_closed():
    """An unevaluated node inside the aperture is not a measurement."""
    coords, q_values, detectable = _grid_map_arrays()
    q_values[2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        runner._rung_metrics(
            coords, coords, q_values, detectable, 0.05, (0.0, 0.0), 0.075
        )


def test_a_map_that_misses_the_aperture_fails_closed():
    """A map holding no aperture node cannot measure the estimands."""
    coords, q_values, detectable = _grid_map_arrays()
    with pytest.raises(ValueError, match="no node inside"):
        runner._rung_metrics(
            coords, coords, q_values, detectable, 0.05, (5.0, 5.0), 0.075
        )


# ---------------------------------------------------------------------------
# Runner plumbing: the artifact
# ---------------------------------------------------------------------------


def _payload_inputs(estimands=None) -> dict:
    """Return the inputs of one finished ladder's artifact payload."""
    extraction = _extraction()
    config = _staged_config(extraction)
    table = [
        {
            "logm": 6.0, "q_max": 0.5, "detectable_area_arcsec2": 0.0,
            "aperture_fraction": 0.0, "perimeter_clipped": False,
            "wall_seconds": 12.5,
        },
        {
            "logm": 6.25, "q_max": 30.0, "detectable_area_arcsec2": 1.5,
            "aperture_fraction": 0.6, "perimeter_clipped": True,
            "wall_seconds": 13.5,
        },
    ]
    return {
        "campaign_uuid": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
        "config_hash_value": "f"*64,
        "system_id": config["run_name"],
        "revision": config["stage0"]["code_revision"],
        "source_asset_path": config["stage0"]["source_asset_path"],
        "source_asset_sha256": config["stage0"]["source_asset_sha256"],
        "ladder": config["ladder"],
        "aperture": config["ladder"]["aperture"],
        "psf": {
            "state": "science35",
            "rms_nm": 35.0,
            "kernel_shape_native": [999, 999],
            "kernel_sha256": "1"*64,
        },
        "table": table,
        "estimands": estimands if estimands is not None else {
            "m_best": walk.Crossing(6.1, 6.0, 6.25, 0.5, 30.0),
            "m10": walk.Crossing(6.05, 6.0, 6.25, 0.0, 0.6),
            "m50": None,
        },
        "stop_reason": walk.STOP_M50,
    }


def test_the_artifact_carries_the_identity_members_the_campaign_validates(
    tmp_path,
):
    """The campaign layer's identity and provenance members are embedded."""
    inputs = _payload_inputs()
    artifact_path = tmp_path/runner.ARTIFACT_NAME
    runner._write_artifact(artifact_path, runner._artifact_payload(**inputs))
    with np.load(artifact_path, allow_pickle=False) as stored:
        assert str(stored["campaign_uuid"]) == inputs["campaign_uuid"]
        assert str(stored["config_hash"]) == inputs["config_hash_value"]
        assert str(stored["code_revision_sha256"]) == inputs["revision"][
            "sha256"
        ]
        assert str(stored["source_asset_sha256"]) == inputs[
            "source_asset_sha256"
        ]
        assert str(stored["system_id"]) == inputs["system_id"]


def test_the_artifact_carries_the_walk_and_its_provenance(tmp_path):
    """The rungs, the estimands, the stop reason and the flags travel."""
    inputs = _payload_inputs()
    artifact_path = tmp_path/runner.ARTIFACT_NAME
    runner._write_artifact(artifact_path, runner._artifact_payload(**inputs))
    with np.load(artifact_path, allow_pickle=False) as stored:
        assert np.array_equal(stored["rung_logm"], [6.0, 6.25])
        assert np.array_equal(stored["rung_q_max"], [0.5, 30.0])
        assert np.array_equal(
            stored["rung_detectable_area_arcsec2"], [0.0, 1.5]
        )
        assert np.array_equal(stored["rung_aperture_fraction"], [0.0, 0.6])
        assert np.array_equal(
            stored["rung_perimeter_clipped"], [False, True]
        )
        assert np.array_equal(stored["rung_wall_seconds"], [12.5, 13.5])
        assert str(stored["stop_reason"]) == walk.STOP_M50
        assert bool(stored["any_perimeter_clipped"]) is True
        assert bool(stored["perimeter_cap_flag"]) is False
        assert str(stored["psf_state"]) == "science35"
        assert float(stored["psf_state_rms_nm"]) == 35.0
        assert np.array_equal(stored["psf_kernel_shape_native"], [999, 999])
        assert str(stored["tier"]) == "selected"
        assert bool(stored["golden"]) is True
        assert float(stored["m_best"]) == 6.1
        assert np.array_equal(stored["m_best_bracket_logm"], [6.0, 6.25])
        assert np.array_equal(stored["m_best_bracket_value"], [0.5, 30.0])


def test_an_uncrossed_estimand_is_recorded_as_null(tmp_path):
    """A ladder that never crossed records not-a-number, not a guess."""
    inputs = _payload_inputs()
    artifact_path = tmp_path/runner.ARTIFACT_NAME
    runner._write_artifact(artifact_path, runner._artifact_payload(**inputs))
    with np.load(artifact_path, allow_pickle=False) as stored:
        assert math.isnan(float(stored["m50"]))
        assert np.all(np.isnan(stored["m50_bracket_logm"]))
        assert np.all(np.isnan(stored["m50_bracket_value"]))


def test_the_artifact_stores_no_map():
    """The re-render doctrine: no electron map and no q map is stored."""
    payload = runner._artifact_payload(**_payload_inputs())
    for member, value in payload.items():
        assert np.asarray(value).ndim <= 1, member
    assert not [
        member for member in payload
        if member.endswith("_2d") or "asimov" in member
    ]


def test_a_capped_system_carries_its_cap_flag():
    """A grid-capped system's aperture estimands travel flagged."""
    inputs = _payload_inputs()
    inputs["aperture"] = dict(inputs["aperture"], perimeter_cap_flag=True)
    payload = runner._artifact_payload(**inputs)
    assert bool(payload["perimeter_cap_flag"]) is True


# ---------------------------------------------------------------------------
# Runner plumbing: the Stage 0 verifications, inherited
# ---------------------------------------------------------------------------


def test_the_stage0_verifications_are_inherited_verbatim():
    """The ladder holds a job to the Stage 0 declarations with Stage 0 code."""
    import run_stage0_observation as stage0

    assert runner._verify_source_asset is stage0._verify_source_asset
    assert runner._verify_code_revision is stage0._verify_code_revision
    assert runner._extract_theta_e_eff is stage0._extract_theta_e_eff


def _asset_config(path, digest):
    """Build the fragment of a staged config the asset check reads."""
    return {
        "lensing": {"source_galaxy": {"light": {"asset_path": str(path)}}},
        "stage0": {
            "source_asset_path": str(path),
            "source_asset_sha256": digest,
        },
    }


def test_changed_asset_bytes_refuse_to_walk(tmp_path):
    """A template whose bytes moved under the design refuses to render."""
    path = tmp_path/"template.npz"
    path.write_bytes(b"prepared source asset bytes")
    config = _asset_config(path, hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_bytes(b"prepared source asset bytes, edited")
    with pytest.raises(ValueError, match="asset bytes moved"):
        runner._verify_source_asset(config)


def test_a_different_source_revision_refuses_to_walk():
    """A resume under moved code refuses rather than mix code states."""
    config = {
        "stage0": {
            "code_revision": {
                "git_hash": "a"*40,
                "git_dirty": False,
                "sha256": "b"*64,
            },
        },
    }
    with pytest.raises(ValueError, match="check out the recorded revision"):
        runner._verify_code_revision(config)


def test_an_extraction_algorithm_this_checkout_lacks_refuses_to_walk():
    """A ladder frozen on another extraction algorithm refuses to render."""
    config = {
        "lensing": {"lens_galaxy": {"mass": {"einstein_radius": 1.0}}},
        "stage0": {
            "system_id": "sys0007",
            "theta_e_extraction": {
                "algorithm_id": "tangential_critical_curve_marching_squares_v0",
                "choice_rule_id": (
                    "largest_area_closed_curve_enclosing_lens_centre"
                ),
                "extraction_grid": {
                    "pixel_scale_arcsec": 0.01,
                    "half_width_factor": 4.0,
                },
                "theta_e_factor": 2.0,
                "computational_margin_fraction": 0.1,
                "guards": {
                    "closure_tolerance_pixels": 0.5,
                    "border_margin_pixels": 2.0,
                    "min_contour_vertices": 32,
                },
            },
            "theta_e_eff_arcsec": 1.0,
            "theta_e_eff_tolerance_fractional": 0.02,
        },
    }
    with pytest.raises(ValueError, match="this checkout implements"):
        runner._extract_theta_e_eff(config)


def test_a_contour_that_leaves_the_generator_curve_refuses_to_walk():
    """A re-extraction that no longer reproduces the bound curve refuses."""
    pytest.importorskip("autolens")

    config = {
        "lensing": {
            "lens_galaxy": {
                "redshift": 0.2,
                "mass": {
                    "type": "Isothermal",
                    "centre": [0.0, 0.0],
                    "ell_comps": [0.0, 0.0],
                    "einstein_radius": 1.0,
                },
            },
        },
        "stage0": {
            "system_id": "sys0007",
            "theta_e_extraction": {
                "algorithm_id": "tangential_critical_curve_marching_squares_v1",
                "choice_rule_id": (
                    "largest_area_closed_curve_enclosing_lens_centre"
                ),
                "extraction_grid": {
                    "pixel_scale_arcsec": 0.01,
                    "half_width_factor": 4.0,
                },
                "theta_e_factor": 2.0,
                "computational_margin_fraction": 0.1,
                "guards": {
                    "closure_tolerance_pixels": 0.5,
                    "border_margin_pixels": 2.0,
                    "min_contour_vertices": 32,
                },
            },
            "theta_e_eff_arcsec": 1.0,
            "theta_e_eff_tolerance_fractional": 0.02,
            "theta_e_contour_sha256": "0"*64,
            "theta_e_aperture_sha256": "1"*64,
        },
    }
    with pytest.raises(ValueError, match="no longer reproduces"):
        runner._extract_theta_e_eff(config)


def test_the_reused_detector_exposes_the_state_a_rung_advances():
    """The rung advance is pinned to the detector state that carries mass.

    Only the template configurations and the cached grid template engine
    depend on the injected mass, so a rename there would leave the
    reused detector rendering every rung at the first rung's mass. This
    fails on the rename instead.
    """
    pytest.importorskip("autolens")

    import inspect

    from hwoslaps.modeling.fisher_detector import FisherDetector

    source = inspect.getsource(FisherDetector.__init__)
    for attribute in (
        "self.full_config",
        "self.map_config_template",
        "self.map_config_template_truth",
        "self._jax_grid_engine",
    ):
        assert f"{attribute} =" in source, attribute
    assert "self._jax_grid_engine" in inspect.getsource(
        FisherDetector._grid_signal_iterator_jax
    )


def _stand_in_psf_data(config, pixel_scale):
    """Return a PSF system carrying only the kernel the detector needs.

    Generating the real PSF pulls in the pupil-side optical stack, which
    this contract has nothing to say about: what is under test is that
    one reused detector renders each rung at that rung's mass.
    """
    from hwoslaps.psf.utils import PSFData, make_pyauto_kernel

    offsets = np.arange(-2, 3, dtype=float)
    values = np.exp(-(offsets[:, None]**2 + offsets[None, :]**2)/2.0)
    return PSFData(
        psf=None,
        wavefront=None,
        telescope_data={},
        kernel=make_pyauto_kernel(values, pixel_scale),
        kernel_pixel_scale=pixel_scale,
        wavelength_nm=500.0,
        pupil_diameter_m=7.225765,
        focal_length_m=144.0,
        pixel_scale_arcsec=pixel_scale,
        sampling_factor=5.0,
        requested_sampling_factor=5.0,
        used_sampling_factor=5.0,
        integer_subsampling_factor=1,
        num_segments=19,
        segment_flat_to_flat_m=1.43,
        segment_point_to_point_m=1.65,
        gap_size_m=0.006,
        num_rings=2,
        config=config["psf"],
    )


def test_one_reused_detector_renders_each_rung_at_its_own_mass(monkeypatch):
    """The whole economics of the job: one detector, many rungs.

    Every expensive part of the detector is baseline work, so the ladder
    builds it once and advances the rung in place. Only the template
    configurations and the cached grid template engine carry the mass,
    and the cache is the trap: leaving it in place reports the new mass
    in the map metadata while still rendering the old one. A heavier
    rung must therefore be louder, and a revisited rung must reproduce
    itself exactly.
    """
    pytest.importorskip("autolens")
    pytest.importorskip("jax")

    monkeypatch.setenv("HWOSLAPS_DISABLE_TQDM", "1")
    monkeypatch.setenv("HWOSLAPS_DISABLE_FISHER_TIMING", "1")

    from hwoslaps.config.validation import validate_or_raise

    pixel_scale = 0.05
    scene_path = PROJECT_ROOT/"configs"/"scenes"/"scene1_smooth_ring.yaml"
    with scene_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["plotting"] = {"enabled": False, "output_dir": "outputs"}
    config["lensing"]["grid"] = {"shape": [60, 60], "pixel_scale": pixel_scale}
    config["lensing"]["subhalo"]["enabled"] = True
    config["psf"]["kernel"]["shape_native"] = [5, 5]
    config["modeling"]["enabled"] = True
    fisher = config["modeling"]["fisher"]
    fisher["mode"] = "map"
    fisher["mask_mode"] = runner.MASK_MODE
    fisher["map"]["type"] = "grid"
    fisher["map"]["engine"] = runner.ENGINE
    fisher["map"]["detection_q_threshold"] = walk.Q_THRESHOLD
    fisher["map"]["grid"] = {
        "spacing_arcsec": 0.2,
        "half_width_arcsec": 0.6,
        "annulus": None,
    }
    validate_or_raise(config)

    runner._enable_float64()
    detector = runner._build_detector(
        config, _stand_in_psf_data(config, pixel_scale)
    )
    measured = []
    for logm in (8.0, 9.5, 8.0):
        runner._point_detector_at_rung(detector, logm)
        grid_map = detector.compute_grid_map()
        assert grid_map.subhalo_mass == pytest.approx(10.0**logm)
        measured.append(runner._rung_metrics(
            grid_map.y_coords,
            grid_map.x_coords,
            grid_map.q_asimov_2d,
            grid_map.detectable_mask_2d,
            grid_map.spacing_arcsec,
            (0.0, 0.0),
            0.5,
        ))
    assert measured[1]["q_max"] > measured[0]["q_max"]
    assert measured[2] == measured[0]


def test_an_existing_artifact_is_not_overwritten(tmp_path):
    """A finished job refuses to be replaced without an explicit force."""
    config = _staged_config()
    config["plotting"]["output_dir"] = str(tmp_path)
    config_path = tmp_path/"config.yaml"
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=True)
    output_dir = runner._output_dir(config)
    output_dir.mkdir(parents=True)
    (output_dir/runner.ARTIFACT_NAME).write_bytes(b"")
    with pytest.raises(ValueError, match="pass --force"):
        runner.main([str(config_path)])


# ---------------------------------------------------------------------------
# The no-random-stream gate
# ---------------------------------------------------------------------------


def test_the_ladder_consumes_no_random_stream(monkeypatch):
    """Every non-rendering path runs with numpy's generators disarmed.

    Fisher ladders are deterministic and the manifest declares that they
    draw nothing, so reaching for a generator anywhere in the walk, the
    estimands, the map reduction or the artifact is a fail-closed
    condition rather than a style question.
    """
    def _refuse(*args, **kwargs):
        raise AssertionError("the ladder runner drew from a random stream")

    monkeypatch.setattr(np.random, "default_rng", _refuse)
    monkeypatch.setattr(np.random, "Generator", _refuse)
    monkeypatch.setattr(np.random, "RandomState", _refuse)
    monkeypatch.setattr(np.random, "seed", _refuse)

    extraction = _extraction()
    config = _staged_config(extraction)
    ladder = runner._verify_ladder_block(config)
    runner._verify_psf_state(config)
    aperture = runner._verify_aperture(config, extraction)
    runner._rung_config(config, ladder, aperture)
    policy = walk.policy_from_mass_ladder(ladder["mass_ladder"])

    coords, q_values, detectable = _grid_map_arrays()
    table = []
    for _ in range(200):
        step = walk.next_rung(table, policy)
        if step.logm is None:
            stop_reason = step.stop_reason
            break
        detected = min(9, max(0, int(round((step.logm - 7.0)*4.0))))
        q_values[:] = 1.0
        detectable[:] = False
        flat = np.argwhere(
            coords[:, None]**2 + coords[None, :]**2 <= 0.075**2
        )
        for index in flat[:detected]:
            detectable[index[0], index[1]] = True
        q_values[2, 2] = 10.0**((step.logm - 8.0)*2.0)
        row = {"logm": step.logm}
        row.update(runner._rung_metrics(
            coords, coords, q_values, detectable, 0.05, (0.0, 0.0), 0.075
        ))
        row["wall_seconds"] = 0.0
        table.append(row)
    else:
        pytest.fail("the walk did not terminate")

    payload = runner._artifact_payload(
        campaign_uuid="",
        config_hash_value="0"*64,
        system_id=str(config["run_name"]),
        revision=config["stage0"]["code_revision"],
        source_asset_path=config["stage0"]["source_asset_path"],
        source_asset_sha256=config["stage0"]["source_asset_sha256"],
        ladder=ladder,
        aperture=aperture,
        psf={
            "state": "science35",
            "rms_nm": 35.0,
            "kernel_shape_native": [999, 999],
            "kernel_sha256": "2"*64,
        },
        table=table,
        estimands={
            "m_best": walk.log_linear_crossing(table, policy),
            "m10": walk.aperture_fraction_crossing(
                table, policy, walk.M10_LEVEL
            ),
            "m50": walk.aperture_fraction_crossing(
                table, policy, walk.M50_LEVEL
            ),
        },
        stop_reason=stop_reason,
    )
    assert str(payload["stop_reason"]) == stop_reason
    assert payload["rung_logm"].size == len(table)


def test_the_jax_compilation_cache_is_off_unless_a_directory_is_named(
    monkeypatch, tmp_path
):
    """The persistent compilation cache is opt-in and never on by default.

    A production ladder must behave the same whether or not a machine
    happens to carry a cache, so the hook does nothing at all until
    `HWOSLAPS_JAX_CACHE_DIR` names a directory, and then points JAX at
    exactly that directory.
    """
    jax = pytest.importorskip("jax")

    monkeypatch.delenv(runner.JAX_CACHE_DIR_ENV, raising=False)
    monkeypatch.setattr(
        jax.config, "update", lambda *args: pytest.fail("cache configured")
    )
    runner._enable_jax_compilation_cache()

    updates = {}
    monkeypatch.setattr(
        jax.config, "update", lambda name, value: updates.__setitem__(name, value)
    )
    monkeypatch.setenv(runner.JAX_CACHE_DIR_ENV, str(tmp_path/"jax-cache"))
    runner._enable_jax_compilation_cache()
    assert updates["jax_compilation_cache_dir"] == str(tmp_path/"jax-cache")
    assert updates["jax_persistent_cache_min_compile_time_secs"] == 0.0
