"""Tests for sparse nonlinear trial selection."""

from __future__ import annotations

from types import SimpleNamespace

from hwoslaps.modeling.nonlinear.trial import SubhaloTrial, trial_from_lensing_truth
from hwoslaps.modeling.nonlinear.trial_selection import (
    TrialSelectionConfig,
    select_trials,
)


def _trial(index: int, q_value: float) -> SubhaloTrial:
    return SubhaloTrial(
        case_id=f"case_{index}",
        mass_msun=1.0e7,
        position_yx_arcsec=(0.0, float(index)),
        model="PointMass",
        profile_class="PointMass",
        lens_redshift=0.2,
        source_redshift=0.6,
        einstein_radius_arcsec=0.001,
        fisher_q=q_value,
    )


def test_trial_from_lensing_truth_preserves_nfw_parameters():
    lensing = SimpleNamespace(
        has_subhalo=True,
        subhalo_model="NFW",
        subhalo_mass=1.0e7,
        subhalo_position=(0.1, -0.2),
        lens_redshift=0.2,
        source_redshift=0.6,
        subhalo_einstein_radius=None,
        subhalo_kappa_s=0.01,
        subhalo_scale_radius_arcsec=0.2,
        subhalo_concentration=25.0,
        subhalo_concentration_model="moline2017_eq7",
        subhalo_concentration_source="source",
        subhalo_concentration_x_sub=1.0,
        subhalo_concentration_h=0.6774,
    )

    trial = trial_from_lensing_truth(lensing)

    assert trial.profile_class == "NFWSph"
    assert trial.kappa_s == 0.01
    assert trial.scale_radius_arcsec == 0.2
    assert trial.concentration == 25.0


def test_select_trials_prioritizes_extremes_and_near_threshold_cases():
    trials = [_trial(index, q_value) for index, q_value in enumerate([1.0, 4.0, 9.5, 10.5, 20.0])]

    selected = select_trials(
        trials,
        TrialSelectionConfig(
            strategy="near_threshold",
            max_cases=4,
            include_min_q=True,
            include_max_q=True,
        ),
    )

    selected_ids = {trial.case_id for trial in selected}
    assert "case_0" in selected_ids
    assert "case_4" in selected_ids
    assert len(selected) == 4


def test_stratified_selection_is_deterministic():
    trials = [_trial(index, float(index)) for index in range(12)]
    config = TrialSelectionConfig(
        strategy="stratified",
        max_cases=6,
        q_bins=(0.0, 4.0, 8.0, 12.0),
        positions_per_bin=1,
        random_seed=99,
    )

    first = select_trials(trials, config)
    second = select_trials(trials, config)

    assert [trial.case_id for trial in first] == [trial.case_id for trial in second]
