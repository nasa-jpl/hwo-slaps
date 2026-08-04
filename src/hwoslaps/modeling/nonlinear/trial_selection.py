"""Select sparse nonlinear validation trials from Fisher results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .trial import (
    SubhaloTrial,
    trial_from_fisher_map_position,
    trial_from_lensing_truth,
)


@dataclass(frozen=True)
class TrialSelectionConfig:
    """Configuration for selecting sparse nonlinear validation cases."""

    strategy: str = "stratified"
    max_cases: int = 24
    q_bins: Sequence[float] = (0.0, 5.0, 10.0, 20.0, np.inf)
    positions_per_bin: int = 3
    include_local_injection: bool = True
    include_max_q: bool = True
    include_min_q: bool = True
    random_seed: int = 11


def trials_from_fisher_local(fisher_data: object, lensing_test: object) -> List[SubhaloTrial]:
    """Create validation trials from the local Fisher output.

    Parameters
    ----------
    fisher_data : `object`
        Fisher result payload.
    lensing_test : `object`
        Lensing data containing the injected subhalo truth.

    Returns
    -------
    trials : `list` [`SubhaloTrial`]
        Local validation trial list.
    """
    if getattr(fisher_data, "local", None) is None:
        return []
    trial = trial_from_lensing_truth(lensing_test, case_id="local_injection")
    local = fisher_data.local
    fisher_q = local.q_asimov_local
    if fisher_q is None:
        fisher_q = local.delta_chi2_profiled
    fisher_z = local.z_asimov_local
    if fisher_z is None and fisher_q is not None and fisher_q >= 0.0:
        fisher_z = fisher_q**0.5

    return [
        SubhaloTrial(
            **{
                **trial.to_dict(),
                "fisher_q": fisher_q,
                "fisher_z": fisher_z,
                "fisher_delta_log_l_equiv": 0.5*fisher_q if fisher_q is not None else None,
                "metadata": {**trial.metadata, "source": "fisher_local"},
            }
        )
    ]


def trials_from_fisher_map(
    fisher_data: object,
    full_config: dict,
    lensing_reference: object,
) -> List[SubhaloTrial]:
    """Create candidate validation trials from a Fisher map.

    Parameters
    ----------
    fisher_data : `object`
        Fisher result payload.
    full_config : `dict`
        Full HWO-SLAPS configuration.
    lensing_reference : `object`
        Lensing data used for redshifts and available profile metadata.

    Returns
    -------
    trials : `list` [`SubhaloTrial`]
        Map-position validation trials.
    """
    fmap = getattr(fisher_data, "map", None)
    if fmap is None:
        return []
    mass_msun = full_config["lensing"]["subhalo"]["mass"]
    q_values = fmap.q_asimov_local_by_position
    if q_values is None:
        q_values = fmap.delta_chi2_profiled_by_position
    trials = []
    for index, position in enumerate(np.asarray(fmap.positions_yx, dtype=float)):
        q_value = float(q_values[index])
        trials.append(
            trial_from_fisher_map_position(
                full_config=full_config,
                lensing_reference=lensing_reference,
                mass_msun=mass_msun,
                position_yx_arcsec=(float(position[0]), float(position[1])),
                fisher_q=q_value,
                case_id=f"map_{index:04d}",
            )
        )
    return trials


def _dedupe_trials(trials: Sequence[SubhaloTrial]) -> List[SubhaloTrial]:
    """Remove duplicate case IDs while preserving order."""
    seen = set()
    unique = []
    for trial in trials:
        if trial.case_id in seen:
            continue
        seen.add(trial.case_id)
        unique.append(trial)
    return unique


def select_trials(
    candidate_trials: Sequence[SubhaloTrial],
    config: TrialSelectionConfig,
) -> List[SubhaloTrial]:
    """Select a sparse validation subset from candidate trials.

    Parameters
    ----------
    candidate_trials : sequence [`SubhaloTrial`]
        Candidate trials with Fisher metrics.
    config : `TrialSelectionConfig`
        Selection configuration.

    Returns
    -------
    selected : `list` [`SubhaloTrial`]
        Selected trial subset.
    """
    if config.strategy not in {"explicit", "local", "stratified", "near_threshold"}:
        raise ValueError("Unsupported trial-selection strategy")
    if config.max_cases <= 0:
        raise ValueError("max_cases must be positive")

    candidates = list(candidate_trials)
    if config.strategy == "explicit":
        return _dedupe_trials(candidates[:config.max_cases])
    if config.strategy == "local":
        return _dedupe_trials(candidates[:1])

    rng = np.random.default_rng(config.random_seed)
    selected = []
    candidates_with_q = [
        trial for trial in candidates if trial.fisher_q is not None and np.isfinite(trial.fisher_q)
    ]
    if not candidates_with_q:
        return _dedupe_trials(candidates[:config.max_cases])

    if config.include_min_q:
        selected.append(min(candidates_with_q, key=lambda trial: trial.fisher_q))
    if config.include_max_q:
        selected.append(max(candidates_with_q, key=lambda trial: trial.fisher_q))

    if config.strategy == "near_threshold":
        threshold_sorted = sorted(
            candidates_with_q,
            key=lambda trial: abs(trial.fisher_q - 10.0),
        )
        selected.extend(threshold_sorted)
        return _dedupe_trials(selected)[:config.max_cases]

    q_bins = list(config.q_bins)
    for lower, upper in zip(q_bins[:-1], q_bins[1:]):
        in_bin = [
            trial
            for trial in candidates_with_q
            if lower <= trial.fisher_q < upper
        ]
        if not in_bin:
            continue
        count = min(config.positions_per_bin, len(in_bin))
        indices = rng.choice(len(in_bin), size=count, replace=False)
        selected.extend([in_bin[int(index)] for index in indices])

    return _dedupe_trials(selected)[:config.max_cases]
