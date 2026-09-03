#!/usr/bin/env python
"""Harvest and review the nonlinear-validation campaign.

Collects every eligible arm artifact of the campaign the manifest
declares, verifies the identity chain fail-closed (job binding,
campaign uuid, code revision, restamped configuration hash, declared
sampler seed re-derived from the freeze rule, arm declaration, fit
settings against the freeze protocol, matched kernels, non-degenerate
mask support), and writes ``harvest/harvest.json`` with one row per fit
pair plus ``harvest/review.json`` with the integrity census and the
declared success criteria: recovery at the first Fisher-positive rung,
below-rung consistency, within-rung rank fidelity, control tallies,
the golden-five bridge comparison, replicate scatter, morphology
transfer and censoring consistency.

An incomplete campaign is reported and fails the run unless
``--allow-incomplete`` is passed (useful mid-campaign); science
summaries are then computed over the rows present.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"scripts"))

from run_nonlinear_validation import (  # noqa: E402
    derive_noise_seed,
    derive_sampler_seed,
    system_index,
)
from generate_nonlinear_validation_campaign import eligible_arms  # noqa: E402

Q_FIT_THRESHOLD = 10.0
DLOGZ_THRESHOLD = 5.0

REPLICATE_ARMS = ("asimov_injected", "asimov_injected_r1", "asimov_injected_r2")

CHECKED_FIT_SETTINGS = (
    "kernel_shape_native",
    "n_live_smooth",
    "n_live_subhalo_search",
    "n_live_subhalo_fixed",
    "maxcall",
    "jax_n_batch",
    "number_of_cores",
    "log10_m200_range",
    "nautilus_training_workers",
)


def spearman_rank_correlation(first, second) -> float:
    """Spearman rank correlation of two equal-length sequences.

    Parameters
    ----------
    first, second : `numpy.ndarray`
        Paired samples.

    Returns
    -------
    correlation : `float`
        The Pearson correlation of the two rank vectors, with ties
        assigned average ranks.
    """
    def ranks(values):
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="stable")
        rank = np.empty(len(values), dtype=float)
        rank[order] = np.arange(1, len(values) + 1, dtype=float)
        for value in np.unique(values):
            matches = values == value
            if int(np.count_nonzero(matches)) > 1:
                rank[matches] = rank[matches].mean()
        return rank

    a = ranks(first)
    b = ranks(second)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a**2).sum()*(b**2).sum()))
    if denominator == 0.0:
        raise ValueError("Rank correlation is undefined for constant input")
    return float((a*b).sum()/denominator)


def _row(job: dict, arm: str, payload: dict) -> dict:
    """Reduce one arm artifact to its harvest row."""
    recovery = (payload.get("case") or {}).get("subhalo_recovery")
    return {
        "system_id": payload["system_id"],
        "tier": job["tier"],
        "report_tiers": job["report_tiers"],
        "template": job["template"],
        "golden": job["golden"],
        "arm": arm,
        "fit_mode": payload["arm_declaration"]["fit_mode"],
        "dataset_kind": payload["arm_declaration"]["dataset_kind"],
        "subhalo_in_truth": payload["arm_declaration"]["subhalo_in_truth"],
        "rung_name": payload["arm_declaration"]["rung"],
        "injection_logm": payload["rung"]["logm"],
        "censored": payload["censored"],
        "position_yx_arcsec": payload["rung"]["position_yx_arcsec"],
        "q_f_matched": payload["rung"]["q_f_matched"],
        "q_f_production": payload["rung"]["q_f_production_at_position"],
        "q_fit": payload["q_fit"],
        "delta_log_evidence": payload["delta_log_evidence"],
        "delta_log_likelihood": payload["delta_log_likelihood"],
        "smooth_status": payload["smooth_status"],
        "subhalo_status": payload["subhalo_status"],
        "quality_flags": payload["quality_flags"],
        "subhalo_recovery": recovery,
        "n_unmasked_pixels": payload["n_unmasked_pixels"],
        "sampler_seed": payload["sampler_seed"],
        "noise_seed": payload.get("noise_seed"),
        "noise_replicate": payload.get("noise_replicate"),
        "ladder_campaign_uuid": payload.get("ladder_campaign_uuid"),
        "ladder_config_hash": payload.get("ladder_config_hash"),
        "fit_pair_s": payload["timings"]["fit_pair_s"],
    }


def expected_provenance(job: dict, arm: str, manifest: dict) -> tuple:
    """Resolve one row's expected code revision and config hash.

    Parameters
    ----------
    job : `dict`
        The manifest job entry.
    arm : `str`
        Arm name of the row.
    manifest : `dict`
        Campaign manifest, optionally carrying ``amendments`` records
        for jobs rerun at a later revision from restaged configurations.

    Returns
    -------
    revision_sha256 : `str`
        Expected artifact code-revision digest.
    config_hash : `str`
        Expected artifact staged-configuration hash.
    """
    label = f"{job['run_name']}/{arm}"
    revision_sha256 = manifest["code_revision"]["sha256"]
    config_hash = job["restamped_config_hash"]
    for amendment in manifest.get("amendments", []):
        entry = amendment["jobs"].get(label)
        if entry is not None:
            revision_sha256 = amendment["code_revision"]["sha256"]
            config_hash = entry["restamped_config_hash"]
    return revision_sha256, config_hash


def _verify_row(
    job: dict,
    arm: str,
    payload: dict,
    manifest: dict,
    protocol: dict,
) -> list:
    """Return the integrity findings of one arm artifact."""
    findings = []
    label = f"{job['run_name']}/{arm}"

    if payload["system_id"] != job["run_name"]:
        findings.append(
            f"{label}: artifact belongs to {payload['system_id']!r}"
        )
    if payload["arm"] != arm:
        findings.append(f"{label}: artifact records arm {payload['arm']!r}")

    declared = protocol["arms"][arm]
    recorded = payload["arm_declaration"]
    for key in ("arm_index", "dataset_kind", "subhalo_in_truth",
                "fit_mode", "rung", "sample"):
        if recorded.get(key) != declared[key]:
            findings.append(
                f"{label}: arm declaration {key} is {recorded.get(key)!r}, "
                f"protocol declares {declared[key]!r}"
            )

    expected_seed = derive_sampler_seed(
        int(protocol["seeds"]["entropy"]),
        system_index(job["run_name"]),
        int(declared["arm_index"]),
    )
    if int(payload["sampler_seed"]) != expected_seed:
        findings.append(
            f"{label}: sampler seed {payload['sampler_seed']} is not the "
            f"declared {expected_seed}"
        )

    if payload["campaign_uuid"] != manifest["campaign_uuid"]:
        findings.append(
            f"{label}: campaign uuid {payload['campaign_uuid']!r}"
        )
    revision_sha256, config_hash = expected_provenance(job, arm, manifest)
    if payload["code_revision"]["sha256"] != revision_sha256:
        findings.append(
            f"{label}: code revision "
            f"{payload['code_revision']['sha256'][:16]}"
        )
    if payload["staged_config_hash"] != config_hash:
        findings.append(
            f"{label}: staged config hash "
            f"{payload['staged_config_hash'][:16]} is not the manifest's "
            f"{config_hash[:16]}"
        )

    for key in ("ladder_campaign_uuid", "ladder_config_hash"):
        if key in job and payload.get(key) != job[key]:
            findings.append(
                f"{label}: {key} {payload.get(key)!r} is not the manifest's "
                f"{job[key]!r}"
            )
    if "censored" in job and payload.get("censored") != job["censored"]:
        findings.append(
            f"{label}: artifact censored {payload.get('censored')!r} is not "
            f"the job's {job['censored']!r}"
        )

    rung = payload.get("rung")
    if not isinstance(rung, dict) or "logm" not in rung:
        findings.append(f"{label}: artifact rung is missing logm")
    else:
        logm = rung["logm"]
        if isinstance(logm, bool) or not isinstance(logm, (int, float)):
            findings.append(f"{label}: artifact rung logm is {logm!r}")
        elif payload.get("censored") is True and declared["rung"] == "top":
            if float(logm) != 9.5:
                findings.append(
                    f"{label}: censored top rung logm {logm!r} is not 9.5"
                )
    if payload.get("censored") is True and declared["rung"] == "below":
        findings.append(f"{label}: censored artifact carries a below rung")

    if manifest.get("campaign") is not None:
        if payload.get("tier") != job["tier"]:
            findings.append(
                f"{label}: artifact tier {payload.get('tier')!r} is not "
                f"the job tier {job['tier']!r}"
            )
        if recorded.get("noise_replicate") != declared.get("noise_replicate"):
            findings.append(
                f"{label}: arm declaration noise_replicate is "
                f"{recorded.get('noise_replicate')!r}, protocol declares "
                f"{declared.get('noise_replicate')!r}"
            )
        if payload.get("schema_version") != 3:
            findings.append(
                f"{label}: artifact schema_version "
                f"{payload.get('schema_version')!r}, expected 3"
            )
        if "staged_global_seed" not in job:
            findings.append(f"{label}: manifest is missing staged_global_seed")
        elif (
            isinstance(job["staged_global_seed"], bool)
            or not isinstance(job["staged_global_seed"], int)
        ):
            findings.append(
                f"{label}: manifest staged_global_seed is "
                f"{job['staged_global_seed']!r}, not an int"
            )
        declaration_replicate = declared.get("noise_replicate")
        payload_noise_seed = payload.get("noise_seed")
        payload_noise_replicate = payload.get("noise_replicate")
        payload_noise_spawn_key = payload.get("noise_spawn_key")
        for key in ("noise_seed", "noise_replicate", "noise_spawn_key"):
            if key not in payload:
                findings.append(f"{label}: artifact is missing {key}")
        if declaration_replicate is None:
            expected_noise_seed = job.get("staged_global_seed")
            expected_replicate = 0
            expected_spawn_key = None
        else:
            expected_noise_seed = derive_noise_seed(
                int(protocol["seeds"]["entropy"]),
                int(declaration_replicate),
                system_index(job["run_name"]),
            )
            expected_replicate = int(declaration_replicate)
            expected_spawn_key = [
                6,
                int(declaration_replicate),
                system_index(job["run_name"]),
            ]
        if (
            isinstance(payload_noise_seed, bool)
            or not isinstance(payload_noise_seed, int)
            or payload_noise_seed != expected_noise_seed
        ):
            findings.append(
                f"{label}: noise seed {payload_noise_seed!r} is not the "
                f"declared {expected_noise_seed!r}"
            )
        if (
            isinstance(payload_noise_replicate, bool)
            or not isinstance(payload_noise_replicate, int)
            or payload_noise_replicate != expected_replicate
        ):
            findings.append(
                f"{label}: noise replicate {payload_noise_replicate!r} is not "
                f"the declared {expected_replicate!r}"
            )
        if payload_noise_spawn_key != expected_spawn_key:
            findings.append(
                f"{label}: noise spawn key {payload_noise_spawn_key!r} is not "
                f"the declared {expected_spawn_key!r}"
            )

    fit_block = protocol["fit"]
    for key in CHECKED_FIT_SETTINGS:
        if payload["fit_settings"].get(key) != fit_block[key]:
            findings.append(
                f"{label}: fit setting {key} is "
                f"{payload['fit_settings'].get(key)!r}, protocol declares "
                f"{fit_block[key]!r}"
            )

    if payload["kernel_sha256"] != payload["truth_kernel_sha256"]:
        findings.append(f"{label}: fit kernel is not the truth kernel")
    if int(payload["n_unmasked_pixels"]) <= 0:
        findings.append(f"{label}: degenerate mask support")
    for side in ("smooth_status", "subhalo_status"):
        if payload[side] != "success":
            findings.append(f"{label}: {side} {payload[side]!r}")
    if (
        payload["smooth_status"] == "success"
        and payload["subhalo_status"] == "success"
        and payload.get("q_fit") is None
    ):
        findings.append(
            f"{label}: both fit statuses are success but q_fit is None"
        )
    return findings


def clopper_pearson(count: int, n: int, confidence: float = 0.95) -> tuple:
    """Return an exact two-sided Clopper-Pearson binomial interval.

    Parameters
    ----------
    count : `int`
        Number of successes.
    n : `int`
        Number of tested observations.
    confidence : `float`, optional
        Requested two-sided confidence level.

    Returns
    -------
    interval : `tuple` [`float`, `float`]
        Lower and upper interval limits.

    Raises
    ------
    ValueError
        Raised when the count, sample size or confidence is invalid.
    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"count must be an integer, got {count!r}")
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if count < 0 or count > n:
        raise ValueError(f"count must lie between 0 and n, got {count} of {n}")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError(
            f"confidence must lie strictly between 0 and 1, got {confidence!r}"
        )
    from scipy.stats import beta

    alpha = 1.0 - float(confidence)
    lower = 0.0 if count == 0 else float(
        beta.ppf(alpha/2.0, count, n - count + 1)
    )
    upper = 1.0 if count == n else float(
        beta.ppf(1.0 - alpha/2.0, count + 1, n - count)
    )
    return lower, upper


def _q_verdict(row) -> bool:
    """Screening-convention verdict of one row."""
    return (
        row["q_fit"] is not None and float(row["q_fit"]) >= Q_FIT_THRESHOLD
    )


def _science_v3(rows) -> dict:
    """Compute the freeze v3 success-criteria summaries."""
    science = {}
    for arm in ("asimov_injected", "noisy_injected"):
        injected = [
            row for row in rows if row["arm"] == arm and not row["censored"]
        ]
        recovered = [row for row in injected if _q_verdict(row)]
        pairs = [
            (float(row["q_fit"]), float(row["q_f_matched"]))
            for row in injected
            if row["q_fit"] is not None
        ]
        overshoot = [
            float(row["q_f_matched"]) - Q_FIT_THRESHOLD for row in injected
        ]
        science[arm] = {
            "n": len(injected),
            "q_fit_none": sum(row["q_fit"] is None for row in injected),
            "recovered_at_first_positive_rung": len(recovered),
            "recovery_fraction": (
                len(recovered)/len(injected) if injected else None
            ),
            "median_q_f_overshoot": (
                float(np.median(overshoot)) if overshoot else None
            ),
            "spearman_within_rung_q_fit_vs_q_f_matched": (
                spearman_rank_correlation(
                    [pair[0] for pair in pairs],
                    [pair[1] for pair in pairs],
                )
                if len(pairs) >= 3
                else None
            ),
            "median_q_fit": (
                float(np.median([pair[0] for pair in pairs]))
                if pairs
                else None
            ),
        }

    below = [row for row in rows if row["arm"] == "asimov_below"]
    below_tested = [row for row in below if row["q_fit"] is not None]
    below_threshold = sum(
        float(row["q_fit"]) < Q_FIT_THRESHOLD for row in below_tested
    )
    science["asimov_below"] = {
        "n": len(below),
        "n_tested": len(below_tested),
        "q_fit_none": len(below) - len(below_tested),
        "below_threshold": below_threshold,
        "below_rung_consistency_fraction": (
            below_threshold/len(below_tested)
            if below_tested
            else None
        ),
        "exceedances": [
            f"{row['system_id']} q_fit {row['q_fit']}"
            for row in below
            if _q_verdict(row)
        ],
    }

    controls = [row for row in rows if row["arm"] == "noisy_control"]
    science["noisy_control"] = {
        "n": len(controls),
        "q_fit_none": sum(row["q_fit"] is None for row in controls),
        "screening_convention_tally_q_fit": sum(
            1 for row in controls if _q_verdict(row)
        ),
        "bayesian_convention_tally_dlogz": sum(
            1
            for row in controls
            if row["delta_log_evidence"] is not None
            and float(row["delta_log_evidence"]) > DLOGZ_THRESHOLD
        ),
        "note": (
            "sample-conditional tallies under the declared protocol, not "
            "calibrated false-positive rates"
        ),
    }

    bridge = [row for row in rows if row["arm"] == "asimov_fixed_bridge"]
    science["asimov_fixed_bridge"] = {
        "n": len(bridge),
        "systems": [
            {
                "system_id": row["system_id"],
                "q_fit_fixed": row["q_fit"],
                "q_f_matched": row["q_f_matched"],
                "q_f_production": row["q_f_production"],
            }
            for row in sorted(bridge, key=lambda row: row["system_id"])
        ],
    }

    replicate_rows = {}
    for row in rows:
        if row["arm"] in REPLICATE_ARMS and row["golden"]:
            replicate_rows.setdefault(row["system_id"], {})[row["arm"]] = row
    replicate_summary = []
    for system_id in sorted(replicate_rows):
        group = replicate_rows[system_id]
        q_values = [
            float(group[arm]["q_fit"])
            for arm in REPLICATE_ARMS
            if arm in group and group[arm]["q_fit"] is not None
        ]
        z_values = [
            float(group[arm]["delta_log_evidence"])
            for arm in REPLICATE_ARMS
            if arm in group
            and group[arm]["delta_log_evidence"] is not None
        ]
        replicate_summary.append({
            "system_id": system_id,
            "n_seeds": len(q_values),
            "q_fit_values": q_values,
            "q_fit_spread": (
                max(q_values) - min(q_values) if q_values else None
            ),
            "dlogz_spread": (
                max(z_values) - min(z_values) if z_values else None
            ),
        })
    science["replicate_scatter_golden"] = {
        "systems": replicate_summary,
        "max_q_fit_spread": max(
            (entry["q_fit_spread"] for entry in replicate_summary
             if entry["q_fit_spread"] is not None),
            default=None,
        ),
    }

    per_template = {}
    for row in rows:
        if row["arm"] != "asimov_injected" or row["censored"]:
            continue
        entry = per_template.setdefault(
            row["template"], {"n": 0, "recovered": 0, "q_fit": []}
        )
        entry["n"] += 1
        entry["recovered"] += int(_q_verdict(row))
        if row["q_fit"] is not None:
            entry["q_fit"].append(float(row["q_fit"]))
    for entry in per_template.values():
        entry["median_q_fit"] = (
            float(np.median(entry["q_fit"])) if entry["q_fit"] else None
        )
        del entry["q_fit"]
    science["per_template_asimov"] = per_template

    censored_rows = [
        row
        for row in rows
        if row["censored"]
        and row["arm"] in ("asimov_injected", "noisy_injected")
    ]
    science["censored"] = {
        "n": len(censored_rows),
        "rows": [
            {
                "system_id": row["system_id"],
                "arm": row["arm"],
                "injection_logm": row["injection_logm"],
                "q_fit": row["q_fit"],
                "verdict": _q_verdict(row),
            }
            for row in sorted(
                censored_rows, key=lambda row: (row["system_id"], row["arm"])
            )
        ],
        "unexpected_detections": [
            f"{row['system_id']}/{row['arm']} q_fit {row['q_fit']}"
            for row in censored_rows
            if _q_verdict(row)
        ],
    }
    return science


def _exceedance_summary(rows, value_key: str, predicate) -> dict:
    """Summarize one threshold over rows, retaining missing values."""
    values = [row.get(value_key) for row in rows]
    tested = [value for value in values if value is not None]
    count = sum(1 for value in tested if predicate(float(value)))
    interval = clopper_pearson(count, len(tested)) if tested else None
    return {
        "n": len(values),
        "n_tested": len(tested),
        "n_none": len(values) - len(tested),
        "count": count,
        "fraction": count/len(tested) if tested else None,
        "interval": list(interval) if interval is not None else None,
        "confidence": 0.95,
        "method": "exact two-sided Clopper-Pearson via scipy.stats.beta.ppf",
    }


def _per_template_recovery(rows, arm: str) -> dict:
    """Summarize non-censored recovery for one arm by template."""
    result = {}
    for row in rows:
        if row["arm"] != arm or row["censored"]:
            continue
        entry = result.setdefault(
            row["template"],
            {"n": 0, "recovered": 0, "q_fit": [], "q_fit_none": 0},
        )
        entry["n"] += 1
        entry["recovered"] += int(_q_verdict(row))
        if row["q_fit"] is None:
            entry["q_fit_none"] += 1
        else:
            entry["q_fit"].append(float(row["q_fit"]))
    for entry in result.values():
        entry["recovery_fraction"] = (
            entry["recovered"]/entry["n"] if entry["n"] else None
        )
        entry["median_q_fit"] = (
            float(np.median(entry["q_fit"])) if entry["q_fit"] else None
        )
        del entry["q_fit"]
    return result


def _quantile_summary(values) -> dict:
    """Return null quantiles and the number of missing statistics."""
    finite = [float(value) for value in values if value is not None]
    quantiles = {}
    for percentile in (50, 90, 95, 99):
        quantiles[str(percentile)] = (
            float(np.percentile(finite, percentile)) if finite else None
        )
    return {
        "n": len(values),
        "n_none": len(values) - len(finite),
        "quantiles": quantiles,
        "max": max(finite) if finite else None,
    }


def _load_declared_source_rows(
    source: dict,
    campaign_dir: Path,
    label: str,
) -> tuple[list, Path, Path]:
    """Load rows from a hash-bound source campaign after integrity checks.

    Parameters
    ----------
    source : `dict`
        Manifest source declaration carrying the source UUID and digests.
    campaign_dir : `pathlib.Path`
        Current campaign directory used to resolve relative source paths.
    label : `str`
        Source label used in failure messages.

    Returns
    -------
    rows : `list`
        Rows from the validated source harvest.
    harvest_path : `pathlib.Path`
        Resolved source harvest path.
    review_path : `pathlib.Path`
        Resolved source review path.

    Raises
    ------
    ValueError
        Raised when the source binding, file digest, UUID or review
        integrity is not valid.
    """
    from hwoslaps.campaign.design_freeze import file_sha256

    if not isinstance(source, dict):
        raise ValueError(f"{label} must be a mapping")
    for key in (
        "campaign",
        "campaign_uuid",
        "harvest",
        "harvest_sha256",
        "review_sha256",
    ):
        if key not in source:
            raise ValueError(f"{label} is missing {key}")
    expected_uuid = source["campaign_uuid"]
    harvest_value = source["harvest"]
    if not isinstance(expected_uuid, str) or not expected_uuid:
        raise ValueError(f"{label}.campaign_uuid must be a non-empty string")
    if not isinstance(harvest_value, str) or not harvest_value:
        raise ValueError(f"{label}.harvest must be a non-empty path")
    declared_path = Path(harvest_value)
    candidates = [declared_path]
    if not declared_path.is_absolute():
        candidates.extend(
            (campaign_dir.parent/declared_path, campaign_dir/declared_path)
        )
    harvest_path = next((path for path in candidates if path.is_file()), None)
    if harvest_path is None:
        raise ValueError(f"Missing {label} harvest {declared_path}")
    review_path = harvest_path.parent/"review.json"
    if not review_path.is_file():
        raise ValueError(f"Missing {label} review {review_path}")
    harvest_sha256 = file_sha256(harvest_path)
    if harvest_sha256 != source["harvest_sha256"]:
        raise ValueError(
            f"{label} harvest sha256 {harvest_sha256} does not match "
            f"declared {source['harvest_sha256']}"
        )
    review_sha256 = file_sha256(review_path)
    if review_sha256 != source["review_sha256"]:
        raise ValueError(
            f"{label} review sha256 {review_sha256} does not match "
            f"declared {source['review_sha256']}"
        )
    with review_path.open(encoding="utf-8") as stream:
        review = json.load(stream)
    if not isinstance(review, dict) or review.get("integrity") != "CLEAN":
        raise ValueError(f"{label} review {review_path} integrity is not CLEAN")
    with harvest_path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f"{harvest_path} must be a mapping")
    if document.get("campaign_uuid") != expected_uuid:
        raise ValueError(
            f"{label} harvest campaign uuid "
            f"{document.get('campaign_uuid')!r} does not match "
            f"declared {expected_uuid!r}"
        )
    source_rows = document.get("rows")
    if not isinstance(source_rows, list):
        raise ValueError(f"{harvest_path} rows must be a list")
    return source_rows, harvest_path, review_path


def _load_replicate_zero_rows(manifest: dict, campaign_dir: Path) -> tuple:
    """Load and validate the v1 replicate-zero control rows."""
    campaign = manifest["campaign"]
    source = campaign.get("replicate_zero_source")
    if not isinstance(source, dict):
        raise ValueError(
            "nonlinear null manifest is missing replicate_zero_source"
        )
    source_rows, harvest_path, _ = _load_declared_source_rows(
        source, campaign_dir, "replicate_zero_source"
    )
    expected_systems = {job["run_name"] for job in manifest["jobs"]}
    if len(expected_systems) != 59:
        raise ValueError(
            f"Nonlinear null manifest has {len(expected_systems)} systems, "
            "expected 59"
        )
    controls = {}
    for row in source_rows:
        if row.get("arm") != "noisy_control":
            continue
        system_id = row.get("system_id")
        if system_id not in expected_systems:
            raise ValueError(
                "Replicate-zero control row has unexpected system "
                f"{system_id!r}"
            )
        if system_id in controls:
            raise ValueError(
                "Replicate-zero harvest has multiple noisy_control rows for "
                f"{system_id}"
            )
        controls[system_id] = copy.deepcopy(row)
        controls[system_id]["system_id"] = system_id
    missing = sorted(expected_systems.difference(controls))
    if missing:
        raise ValueError(
            "Replicate-zero harvest is missing noisy_control rows for "
            f"{missing[:5]}"
        )
    zero_rows = []
    for system_id in sorted(controls):
        row = controls[system_id]
        row["noise_replicate"] = 0
        zero_rows.append(row)
    return zero_rows, source_rows, harvest_path


def _per_template_pooled(
    source_rows: list,
    current_rows: list,
    arm: str,
    source_campaign: str,
    source_uuid: str,
    current_campaign: str,
    current_uuid: str,
) -> dict:
    """Summarize one injected arm over source and current rows.

    Parameters
    ----------
    source_rows : `list` [`dict`]
        Rows from the declared pooled source campaign.
    current_rows : `list` [`dict`]
        Rows from the current validation campaign.
    arm : `str`
        Injected arm to summarize.
    source_campaign, source_uuid : `str`
        Source campaign name and UUID for provenance.
    current_campaign, current_uuid : `str`
        Current campaign name and UUID for provenance.

    Returns
    -------
    summaries : `dict`
        Per-template pooled recovery summaries.
    """
    provenance = (
        f"pooled from {source_campaign} ({source_uuid}) and "
        f"{current_campaign} ({current_uuid})"
    )
    templates = sorted(
        {
            row["template"]
            for row in source_rows + current_rows
            if row.get("arm") == arm and not row.get("censored", False)
        }
    )
    summaries = {}
    for template in templates:
        source = [
            row
            for row in source_rows
            if row.get("arm") == arm
            and row.get("template") == template
            and not row.get("censored", False)
        ]
        current = [
            row
            for row in current_rows
            if row.get("arm") == arm
            and row.get("template") == template
            and not row.get("censored", False)
        ]
        combined = source + current
        q_values = [row["q_fit"] for row in combined if row["q_fit"] is not None]
        recovered = sum(1 for row in combined if _q_verdict(row))
        summaries[template] = {
            "n": len(combined),
            "n_source": len(source),
            "n_this_campaign": len(current),
            "recovered": recovered,
            "recovery_fraction": recovered/len(combined) if combined else None,
            "median_q_fit": (
                float(np.median([float(value) for value in q_values]))
                if q_values
                else None
            ),
            "q_fit_none": len(combined) - len(q_values),
            "provenance": provenance,
        }
    return summaries


def _morphology_template_summary(rows: list) -> dict:
    """Summarize the Asimov freed-to-Fisher transfer by template.

    Parameters
    ----------
    rows : `list` [`dict`]
        Harvest rows from one campaign or a pooled set.

    Returns
    -------
    summaries : `dict`
        Per-template transfer summaries and their two orderings.
    """
    per_template = {}
    for row in rows:
        if row.get("arm") != "asimov_injected" or row.get("censored", False):
            continue
        entry = per_template.setdefault(
            row["template"],
            {
                "n": 0,
                "recovered": 0,
                "q_fit": [],
                "q_f_matched": [],
                "injection_logm": [],
            },
        )
        entry["n"] += 1
        entry["recovered"] += int(_q_verdict(row))
        if row.get("q_fit") is None:
            entry.setdefault("q_fit_none", 0)
            entry["q_fit_none"] += 1
        else:
            entry["q_fit"].append(float(row["q_fit"]))
        entry["q_f_matched"].append(float(row["q_f_matched"]))
        entry["injection_logm"].append(float(row["injection_logm"]))
    for entry in per_template.values():
        entry.setdefault("q_fit_none", 0)
        entry["recovery_fraction"] = (
            entry["recovered"]/entry["n"] if entry["n"] else None
        )
        entry["median_q_fit"] = (
            float(np.median(entry["q_fit"])) if entry["q_fit"] else None
        )
        entry["median_q_f_matched"] = (
            float(np.median(entry["q_f_matched"]))
            if entry["q_f_matched"]
            else None
        )
        entry["median_injection_logm"] = (
            float(np.median(entry["injection_logm"]))
            if entry["injection_logm"]
            else None
        )
        del entry["q_fit"]
        del entry["q_f_matched"]
        del entry["injection_logm"]

    def descending_key(name, field):
        """Sort values descending while placing missing values last."""
        value = per_template[name][field]
        return (value is None, -float(value) if value is not None else 0.0, name)

    by_freed = sorted(
        per_template,
        key=lambda name: (
            per_template[name]["recovery_fraction"] is None,
            -float(per_template[name]["recovery_fraction"])
            if per_template[name]["recovery_fraction"] is not None
            else 0.0,
            -float(per_template[name]["median_q_fit"])
            if per_template[name]["median_q_fit"] is not None
            else 0.0,
            name,
        ),
    )
    by_fisher = sorted(
        per_template, key=lambda name: descending_key(name, "median_q_f_matched")
    )
    paired = [
        (entry["median_q_fit"], entry["median_q_f_matched"])
        for entry in per_template.values()
        if entry["median_q_fit"] is not None
        and entry["median_q_f_matched"] is not None
    ]
    if len(paired) < 2:
        correlation = None
        correlation_note = (
            "None because fewer than two templates have both medians"
        )
    elif len({pair[0] for pair in paired}) < 2:
        correlation = None
        correlation_note = "None because median q_fit is constant"
    elif len({pair[1] for pair in paired}) < 2:
        correlation = None
        correlation_note = "None because median q_f_matched is constant"
    else:
        correlation = spearman_rank_correlation(
            [pair[0] for pair in paired], [pair[1] for pair in paired]
        )
        correlation_note = None
    return {
        "per_template": per_template,
        "rank_order_by_freed_recovery": by_freed,
        "rank_order_by_median_q_f_matched": by_fisher,
        "spearman_median_q_fit_vs_median_q_f_matched": correlation,
        "spearman_note": correlation_note,
    }


def _science_null(
    rows,
    manifest: dict,
    campaign_dir: Path,
    freeze: dict | None = None,
) -> dict:
    """Compute the calibrated null summaries for the v4 null campaign."""
    if freeze is None:
        from hwoslaps.campaign.design_freeze import load_design_freeze

        freeze = load_design_freeze()
    zero_rows, source_rows, source_harvest = _load_replicate_zero_rows(
        manifest, campaign_dir
    )
    null_rows = zero_rows + [copy.deepcopy(row) for row in rows]
    replicate_indices = freeze["seeds"]["streams"]["null_noise"][
        "replicate_indices"
    ]
    expected_replicates = len(replicate_indices) + 1
    expected_draws = expected_replicates*59
    observed_draws = len(null_rows)

    def group_summary(group):
        """Summarize both declared null thresholds for one group."""
        return {
            "q_fit_ge_10": _exceedance_summary(
                group, "q_fit", lambda value: value >= Q_FIT_THRESHOLD
            ),
            "delta_log_evidence_gt_5": _exceedance_summary(
                group,
                "delta_log_evidence",
                lambda value: value > DLOGZ_THRESHOLD,
            ),
        }

    by_template = {}
    for template in sorted({row["template"] for row in null_rows}):
        by_template[template] = group_summary(
            [row for row in null_rows if row["template"] == template]
        )
    by_tier = {}
    for tier in sorted(
        {
            report_tier
            for row in null_rows
            for report_tier in row.get("report_tiers", [row["tier"]])
        }
    ):
        by_tier[tier] = group_summary(
            [
                row
                for row in null_rows
                if tier in row.get("report_tiers", [row["tier"]])
            ]
        )

    by_system = {}
    for system_id in sorted({row["system_id"] for row in null_rows}):
        system_rows = [row for row in null_rows if row["system_id"] == system_id]
        q_values = [
            row["q_fit"] for row in system_rows if row["q_fit"] is not None
        ]
        exceedances = sum(
            1 for value in q_values if float(value) >= Q_FIT_THRESHOLD
        )
        by_system[system_id] = {
            "max_q_fit": max(
                (float(value) for value in q_values), default=None
            ),
            "q_fit_none": len(system_rows) - len(q_values),
            "q_fit_exceedance_count": exceedances,
            "any_q_fit_exceedance": bool(exceedances),
        }
    any_count = sum(
        int(entry["any_q_fit_exceedance"]) for entry in by_system.values()
    )
    pooled_q = [row["q_fit"] for row in null_rows]
    q_summary = {
        "pooled": _quantile_summary(pooled_q),
        "per_template": {
            template: _quantile_summary(
                [
                    row["q_fit"]
                    for row in null_rows
                    if row["template"] == template
                ]
            )
            for template in sorted(by_template)
        },
    }
    q_values = [float(value) for value in pooled_q if value is not None]
    q95 = float(np.percentile(q_values, 95)) if q_values else None
    q99 = float(np.percentile(q_values, 99)) if q_values else None
    if q_values:
        from scipy.stats import chi2, kstest

        ks_result = kstest(q_values, chi2(3).cdf)
        ks = {
            "distance": float(ks_result.statistic),
            "p_value": float(ks_result.pvalue),
        }
    else:
        ks = {"distance": None, "p_value": None}

    boundary_tally = {}
    for row in null_rows:
        recovery = row.get("subhalo_recovery") or {}
        for key in ("mass_at_lower_bound", "mass_at_upper_bound"):
            boundary_tally[key] = boundary_tally.get(key, 0) + int(
                recovery.get(key) is True
            )
    quality_flag_tally = {}
    for row in null_rows:
        for flag in row.get("quality_flags", []):
            quality_flag_tally[flag] = quality_flag_tally.get(flag, 0) + 1

    transfer = {}
    for arm in ("asimov_injected", "noisy_injected"):
        candidates = [
            row
            for row in source_rows
            if row.get("arm") == arm and not row.get("censored", False)
        ]
        tested = [row for row in candidates if row.get("q_fit") is not None]
        recovered = [
            row for row in tested
            if q99 is not None
            and float(row["q_fit"]) >= q99
        ]
        transfer[arm] = {
            "n": len(candidates),
            "n_tested": len(tested),
            "q_fit_none": sum(
                row.get("q_fit") is None for row in candidates
            ),
            "recovered": len(recovered),
            "recovery_fraction": (
                len(recovered)/len(tested) if tested else None
            ),
            "threshold_q_fit": q99,
        }

    return {
        "null_exceedances": {
            "pooled": group_summary(null_rows),
            "per_template": by_template,
            "per_tier": by_tier,
            "note": (
                "Per-template and per-tier counts use each row's report_tiers; "
                "the overlap member is counted in both parent and selected. "
                f"The pooled Clopper-Pearson interval treats the "
                f"{observed_draws} observed draws as independent "
                f"(expected {expected_draws}), while draws within a system "
                "are dependent."
            ),
        },
        "per_system": {
            "systems": by_system,
            "n_systems": len(by_system),
            "n_with_any_q_fit_exceedance": any_count,
            "fraction_with_any_q_fit_exceedance": (
                any_count/len(by_system) if by_system else None
            ),
        },
        "null_q_fit_quantiles": q_summary,
        "null_implied_thresholds": {
            "q_fit_5_percent": q95,
            "q_fit_1_percent": q99,
            "screening_convention_q_fit": Q_FIT_THRESHOLD,
            "bayesian_convention_delta_log_evidence": DLOGZ_THRESHOLD,
        },
        "null_q_fit_chi2_3": {
            "distance": ks["distance"],
            "p_value": ks["p_value"],
            "distribution": "chi-square with 3 degrees of freedom",
            "method": "scipy.stats.kstest with scipy.stats.chi2(3).cdf",
            "context_only": True,
        },
        "boundary_tally": boundary_tally,
        "quality_flag_tally": quality_flag_tally,
        "post_hoc_threshold_transfer": {
            "label": "post hoc",
            "source_harvest": str(source_harvest),
            "threshold_q_fit": q99,
            "arms": transfer,
        },
        "expected_replicates": expected_replicates,
        "expected_draws": expected_draws,
        "draw_census": {
            "expected": expected_draws,
            "observed": observed_draws,
            "complete": observed_draws == expected_draws,
        },
        "note": (
            "This is a calibrated null for the declared truth-centred "
            "matched-PSF convention, not a blind-search false-positive rate. "
            f"The {observed_draws} observed draws are dependent through the "
            "shared scene and position prior, so the pooled interval is a "
            "per-draw summary and the per-system any-exceedance fraction is "
            "the dependence-honest companion statistic."
        ),
    }


def _campaign_findings(
    manifest: dict,
    protocol: dict,
    manifest_name: str = "<manifest>",
) -> list:
    """Return integrity findings for a v4 manifest's campaign contract.

    Parameters
    ----------
    manifest : `dict`
        Campaign manifest to check.
    protocol : `dict`
        Validated nonlinear-validation freeze block.
    manifest_name : `str`, optional
        Manifest path used when a required campaign mapping is absent.

    Returns
    -------
    findings : `list` [`str`]
        Per-job campaign integrity findings.

    Raises
    ------
    ValueError
        Raised when a schema-3 or version-4 manifest omits its campaign
        mapping.
    """
    schema_version = manifest.get("schema_version")
    design_freeze = manifest.get("design_freeze")
    design_version = (
        design_freeze.get("version")
        if isinstance(design_freeze, dict)
        else None
    )
    requires_campaign = (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version >= 3
    ) or (
        isinstance(design_version, (int, float))
        and not isinstance(design_version, bool)
        and design_version >= 4
    )
    campaign = manifest.get("campaign")
    if requires_campaign and not isinstance(campaign, dict):
        raise ValueError(
            f"Manifest {manifest_name} requires a campaign mapping for its "
            f"schema_version {schema_version!r} and design-freeze version "
            f"{design_version!r}"
        )
    if not requires_campaign or campaign is None:
        return []

    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return [f"{manifest_name}: jobs must be a list"]

    findings = []

    def add_for_jobs(message: str) -> None:
        """Add one campaign finding for each manifest job."""
        if jobs:
            for index, job in enumerate(jobs):
                label = (
                    str(job.get("run_name", f"job[{index}]"))
                    if isinstance(job, dict)
                    else f"job[{index}]"
                )
                findings.append(f"{label}: {message}")
        else:
            findings.append(f"{manifest_name}: {message}")

    campaign_name = manifest.get("name")
    freeze_campaigns = protocol.get("campaigns")
    frozen_campaign = (
        freeze_campaigns.get(campaign_name)
        if isinstance(freeze_campaigns, dict)
        else None
    )
    if not isinstance(frozen_campaign, dict):
        add_for_jobs(
            f"campaign {campaign_name!r} is not declared in the nonlinear "
            "freeze"
        )
        return findings

    manifest_arms = campaign.get("arms")
    frozen_arms = frozen_campaign.get("arms")
    if manifest_arms != frozen_arms:
        add_for_jobs(
            f"manifest campaign arms {manifest_arms!r} do not equal the "
            f"freeze arms {frozen_arms!r}"
        )

    frozen_member_set = frozen_campaign.get("member_set")
    if campaign.get("member_set") != frozen_member_set:
        add_for_jobs(
            f"manifest member_set {campaign.get('member_set')!r} does not "
            f"equal the freeze member_set {frozen_member_set!r}"
        )
    member_sets = protocol.get("member_sets")
    member_set = (
        member_sets.get(frozen_member_set)
        if isinstance(member_sets, dict)
        else None
    )
    expected_n_systems = (
        member_set.get("n_systems")
        if isinstance(member_set, dict)
        else None
    )
    if manifest.get("n_systems") != expected_n_systems:
        add_for_jobs(
            f"manifest n_systems {manifest.get('n_systems')!r} does not "
            f"equal the freeze member-set size {expected_n_systems!r}"
        )

    if not isinstance(manifest_arms, list):
        add_for_jobs("manifest campaign arms must be a list")
        return findings
    protocol_arms = protocol.get("arms")
    if not isinstance(protocol_arms, dict):
        add_for_jobs("protocol arms must be a mapping")
        return findings
    effective_campaign_arms = (
        [arm_name for arm_name in frozen_arms if arm_name in protocol_arms]
        if isinstance(frozen_arms, list)
        else []
    )
    for index, job in enumerate(jobs):
        label = (
            str(job.get("run_name", f"job[{index}]"))
            if isinstance(job, dict)
            else f"job[{index}]"
        )
        if not isinstance(job, dict):
            findings.append(f"{label}: job must be a mapping")
            continue
        actual_arms = job.get("arms")
        if not isinstance(actual_arms, dict):
            findings.append(f"{label}: job arms must be a mapping")
            continue
        missing_flags = [
            key for key in ("censored", "golden") if key not in job
        ]
        if missing_flags:
            findings.append(
                f"{label}: job is missing eligibility fields {missing_flags}"
            )
            continue
        expected_arms = eligible_arms(
            job, protocol_arms, effective_campaign_arms
        )
        if sorted(actual_arms) != sorted(expected_arms):
            findings.append(
                f"{label}: job arms {sorted(actual_arms)!r} do not equal "
                f"eligible arms {sorted(expected_arms)!r}"
            )
    return findings


def _science(
    rows,
    manifest: dict | None = None,
    protocol: dict | None = None,
    campaign_dir: Path | None = None,
    freeze: dict | None = None,
) -> dict:
    """Dispatch legacy and v4 campaign science summaries."""
    if manifest is None or manifest.get("campaign") is None:
        return _science_v3(rows)
    campaign_name = manifest.get("name")
    if campaign_name == "nonlinear_validation100_v1":
        core = _science_v3(rows)
        science = {
            key: core[key]
            for key in (
                "asimov_injected",
                "noisy_injected",
                "asimov_below",
                "noisy_control",
                "per_template_asimov",
                "censored",
            )
        }
        science["per_template"] = _per_template_recovery(
            rows, "noisy_injected"
        )
        source = manifest["campaign"].get("pooled_source")
        if not isinstance(source, dict):
            raise ValueError(
                "nonlinear_validation100_v1 manifest is missing pooled_source"
            )
        if campaign_dir is None:
            raise ValueError(
                "nonlinear_validation100_v1 science requires campaign_dir"
            )
        source_rows, source_harvest, _ = _load_declared_source_rows(
            source, campaign_dir, "pooled_source"
        )
        provenance = (
            f"pooled from {source['campaign']} ({source['campaign_uuid']}) "
            f"and {manifest['name']} ({manifest['campaign_uuid']})"
        )
        science["per_template_pooled"] = {
            "asimov_injected": _per_template_pooled(
                source_rows,
                rows,
                "asimov_injected",
                source["campaign"],
                source["campaign_uuid"],
                manifest["name"],
                manifest["campaign_uuid"],
            ),
            "noisy_injected": _per_template_pooled(
                source_rows,
                rows,
                "noisy_injected",
                source["campaign"],
                source["campaign_uuid"],
                manifest["name"],
                manifest["campaign_uuid"],
            ),
            "provenance": provenance,
            "source_harvest": str(source_harvest),
        }
        science["morphology_transfer"] = {
            "this_campaign": _morphology_template_summary(rows),
            "pooled": _morphology_template_summary(source_rows + rows),
            "provenance": provenance,
        }
        science["censoring_consistency"] = {
            "expected_logm": 9.5,
            **copy.deepcopy(science["censored"]),
        }
        return science
    if campaign_name == "nonlinear_null_v1":
        if campaign_dir is None:
            raise ValueError(
                "nonlinear_null_v1 science requires campaign_dir"
            )
        if freeze is None:
            from hwoslaps.campaign.design_freeze import load_design_freeze

            freeze = load_design_freeze()
        return _science_null(rows, manifest, campaign_dir, freeze)
    raise ValueError(f"Unsupported nonlinear campaign {campaign_name!r}")


def main(argv=None) -> None:
    """Harvest the campaign and write the review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", help="Campaign directory")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Summarize the rows present instead of failing on gaps",
    )
    args = parser.parse_args(argv)
    campaign_dir = Path(args.campaign_dir)
    manifest_path = campaign_dir/"manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    from hwoslaps.campaign.design_freeze import load_design_freeze

    freeze = load_design_freeze()
    protocol = freeze["nonlinear_validation"]

    rows = []
    findings = _campaign_findings(manifest, protocol, str(manifest_path))
    missing = []
    for job in manifest["jobs"]:
        for arm in sorted(
            job["arms"], key=lambda name: job["arms"][name]["arm_index"]
        ):
            artifact = (
                Path(job["output_dir"])/f"nonlinear_validation_{arm}.json"
            )
            if not artifact.is_file():
                missing.append(f"{job['run_name']}/{arm}")
                continue
            with open(artifact, encoding="utf-8") as stream:
                payload = json.load(stream)
            findings.extend(
                _verify_row(job, arm, payload, manifest, protocol)
            )
            rows.append(_row(job, arm, payload))

    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"Campaign incomplete: {len(missing)} arm artifacts missing "
            f"(first: {missing[:5]}); pass --allow-incomplete to summarize"
        )

    review = {
        "schema_version": 2,
        "campaign_uuid": manifest["campaign_uuid"],
        "code_revision": manifest["code_revision"],
        "amendments": manifest.get("amendments", []),
        "rows": len(rows),
        "expected_rows": manifest["n_fit_pairs"],
        "missing": missing,
        "integrity_findings": findings,
        "integrity": "CLEAN" if not findings and not missing else "FINDINGS",
        "quality_flag_tally": {
            flag: sum(1 for row in rows if flag in row["quality_flags"])
            for row in rows
            for flag in row["quality_flags"]
        },
        "total_fit_wall_hours": float(
            sum(row["fit_pair_s"] for row in rows)/3600.0
        ),
        "science": _science(rows, manifest, protocol, campaign_dir, freeze),
    }

    harvest_dir = campaign_dir/"harvest"
    harvest_dir.mkdir(exist_ok=True)
    (harvest_dir/"harvest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "campaign_uuid": manifest["campaign_uuid"],
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (harvest_dir/"review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(review, indent=2, sort_keys=True))
    if findings:
        raise SystemExit(f"{len(findings)} integrity findings")


if __name__ == "__main__":
    main()
