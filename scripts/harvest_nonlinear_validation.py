#!/usr/bin/env python
"""Harvest and review the nonlinear-validation campaign.

Collects every eligible arm artifact of the campaign the manifest
declares, verifies the identity chain fail-closed (job binding,
campaign uuid, code revision, restamped configuration hash, declared
sampler seed re-derived from the freeze rule, arm declaration, fit
settings against the freeze protocol, matched kernels, non-degenerate
mask support), and writes ``harvest/harvest.json`` with one row per fit
pair plus ``harvest/review.json`` with the integrity census and the
freeze v3 success criteria: recovery at the first Fisher-positive rung,
below-rung consistency, within-rung rank fidelity, control tallies,
the golden-five bridge comparison, replicate scatter, morphology
transfer and censoring consistency.

An incomplete campaign is reported and fails the run unless
``--allow-incomplete`` is passed (useful mid-campaign); science
summaries are then computed over the rows present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"scripts"))

from run_nonlinear_validation import (  # noqa: E402
    derive_sampler_seed,
    load_protocol,
    system_index,
)

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
    return findings


def _q_verdict(row) -> bool:
    """Screening-convention verdict of one row."""
    return (
        row["q_fit"] is not None and float(row["q_fit"]) >= Q_FIT_THRESHOLD
    )


def _science(rows) -> dict:
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
    science["asimov_below"] = {
        "n": len(below),
        "below_threshold": sum(1 for row in below if not _q_verdict(row)),
        "below_rung_consistency_fraction": (
            sum(1 for row in below if not _q_verdict(row))/len(below)
            if below
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
        "unexpected_detections": [
            f"{row['system_id']}/{row['arm']} q_fit {row['q_fit']}"
            for row in censored_rows
            if _q_verdict(row)
        ],
    }
    return science


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
    with open(campaign_dir/"manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    protocol = load_protocol()

    rows = []
    findings = []
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
        "science": _science(rows),
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
