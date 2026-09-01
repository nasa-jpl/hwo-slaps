#!/usr/bin/env python
"""Harvest and review the nonlinear-validation campaign.

Collects every arm artifact of the campaign the manifest declares,
verifies the identity chain fail-closed (campaign uuid, code revision,
and the declared sampler seed re-derived from the freeze rule), and
writes ``harvest/harvest.json`` with one row per fit pair plus
``harvest/review.json`` with the integrity census and the freeze v3
success criteria: crossing agreement, rank fidelity, control false
positives, morphology transfer and censoring consistency.

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
    ARMS,
    derive_sampler_seed,
    system_index,
)

Q_FIT_THRESHOLD = 10.0
DLOGZ_THRESHOLD = 5.0


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


def _row(job: dict, arm: str, payload: dict, template: str) -> dict:
    """Reduce one arm artifact to its harvest row."""
    return {
        "system_id": payload["system_id"],
        "tier": job["tier"],
        "report_tiers": job["report_tiers"],
        "template": template,
        "arm": arm,
        "dataset_kind": payload["dataset_kind"],
        "subhalo_in_truth": payload["subhalo_in_truth"],
        "injection_logm": payload["injection_logm"],
        "censored": payload["censored"],
        "position_yx_arcsec": payload["position_yx_arcsec"],
        "fisher_q_at_position": payload["fisher_q_at_position"],
        "q_fit": payload["q_fit"],
        "delta_log_evidence": payload["delta_log_evidence"],
        "smooth_status": payload["smooth_status"],
        "subhalo_status": payload["subhalo_status"],
        "quality_flags": payload["quality_flags"],
        "sampler_seed": payload["sampler_seed"],
        "fit_pair_s": payload["timings"]["fit_pair_s"],
        "smooth_runtime_s": payload["smooth_runtime_s"],
        "subhalo_runtime_s": payload["subhalo_runtime_s"],
    }


def _verify_row(job: dict, arm: str, payload: dict, manifest: dict) -> list:
    """Return the integrity findings of one arm artifact."""
    findings = []
    expected_seed = derive_sampler_seed(
        system_index(payload["system_id"]), ARMS[arm]["arm_index"]
    )
    if int(payload["sampler_seed"]) != expected_seed:
        findings.append(
            f"{payload['system_id']}/{arm}: sampler seed "
            f"{payload['sampler_seed']} is not the declared {expected_seed}"
        )
    if payload["campaign_uuid"] != manifest["campaign_uuid"]:
        findings.append(
            f"{payload['system_id']}/{arm}: campaign uuid "
            f"{payload['campaign_uuid']!r}"
        )
    if payload["code_revision"]["sha256"] != manifest["code_revision"]["sha256"]:
        findings.append(
            f"{payload['system_id']}/{arm}: code revision "
            f"{payload['code_revision']['sha256'][:16]}"
        )
    if payload["kernel_sha256"] != payload["truth_kernel_sha256"]:
        findings.append(
            f"{payload['system_id']}/{arm}: fit kernel is not the truth kernel"
        )
    for side in ("smooth_status", "subhalo_status"):
        if payload[side] != "success":
            findings.append(
                f"{payload['system_id']}/{arm}: {side} {payload[side]!r}"
            )
    return findings


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

    rows = []
    findings = []
    missing = []
    for job in manifest["jobs"]:
        with open(job["restamped_config"], encoding="utf-8") as stream:
            template = yaml.safe_load(stream)["stage0"]["source_template"]
        for arm in sorted(ARMS):
            artifact = (
                Path(job["output_dir"])/f"nonlinear_validation_{arm}.json"
            )
            if not artifact.is_file():
                missing.append(f"{job['run_name']}/{arm}")
                continue
            with open(artifact, encoding="utf-8") as stream:
                payload = json.load(stream)
            findings.extend(_verify_row(job, arm, payload, manifest))
            rows.append(_row(job, arm, payload, template))

    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"Campaign incomplete: {len(missing)} arm artifacts missing "
            f"(first: {missing[:5]}); pass --allow-incomplete to summarize"
        )

    def q_verdict(row):
        return (
            row["q_fit"] is not None
            and float(row["q_fit"]) >= Q_FIT_THRESHOLD
        )

    science = {}
    for arm in ("asimov_injected", "noisy_injected"):
        injected = [
            row for row in rows if row["arm"] == arm and not row["censored"]
        ]
        agree = [row for row in injected if q_verdict(row)]
        pairs = [
            (float(row["q_fit"]), float(row["fisher_q_at_position"]))
            for row in injected
            if row["q_fit"] is not None
        ]
        science[arm] = {
            "n": len(injected),
            "nonlinear_detections_at_injection": len(agree),
            "crossing_agreement_fraction": (
                len(agree)/len(injected) if injected else None
            ),
            "spearman_q_fit_vs_q_f": (
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

    controls = [row for row in rows if row["arm"] == "noisy_control"]
    science["noisy_control"] = {
        "n": len(controls),
        "false_positives_q_fit": sum(1 for row in controls if q_verdict(row)),
        "false_positives_dlogz": sum(
            1
            for row in controls
            if row["delta_log_evidence"] is not None
            and float(row["delta_log_evidence"]) > DLOGZ_THRESHOLD
        ),
    }

    per_template = {}
    for row in rows:
        if row["arm"] != "asimov_injected" or row["censored"]:
            continue
        entry = per_template.setdefault(
            row["template"], {"n": 0, "detected": 0, "q_fit": []}
        )
        entry["n"] += 1
        entry["detected"] += int(q_verdict(row))
        if row["q_fit"] is not None:
            entry["q_fit"].append(float(row["q_fit"]))
    for entry in per_template.values():
        entry["median_q_fit"] = (
            float(np.median(entry["q_fit"])) if entry["q_fit"] else None
        )
        del entry["q_fit"]

    censored_rows = [
        row
        for row in rows
        if row["censored"] and row["arm"] in ("asimov_injected", "noisy_injected")
    ]
    science["censored"] = {
        "n": len(censored_rows),
        "unexpected_detections": [
            f"{row['system_id']}/{row['arm']} q_fit {row['q_fit']}"
            for row in censored_rows
            if q_verdict(row)
        ],
    }
    science["per_template_asimov"] = per_template

    review = {
        "schema_version": 1,
        "campaign_uuid": manifest["campaign_uuid"],
        "code_revision": manifest["code_revision"],
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
        "science": science,
    }

    harvest_dir = campaign_dir/"harvest"
    harvest_dir.mkdir(exist_ok=True)
    (harvest_dir/"harvest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
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
