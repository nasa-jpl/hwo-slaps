#!/usr/bin/env python
"""Pre-registered three-curve selector comparison on the validation sample.

Implements section 3 of configs/design/selection_rule_v2.md on the
100-member pseudo-random validation sample of ladder_validation_v1, the
unenriched measured set the Sol Pro production review asked for. The
frozen ``s_only`` and ``s_plus_c`` rankings come from the Stage 0
selection report and are restricted to the sample; the oracle is formed
from the measured ladders and reported as a labelled upper bound. Also
completes the pre-registered S/N-only control: the frozen ``s_only``
top-12 and ``s_plus_c`` top-12 are both fully measured once the
validation ladders exist, and their measured estimands are tabulated
side by side.

``M_lim`` is ``m_best``, the aperture ``q_max`` crossing of ``q_F = 10``,
the convention every production table uses. Right-censored members
(no crossing below the ladder ceiling) are handled under two declared
conventions, ``finite_only`` and ``censored_last``, because the
pre-registration does not fix one; both are reported.

Reads only harvested artifacts. No GPU, no rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from hwoslaps.analysis import selection_score as ss

VARIANTS = ("s_only", "s_plus_c")
ESTIMANDS = ("m_best", "m10", "m50")
PRIMARY_ESTIMAND = "m_best"
CONVENTIONS = ("finite_only", "censored_last")
M_LIM_CONVENTION = (
    "m_best: aperture q_max crossing of q_F = 10 (W3 M_lim convention); "
    "censored members never crossed below the logM 9.5 ladder ceiling"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _short_id(system_id: str) -> str:
    if "sys" not in system_id:
        raise ValueError(f"Unexpected system id {system_id!r}")
    return "sys" + system_id.split("sys")[1]


def restrict_ranking(ranking, members) -> list:
    """Restrict a full ranking to ``members``, preserving order.

    Every member must appear in the ranking; a member outside it (a
    floor-cut failure) is a fail-closed condition, not a silent drop.
    """
    wanted = set(members)
    restricted = [system_id for system_id in ranking if system_id in wanted]
    missing = wanted - set(restricted)
    if missing:
        raise ValueError(f"{len(missing)} members are absent from the ranking: {sorted(missing)[:5]}")
    return restricted


def censored_last_values(values) -> list:
    """Map right-censored (non-finite) entries to one value above every finite one.

    The tie-aware Spearman then ranks every censored member equal and
    last, which is the ordering the ladders actually establish.
    """
    finite = [value for value in values if _finite(value)]
    if not finite:
        raise ValueError("A censored-last vector needs at least one finite entry.")
    ceiling = max(finite) + 1.0
    return [value if _finite(value) else ceiling for value in values]


def spearman_by_convention(positions, values) -> dict:
    """Spearman of ranking positions against measured masses under both conventions.

    A working score ranks sensitive systems first (low position) and
    those systems have low ``M_lim``, so a working score gives a
    POSITIVE correlation here; the pre-registration's negative sign
    refers to the score itself, which is monotone decreasing in position.
    """
    out = {}
    finite_pairs = [(p, v) for p, v in zip(positions, values) if _finite(v)]
    out["finite_only"] = {
        "n": len(finite_pairs),
        "spearman": _safe_spearman([p for p, _ in finite_pairs], [v for _, v in finite_pairs]),
    }
    out["censored_last"] = {
        "n": len(values),
        "n_censored": sum(1 for v in values if not _finite(v)),
        "spearman": _safe_spearman(list(positions), censored_last_values(values)),
    }
    return out


def _safe_spearman(x, y):
    try:
        return ss.spearman_rank_correlation(x, y)
    except ValueError:
        return None


def oracle_ranking(members, m_lim) -> list:
    """Ascending measured ``M_lim`` over the finite members; censored members last, by id."""
    finite_ids = [system_id for system_id in members if _finite(m_lim[system_id])]
    censored_ids = sorted(system_id for system_id in members if not _finite(m_lim[system_id]))
    ranked = list(ss.rank_by_sensitivity(finite_ids, [m_lim[system_id] for system_id in finite_ids]))
    return ranked + censored_ids


def three_curve_comparison(sample, rankings, rows, template_of, k: int) -> dict:
    """Section 3 of the pre-registration on one unenriched measured sample."""
    sample = list(sample)
    restricted = {variant: restrict_ranking(rankings[variant], sample) for variant in VARIANTS}
    m_lim = {system_id: rows[system_id][PRIMARY_ESTIMAND] for system_id in sample}
    n_finite = sum(1 for value in m_lim.values() if _finite(value))
    if n_finite < k:
        raise ValueError(f"Only {n_finite} finite M_lim values, fewer than the tier of {k}.")
    oracle = oracle_ranking(sample, m_lim)
    oracle_top = oracle[:k]

    report = {
        "sample_size": len(sample),
        "tier_size": k,
        "m_lim_convention": M_LIM_CONVENTION,
        "n_censored": len(sample) - n_finite,
        "oracle": {
            "status": "labelled_upper_bound",
            "top_k": [
                {
                    "system_id": system_id,
                    "template": template_of[system_id],
                    **{key: rows[system_id][key] for key in ESTIMANDS},
                }
                for system_id in oracle_top
            ],
        },
        "curves": {},
        "operational_top_k_jaccard": ss.top_k_jaccard(
            restricted["s_only"], restricted["s_plus_c"], k
        ),
    }
    for variant in VARIANTS:
        ranking = restricted[variant]
        positions = ss.ranking_positions(ranking)
        ordered_positions = [positions[system_id] for system_id in sample]
        per_estimand = {}
        for key in ESTIMANDS:
            values = [rows[system_id][key] for system_id in sample]
            per_estimand[key] = spearman_by_convention(ordered_positions, values)
        per_template = {}
        for template in sorted(set(template_of[system_id] for system_id in sample)):
            ids = [system_id for system_id in sample if template_of[system_id] == template]
            per_template[template] = {
                "n": len(ids),
                **spearman_by_convention(
                    [positions[system_id] for system_id in ids],
                    [rows[system_id][PRIMARY_ESTIMAND] for system_id in ids],
                ),
            }
        top = ranking[:k]
        report["curves"][variant] = {
            "top_k": [
                {
                    "system_id": system_id,
                    "template": template_of[system_id],
                    PRIMARY_ESTIMAND: rows[system_id][PRIMARY_ESTIMAND],
                    "in_oracle_top_k": system_id in set(oracle_top),
                }
                for system_id in top
            ],
            "spearman_position_vs_estimand": per_estimand,
            "spearman_position_vs_m_lim_per_template": per_template,
            "oracle_recovered_fraction": ss.oracle_recovered_fraction(ranking, oracle, k),
            "top_k_jaccard_vs_oracle": ss.top_k_jaccard(ranking, oracle, k),
            "top_k_m_lim_median_finite": _median_finite(
                [rows[system_id][PRIMARY_ESTIMAND] for system_id in top]
            ),
        }
    return report


def _median_finite(values):
    finite = [value for value in values if _finite(value)]
    return float(np.median(finite)) if finite else None


def frozen_top_k_outcomes(rankings, rows, template_of, tier_of, k: int) -> dict:
    """Measured outcomes of the two frozen full-pool top-k sets (the S/N-only control)."""
    out = {"tier_size": k, "sets": {}}
    tops = {variant: rankings[variant][:k] for variant in VARIANTS}
    for variant, top in tops.items():
        missing = [system_id for system_id in top if system_id not in rows]
        if missing:
            raise ValueError(f"{variant} top-{k} has unmeasured members: {missing}")
        members = [
            {
                "system_id": system_id,
                "template": template_of[system_id],
                "measured_in": tier_of[system_id],
                **{key: rows[system_id][key] for key in ESTIMANDS},
            }
            for system_id in top
        ]
        out["sets"][variant] = {
            "members": members,
            "medians_finite_only": {
                key: _median_finite([rows[system_id][key] for system_id in top])
                for key in ESTIMANDS
            },
            "n_censored": {
                key: sum(1 for system_id in top if not _finite(rows[system_id][key]))
                for key in ESTIMANDS
            },
        }
    out["top_k_jaccard"] = ss.top_k_jaccard(tops["s_only"], tops["s_plus_c"], k)
    out["shared_members"] = sorted(set(tops["s_only"]) & set(tops["s_plus_c"]))
    return out


def pool_context(sample_rows, selected_rows) -> dict:
    """Where the selected tier sits against the unenriched sample's M_lim distribution."""
    sample_values = [row[PRIMARY_ESTIMAND] for row in sample_rows]
    finite_sorted = sorted(value for value in sample_values if _finite(value))
    n = len(sample_values)
    out = {
        "sample_size": n,
        "sample_n_censored": n - len(finite_sorted),
        "sample_median_finite_only": float(np.median(finite_sorted)),
        "sample_median_censoring_aware": _censored_median(sample_values),
        "selected": [],
    }
    for row in selected_rows:
        value = row[PRIMARY_ESTIMAND]
        below = sum(1 for other in finite_sorted if other <= value)
        out["selected"].append(
            {
                "system_id": _short_id(row["system_id"]),
                PRIMARY_ESTIMAND: value,
                "sample_fraction_at_or_below": below / n,
            }
        )
    return out


def _censored_median(values):
    """Median with censored entries above every finite one; None if it lands on one."""
    finite = sorted(value for value in values if _finite(value))
    n = len(values)
    position = 0.5 * (n - 1)
    upper = min(math.ceil(position), n - 1)
    if upper >= len(finite):
        return None
    lower = math.floor(position)
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def load_estimands(path: Path) -> list:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not hold a non-empty estimand list.")
    return rows


def build_inputs(root: Path) -> dict:
    paths = {
        "selection_report": root/"stage0_pool_v3"/"layer2"/"stability"/"selection_report.json",
        "pool_members": root/"stage0_pool_v3"/"layer2"/"selection"/"stage0_pool_members.json",
        "parent_estimands": root/"ladder_parent_v1"/"run"/"harvest"/"estimands.json",
        "selected_estimands": root/"ladder_selected_v1"/"run"/"harvest"/"estimands.json",
        "validation_estimands": root/"ladder_validation_v1"/"run"/"harvest"/"estimands.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = build_inputs(args.campaigns_root)
    out_dir = args.out_dir or args.campaigns_root/"reporting_v1"/"selector_validation"

    report = json.loads(paths["selection_report"].read_text())
    k = int(report["tier_size"])
    if k != ss.SELECTED_TIER_SIZE:
        raise ValueError(f"Report tier size {k} disagrees with the module constant {ss.SELECTED_TIER_SIZE}.")
    rankings = {variant: list(report["curves"][variant]["ranking"]) for variant in VARIANTS}
    survivors = set(report["curves"]["s_plus_c"]["survivor_ids"])
    if set(rankings["s_only"]) != survivors or set(rankings["s_plus_c"]) != survivors:
        raise ValueError("The two frozen rankings do not cover the same survivor pool.")

    pool = {row["system_id"]: row for row in json.loads(paths["pool_members"].read_text())}
    template_of = {system_id: row["source_template"] for system_id, row in pool.items()}

    rows: dict = {}
    tier_of: dict = {}
    for tier, name in (("parent", "parent_estimands"), ("selected", "selected_estimands"),
                       ("validation", "validation_estimands")):
        for row in load_estimands(paths[name]):
            system_id = _short_id(row["system_id"])
            if system_id in rows:
                for key in ESTIMANDS:
                    same = rows[system_id][key] == row[key] or (
                        not _finite(rows[system_id][key]) and not _finite(row[key]))
                    if not same:
                        raise ValueError(f"Overlap mismatch on {system_id} {key}")
                tier_of[system_id] = f"{tier_of[system_id]}+{tier}"
                continue
            rows[system_id] = row
            tier_of[system_id] = tier

    validation_rows = load_estimands(paths["validation_estimands"])
    sample = sorted(_short_id(row["system_id"]) for row in validation_rows if row["validation_sample_member"])
    if len(sample) != 100:
        raise ValueError(f"Expected 100 validation sample members, found {len(sample)}.")
    production = {_short_id(row["system_id"]) for name in ("parent_estimands", "selected_estimands")
                  for row in load_estimands(paths[name])}
    if production & set(sample):
        raise ValueError("Validation sample overlaps the production tiers.")
    if not set(sample) <= survivors:
        raise ValueError("Validation sample members must all be post-cut survivors.")

    selected_rows = [row for row in load_estimands(paths["selected_estimands"])]
    if len(selected_rows) != k:
        raise ValueError(f"Selected tier holds {len(selected_rows)} rows, expected {k}.")

    result = {
        "pre_registration": report["definitions"]["pre_registration"],
        "inputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()},
        "campaign_uuids": {
            tier: sorted({row["campaign_uuid"] for row in load_estimands(paths[name])})
            for tier, name in (("parent", "parent_estimands"), ("selected", "selected_estimands"),
                               ("validation", "validation_estimands"))
        },
        "validation_sample": three_curve_comparison(sample, rankings, rows, template_of, k),
        "frozen_top_k_outcomes": frozen_top_k_outcomes(rankings, rows, template_of, tier_of, k),
        "pool_context": pool_context([rows[system_id] for system_id in sample], selected_rows),
        "labels": [
            "validation-sample conditional: the sample is the sha256-sorted first 100 of the 906 "
            "unmeasured post-cut survivors, unenriched by construction",
            "oracle is a labelled upper bound formed after the ladders were measured, never operational",
            "the selector remains an idealized no-subhalo truth-proxy selection; the pre-registered "
            "noise-stability test failed and this analysis does not revisit it",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"selector_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir/"selector_validation.md").write_text(render_markdown(result))
    print(f"wrote {out_dir/'selector_validation.json'}")
    print(render_markdown(result))
    return 0


def render_markdown(result: dict) -> str:
    vs = result["validation_sample"]
    k = vs["tier_size"]
    lines = [
        "# Three-curve selector comparison on the validation sample",
        "",
        f"Sample: {vs['sample_size']} members, {vs['n_censored']} right-censored in m_best. "
        f"Tier size {k}. Labels: " + "; ".join(result["labels"]) + ".",
        "",
        "## Spearman of ranking position against measured estimand (positive = working score)",
        "",
        "| curve | estimand | finite-only rho (n) | censored-last rho (n, censored) |",
        "|---|---|---|---|",
    ]
    for variant in VARIANTS:
        per = vs["curves"][variant]["spearman_position_vs_estimand"]
        for key in ESTIMANDS:
            f, c = per[key]["finite_only"], per[key]["censored_last"]
            lines.append(
                f"| {variant} | {key} | {_fmt(f['spearman'])} ({f['n']}) | "
                f"{_fmt(c['spearman'])} ({c['n']}, {c['n_censored']}) |"
            )
    lines += ["", "## Per-template Spearman against m_best", "", "| template | n | s_only finite / censored-last | s_plus_c finite / censored-last |", "|---|---|---|---|"]
    templates = vs["curves"]["s_only"]["spearman_position_vs_m_lim_per_template"]
    for template, block in templates.items():
        other = vs["curves"]["s_plus_c"]["spearman_position_vs_m_lim_per_template"][template]
        lines.append(
            f"| {template} | {block['n']} | {_fmt(block['finite_only']['spearman'])} / "
            f"{_fmt(block['censored_last']['spearman'])} | {_fmt(other['finite_only']['spearman'])} / "
            f"{_fmt(other['censored_last']['spearman'])} |"
        )
    lines += ["", f"## Oracle recovery within the sample (oracle top-{k} is a labelled upper bound)", "",
              "| curve | oracle-recovered fraction | top-k Jaccard vs oracle | top-k m_best median |", "|---|---|---|---|"]
    for variant in VARIANTS:
        block = vs["curves"][variant]
        lines.append(
            f"| {variant} | {_fmt(block['oracle_recovered_fraction'])} | "
            f"{_fmt(block['top_k_jaccard_vs_oracle'])} | {_fmt(block['top_k_m_lim_median_finite'])} |"
        )
    lines.append(f"| oracle | 1.000 | 1.000 | {_fmt(_median_finite([r['m_best'] for r in vs['oracle']['top_k']]))} |")
    lines.append(f"\nOperational top-{k} Jaccard (s_only vs s_plus_c) within the sample: {_fmt(vs['operational_top_k_jaccard'])}.")
    fo = result["frozen_top_k_outcomes"]
    lines += ["", f"## Frozen full-pool top-{k} sets, measured outcomes (S/N-only control)", "",
              "| set | m_best median | m10 median | m50 median | censored (m_best/m10/m50) |", "|---|---|---|---|---|"]
    for variant in VARIANTS:
        block = fo["sets"][variant]
        med, cen = block["medians_finite_only"], block["n_censored"]
        lines.append(
            f"| {variant} | {_fmt(med['m_best'])} | {_fmt(med['m10'])} | {_fmt(med['m50'])} | "
            f"{cen['m_best']}/{cen['m10']}/{cen['m50']} |"
        )
    lines.append(f"\nShared members: {', '.join(fo['shared_members'])} (Jaccard {_fmt(fo['top_k_jaccard'])}).")
    lines += ["", "| set | system | template | measured in | m_best |", "|---|---|---|---|---|"]
    for variant in VARIANTS:
        for member in fo["sets"][variant]["members"]:
            lines.append(f"| {variant} | {member['system_id']} | {member['template']} | {member['measured_in']} | {_fmt(member['m_best'])} |")
    pc = result["pool_context"]
    lines += ["", "## Selected tier against the validation sample", "",
              f"Sample m_best median: finite-only {_fmt(pc['sample_median_finite_only'])}, censoring-aware "
              f"{_fmt(pc['sample_median_censoring_aware'])} ({pc['sample_n_censored']} censored of {pc['sample_size']}).", "",
              "| selected system | m_best | fraction of sample at or below |", "|---|---|---|"]
    for entry in pc["selected"]:
        lines.append(f"| {entry['system_id']} | {_fmt(entry['m_best'])} | {_fmt(entry['sample_fraction_at_or_below'])} |")
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if value is None:
        return "undefined"
    return f"{value:.3f}"


if __name__ == "__main__":
    sys.exit(main())
