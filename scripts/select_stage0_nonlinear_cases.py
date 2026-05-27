#!/usr/bin/env python
"""Select a bounded nonlinear-validation subset from Stage 0 Fisher results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _float(row: Dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value not in ("", None) else default
    except ValueError:
        return default


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row.get("status") == "success" and math.isfinite(_float(row, "q_f"))
        ]


def _row_key(row: Dict[str, str]) -> str:
    return str(row["run_name"])


def _group_key(row: Dict[str, str]) -> Tuple[float, str, float]:
    return (
        round(_float(row, "mass_msun"), 6),
        str(row.get("psf_family", "")),
        round(_float(row, "psf_amplitude"), 6),
    )


def _median_q_row(rows: Sequence[Dict[str, str]]) -> Dict[str, str]:
    ordered = sorted(rows, key=lambda row: _float(row, "q_f"))
    return ordered[len(ordered) // 2]


def _add_candidate(
    selected: Dict[str, Dict[str, str]],
    priorities: Dict[str, Tuple[int, float, str]],
    row: Dict[str, str],
    *,
    priority: int,
    score: float,
    reason: str,
) -> None:
    key = _row_key(row)
    if key not in selected or (priority, score, key) < priorities[key]:
        selected[key] = row
        priorities[key] = (priority, score, reason)


def select_cases(
    rows: Sequence[Dict[str, str]],
    *,
    max_cases: int,
    near_low: float,
    near_high: float,
    max_near_per_group: int,
    endpoint_amplitudes: Iterable[float],
    include_perfect: bool,
) -> List[str]:
    selected: Dict[str, Dict[str, str]] = {}
    priorities: Dict[str, Tuple[int, float, str]] = {}

    if include_perfect:
        for row in rows:
            if row.get("psf_family") == "none":
                _add_candidate(
                    selected,
                    priorities,
                    row,
                    priority=0,
                    score=abs(_float(row, "q_f") - 10.0),
                    reason="perfect_reference",
                )

    groups: Dict[Tuple[float, str, float], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("psf_family") != "none":
            groups[_group_key(row)].append(row)

    for group_rows in groups.values():
        near_rows = [
            row
            for row in group_rows
            if near_low <= _float(row, "q_f") <= near_high
        ]
        near_rows.sort(key=lambda row: abs(_float(row, "q_f") - 10.0))
        for row in near_rows[:max(0, int(max_near_per_group))]:
            _add_candidate(
                selected,
                priorities,
                row,
                priority=1,
                score=abs(_float(row, "q_f") - 10.0),
                reason="near_threshold",
            )

    endpoint_set = {float(value) for value in endpoint_amplitudes}
    for (_mass, _family, amplitude), group_rows in groups.items():
        if amplitude not in endpoint_set:
            continue
        worst = min(group_rows, key=lambda row: _float(row, "q_f"))
        median = _median_q_row(group_rows)
        _add_candidate(
            selected,
            priorities,
            worst,
            priority=2,
            score=_float(worst, "q_f"),
            reason="endpoint_worst",
        )
        _add_candidate(
            selected,
            priorities,
            median,
            priority=3,
            score=abs(_float(median, "q_f") - 10.0),
            reason="endpoint_median",
        )

    mass_family_groups: Dict[Tuple[float, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("psf_family") != "none":
            mass_family_groups[(round(_float(row, "mass_msun"), 6), str(row.get("psf_family")))].append(row)
    for group_rows in mass_family_groups.values():
        row = min(group_rows, key=lambda item: abs(_float(item, "q_f") - 10.0))
        _add_candidate(
            selected,
            priorities,
            row,
            priority=4,
            score=abs(_float(row, "q_f") - 10.0),
            reason="mass_family_closest_threshold",
        )

    ordered = sorted(
        selected.values(),
        key=lambda row: (
            priorities[_row_key(row)][0],
            priorities[_row_key(row)][1],
            _float(row, "mass_msun"),
            str(row.get("psf_family", "")),
            _float(row, "psf_amplitude"),
            _row_key(row),
        ),
    )
    return [_row_key(row) for row in ordered[: max(1, int(max_cases))]]


def select_false_positive_templates(
    rows: Sequence[Dict[str, str]],
    *,
    max_cases: int,
    min_amplitude: float,
) -> List[str]:
    candidates = [
        row
        for row in rows
        if row.get("psf_family") != "none" and _float(row, "psf_amplitude") >= min_amplitude
    ]
    candidates.sort(
        key=lambda row: (
            -_float(row, "psf_amplitude"),
            _float(row, "q_f"),
            _float(row, "mass_msun"),
            str(row.get("psf_family", "")),
            _row_key(row),
        )
    )
    selected: List[str] = []
    seen: set[Tuple[float, str, float]] = set()
    for row in candidates:
        key = _group_key(row)
        if key in seen:
            continue
        selected.append(_row_key(row))
        seen.add(key)
        if len(selected) >= max(0, int(max_cases)):
            break
    return selected


def _write_lines(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv")
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--false-positive-output", required=True)
    parser.add_argument("--max-cases", type=int, default=160)
    parser.add_argument("--max-false-positive-cases", type=int, default=18)
    parser.add_argument("--near-low", type=float, default=8.0)
    parser.add_argument("--near-high", type=float, default=12.0)
    parser.add_argument("--max-near-per-group", type=int, default=2)
    parser.add_argument("--endpoint-amplitudes", nargs="*", type=float, default=[50.0, 100.0])
    parser.add_argument("--false-positive-min-amplitude", type=float, default=50.0)
    parser.add_argument("--no-perfect", dest="include_perfect", action="store_false", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _read_rows(Path(args.results_csv))
    cases = select_cases(
        rows,
        max_cases=args.max_cases,
        near_low=args.near_low,
        near_high=args.near_high,
        max_near_per_group=args.max_near_per_group,
        endpoint_amplitudes=args.endpoint_amplitudes,
        include_perfect=bool(args.include_perfect),
    )
    false_positive_cases = select_false_positive_templates(
        rows,
        max_cases=args.max_false_positive_cases,
        min_amplitude=args.false_positive_min_amplitude,
    )
    _write_lines(Path(args.cases_output), cases)
    _write_lines(Path(args.false_positive_output), false_positive_cases)
    print(f"Selected injected nonlinear cases: {len(cases)}")
    print(f"Selected false-positive templates: {len(false_positive_cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
