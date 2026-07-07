"""Ensemble detection summaries with binomial uncertainties.

Turns per-case detection tables (lists of row mappings, e.g. from an
aggregate CSV) into grouped completeness summaries with Wilson score
intervals, as used for the SPIE mass-completeness results.
"""

from __future__ import annotations

__all__ = ("completeness_summary", "wilson_interval")

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.0) -> Tuple[float, float, float]:
    """Return a binomial fraction with its Wilson score interval.

    Parameters
    ----------
    k : `int`
        Number of successes.
    n : `int`
        Number of trials.
    z : `float`, optional
        Interval half-width in standard normal quantiles; the default of
        ``1.0`` gives a one-sigma interval.

    Returns
    -------
    fraction : `float`
        Observed fraction ``k/n``, or NaN when ``n == 0``.
    lower : `float`
        Lower Wilson bound, clipped to ``[0, 1]``.
    upper : `float`
        Upper Wilson bound, clipped to ``[0, 1]``.
    """
    if n == 0:
        return math.nan, math.nan, math.nan
    phat = k/n
    denom = 1.0 + z*z/n
    center = (phat + z*z/(2.0*n))/denom
    half_width = z*math.sqrt((phat*(1.0 - phat) + z*z/(4.0*n))/n)/denom
    return phat, max(0.0, center - half_width), min(1.0, center + half_width)


def completeness_summary(
    rows: Sequence[Mapping[str, Any]],
    group_cols: Sequence[str],
    detection_cols: Sequence[str],
    value_cols: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Summarize detection completeness over groups of cases.

    Parameters
    ----------
    rows : `list` of `dict`
        Per-case records, e.g. rows of an aggregate results table.
    group_cols : `list` of `str`
        Columns whose value combinations define the groups.
    detection_cols : `list` of `str`
        Boolean columns to summarize. Each contributes ``<col>_count``,
        ``<col>_fraction``, and one-sigma Wilson bounds
        ``<col>_wilson_lo_1sigma`` / ``<col>_wilson_hi_1sigma``.
    value_cols : `list` of `str`, optional
        Numeric columns to summarize with ``median_<col>``, ``p16_<col>``,
        and ``p84_<col>``.

    Returns
    -------
    summary : `list` of `dict`
        One record per group, sorted by the group key values, each with
        the group columns, the case count ``n``, and the per-column
        statistics described above.
    """
    groups: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        groups.setdefault(key, []).append(row)

    summary: List[Dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        record: Dict[str, Any] = dict(zip(group_cols, key))
        record["n"] = len(group)
        for col in detection_cols:
            count = sum(bool(row[col]) for row in group)
            fraction, lower, upper = wilson_interval(count, len(group))
            record[f"{col}_count"] = count
            record[f"{col}_fraction"] = fraction
            record[f"{col}_wilson_lo_1sigma"] = lower
            record[f"{col}_wilson_hi_1sigma"] = upper
        for col in value_cols:
            values = np.asarray([float(row[col]) for row in group], dtype=float)
            record[f"median_{col}"] = float(np.median(values))
            record[f"p16_{col}"] = float(np.quantile(values, 0.16))
            record[f"p84_{col}"] = float(np.quantile(values, 0.84))
        summary.append(record)
    return summary
