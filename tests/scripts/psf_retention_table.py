"""Reduce the archived PSF knowledge-error masks to the Fisher columns of RASTI Table 5.

For each system rung (the M10 and M50 upper brackets of the 12 highest-ranked systems)
and each drift direction, R = |mismatch| / |matched| is the declared total-area ratio and
K = |mismatch & matched| / |matched| is the fraction of correct-PSF sensitive cells that
stay sensitive with the wrong PSF. Masks are restricted to the aperture; the matched mask
is the detectable mask at zero error. A rung passes an amplitude if Q10 over the eight
directions is at least 0.9 and Q90[F] is at most 0.1, as in the declared gate. Medians over
the 12 systems use the inverse empirical CDF (the Table 3 convention).

The campaign directory is the archived ``psf_knowledge_fisher_v1`` campaign: it holds
``harvest/review.json`` and ``outputs/<system>/psf_knowledge_map_*.npz``::

    python tests/scripts/psf_retention_table.py --campaign CAMPAIGN --output psf_retention_table.json

This is the reducer behind the submitted table; its output is the ``source_table`` that
``tests/scripts/extract_table5_knowledge.py`` checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np

AMPLITUDES = [1, 2, 5, 10, 20]
TABLE_AMPLITUDES = [2, 5, 10, 20]
DIRECTIONS = range(1, 9)
RETAIN, SPURIOUS = 0.9, 0.1


def unpack(archive, name):
    shape = tuple(archive[name + "_shape"])
    return np.unpackbits(archive[name + "_packed"])[:np.prod(shape)].astype(bool).reshape(shape)


def masks(campaign, system_id, logm, amplitude, direction):
    return np.load(campaign / f"outputs/{system_id}/psf_knowledge_map_m{logm:.2f}_delta{amplitude}_dir{direction}.npz")


def median(values):
    return float(np.quantile(values, .5, method="inverted_cdf"))


def tolerance(passing):
    """The largest amplitude of the contiguous passing run that starts at the smallest one."""
    best = 0
    for amplitude in AMPLITUDES:
        if amplitude not in passing:
            break
        best = amplitude
    return best


def reduce_rung_class(campaign, review, rung_class):
    systems = []
    for record in review["per_system_rung_class"][rung_class]:
        system_id, logm = record["system_id"], record["logm"]
        if record["below_ratio_floor"]:
            raise ValueError(f"{system_id}: rung is below the ratio floor")
        reference = masks(campaign, system_id, logm, 0, 0)
        aperture = unpack(reference, "aperture_mask")
        matched = unpack(reference, "detectable_mask") & aperture
        if matched.sum() != record["matched_cells"]:
            raise ValueError(f"{system_id}: matched cell count differs from the harvest review")
        per_amplitude = {}
        for amplitude in AMPLITUDES:
            ratio_r, ratio_k, ratio_f = [], [], []
            for direction in DIRECTIONS:
                archive = masks(campaign, system_id, logm, amplitude, direction)
                mismatch = unpack(archive, "mismatch_detectable_mask") & aperture
                spurious = unpack(archive, "false_positive_mask") & aperture
                ratio_r.append(mismatch.sum() / matched.sum())
                ratio_k.append((mismatch & matched).sum() / matched.sum())
                ratio_f.append(spurious.sum() / matched.sum())
            archived = record["per_delta"][str(amplitude)]
            archived_r = [direction["R"] for direction in archived["directions"]]
            if archived_r != ratio_r and not np.allclose(archived_r, ratio_r, rtol=0, atol=1e-12):
                raise ValueError(f"{system_id}/{amplitude}: R differs from the harvest review")
            if abs(np.quantile(ratio_f, .9) - archived["quantiles"]["F"]["q90"]) >= 1e-12:
                raise ValueError(f"{system_id}/{amplitude}: Q90[F] differs from the harvest review")
            q10_r, q10_k, q90_f = (float(np.quantile(values, quantile))
                                   for values, quantile in [(ratio_r, .1), (ratio_k, .1), (ratio_f, .9)])
            per_amplitude[amplitude] = dict(q10_R=q10_r, q10_K=q10_k, q90_F=q90_f,
                                            pass_R=q10_r >= RETAIN and q90_f <= SPURIOUS,
                                            pass_K=q10_k >= RETAIN and q90_f <= SPURIOUS)
        passing_r = [amplitude for amplitude in AMPLITUDES if per_amplitude[amplitude]["pass_R"]]
        passing_k = [amplitude for amplitude in AMPLITUDES if per_amplitude[amplitude]["pass_K"]]
        delta_star = record["delta_star"]["default"]
        if passing_r != [int(value) for value in delta_star["passing_deltas"]]:
            raise ValueError(f"{system_id}: passing amplitudes differ from the harvest review")
        if tolerance(passing_r) != delta_star["delta_star"]:
            raise ValueError(f"{system_id}: tolerance differs from the harvest review")
        systems.append(dict(system_id=system_id, logm=logm, matched_cells=int(matched.sum()),
                            per_amp=per_amplitude, tolerance_R=tolerance(passing_r),
                            tolerance_K=tolerance(passing_k),
                            monotone_R=passing_r == AMPLITUDES[:len(passing_r)],
                            monotone_K=passing_k == AMPLITUDES[:len(passing_k)]))
    tolerances_r = [system["tolerance_R"] for system in systems]
    tolerances_k = [system["tolerance_K"] for system in systems]
    if median(tolerances_r) != review["tier_summary"][rung_class]["default"]["median_delta_star"]:
        raise ValueError(f"{rung_class}: median tolerance differs from the harvest review")
    table = {}
    for amplitude in AMPLITUDES:
        rows = [system["per_amp"][amplitude] for system in systems]
        table[amplitude] = dict(median_q10_K=median([row["q10_K"] for row in rows]),
                                pass_K=sum(row["pass_K"] for row in rows),
                                median_q10_R=median([row["q10_R"] for row in rows]),
                                pass_R=sum(row["pass_R"] for row in rows),
                                max_q90_F=max(row["q90_F"] for row in rows))
    return dict(systems=systems, table=table,
                tolerance_K=dict(median=median(tolerances_k), range=[min(tolerances_k), max(tolerances_k)],
                                 values=sorted(tolerances_k)),
                tolerance_R=dict(median=median(tolerances_r), range=[min(tolerances_r), max(tolerances_r)],
                                 values=sorted(tolerances_r)),
                all_monotone=all(system["monotone_R"] and system["monotone_K"] for system in systems))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads((args.campaign / "harvest/review.json").read_text())["science"]
    if review["gates"]["default"] != [RETAIN, SPURIOUS]:
        raise ValueError("the harvest review used a different gate")
    out = {rung_class: reduce_rung_class(args.campaign, review, rung_class) for rung_class in ("m10", "m50")}
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    for rung_class, reduced in out.items():
        table = reduced["table"]
        print(rung_class, "tolerance K", reduced["tolerance_K"], "| R", reduced["tolerance_R"],
              "| monotone", reduced["all_monotone"])
        for amplitude in TABLE_AMPLITUDES:
            row = table[amplitude]
            print(f"  {amplitude:2d} nm  K {row['median_q10_K']:.3f} pass {row['pass_K']:2d}/12   "
                  f"R {row['median_q10_R']:.3f} pass {row['pass_R']:2d}/12   max Q90F {row['max_q90_F']:.3f}")


if __name__ == "__main__":
    main()
