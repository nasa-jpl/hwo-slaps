#!/usr/bin/env python
"""Run the pre-registered T4 rank-stability harness over one pool.

Reads ``configs/design/design_freeze_v1.yaml``, verifies every artifact
the freeze binds by digest, and runs
`hwoslaps.analysis.rank_stability` over a directory of member ``.npz``
records. The replicate count, the replicate indices, the per-system
noise seeds and the reported tier size all come from the freeze, so a
clean clone regenerates the identical rankings from the identical
declarations.

The report is written as JSON beside nothing else this script owns; the
output path is given on the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT/"src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT/"src"))


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "members_dir",
        help="Directory of member .npz records following the harness contract",
    )
    parser.add_argument(
        "output",
        help="Path the JSON report is written to",
    )
    parser.add_argument(
        "--label",
        default="stage0_pool",
        help="Name of this run, carried into the report",
    )
    parser.add_argument(
        "--design-freeze",
        default=None,
        help="Design freeze artifact; defaults to the committed one",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Report progress after every replicate",
    )
    return parser


def _progress_reporter(enabled: bool):
    """Return a progress callable, or None when reporting is off."""
    if not enabled:
        return None

    def report(done: int, total: int) -> None:
        """Print one progress line per completed replicate."""
        print(f"  replicate {done}/{total} done", flush=True)

    return report


def main(argv=None) -> None:
    """Run the harness and write its report."""
    args = _build_parser().parse_args(argv)

    from hwoslaps.analysis.rank_stability import run_rank_stability
    from hwoslaps.campaign.design_freeze import (
        design_freeze_digest,
        load_design_freeze,
        verify_bound_artifacts,
    )

    freeze = load_design_freeze(args.design_freeze)
    bound = verify_bound_artifacts(freeze)
    report = run_rank_stability(
        args.members_dir,
        freeze,
        args.label,
        progress=_progress_reporter(args.progress),
    )
    report["design_freeze_sha256"] = design_freeze_digest(args.design_freeze)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    stability = report["stability"]
    print(f"design freeze          : {report['design_freeze_sha256']}")
    print(f"bound artifacts        : {len(bound['verified'])} verified, absent "
          f"{bound['absent']}")
    print(f"pool size              : {report['pool_size']}")
    print(f"replicates             : {stability['replicates']}")
    print(f"tier size k            : {report['tier_size']}")
    print(f"oracle available       : {report['oracle_available']}")
    print(f"report                 : {output}")
    for variant in report["curves"]:
        summary = stability["summary"][variant]
        print(f"[{variant}] Spearman vs noiseless "
              f"min {summary['spearman_vs_noiseless']['min']:.4f} "
              f"median {summary['spearman_vs_noiseless']['median']:.4f}; "
              f"top-{report['tier_size']} Jaccard median "
              f"{summary['top_k_jaccard_vs_noiseless']['median']:.4f}")
    ratios = list(stability["estimator_ratios"].values())
    print("[noisy estimators] median over replicates of the pool-median "
          "noisy/noiseless ratio: "
          f"S {median(entry['arc_snr_ratio_median'] for entry in ratios):.4f}, "
          f"G {median(entry['gradient_power_ratio_median'] for entry in ratios):.4f}")
    print(f"frozen goldens         : {report['frozen_selection']['golden_ids']}")


if __name__ == "__main__":
    main()
