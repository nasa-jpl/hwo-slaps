#!/usr/bin/env python
"""Run one standalone dark-matter fold and write its forecast artifacts.

This CLI is intentionally a heavy runtime entry point. Provenance capture
imports engine packages to record their versions even though the fold itself
is CPU-only post-processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold_spec", help="Standalone subhalo-fold YAML path")
    parser.add_argument("output_dir", help="Forecast output directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing subhalo_forecast.npz",
    )
    return parser


def main(argv=None) -> None:
    """Validate, run, and persist one subhalo forecast."""
    args = _build_parser().parse_args(argv)
    spec_path = Path(args.fold_spec).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    artifact_path = output_dir / "subhalo_forecast.npz"
    if artifact_path.exists() and not args.force:
        raise ValueError(
            f"Refusing to overwrite {artifact_path}; pass --force to replace it"
        )

    with spec_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    from hwoslaps.analysis.subhalo_forecast import (
        run_subhalo_forecast,
        save_subhalo_forecast_npz,
        validate_subhalo_forecast_config,
    )
    from hwoslaps.plotting.subhalo_forecast import (
        plot_expected_detections_vs_mhm,
        plot_lenses_to_discriminate,
    )
    from hwoslaps.provenance import write_provenance

    normalized = validate_subhalo_forecast_config(config)
    data = run_subhalo_forecast(normalized)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_subhalo_forecast_npz(data, artifact_path)
    plot_expected_detections_vs_mhm(
        data,
        output_dir / "expected_detections_vs_mhm.png",
    )
    plot_lenses_to_discriminate(
        data,
        output_dir / "lenses_to_discriminate.png",
    )
    write_provenance(
        output_dir / "provenance.yaml",
        config=config,
        command=sys.argv if argv is None else [sys.argv[0], *argv],
    )


if __name__ == "__main__":
    main()
