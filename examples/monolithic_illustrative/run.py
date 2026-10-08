"""Forecast one halo mass for the illustrative monolithic instrument."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_support import positive_number, run_product


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-threshold", type=positive_number, required=True)
    parser.add_argument("--reference-workers", type=int, default=8, help="CPU reference workers")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.reference_workers < 1:
        parser.error("--reference-workers must be positive")
    from hwoslaps.config.schema import load_config

    directory = Path(__file__).resolve().parent
    config = load_config([directory / name for name in ("scene.yaml", "instrument.yaml", "forecast.yaml")])
    run_product(config, args.output, masses=[1e8], engine="reference", q_threshold=args.q_threshold,
                budget_s=120, command=sys.argv if argv is None else [str(__file__), *argv], plot=args.plot, started_at=started,
                reference_workers=args.reference_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
