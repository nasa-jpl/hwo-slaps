"""Run the HWO reference and check its input-first photometric targets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_support import positive_number, run_product, verify_sei


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference-workers", type=int, default=8, help="CPU workers for --quick")
    parser.add_argument("--seed", type=int, default=11, help="seed for the smooth noisy observation")
    parser.add_argument("--q-threshold", type=positive_number, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.reference_workers < 1:
        parser.error("--reference-workers must be positive")
    from hwoslaps.config.schema import load_config

    directory = Path(__file__).resolve().parent
    sei = verify_sei(directory / "sei_v0.1.9")
    config = load_config([directory / name for name in ("scene_smooth_ring.yaml", "instrument.yaml", "forecast.yaml")]
                         + args.overlay)
    if args.quick:
        config = config.replace({"psf": {"truth": {"kernel_shape": [101, 101]}},
                                 "forecast": {"positions": {"spacing_arcsec": .3}}})
    run_product(config, args.output, masses=[1e7, 1e8, 1e9], engine="reference" if args.quick else "jax",
                q_threshold=args.q_threshold, budget_s=120 if args.quick else 600,
                command=sys.argv if argv is None else [str(__file__), *argv], noise_seed=args.seed,
                extra={"quick": args.quick, "sei": sei}, plot=args.plot, started_at=started,
                reference_workers=args.reference_workers if args.quick else 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
