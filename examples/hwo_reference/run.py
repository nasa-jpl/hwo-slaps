"""Run the HWO reference and check its input-first photometric targets."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_support import positive_number, run_product, verify_sei


def check_reference(prepared):
    record = prepared.observation.photometry
    if not math.isclose(record.collecting_area_m2, 33.606448937520405, rel_tol=1e-12, abs_tol=0):
        raise ValueError("HWO collecting area differs from the pinned pupil target")
    component = prepared.config.scene.source.light[0]
    if component.flux is not None and component.flux.ab_mag is not None:
        rate = record.components["source.disk"]["rate_e_per_s"]
        if not math.isclose(rate, 8.951505744562876, rel_tol=1e-9, abs_tol=0):
            raise ValueError("HWO source rate differs from the pinned AB target")
        intensity = prepared.scene.spec.source.light[0].values["intensity"]
        if not math.isclose(intensity, 0.003174147284617635, rel_tol=2e-7, abs_tol=0):
            raise ValueError("HWO continuous normalization differs from the discrete paper target")
    sky = prepared.observation.exposure.sky_rate_e_per_s
    if not math.isclose(sky, 0.002510279845963486, rel_tol=1e-9, abs_tol=0):
        raise ValueError("HWO sky rate differs from the pinned target")
    if not math.isclose(prepared.observation.exposure.blank_variance_e2, 9.100559691926973, rel_tol=1e-15, abs_tol=0):
        raise ValueError("HWO blank detector variance differs from the pinned target")


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=11, help="seed for the smooth noisy observation")
    parser.add_argument("--q-threshold", type=positive_number, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
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
                check=check_reference, extra={"quick": args.quick, "sei": sei}, plot=args.plot, started_at=started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
