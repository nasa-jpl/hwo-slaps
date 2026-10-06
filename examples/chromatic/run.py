"""Run one chromatic product, or one of the prescribed convergence arms."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_support import positive_number, run_product, verify_sei


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-threshold", type=positive_number, required=True)
    parser.add_argument("--variant", choices=("11_901", "22_901", "11_601"), default="11_901")
    parser.add_argument("--arm", choices=("matched", "monochromatic"), default="matched")
    parser.add_argument("--ring", action="store_true", help="use the 36-position convergence layout")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    from hwoslaps.config.loading import load_config

    directory = Path(__file__).resolve().parent
    hwo = directory.parent / "hwo_reference"
    sei = verify_sei(hwo / "sei_v0.1.9")
    paths = [hwo / "instrument.yaml", directory / "instrument_sei_chain.yaml", directory / "scene_two_colour.yaml",
             directory / "ring.yaml" if args.ring else hwo / "forecast.yaml"]
    if args.variant == "22_901":
        paths.append(directory / "wavelengths_22.yaml")
    elif args.variant == "11_601":
        paths.append(directory / "support_601.yaml")
    if args.arm == "monochromatic":
        paths.append(directory / "monochromatic.yaml")
    config = load_config(paths)
    run_product(config, args.output, masses=[1e8, 1e9], engine="jax", q_threshold=args.q_threshold,
                budget_s=600, command=sys.argv if argv is None else [str(__file__), *argv], plot=args.plot,
                extra={"variant": args.variant, "arm": args.arm, "ring": args.ring, "sei": sei})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
