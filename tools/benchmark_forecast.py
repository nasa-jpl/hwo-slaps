"""Time forecast preparation and a first and a repeated forecast for one configuration.

YAML files compose in order. Pin BLAS threads for comparable timings, for example::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tools/benchmark_forecast.py \\
        configs/minimal.yaml --engine reference --masses 1e7 1e8 1e9

With ``--engine jax`` the first forecast includes compilation and the repeated one
does not. The printed ``engine`` record describes the engine that ran: a scene with a
PointMass or SIS subhalo or perturber runs on the CPU reference engine even when JAX is
requested, and its record then shows ``jax_accelerated: false``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--engine", choices=("reference", "jax"), default="reference")
    parser.add_argument("--masses", type=float, nargs="+", default=[1.0e8])
    args = parser.parse_args(argv)
    from hwoslaps.fisher.api import Execution, forecast, prepare_forecast

    start = perf_counter()
    with prepare_forecast(args.configs, execution=Execution(engine=args.engine)) as prepared:
        seconds = {"prepare": perf_counter() - start}
        start = perf_counter()
        result = forecast(prepared, masses_msun=args.masses)
        seconds["forecast_first"] = perf_counter() - start
        start = perf_counter()
        forecast(prepared, masses_msun=args.masses)
        seconds["forecast_repeat"] = perf_counter() - start
    nodes = len(result.positions_yx) * len(args.masses)
    print(json.dumps({"configs": [str(path) for path in args.configs], "engine": result.provenance["engine"], "nodes": nodes,
                      "seconds": seconds, "nodes_per_second_repeat": nodes / seconds["forecast_repeat"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
