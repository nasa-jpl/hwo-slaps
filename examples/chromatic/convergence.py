"""Read the six ring forecasts and report the prescribed chromatic convergence gates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_support import positive_number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-threshold", type=positive_number, required=True)
    args = parser.parse_args(argv)
    import numpy as np
    from hwoslaps.analysis.reductions import summarize
    from hwoslaps.artifacts import load_forecast, write_json
    from hwoslaps.identity import file_digest, mapping_digest

    variants = {"11_901": (11, 901), "22_901": (22, 901), "11_601": (11, 601)}
    products, records, peaks, spurious = {}, {}, {}, {}
    common_identity = None
    reference = None
    for variant, (nodes, support) in variants.items():
        for arm in ("matched", "monochromatic"):
            name = variant + "_" + arm
            directory = args.directory / name
            result = load_forecast(directory / "forecast.npz")
            record = json.loads((directory / "run.json").read_text())
            engine = result.provenance["engine"]
            if (engine["kind"] != "jax" or engine.get("device") not in record["execution"].get("devices", [])
                    or record["execution"]["engine"] != "jax" or record["engine"] != engine
                    or record["forecast_sha256"] != file_digest(directory / "forecast.npz")):
                raise ValueError(f"{name}: actual forecast engine or artifact differs from its run record")
            bindings = {"config_digest": result.provenance["config_digest"],
                        "comparison_digest": result.provenance["comparison_digest"],
                        "input_files": result.provenance["file_digests"],
                        "masses_msun": result.masses_msun.tolist(), "positions_count": len(result.positions),
                        "metric": result.detection_metric, "sampling": result.provenance["sampling"],
                        "spectral": result.provenance["spectral"], "photometry": result.provenance["photometry"]}
            if any(record[key] != value for key, value in bindings.items()):
                raise ValueError(f"{name}: run record does not describe its actual forecast inputs")
            if (record["budget_s"] != 600 or not np.isfinite(record["elapsed_s"]) or record["elapsed_s"] < 0
                    or record["within_budget"] != (record["elapsed_s"] <= 600)):
                raise ValueError(f"{name}: run record does not describe the prescribed runtime budget")
            config = deepcopy(dict(result.config))
            truth = config["psf"]["truth"]
            if (truth["wavelength_samples"] != nodes or truth["kernel_shape"] != [support, support]
                    or config["psf"]["model"]["kind"] != arm or record["execution"]["lane"] != "gpu"
                    or record["q_threshold"] != args.q_threshold or not record["extra"]["ring"]
                    or record["config_digest"] != result.provenance["config_digest"]
                    or result.positions.kind != "ring" or len(result.positions) != 36):
                raise ValueError(f"{name}: product does not have the prescribed ring/PSF/threshold/lane inputs")
            if arm == "monochromatic" and config["psf"]["model"]["wavelength_nm"] is not None:
                raise ValueError(f"{name}: model must use the group photon-weighted mean wavelengths")
            truth.pop("wavelength_samples")
            truth.pop("kernel_shape")
            config["psf"].pop("model")
            identity = mapping_digest(config)
            if common_identity is None:
                common_identity = identity
                reference = result
            elif (identity != common_identity
                  or result.provenance["file_digests"] != reference.provenance["file_digests"]
                  or result.masses_msun.dtype != reference.masses_msun.dtype
                  or result.masses_msun.tobytes() != reference.masses_msun.tobytes()
                  or result.positions_yx.dtype != reference.positions_yx.dtype
                  or result.positions_yx.tobytes() != reference.positions_yx.tobytes()):
                raise ValueError(f"{name}: science inputs differ beyond the prescribed PSF changes")
            products[name] = result
            records[name] = {"sampling": record["sampling"], "spectral": record["spectral"],
                             "elapsed_s": record["elapsed_s"], "budget_s": record["budget_s"],
                             "within_budget": record["within_budget"], "engine": engine,
                             "forecast_sha256": record["forecast_sha256"], "environment": record["environment"],
                             "config_digest": result.provenance["config_digest"],
                             "input_files": result.provenance["file_digests"]}
            peaks[name] = summarize(result, q_threshold=args.q_threshold).q_max
            if arm == "monochromatic":
                spurious[name] = summarize(result, q_threshold=args.q_threshold, metric="q_spurious").q_max
    gates = {}
    matched_base = peaks["11_901_matched"]
    if np.any(matched_base <= 0):
        raise ValueError("matched reference q_max must be positive for relative convergence")
    for variant in ("22_901", "11_601"):
        changes = {}
        passed = np.ones(matched_base.shape, dtype=bool)
        for arm in ("matched", "monochromatic"):
            base = peaks["11_901_" + arm]
            change = np.full(base.shape, np.nan)
            np.divide(np.abs(peaks[variant + "_" + arm] - base), base, out=change, where=base > 0)
            changes[arm + "_q_max_relative"] = [float(item) if np.isfinite(item) else None for item in change]
            passed &= np.isfinite(change) & (change <= 1e-2)
        error = np.abs(spurious[variant + "_monochromatic"] - spurious["11_901_monochromatic"]) / matched_base
        changes["q_spurious_change_over_matched_q_max"] = error.tolist()
        changes["passed"] = (passed & (error <= 1e-2)).tolist()
        gates[variant] = changes
    record = {"masses_msun": reference.masses_msun.tolist(), "q_threshold": args.q_threshold,
              "tolerance": 1e-2, "gates": gates, "products": records,
              "all_products_within_budget": all(row["within_budget"] for row in records.values()),
              "passed_scope": "numerical convergence only; runtime budgets are reported separately",
              "passed": all(all(row["passed"]) for row in gates.values())}
    write_json(args.output, record)
    print(json.dumps(record, indent=2, allow_nan=False))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
