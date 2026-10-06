"""Compare illustrative matched and mismatched kernel forecasts."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from example_support import new_output, positive_number, run_product


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q-threshold", type=positive_number, required=True)
    parser.add_argument("--min-reference-count", type=int, required=True)
    args = parser.parse_args(argv)
    if args.min_reference_count < 1:
        parser.error("--min-reference-count must be at least 1")
    import numpy as np
    from hwoslaps.analysis.knowledge_error import knowledge_error_areas
    from hwoslaps.artifacts import write_json
    from hwoslaps.config.schema import load_config

    directory = Path(__file__).resolve().parent
    minimal = directory.parents[1] / "configs" / "minimal.yaml"
    matched_config = load_config([minimal, directory / "truth.yaml"])
    model_config = load_config([minimal, directory / "truth.yaml", directory / "knowledge_error.yaml"])
    output = new_output(args.output)
    results, records = [], {}
    for arm, config in (("matched", matched_config), ("mismatched", model_config)):
        result, record = run_product(config, output / arm, masses=[1e7, 1e8, 1e9], engine="reference",
                                     q_threshold=args.q_threshold, budget_s=120,
                                     command=sys.argv if argv is None else [str(__file__), *argv])
        results.append(result)
        records[arm] = record
    areas = knowledge_error_areas(*results, q_threshold=args.q_threshold,
                                 min_reference_count=args.min_reference_count)
    eligible = areas.reference_count >= args.min_reference_count
    values = {}
    for field in fields(areas):
        value = getattr(areas, field.name)
        if isinstance(value, np.ndarray):
            if field.name in {"retention", "detected_area_ratio", "spurious_ratio"}:
                values[field.name] = [float(item) if use else None for item, use in zip(value, eligible, strict=True)]
            else:
                values[field.name] = value.tolist()
        else:
            values[field.name] = value
    values["eligible"] = eligible.tolist()
    values["sampling"] = {arm: record["sampling"] for arm, record in records.items()}
    elapsed = time.perf_counter() - started
    values.update(elapsed_s=elapsed, budget_s=120, within_budget=elapsed <= 120)
    write_json(output / "knowledge_error.json", values)
    print(values)
    if elapsed > 120:
        raise RuntimeError(f"paired kernel example exceeds 120 s: {elapsed:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
