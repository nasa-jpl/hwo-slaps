"""Explicit scientific operations and command-owned output artifacts."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import yaml

from .config import load_config


class _Tee:
    """Write to the caller's stream and a line-buffered operation log."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing any optical or inference backend."""
    parser = argparse.ArgumentParser(description="Strong-lensing simulations and forecasts")
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation, description in (
        ("validate", "Compose and validate scientific configuration"),
        ("simulate", "Simulate one observation"),
        ("forecast", "Evaluate a prepared sensitivity forecast"),
    ):
        command = commands.add_parser(operation, help=description)
        command.add_argument("-c", "--config", action="append", required=True, metavar="YAML")
        command.add_argument("--base-dir", type=Path, help="Explicit base for file-declared relative paths")
        if operation != "validate":
            command.add_argument(
                "--output-dir", required=True, type=Path,
                help="New directory for operation artifacts",
            )
        if operation == "forecast":
            command.add_argument(
                "--masses", nargs="+", type=float, metavar="MSUN",
                help="Explicit subhalo masses in solar masses",
            )
            command.add_argument(
                "--positions", type=Path, metavar="JSON",
                help="JSON array of [y, x] positions in arcseconds",
            )
    return parser


def _read_positions(path: Path | None):
    if path is None:
        return None
    with path.expanduser().open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    positions = np.asarray(values, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] == 0:
        raise ValueError("positions must contain a non-empty array of [y, x] coordinates")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain finite coordinates")
    return positions


def _metadata_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("observation metadata must contain JSON-compatible values")


def _save_observation(observation: Any, path: Path) -> None:
    """Write simulated arrays, dataset PSF, and explicit observation identity."""
    metadata = json.dumps(observation.metadata, sort_keys=True, default=_metadata_value, allow_nan=False)
    np.savez_compressed(
        path,
        data_adu=np.asarray(observation.data.native),
        noise_adu=np.asarray(observation.noise_map.native),
        noiseless_source_eps=np.asarray(observation.noiseless_source_eps),
        imaging_psf_kernel=np.asarray(observation.psf.native),
        pixel_scale_arcsec=float(observation.pixel_scale),
        metadata_json=np.asarray(metadata),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run a public engine operation with explicit command-owned I/O."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, base_dir=args.base_dir)
        if args.operation == "forecast" and "modeling" not in config:
            raise ValueError("forecast requires modeling.fisher settings")
        masses = getattr(args, "masses", None)
        if masses is not None and (
            not np.all(np.isfinite(masses)) or np.any(np.asarray(masses) <= 0)
        ):
            raise ValueError("masses must be positive finite solar masses")
        positions = _read_positions(getattr(args, "positions", None))
        if args.operation == "validate":
            print("Configuration valid")
            return 0

        output = args.output_dir.expanduser().resolve()
        try:
            output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Output directory already exists: {output}; choose a new output directory"
            ) from exc
        with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                from .provenance import write_provenance

                command = list(sys.argv) if argv is None else ["hwoslaps", *argv]
                if args.operation == "simulate":
                    from . import simulate

                    observation = simulate(config)
                    with (output / "config_used.yaml").open("w", encoding="utf-8") as stream:
                        yaml.safe_dump(config, stream, sort_keys=False)
                    write_provenance(output / "provenance.yaml", config=config, command=command)
                    _save_observation(observation, output / "observation.npz")
                else:
                    from . import forecast, prepare_forecast

                    prepared = prepare_forecast(config)
                    effective = prepared.config
                    with (output / "config_used.yaml").open("w", encoding="utf-8") as stream:
                        yaml.safe_dump(effective, stream, sort_keys=False)
                    write_provenance(output / "provenance.yaml", config=effective, command=command)
                    result = forecast(prepared, masses=masses, positions=positions)
                    result.save_npz(output / "forecast.npz")
                print(f"Artifacts: {output}")
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
