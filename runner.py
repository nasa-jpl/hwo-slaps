#!/usr/bin/env python
"""Simple runner for HWO-SLAPS pipeline with run artifact capture."""

import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


class _Tee:
    """Write stream data to both terminal and a log file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description='Run HWO-SLAPS pipeline')
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_name = config.get('run_name', 'run')
    output_root = Path(config.get('plotting', {}).get('output_dir', 'outputs')).expanduser()
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = run_dir / 'config_used.yaml'
    with open(snapshot_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    log_path = run_dir / 'run.log'
    with open(log_path, 'w', buffering=1) as log_file:
        tee_stdout = _Tee(sys.__stdout__, log_file)
        tee_stderr = _Tee(sys.__stderr__, log_file)
        with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
            print(f"Run log: {log_path}")
            print(f"Config snapshot: {snapshot_path}")

            # Import here so third-party import-time logs are also captured.
            from hwoslaps.pipeline import run_enhanced_pipeline

            # Run enhanced pipeline (automatically detects standard vs detection mode)
            run_enhanced_pipeline(str(config_path), verbose=not args.quiet)


if __name__ == '__main__':
    main()
