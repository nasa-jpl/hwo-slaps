#!/usr/bin/env python3
"""Launch two HWO-SLAPS configs in isolated, GPU-pinned processes.

This launcher intentionally imports only the Python standard library. Each
child receives ``CUDA_VISIBLE_DEVICES`` before it starts the HWO-SLAPS runner,
so JAX cannot initialize against the wrong device first.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _arguments() -> argparse.Namespace:
    """Parse two config paths, two physical GPU indices, and output paths."""
    parser = argparse.ArgumentParser(
        description="Run two HWO-SLAPS configs on distinct pinned GPUs",
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config path; pass exactly twice",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        required=True,
        help="Physical GPU index; pass exactly twice",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used by each child",
    )
    parser.add_argument(
        "--runner",
        default="runner.py",
        help="HWO-SLAPS runner script",
    )
    parser.add_argument(
        "--log-dir",
        default="scratch/item7b_validation/v7_two_gpu",
        help="Directory for one combined stdout/stderr log per child",
    )
    arguments = parser.parse_args()
    if len(arguments.config) != 2 or len(arguments.gpu) != 2:
        parser.error("pass --config and --gpu exactly twice each")
    if len(set(arguments.gpu)) != 2:
        parser.error("the two --gpu values must be distinct")
    return arguments


def main() -> int:
    """Start both child processes and return their aggregate status."""
    arguments = _arguments()
    log_dir = Path(arguments.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    children = []

    try:
        for index, (config, gpu) in enumerate(
            zip(arguments.config, arguments.gpu),
            start=1,
        ):
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            log_path = log_dir / f"child_{index}_gpu_{gpu}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            command = [
                arguments.python,
                arguments.runner,
                "--config",
                config,
            ]
            try:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            except Exception:
                log_handle.close()
                raise
            children.append((process, log_handle, log_path, gpu))
            print(
                f"started pid={process.pid} gpu={gpu} log={log_path}",
                flush=True,
            )
    except Exception:
        for process, log_handle, _, _ in children:
            try:
                if process.poll() is None:
                    process.terminate()
            except OSError:
                pass
            try:
                process.wait()
            except OSError:
                pass
            log_handle.close()
        raise

    failed = False
    for process, log_handle, log_path, gpu in children:
        try:
            return_code = process.wait()
        finally:
            log_handle.close()
        print(
            f"finished pid={process.pid} gpu={gpu} status={return_code} "
            f"log={log_path}",
            flush=True,
        )
        failed = failed or return_code != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
