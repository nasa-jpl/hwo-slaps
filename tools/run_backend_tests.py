"""Run optional scientific-backend tests with explicit process-local setup."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numba-jit", choices=("enabled", "disabled"), default="disabled",
        help="Backend-only policy; disabled matches the validated reference lane",
    )
    parser.add_argument("--require-gpu", action="store_true", help="Fail rather than skip when CUDA is unavailable")
    args, pytest_args = parser.parse_known_args(argv)
    os.environ["NUMBA_DISABLE_JIT"] = "1" if args.numba_jit == "disabled" else "0"
    root = Path(__file__).resolve().parents[1]
    resolved = []
    for value in pytest_args:
        name, separator, node = value.partition("::")
        if not value.startswith("-") and (root / name).exists():
            value = str((root / name).resolve()) + (separator + node if separator else "")
        resolved.append(value)
    if not resolved:
        resolved = [str(root / "tests"), "-q"]
    with tempfile.TemporaryDirectory(prefix="hwoslaps-backend-tests-") as directory:
        os.chdir(directory)
        from autoconf import conf

        autoarray = importlib.util.find_spec("autoarray")
        if autoarray is None or autoarray.origin is None:
            raise RuntimeError("The backend lane requires the supported PyAutoLabs environment")
        conf.instance.push(str(Path(autoarray.origin).resolve().parent / "config"), keep_first=True)
        if args.require_gpu:
            import jax
            if not jax.devices("gpu"):
                raise RuntimeError("The GPU lane requires an available CUDA backend")
        import pytest

        return pytest.main(resolved)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
