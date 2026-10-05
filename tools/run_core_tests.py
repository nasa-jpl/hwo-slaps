"""Run the dependency-light test lane with every scientific backend import blocked."""
from __future__ import annotations

import importlib.abc
from pathlib import Path
import sys

BACKENDS = frozenset(("autolens", "autogalaxy", "autoarray", "autofit", "autoconf", "hcipy", "jax", "jaxlib",
                      "nautilus", "numba", "matplotlib"))
CORE_MARKERS = "not backend and not xtx_gpu and not xtx_multi_gpu"


class BackendBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BACKENDS:
            raise ImportError(f"the core lane forbids backend import {fullname!r}")
        return None


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value == "-m" or value.startswith("--markexpr") or
           (value.startswith("-m") and not value.startswith("--")) for value in arguments):
        raise SystemExit("the core lane fixes its own marker expression")
    loaded = [name for name in sys.modules if name.split(".", 1)[0] in BACKENDS]
    if loaded:
        raise RuntimeError(f"a backend loaded before the core blocker: {loaded}")
    root = Path(__file__).resolve().parents[1]
    resolved = []
    for value in arguments or ["tests"]:
        name, separator, node = value.partition("::")
        if not value.startswith("-") and (root / name).exists():
            value = str((root / name).resolve()) + (separator + node if separator else "")
        resolved.append(value)
    blocker = BackendBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        import pytest
        return pytest.main([*resolved, "-m", CORE_MARKERS])
    finally:
        sys.meta_path.remove(blocker)


if __name__ == "__main__":
    raise SystemExit(main())
