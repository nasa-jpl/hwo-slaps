"""Keep study imports outside the reusable scientific dependency graph."""

from __future__ import annotations

import ast
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]


def test_engine_has_no_study_imports():
    for path in (ROOT / "src/hwoslaps").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            assert not any(name == "studies" or name.startswith("studies.")
                           for name in modules), str(path)
