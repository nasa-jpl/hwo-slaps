"""Keep paper reproduction contracts outside the installed engine."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest


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


def test_campaign_import_needs_only_installed_source(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import hwoslaps.campaign as c; "
         "assert callable(c.validate_campaign_manifest); "
         "assert not any(n == 'studies' or n.startswith('studies.') for n in sys.modules)"],
        cwd=tmp_path, env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("command", [
    "generate_stage0_campaign.py", "generate_ladder_campaign.py",
])
def test_study_commands_bootstrap_the_checkout_from_any_directory(tmp_path, command):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "studies/rasti/scripts" / command), "--help"],
        cwd=tmp_path, env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_source_only_study_namespaces_import_without_a_science_backend(tmp_path):
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import studies.rasti.campaign; import studies.rasti.scripts; "
         "assert 'autolens' not in sys.modules and 'jax' not in sys.modules"],
        cwd=tmp_path, env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
