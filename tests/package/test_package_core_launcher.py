"""The real core launcher allows dependency discovery and refuses actual backend loads."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("probe,expected", [
    ("from astropy.cosmology import Planck15; assert Planck15.H0.value > 0", 0),
    ("import importlib.util; assert importlib.util.find_spec('matplotlib') is not None", 0),
    ("import matplotlib", 1),
    ("import importlib; importlib.import_module('jax')", 1),
])
def test_core_launcher_separates_discovery_from_loading(tmp_path, probe, expected):
    suite = tmp_path / "test_core_probe.py"
    suite.write_text("def test_probe():\n    " + probe + "\n")
    completed = subprocess.run([sys.executable, str(ROOT / "tools/run_core_tests.py"), str(suite), "-q"],
                               capture_output=True, text=True, timeout=60)
    assert completed.returncode == expected, completed.stdout + completed.stderr
    if expected:
        assert "forbids backend import" in completed.stdout + completed.stderr
