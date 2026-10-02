"""Dependency boundaries for lightweight nonlinear configuration imports."""

import subprocess
import sys


def test_package_and_profile_settings_do_not_import_execution_backends():
    code = """
import sys
import hwoslaps.modeling.nonlinear as nonlinear
from hwoslaps.modeling.nonlinear.profile_settings import FreshProfileSettings
assert nonlinear.FreshProfileSettings is FreshProfileSettings
assert FreshProfileSettings().maxiter == 500
for name in ("autolens", "autofit", "autoarray", "jax", "matplotlib"):
    assert not any(item == name or item.startswith(name + ".") for item in sys.modules), name
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
