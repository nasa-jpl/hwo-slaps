"""Fresh-process import integrity across lightweight test and package owners."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('helper_first', [True, False])
def test_physics_fixture_import_preserves_real_lazy_public_packages(helper_first):
    """A fixture import must neither shadow the package nor load science backends."""
    script = r'''
import importlib
import importlib.abc
import sys

class ForbidOptionalBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'astropy', 'autolens', 'autogalaxy', 'autoarray', 'hcipy', 'matplotlib'}:
            raise AssertionError('lightweight import loaded optional backend: ' + fullname)

sys.meta_path.insert(0, ForbidOptionalBackends())
sys.path.insert(0, sys.argv[1])
if sys.argv[2] == 'True':
    import _lensing_physics_helpers

import hwoslaps
from hwoslaps import prepare_forecast, sample_population
from hwoslaps.config import validate_or_raise
from hwoslaps.lensing.utils import get_einstein_ring_position
from hwoslaps.observation import detector_moments
for name in ('hwoslaps', 'hwoslaps.config', 'hwoslaps.lensing', 'hwoslaps.modeling', 'hwoslaps.observation', 'hwoslaps.plotting'):
    module = importlib.import_module(name)
    assert module.__file__ and module.__spec__.origin == module.__file__, name

import _lensing_physics_helpers
assert callable(prepare_forecast) and callable(validate_or_raise)
assert sample_population({'x': {'distribution': 'constant', 'value': 3}}, 1, seed=7) == [{'x': 3}]
assert get_einstein_ring_position(0, 2) == (0, 2)
assert detector_moments([[0]], 1, {'gain': 1, 'read_noise': 2, 'sky_background': 0, 'dark_current': 0}).variance_e2[0, 0] == 4
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(Path(__file__).resolve().parent), str(helper_first)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
