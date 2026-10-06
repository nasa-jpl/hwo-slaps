"""Optional dependency and caller-owned output boundaries, without fake backends."""

import os
import subprocess
import sys

import pytest


@pytest.mark.backend
def test_plot_functions_leave_saving_and_console_output_to_the_caller(plt, forecast_product, area_product,
                                                                    observation_product, monkeypatch, capsys):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from hwoslaps.analysis.reductions import summarize
    from hwoslaps.plotting import (plot_detection_map, plot_kernel, plot_knowledge_error,
                                  plot_mass_curve, plot_observation, plot_pupil, plot_statistic_map)
    from hwoslaps.optics.pupils import build_pupil, parse_pupil
    from hwoslaps.optics.kernels import DetectorPSF

    def forbidden_save(*args, **kwargs):
        raise AssertionError("plot functions must leave saving to the caller")

    monkeypatch.setattr(Figure, "savefig", forbidden_save)
    pupil = build_pupil(parse_pupil({"kind": "circular", "diameter_m": 1., "pixels": 16}, "pupil"))
    kernel = DetectorPSF.from_array([[0., 1, 0], [1, 4, 1], [0, 1, 0]], .05)
    capsys.readouterr()
    artists = [plot_statistic_map(forecast_product, "q_asimov", mass_index=0),
               plot_detection_map(forecast_product, q_threshold=1, mass_index=0),
               plot_mass_curve(summarize(forecast_product, q_threshold=1), "q_max"),
               plot_knowledge_error(area_product), plot_kernel(kernel, log=False),
               plot_pupil(pupil), plot_observation(observation_product, "data")]
    assert all(isinstance(ax, Axes) for ax in artists)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_importing_plot_callables_does_not_require_graphics_or_fitting_backends():
    code = '''
import importlib.abc
import sys
banned = {'matplotlib', 'autolens', 'autoarray', 'autogalaxy', 'autofit', 'autoconf', 'hcipy', 'jax', 'jaxlib'}
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in banned:
            raise ImportError('unexpected plotting import: ' + fullname)
sys.meta_path.insert(0, Refuse())
from hwoslaps.plotting import plot_statistic_map, plot_detection_map, plot_mass_curve, plot_knowledge_error
from hwoslaps.plotting import plot_kernel, plot_pupil, plot_observation
assert callable(plot_statistic_map) and callable(plot_kernel) and callable(plot_observation)
assert not banned.intersection(name.split('.')[0] for name in sys.modules)
'''
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=True, timeout=30)
