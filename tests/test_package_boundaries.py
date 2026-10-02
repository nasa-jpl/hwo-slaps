"""Built distribution and real public entry-point isolation."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_built_wheel_runs_public_validate_command_without_optional_backends(tmp_path):
    """Use wheel metadata and its real command outside the source checkout."""
    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(
        ROOT / "src", project / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    for name in ("setup.py", "pyproject.toml"):
        shutil.copy2(ROOT / name, project / name)
    distribution = tmp_path / "wheel"
    built = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-index",
         str(project), "--wheel-dir", str(distribution)],
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert built.returncode == 0, built.stderr
    wheel, = distribution.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.startswith(("studies/", "scratch/")) for name in archive.namelist())
    config = yaml.safe_load((ROOT / "configs/master_config.yaml").read_text())
    config.pop("run_name", None)
    config.pop("plotting", None)
    config.pop("modeling", None)
    path = tmp_path / "science.yaml"
    path.write_text(yaml.safe_dump(config))
    code = f'''
import importlib.abc
import importlib.metadata
import sys
class BlockBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'autolens', 'autogalaxy', 'autoarray', 'autofit', 'hcipy', 'jax', 'studies'}}:
            raise ImportError('optional backend import forbidden: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
import hwoslaps
assert str(hwoslaps.__file__).startswith({str(wheel)!r}), hwoslaps.__file__
from hwoslaps import simulate, prepare_forecast, forecast
from hwoslaps.modeling.forecast_results import ForecastResult
entries = list(importlib.metadata.distributions(path=[{str(wheel)!r}]))
entry, = [e for d in entries for e in d.entry_points if e.group == 'console_scripts' and e.name == 'hwoslaps']
sys.argv = ['hwoslaps', 'validate', '-c', {str(path)!r}]
assert entry.load()() == 0
'''
    environment = dict(os.environ, PYTHONPATH=str(wheel))
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=tmp_path,
        env=environment, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Configuration valid" in result.stdout
