"""The scientific dependency graph and the installed wheel's real command boundary."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
GLUE = {"scene.image_profile", "scene.multipole_profile", "scene.halo_profiles", "fisher.engines.jax_profiles",
        "fisher.engines.jax_templates", "inference.backend", "inference.subhalo_classes",
        "inference.light_profiles", "inference.mass_profiles"}
BLOCKER = '''
import importlib.abc, importlib.machinery, sys
forbidden = {'autolens','autogalaxy','autoarray','autofit','autoconf','hcipy','jax','jaxlib','nautilus','numba','matplotlib'}
class Block(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'autolens','autogalaxy','autoarray','autofit','autoconf','hcipy','jax','jaxlib','nautilus','numba','matplotlib'}:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise ModuleNotFoundError('blocked scientific backend: ' + module.__name__, name=module.__name__)
sys.meta_path.insert(0, Block())
'''
NO_BACKENDS = "\nassert not any(name.split('.')[0] in forbidden for name in sys.modules)"


def modules():
    found = []
    for path in (SOURCE / "hwoslaps").rglob("*.py"):
        parts = list(path.relative_to(SOURCE).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found.append((path, ".".join(parts)))
    return found


@pytest.mark.parametrize("module", [name for _, name in modules()
                                    if name.partition("hwoslaps.")[2] not in GLUE and not name.endswith(".__main__")])
def test_modules_import_without_backends(module):
    program = BLOCKER + "\nimport importlib; importlib.import_module(" + repr(module) + ")" + NO_BACKENDS
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, timeout=30)


def test_no_private_cross_module_or_study_imports():
    for path, name in modules():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package.split(".")[:len(package.split(".")) - node.level + 1]
                    target = ".".join(base + ([] if node.module is None else node.module.split(".")))
                else:
                    target = node.module or ""
                assert target != "studies" and not target.startswith("studies."), str(path)
                if target.startswith("hwoslaps"):
                    assert all(not part.startswith("_") or target == "hwoslaps._version"
                               for part in target.split(".")[1:]), str(path)
                    assert all(not alias.name.startswith("_") or alias.name == "__version__" for alias in node.names), str(path)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "studies" and not alias.name.startswith("studies."), str(path)
                    if alias.name.startswith("hwoslaps."):
                        assert all(not part.startswith("_") or alias.name == "hwoslaps._version"
                                   for part in alias.name.split(".")[1:]), str(path)


def test_wheel_contains_priors_console_script_and_backend_free_command(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(SOURCE, project / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, project / filename)
    wheels = tmp_path / "wheels"
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", "--wheel-dir",
                    str(wheels), str(project)], check=True, capture_output=True, timeout=120)
    wheel, = wheels.glob("hwoslaps-*.whl")
    installed = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.startswith(("studies/", "scratch/")) for name in archive.namelist())
        archive.extractall(installed)
    info, = installed.glob("hwoslaps-*.dist-info")
    assert "hwoslaps = hwoslaps.cli:main" in (info / "entry_points.txt").read_text()
    assert (installed / "hwoslaps/optics/priors/jwst_wss_static_v1.yaml").read_bytes() == (
        SOURCE / "hwoslaps/optics/priors/jwst_wss_static_v1.yaml").read_bytes()
    assert (installed / "hwoslaps/optics/priors/jwst_wss_drift_v1.yaml").read_bytes() == (
        SOURCE / "hwoslaps/optics/priors/jwst_wss_drift_v1.yaml").read_bytes()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    environment = dict(os.environ, PYTHONPATH=str(installed))
    config = ROOT / "configs/minimal.yaml"
    origin = "\nimport hwoslaps; from pathlib import Path; assert Path(hwoslaps.__file__).resolve().is_relative_to(Path(" + repr(str(installed)) + ").resolve()), hwoslaps.__file__"
    program = BLOCKER + origin + "\nfrom hwoslaps.cli import main; result=main(['validate', " + repr(str(config)) + "])" + NO_BACKENDS + "; raise SystemExit(result)"
    completed = subprocess.run([sys.executable, "-c", program], cwd=foreign, env=environment,
                               check=True, capture_output=True, text=True, timeout=30)
    assert "valid, digest" in completed.stdout
    version = subprocess.run([sys.executable, "-c", BLOCKER + origin + "\nprint(hwoslaps.__version__)"],
                             cwd=foreign, env=environment, check=True, capture_output=True, text=True, timeout=30)
    assert version.stdout.strip() == "1.0.0"


@pytest.mark.backend
@pytest.mark.parametrize("package", ["hwoslaps", "hwoslaps.config", "hwoslaps.scene", "hwoslaps.optics",
                                     "hwoslaps.observation", "hwoslaps.fisher", "hwoslaps.inference", "hwoslaps.analysis"])
def test_exports_never_shadow_child_modules_in_either_import_order(package):
    program = '''
import importlib, json, pkgutil, sys, types
package = importlib.import_module(sys.argv[1])
children = [name for _, name, _ in pkgutil.iter_modules(package.__path__)]
exports = list(package.__all__)
assert not set(children) & set(exports), (children, exports)
if sys.argv[2] == 'children-first':
    for child in children:
        importlib.import_module(package.__name__ + '.' + child)
values = {}
for name in exports:
    value = getattr(package, name)
    assert not isinstance(value, types.ModuleType), name
    values[name] = (type(value).__module__, type(value).__qualname__,
                    getattr(value, '__module__', None), getattr(value, '__qualname__', None))
if sys.argv[2] == 'exports-first':
    for child in children:
        importlib.import_module(package.__name__ + '.' + child)
    for name in exports:
        value = getattr(package, name)
        assert not isinstance(value, types.ModuleType), name
        assert values[name] == (type(value).__module__, type(value).__qualname__,
                                getattr(value, '__module__', None), getattr(value, '__qualname__', None)), name
print(json.dumps(values, sort_keys=True))
'''
    snapshots = []
    for order in ("children-first", "exports-first"):
        completed = subprocess.run([sys.executable, "-c", program, package, order], check=True,
                                   capture_output=True, text=True, timeout=60)
        snapshots.append(completed.stdout.strip().splitlines()[-1])
    assert snapshots[0] == snapshots[1]


def test_library_does_not_print_or_read_engine_behavior_from_the_environment():
    environment_owners = {"hwoslaps.provenance", "hwoslaps.fisher.engines.reference", "hwoslaps.inference.backend",
                          "hwoslaps.batch.runner"}
    for path, name in modules():
        if name in ("hwoslaps.cli", "hwoslaps.__main__"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", str(path)
            if name not in environment_owners:
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                    assert node.attr not in ("environ", "getenv", "putenv", "unsetenv"), str(path)
                if isinstance(node, ast.Name):
                    assert node.id not in ("getenv", "putenv", "unsetenv"), str(path)


def layer(name):
    relative = name.removeprefix("hwoslaps.")
    if relative in ("constants", "identity", "seeding", "provenance", "_version", "config.checks"):
        return 0
    if relative == "config.loading" or relative == "spectra" or relative.startswith("spectra."):
        return 1
    if relative in ("instrument", "scene", "optics") or relative.startswith(("scene.", "optics.")):
        return 2
    if relative == "observation" or relative.startswith("observation."):
        return 3
    if relative == "fisher" or relative.startswith("fisher.") or relative in ("config.schema", "simulation"):
        return 4
    if relative == "inference" or relative.startswith("inference."):
        return 5
    if relative in ("analysis", "population", "artifacts") or relative.startswith(("analysis.", "population.")):
        return 6
    return 7


def test_package_imports_follow_layer_direction():
    for path, name in modules():
        if path.name == "__init__.py":
            continue
        package = name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("hwoslaps."):
                        assert layer(alias.name) <= layer(name), (name, alias.name)
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = package.split(".")[:len(package.split(".")) - node.level + 1]
                target = ".".join(base + ([] if node.module is None else node.module.split(".")))
            else:
                target = node.module or ""
            if target.startswith("hwoslaps.") and not target.endswith(".__init__"):
                assert layer(target) <= layer(name), (name, target)


def test_class_names_are_unique_across_the_package():
    locations = {}
    for path, _ in modules():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef):
                assert node.name not in locations, (node.name, locations.get(node.name), path)
                locations[node.name] = path
