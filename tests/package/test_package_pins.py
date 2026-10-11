"""Dependency pins at the sole packaging source match the pinned parity environment."""
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_project_pins_match_the_paper_parity_runtime():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    manifest = json.loads((ROOT / "tests/fixtures/paper_parity/manifest.json").read_text())
    versions = manifest["scenes"]["p1_optical_matched"]["lanes"]["reference"]["versions"]
    requirements = [value for group in project["optional-dependencies"].values() for value in group]
    pinned = {}
    for value in requirements:
        match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\[[^]]+\])?==(.+)", value)
        if match:
            pinned[match.group(1).lower()] = match.group(2)
    for name in ("numpy", "scipy", "astropy", "numba", "jax", "jaxlib", "autoarray", "autogalaxy", "autofit", "autoconf"):
        assert pinned[name] == versions[name]
    for name, expected in (("nautilus-sampler", "1.0.5"), ("scikit-learn", "1.8.0"),
                           ("threadpoolctl", "3.6.0"), ("jax-cuda12-plugin", versions["jax"]),
                           ("jax-cuda12-pjrt", versions["jax"])):
        assert pinned[name] == expected
    for extra, name in (("lensing", "autolens"), ("optics", "hcipy")):
        value = next(value for value in project["optional-dependencies"][extra] if value.startswith(name + " @"))
        assert re.search(r"git\+https://.+@[0-9a-f]{40}$", value), value
    assert next(value for value in requirements if value.startswith("hcipy @")).endswith(
        "@cc853b392463c33f02db6d20ce16dce0f7d10e2e")
    assert next(value for value in requirements if value.startswith("autolens @")).endswith(
        "@10bfea51ea95fb02087147190da562e186535d7f")
    assert project["scripts"] == {"hwoslaps": "hwoslaps.cli:main"}
    assert not (ROOT / "setup.py").exists() and not (ROOT / "setup.cfg").exists()


def test_validated_constraints_and_cuda_runtime_match_the_packaging_recipe():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    constraints = dict(line.strip().lower().split("==", 1) for line in
                       (ROOT / "constraints/validated.txt").read_text().splitlines()
                       if line.strip() and not line.startswith("#"))
    for requirement in project["optional-dependencies"]["validated"]:
        name, version = requirement.lower().split("==", 1)
        assert constraints[name] == version
    assert "jax-cuda12-plugin[with_cuda]==0.4.38" in project["optional-dependencies"]["cuda12"]
