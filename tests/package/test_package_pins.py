"""Dependency pins at the sole packaging source match the validated constraints recipe."""
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_project_pins_match_the_validated_packaging_recipe():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = [value for group in project["optional-dependencies"].values() for value in group]
    pinned = {}
    for value in requirements:
        match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\[[^]]+\])?==(.+)", value)
        if match:
            pinned[match.group(1).lower()] = match.group(2)
    for name in ("autolens", "autogalaxy", "autoarray", "autofit", "autonerves"):
        assert pinned[name] == "2026.11.20.1"
    for name in ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt"):
        assert pinned[name] == "0.12.1"
    assert not any(value.startswith("autolens @") for value in project["optional-dependencies"]["lensing"])
    hcipy_req = next(value for value in project["optional-dependencies"]["optics"] if value.startswith("hcipy @"))
    assert re.search(r"git\+https://.+@[0-9a-f]{40}$", hcipy_req), hcipy_req
    assert hcipy_req.endswith("@cc853b392463c33f02db6d20ce16dce0f7d10e2e")
    assert project["scripts"] == {"hwoslaps": "hwoslaps.cli:main"}
    assert not (ROOT / "setup.py").exists() and not (ROOT / "setup.cfg").exists()


def test_validated_constraints_and_cuda_runtime_match_the_packaging_recipe():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    constraints = dict(line.strip().lower().split("==", 1) for line in
                       (ROOT / "constraints/validated.txt").read_text().splitlines()
                       if line.strip() and not line.startswith("#"))
    for extra in ("validated", "lensing", "jax", "inference"):
        for requirement in project["optional-dependencies"][extra]:
            match = re.fullmatch(r"([A-Za-z0-9_-]+)(?:\[[^]]+\])?==(.+)", requirement)
            if match:
                assert constraints[match.group(1).lower()] == match.group(2)
    assert "jax-cuda12-plugin[with-cuda]==0.12.1" in project["optional-dependencies"]["cuda12"]
