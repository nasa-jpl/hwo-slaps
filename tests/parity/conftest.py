"""Paper-side oracles and reusable final-API preparations for this lane."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity"


@pytest.fixture(scope="session")
def manifest():
    return json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def paper_digest():
    def array(values):
        value = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        return hashlib.sha256(repr(value.shape).encode() + value.tobytes()).hexdigest()

    def kernel(values):
        value = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        prefix = f"{value.shape[0]}x{value.shape[1]}:".encode()
        return hashlib.sha256(prefix + value.tobytes()).hexdigest()

    return SimpleNamespace(array=array, kernel=kernel)


@pytest.fixture(scope="session")
def final_name(manifest):
    def mapped(name):
        if name.startswith("lens."):
            return "lens.mass.mass." + name.removeprefix("lens.")
        if name.startswith("source."):
            return "source.light.light." + name.removeprefix("source.")
        return name.replace("psf.global_zernikes", "psf.zernikes")

    return {scene: {name: mapped(name) for name in entry.get("profiled_nuisance_names", ())}
            for scene, entry in manifest["scenes"].items()}


@pytest.fixture(scope="module")
def prepared():
    from hwoslaps.fisher.api import Execution, prepare_forecast

    cache = {}
    def get(scene, engine):
        if (scene, engine) not in cache:
            path = FIXTURES / "engine" / f"{scene}.yaml"
            cache[scene, engine] = prepare_forecast(path, execution=Execution(engine=engine))
        return cache[scene, engine]
    yield get
    for value in cache.values():
        value.close()
