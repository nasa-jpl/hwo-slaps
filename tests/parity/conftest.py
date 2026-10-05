"""Paper-side oracles and reusable final-API preparations for this lane."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "paper_parity"
_NAME_MAP = {
    "lens.centre_y": "lens.mass.mass.centre_y", "lens.centre_x": "lens.mass.mass.centre_x",
    "lens.einstein_radius": "lens.mass.mass.einstein_radius", "lens.ell_comp_1": "lens.mass.mass.ell_comp_1",
    "lens.ell_comp_2": "lens.mass.mass.ell_comp_2", "source.centre_y": "source.light.light.centre_y",
    "source.centre_x": "source.light.light.centre_x", "source.ell_comp_1": "source.light.light.ell_comp_1",
    "source.ell_comp_2": "source.light.light.ell_comp_2", "source.intensity": "source.light.light.intensity",
    "source.effective_radius": "source.light.light.effective_radius",
    "observation.background_offset_adu": "observation.background_offset_adu",
    "psf.segment_hexikes[0][1]": "psf.segment_hexikes[0][1]", "psf.segment_hexikes[0][2]": "psf.segment_hexikes[0][2]",
    "psf.segment_hexikes[3][1]": "psf.segment_hexikes[3][1]", "psf.segment_hexikes[3][2]": "psf.segment_hexikes[3][2]",
    "psf.global_zernikes[4]": "psf.zernikes[4]", "psf.global_zernikes[5]": "psf.zernikes[5]",
}


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
def final_name():
    image = dict(_NAME_MAP, **{"source.intensity": "source.light.light.flux_scale",
                              "source.effective_radius": "source.light.light.size_scale"})
    return {"p1_optical_matched": _NAME_MAP, "p2_delta_knowledge_error": _NAME_MAP,
            "p3_image_source_kernel": image, "p4_subhalo_sis": _NAME_MAP, "p4_subhalo_pointmass": _NAME_MAP}


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
