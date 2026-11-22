"""Paper-side oracles and reusable final-API preparations for this lane."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dataclasses import replace

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
def paper_kernel(paper_digest):
    """Check a live kernel against the archived kernel with paper digest ``digest``.

    An optical kernel is compared with ``archived_kernels.npz`` (whole, or its central block
    for a 999x999 kernel) within KERNEL_TOLERANCE of the peak; a kernel read from a file must
    still match its digest exactly. Returns both digests for a receipt.
    """
    from current_oracles import KERNEL_TOLERANCE

    anchor = json.loads((FIXTURES / "k1_paper_kernel.json").read_text(encoding="utf-8"))
    oracle = json.loads((FIXTURES / "production" / "oracle.json").read_text(encoding="utf-8"))
    with np.load(FIXTURES / "archived_kernels.npz", allow_pickle=False) as archive:
        stored = {name: archive[name] for name in archive.files}
    archived = {paper_digest.kernel(stored[name]): (stored[name], stored[name].shape)
                for name in ("fisher_optical_17x17", "fisher_knowledge_error_17x17", "k1_51x51")}
    archived[anchor["kernels"]["999x999"]["sha256"]] = (stored["k1_999x999_core"], (999, 999))
    archived[oracle["truth_kernel_sha256"]] = (stored["production_999x999_core"], (999, 999))

    def check(values, digest):
        current = paper_digest.kernel(values)
        if digest not in archived:
            assert current == digest
        else:
            expected, shape = archived[digest]
            live = np.asarray(values, dtype=float)
            assert live.shape == shape
            (y, x), (ny, nx) = ((n - m) // 2 for n, m in zip(shape, expected.shape)), expected.shape
            np.testing.assert_allclose(live[y:y + ny, x:x + nx], expected, rtol=0.0,
                                       atol=KERNEL_TOLERANCE * float(np.max(expected)))
        return {"historical_digest": digest, "current_digest": current}

    return check


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
            if engine == "reference":
                path = FIXTURES / "engine" / f"{scene}.yaml"
                cache[scene, engine] = prepare_forecast(path, execution=Execution(engine=engine))
            else:
                # Compare actual execution engines on one checked physical
                # preparation, avoiding duplicate costly singular light integrals.
                from hwoslaps.fisher.engines.base import make_engine
                reference = get(scene, "reference")
                execution = Execution(engine=engine)
                actual = make_engine(engine, reference.engine.context, execution)
                cache[scene, engine] = replace(reference, engine=actual, execution=execution)
        return cache[scene, engine]
    yield get
    for value in cache.values():
        value.close()


@pytest.fixture(scope="module")
def current_forecast_oracle(prepared):
    from current_oracles import forecast_oracle
    cache = {}
    def get(scene, masses, positions):
        key = (scene, tuple(masses), tuple(map(tuple, positions)))
        if key not in cache:
            cache[key] = forecast_oracle(prepared(scene, "reference"), masses, positions)
        return cache[key]
    return get
