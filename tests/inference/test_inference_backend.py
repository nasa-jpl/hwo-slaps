"""Backend glue: process state of one search scope, the session pool, the analysis class, JAX x64."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import multiprocessing
import os
import subprocess
import sys
import threading
import textwrap

import numpy as np
import pytest

from hwoslaps.inference.fit_model import autofit_model

pytestmark = pytest.mark.backend

VARIABLE = "PYAUTO_SKIP_VISUALIZATION"
ABSENT = object()


def _process_state():
    from autoconf import conf
    from nautilus.neural import NeuralNetworkEmulator

    output = conf.instance["output"]
    return (os.environ.get(VARIABLE, ABSENT), output["search_internal"] if "search_internal" in output else ABSENT,
            NeuralNetworkEmulator.__dict__["train"])


def _set_prior_state(monkeypatch, variable, key):
    from autoconf import conf

    if variable is ABSENT:
        monkeypatch.delenv(VARIABLE, raising=False)
    else:
        monkeypatch.setenv(VARIABLE, variable)
    if key is ABSENT:
        monkeypatch.delitem(conf.instance["output"], "search_internal", raising=False)
    else:
        monkeypatch.setitem(conf.instance["output"], "search_internal", key)


def _record_process_pool_maps(monkeypatch):
    """Item counts of every ``map`` call on a real process pool (the session's), delegated unchanged."""
    from multiprocessing.pool import Pool

    counts = []
    pool_map = Pool.map

    def recording_map(self, function, iterable, *args, **kwargs):
        items = list(iterable)
        counts.append(len(items))
        return pool_map(self, function, items, *args, **kwargs)

    monkeypatch.setattr(Pool, "map", recording_map)
    return counts


@pytest.mark.parametrize(("variable", "key", "retain", "workers", "cores", "raises"), [
    ("0", False, True, 1, 1, False),
    ("0", False, True, 1, 1, True),
    (ABSENT, False, False, 1, 1, False),
    ("0", ABSENT, True, 1, 1, False),
    ("0", False, True, 2, 1, False),
    ("0", False, True, 2, 2, False),
], ids=["normal-exit", "exception-inside", "variable-absent", "autoconf-key-absent", "pooled-training",
        "several-sampler-cores"])
def test_search_scope_restores_process_state(monkeypatch, variable, key, retain, workers, cores, raises):
    from hwoslaps.inference.backend import BackendSession

    _set_prior_state(monkeypatch, variable, key)
    before = _process_state()
    seen = []
    with BackendSession(training_workers=workers) as session:
        try:
            with session.search_scope(retain_search_internal=retain, number_of_cores=cores) as training_workers:
                seen = [*_process_state(), training_workers]
                if raises:
                    raise ZeroDivisionError("inside the scope")
        except ZeroDivisionError:
            assert raises
    pooled = workers > 1 and cores == 1
    assert seen[0] == "1"
    assert seen[1] == (True if retain else before[1])
    assert (seen[2] is not before[2]) is pooled
    assert seen[3] == (workers if pooled else 1)
    after = _process_state()
    assert after[:2] == before[:2] and after[2] is before[2]


def test_session_pool_outlives_scopes_and_scopes_are_exclusive(monkeypatch):
    """CP2-06: one pool across searches and exceptions; one search scope at a time per process."""
    from hwoslaps.inference.backend import BackendSession

    _set_prior_state(monkeypatch, "0", False)
    before_state = _process_state()
    outside = {child.pid for child in multiprocessing.active_children()}
    with BackendSession(training_workers=2) as session:
        pool = {child.pid for child in multiprocessing.active_children()} - outside
        assert len(pool) == 2
        for _ in range(2):
            with session.search_scope(retain_search_internal=True, number_of_cores=1):
                pass
            assert {child.pid for child in multiprocessing.active_children()} - outside == pool
        with pytest.raises(KeyError):
            with session.search_scope(retain_search_internal=True, number_of_cores=1):
                raise KeyError("a failed search")
        assert session.active and {child.pid for child in multiprocessing.active_children()} - outside == pool
        with session.search_scope(retain_search_internal=False, number_of_cores=1):
            inside = _process_state()
            with pytest.raises(RuntimeError, match="another search scope"):
                with BackendSession() as other, other.search_scope(retain_search_internal=True, number_of_cores=1):
                    pass
            errors = []

            def concurrent():
                try:
                    with session.search_scope(retain_search_internal=True, number_of_cores=1):
                        pass
                except RuntimeError as error:
                    errors.append(str(error))

            thread = threading.Thread(target=concurrent)
            thread.start()
            thread.join()
            assert errors and "another search scope" in errors[0]
            assert _process_state()[:2] == inside[:2] and _process_state()[2] is inside[2]
    assert not session.active
    assert {child.pid for child in multiprocessing.active_children()} - outside == set()
    with pytest.raises(RuntimeError, match="entered BackendSession"):
        with session.search_scope(retain_search_internal=False, number_of_cores=1):
            pass
    assert _process_state()[:2] == before_state[:2]


@pytest.mark.parametrize("caller_pool", [False, True], ids=["session-pool", "caller-pool"])
def test_pooled_scope_trains_through_the_session_pool_unless_the_caller_gives_one(monkeypatch, caller_pool):
    """Nautilus passes ``pool=None`` with one sampler core; inside a pooled scope its networks then
    train through the session's pool, and a pool the caller passes is kept."""
    from nautilus.neural import NeuralNetworkEmulator
    from hwoslaps.inference.backend import BackendSession

    session_maps, caller_maps = _record_process_pool_maps(monkeypatch), []

    class CallerPool:
        def map(self, function, iterable):
            items = list(iterable)
            caller_maps.append(len(items))
            return list(map(function, items))

    rng = np.random.default_rng(3)
    with BackendSession(training_workers=2) as session:
        with session.search_scope(retain_search_internal=False, number_of_cores=1):
            NeuralNetworkEmulator.train(rng.random((50, 3)), rng.random(50), n_networks=2,
                                        pool=CallerPool() if caller_pool else None)
    assert (session_maps, caller_maps) == (([], [2]) if caller_pool else ([2], []))


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2"], ids=str)
def test_session_refuses_invalid_training_worker_counts(workers):
    from hwoslaps.inference.backend import BackendSession

    with pytest.raises(ValueError, match="training_workers must be an integer >= 1"):
        BackendSession(training_workers=workers)


def test_make_analysis_uses_the_given_cosmology_and_writes_no_autolens_class(raw_imaging, light_model):
    import autogalaxy as ag
    import autolens as al
    from hwoslaps.inference.backend import ANALYSIS_CLASS, make_analysis

    before = dict(vars(al.AnalysisImaging))
    cosmology = ag.cosmo.Planck15()
    analysis = make_analysis(raw_imaging, cosmology=cosmology, use_jax=False)
    instance = autofit_model(light_model).instance_from_vector(vector=list(light_model.truth))
    fit = analysis.fit_from(instance=instance)
    assert isinstance(analysis, ANALYSIS_CLASS) and analysis._use_jax is False
    assert analysis.cosmology is cosmology
    assert np.isfinite(fit.log_likelihood)
    assert dict(vars(al.AnalysisImaging)).keys() == before.keys()
    assert all(vars(al.AnalysisImaging)[name] is value for name, value in before.items())


def _backend_binding_args():
    path = Path(__file__).resolve().parents[2] / "src/hwoslaps/inference/backend.py"
    return str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest()


X64_CHILD = textwrap.dedent("""
    import json, sys
    import numpy as np
    import jax
    import autogalaxy as ag
    from autofit.non_linear.fitness import Fitness
    from hwoslaps.inference import backend
    make_analysis = backend.make_analysis
    from hwoslaps.inference.fit_model import autofit_model
    sys.path.insert(0, sys.argv[1])
    from conftest import make_light_model, make_raw_imaging
    import hashlib
    from pathlib import Path
    loaded = Path(backend.__file__).resolve()
    binding = {"backend_path": str(loaded), "backend_sha256": hashlib.sha256(loaded.read_bytes()).hexdigest()}
    assert binding == {"backend_path": str(Path(sys.argv[2]).resolve()), "backend_sha256": sys.argv[3]}
    print(json.dumps({"binding": binding}), flush=True)
    kernel = np.exp(-(np.mgrid[-3:4, -3:4].astype(float) ** 2).sum(axis=0) / 2.0)
    fit_model = make_light_model()
    before = bool(jax.config.jax_enable_x64)
    lower, upper = np.asarray(fit_model.lower), np.asarray(fit_model.upper)
    vectors = lower + np.random.default_rng(20261005).random((4, lower.size)) * (upper - lower)
    reports = []
    for iteration in (1, 2):
        analysis = make_analysis(make_raw_imaging(kernel / kernel.sum()), cosmology=ag.cosmo.Planck15(), use_jax=True)
        fitness = Fitness(model=autofit_model(fit_model), analysis=analysis, paths=None, fom_is_log_likelihood=True,
                          resample_figure_of_merit=-1.0e99, use_jax_vmap=True, batch_size=4)
        values = np.asarray(fitness.call_wrap(vectors))
        report = {"after": bool(jax.config.jax_enable_x64), "dtype": str(values.dtype),
                  "finite": bool(np.all(np.isfinite(values)))}
        reports.append(report)
        print(json.dumps({"iteration": iteration, "before": before, **report}), flush=True)
    json.dump({"before": before, "analyses": reports, "binding": binding}, sys.stdout)
""")


@pytest.mark.xtx_gpu
def test_jax_analysis_enables_x64_and_returns_float64():
    """A fresh process from x64 off: both real analyses keep x64 on and return finite float64 Fitness."""
    environment = {name: value for name, value in os.environ.items() if name != "JAX_ENABLE_X64"}
    lane = os.path.dirname(os.path.abspath(__file__))
    expected_path, expected_sha = _backend_binding_args()
    completed = subprocess.run([sys.executable, "-c", X64_CHILD, lane, expected_path, expected_sha], env=environment, capture_output=True,
                               text=True, check=True, timeout=600)
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report.pop("binding") == {"backend_path": expected_path, "backend_sha256": expected_sha}
    assert report == {"before": False, "analyses": [
        {"after": True, "dtype": "float64", "finite": True},
        {"after": True, "dtype": "float64", "finite": True},
    ]}


GUARD_CHILD = textwrap.dedent("""
    import json, sys
    import numpy as np
    import jax
    import autofit as af
    import autolens as al
    import autogalaxy as ag
    from autofit.non_linear import fitness as fitness_module
    from hwoslaps.inference import backend
    sys.path.insert(0, sys.argv[1])
    from conftest import make_raw_imaging
    mode = sys.argv[2]
    import hashlib
    from pathlib import Path
    loaded = Path(backend.__file__).resolve()
    binding = {"backend_path": str(loaded), "backend_sha256": hashlib.sha256(loaded.read_bytes()).hexdigest()}
    assert binding == {"backend_path": str(Path(sys.argv[3]).resolve()), "backend_sha256": sys.argv[4]}
    original_update = jax.config.update
    original_fitness = fitness_module.Fitness
    original_instance = af.Model.instance_from_vector
    original_analysis = backend.ANALYSIS_CLASS
    constructions = []
    jax.config.update("jax_enable_x64", False)
    before = bool(jax.config.jax_enable_x64)

    class ObservedAnalysis(original_analysis):
        def __init__(self, *args, **kwargs):
            constructions.append(True)
            super().__init__(*args, **kwargs)

    class DelegatingFitness(original_fitness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def delegating_instance(self, *args, **kwargs):
        return original_instance(self, *args, **kwargs)

    def ineffective_update(name, value, *args, **kwargs):
        if name == "jax_enable_x64":
            return None
        return original_update(name, value, *args, **kwargs)

    backend.ANALYSIS_CLASS = ObservedAnalysis
    if mode == "ineffective_enable":
        jax.config.update = ineffective_update
    elif mode == "missing_apis":
        fitness_module.Fitness = DelegatingFitness
        af.Model.instance_from_vector = delegating_instance
    else:
        raise ValueError(mode)
    error_type, message = None, None
    try:
        kernel = np.exp(-(np.mgrid[-3:4, -3:4].astype(float) ** 2).sum(axis=0) / 2.0)
        analysis = backend.make_analysis(make_raw_imaging(kernel / kernel.sum()),
                                         cosmology=ag.cosmo.Planck15(), use_jax=True)
    except RuntimeError as error:
        error_type, message = type(error).__name__, str(error)
    finally:
        flag = bool(jax.config.jax_enable_x64)
        jax.config.update = original_update
        fitness_module.Fitness = original_fitness
        af.Model.instance_from_vector = original_instance
        backend.ANALYSIS_CLASS = original_analysis
        jax.config.update("jax_enable_x64", before)
    json.dump({"binding": binding, "before": before, "after": flag, "error_type": error_type, "error": message,
               "analysis_constructions": len(constructions),
               "versions": {"autofit": str(af.__version__), "autolens": str(al.__version__)}}, sys.stdout)
""")


@pytest.mark.xtx_gpu
@pytest.mark.parametrize("fault", ["ineffective_enable", "missing_apis"])
def test_jax_analysis_refuses_failed_enablement_and_missing_installed_apis(fault):
    """Real installed guards refuse before any delegating real analysis construction."""
    environment = {name: value for name, value in os.environ.items() if name != "JAX_ENABLE_X64"}
    lane = os.path.dirname(os.path.abspath(__file__))
    expected_path, expected_sha = _backend_binding_args()
    completed = subprocess.run([sys.executable, "-c", GUARD_CHILD, lane, fault, expected_path, expected_sha], env=environment,
                               capture_output=True, text=True, check=True, timeout=90)
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report.pop("binding") == {"backend_path": expected_path, "backend_sha256": expected_sha}
    assert report["before"] is False and report["analysis_constructions"] == 0
    assert report["error_type"] == "RuntimeError"
    if fault == "ineffective_enable":
        assert report["after"] is False
        assert report["error"] == "JAX 64-bit mode could not be enabled"
    else:
        assert report["after"] is True
        for name in ("Fitness.use_jax_vmap", "Fitness.batch_size", "Model.instance_from_vector(xp=...)"):
            assert name in report["error"]
        for package, version in report["versions"].items():
            assert version and f"{package} {version}" in report["error"]


@pytest.mark.xtx_gpu
def test_pooled_emulator_training_equals_serial(monkeypatch):
    """Networks trained through the session pool are byte-equal to serial training."""
    from nautilus.neural import NeuralNetworkEmulator
    from hwoslaps.inference.backend import BackendSession

    rng = np.random.default_rng(3)
    x, y = rng.random((200, 6)), rng.random(200)
    serial = NeuralNetworkEmulator.train(x, y, n_networks=2)
    pool_maps = _record_process_pool_maps(monkeypatch)
    with BackendSession(training_workers=2) as session:
        with session.search_scope(retain_search_internal=False, number_of_cores=1) as workers:
            assert workers == 2
            pooled = NeuralNetworkEmulator.train(x, y, n_networks=2)
    assert pool_maps == [2]

    def weights(emulator):
        return [array for network in emulator.neural_networks for array in [*network.coefs_, *network.intercepts_]]

    assert len(weights(serial)) == len(weights(pooled))
    assert all(np.array_equal(a, b) for a, b in zip(weights(serial), weights(pooled)))
    assert np.array_equal(serial.predict(x), pooled.predict(x))
