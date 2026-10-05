"""Every interaction of inference with process state of AutoLens, AutoFit, autoconf, Nautilus and JAX.

Backend glue: this module imports AutoLens when imported, so other modules import it inside
functions only. No AutoLens class attribute is ever written. The process-global effects are:
JAX x64 (enabled for good by ``ensure_jax_x64``), and, for the duration of one search inside
``BackendSession.search_scope`` and restored afterwards, ``PYAUTO_SKIP_VISUALIZATION``, the
autoconf ``output.search_internal`` value and the Nautilus emulator-training replacement.
"""

from __future__ import annotations

import inspect
import multiprocessing
import os
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import autolens as al
from autolens.imaging.fit_imaging import FitImaging

__all__ = ["ANALYSIS_CLASS", "BackendSession", "CompatAnalysisImaging", "ensure_jax_x64", "make_analysis",
           "require_jax_fitness_api"]

_VISUALIZATION_VARIABLE = "PYAUTO_SKIP_VISUALIZATION"
_SCOPE_LOCK = threading.Lock()
_FIT_IMAGING_KEYWORDS = frozenset(inspect.signature(FitImaging).parameters)


class CompatAnalysisImaging(al.AnalysisImaging):
    """``AnalysisImaging`` whose ``fit_from`` matches the installed AutoGalaxy adapt-image signature.

    The pinned PyAutoLens passes ``dataset_model`` and ``xp`` to
    ``adapt_images_via_instance_from``, which the pinned AutoGalaxy does not accept; this
    subclass makes the same fit with the keywords both sides accept.
    """

    def fit_from(self, instance: Any) -> Any:
        if self._use_jax:
            self._register_fit_imaging_pytrees()
        tracer = self.tracer_via_instance_from(instance=instance)
        dataset_model = self.dataset_model_via_instance_from(instance=instance)
        adapt_images = self.adapt_images_via_instance_from(instance=instance, galaxies=tracer.galaxies)
        keywords = {"dataset": self.dataset, "tracer": tracer, "dataset_model": dataset_model,
                    "adapt_images": adapt_images, "settings": self.settings, "xp": self._xp}
        return FitImaging(**{name: value for name, value in keywords.items()
                             if name in _FIT_IMAGING_KEYWORDS and value is not None})


ANALYSIS_CLASS: type = (
    al.AnalysisImaging
    if "dataset_model" in inspect.signature(al.AnalysisImaging.adapt_images_via_instance_from).parameters
    else CompatAnalysisImaging
)
"""The analysis class fits use: ``CompatAnalysisImaging`` while the installed AutoGalaxy lacks the
``dataset_model`` keyword, ``al.AnalysisImaging`` once it has it."""


def ensure_jax_x64() -> None:
    """Enable JAX 64-bit mode for the process and check that it took effect."""
    import jax

    jax.config.update("jax_enable_x64", True)
    if not jax.config.jax_enable_x64:
        raise RuntimeError("JAX 64-bit mode could not be enabled")


def require_jax_fitness_api() -> None:
    """Require the AutoFit traced-vector API the JAX likelihood uses."""
    import autofit as af
    from autofit.non_linear.fitness import Fitness

    missing = [name for name, present in (
        ("Fitness.use_jax_vmap", "use_jax_vmap" in inspect.signature(Fitness).parameters),
        ("Fitness.batch_size", "batch_size" in inspect.signature(Fitness).parameters),
        ("Model.instance_from_vector(xp=...)",
         "xp" in inspect.signature(af.Model.instance_from_vector).parameters),
    ) if not present]
    if missing:
        raise RuntimeError(f"the installed autofit {af.__version__} and autolens {al.__version__} "
                           f"lack the JAX likelihood API: {', '.join(missing)}")


def make_analysis(imaging: Any, *, cosmology: Any, use_jax: bool) -> Any:
    """The analysis of ``imaging`` under ``cosmology``, the instance the truth scene traced with.

    AutoLens falls back to Planck15 when no cosmology is given, which is wrong for any other
    cosmology, so the argument is required.
    """
    if use_jax:
        ensure_jax_x64()
        require_jax_fitness_api()
    analysis = ANALYSIS_CLASS(dataset=imaging, cosmology=cosmology, use_jax=use_jax)
    if use_jax and analysis._use_jax is not True:
        raise RuntimeError("the JAX likelihood was requested, but the analysis does not report _use_jax")
    return analysis


@contextmanager
def _skipped_visualization() -> Iterator[None]:
    previous = os.environ.get(_VISUALIZATION_VARIABLE)
    os.environ[_VISUALIZATION_VARIABLE] = "1"
    try:
        yield
    finally:
        if previous is None:
            del os.environ[_VISUALIZATION_VARIABLE]
        else:
            os.environ[_VISUALIZATION_VARIABLE] = previous


@contextmanager
def _retained_search_internal() -> Iterator[None]:
    """AutoFit's post-fit cleanup keeps the raw sampler state while output.search_internal is True."""
    from autoconf import conf

    output = conf.instance["output"]
    existed = "search_internal" in output
    previous = output["search_internal"] if existed else None
    output["search_internal"] = True
    try:
        yield
    finally:
        if existed:
            output["search_internal"] = previous
        else:
            del output["search_internal"]


@contextmanager
def _pooled_training(pool: Any) -> Iterator[None]:
    """Route Nautilus emulator training through ``pool`` when the caller passes none.

    With one sampler core Nautilus trains every bound's networks with ``pool=None``; each
    network is a pure function of its arrays and index on one BLAS thread, so the trained
    weights do not depend on the pool.
    """
    from nautilus.neural import NeuralNetworkEmulator

    original = NeuralNetworkEmulator.__dict__["train"]
    train = original.__func__

    def train_in_pool(cls: type, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("pool") is None:
            kwargs["pool"] = pool
        return train(cls, *args, **kwargs)

    NeuralNetworkEmulator.train = classmethod(train_in_pool)
    try:
        yield
    finally:
        NeuralNetworkEmulator.train = original


class BackendSession:
    """The backend context of a run of searches: owns the emulator-training process pool.

    The pool (``training_workers`` spawned processes, none for one worker) lives from
    ``__enter__`` to ``__exit__`` across every search of the session. One search runs at a time
    per process: ``search_scope`` holds a process-wide lock.
    """

    def __init__(self, *, training_workers: int = 1) -> None:
        if isinstance(training_workers, bool) or not isinstance(training_workers, int) or training_workers < 1:
            raise ValueError(f"training_workers must be an integer >= 1, got {training_workers!r}")
        self._training_workers = training_workers
        self._pool: Any = None
        self._active = False

    @property
    def training_workers(self) -> int:
        return self._training_workers

    @property
    def active(self) -> bool:
        """True between ``__enter__`` and ``__exit__``."""
        return self._active

    def __enter__(self) -> BackendSession:
        if self._active:
            raise RuntimeError("this BackendSession is already entered")
        if self._training_workers > 1:
            self._pool = multiprocessing.get_context("spawn").Pool(self._training_workers)
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        pool, self._pool, self._active = self._pool, None, False
        if pool is None:
            return
        if exc_type is not None:
            pool.terminate()
            pool.join()
            return
        try:
            pool.close()
            pool.join()
        except BaseException:
            pool.terminate()
            pool.join()
            raise

    @contextmanager
    def search_scope(self, *, retain_search_internal: bool, number_of_cores: int) -> Iterator[int]:
        """The process state of exactly one ``search.fit``; yields the emulator-training worker count.

        Raises RuntimeError outside an entered session, and when another scope is active in this
        process (nested, or from another thread), before changing any state. Everything is
        restored in reverse order and the lock released last, whatever the search raises.
        """
        if not self._active:
            raise RuntimeError("search_scope needs an entered BackendSession")
        if not _SCOPE_LOCK.acquire(blocking=False):
            raise RuntimeError("another search scope is active in this process; one search runs at a time")
        try:
            with ExitStack() as stack:
                stack.enter_context(_skipped_visualization())
                if retain_search_internal:
                    stack.enter_context(_retained_search_internal())
                workers = 1
                if self._pool is not None and number_of_cores == 1:
                    stack.enter_context(_pooled_training(self._pool))
                    workers = self._training_workers
                yield workers
        finally:
            _SCOPE_LOCK.release()
