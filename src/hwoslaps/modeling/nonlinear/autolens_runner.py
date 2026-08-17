"""Runtime wrapper for PyAutoLens nonlinear validation fits."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .output_schema import NonlinearFitSummary

_VISUALIZATION_ENV = "PYAUTO_SKIP_VISUALIZATION"


@dataclass(frozen=True)
class NonlinearSearchSettings:
    """Settings for PyAutoLens/PyAutoFit validation searches.

    Parameters
    ----------
    engine : `str`, optional
        Search engine name. The initial supported value is ``"Nautilus"``.
    n_live_smooth : `int`, optional
        Live points for smooth-model fits.
    n_live_subhalo_fixed : `int`, optional
        Live points for fixed-template subhalo fits.
    n_live_subhalo_search : `int`, optional
        Live points reserved for future local-search fits.
    number_of_cores : `int`, optional
        PyAutoFit process count.
    iterations_per_quick_update : `int`, optional
        PyAutoFit quick-update cadence.
    iterations_per_full_update : `int`, optional
        PyAutoFit full-update cadence.
    maxcall : `int`, optional
        Maximum likelihood calls if supported by the search backend.
    seed : `int`, optional
        Nautilus RNG seed for reproducible searches.
    path_prefix : `str`, optional
        PyAutoFit output path prefix.
    unique_tag : `str`, optional
        Optional PyAutoFit unique tag.
    use_jax : `bool`, optional
        Whether to request PyAutoLens JAX execution.
    jax_n_batch : `int`, optional
        Vectorized AutoFit likelihood batch size for JAX execution only.
    disable_visualization : `bool`, optional
        Whether to disable AutoFit in-search visualization by setting
        ``PYAUTO_SKIP_VISUALIZATION=1`` around the search; ``False``
        sets ``0`` so plots run regardless of the ambient environment.
        The prior value is restored after every fit.
    n_eff : `float`, optional
        Minimum effective posterior sample size before the sampler
        stops. None delegates to the installed backend default; the
        effective value is recorded in the fit summary.
    n_shell : `int`, optional
        Minimum number of points in the sampler shell before stopping.
        None delegates to the installed backend default; the effective
        value is recorded in the fit summary.
    discard_exploration : `bool`, optional
        Whether the sampler discards exploration-phase points when
        estimating the posterior and evidence. None delegates to the
        installed backend default; the effective value is recorded in
        the fit summary.
    retain_search_internal : `bool`, optional
        Whether to keep the raw Nautilus search-internal state on disk
        after the fit instead of letting AutoFit post-fit cleanup
        remove it. Required for evidence-convergence audit cells.

    Notes
    -----
    AutoFit resumes completed or interrupted searches from their output path;
    no explicit resume setting is required.
    """

    engine: str = "Nautilus"
    n_live_smooth: int = 100
    n_live_subhalo_fixed: int = 100
    n_live_subhalo_search: int = 200
    number_of_cores: int = 1
    iterations_per_quick_update: Optional[int] = None
    iterations_per_full_update: Optional[int] = None
    maxcall: Optional[int] = None
    seed: Optional[int] = None
    path_prefix: str = "nonlinear"
    unique_tag: Optional[str] = None
    use_jax: bool = False
    jax_n_batch: int = 100
    disable_visualization: bool = True
    n_eff: Optional[float] = None
    n_shell: Optional[int] = None
    discard_exploration: Optional[bool] = None
    retain_search_internal: bool = False

    def __post_init__(self) -> None:
        """Validate execution settings before analysis or search setup."""
        if (
            isinstance(self.jax_n_batch, bool)
            or not isinstance(self.jax_n_batch, int)
            or self.jax_n_batch <= 0
        ):
            raise ValueError("jax_n_batch must be a positive integer")
        if not isinstance(self.disable_visualization, bool):
            raise ValueError("disable_visualization must be a boolean")
        if self.n_eff is not None and (
            isinstance(self.n_eff, bool)
            or not isinstance(self.n_eff, (int, float))
            or not np.isfinite(self.n_eff)
            or self.n_eff <= 0
        ):
            raise ValueError("n_eff must be None or a positive finite number")
        if self.n_shell is not None and (
            isinstance(self.n_shell, bool)
            or not isinstance(self.n_shell, int)
            or self.n_shell <= 0
        ):
            raise ValueError("n_shell must be None or a positive integer")
        if self.discard_exploration is not None and not isinstance(
            self.discard_exploration,
            bool,
        ):
            raise ValueError("discard_exploration must be None or a boolean")
        if not isinstance(self.retain_search_internal, bool):
            raise ValueError("retain_search_internal must be a boolean")


def ensure_jax_x64() -> None:
    """Import JAX, enable x64, and verify the effective precision mode."""
    try:
        import jax
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "JAX likelihood was requested, but JAX could not be imported"
        ) from exc

    try:
        jax.config.update("jax_enable_x64", True)
        enabled = bool(jax.config.jax_enable_x64)
    except Exception as exc:
        raise RuntimeError(
            "JAX likelihood was requested, but 64-bit mode could not be enabled"
        ) from exc
    if not enabled:
        raise RuntimeError(
            "JAX likelihood was requested, but 64-bit mode is not enabled"
        )


def _installed_version(distribution: str) -> str:
    """Return installed distribution version text for an error message."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _accepts_keyword(callable_obj: Any, name: str) -> bool:
    """Return whether a callable explicitly exposes one keyword."""
    try:
        return name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def ensure_target_jax_backend() -> None:
    """Require the installed AutoFit traced-vector Fitness API seam."""
    try:
        import autofit as af
        from autofit.non_linear.fitness import Fitness
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "JAX likelihood was requested, but AutoFit could not be imported"
        ) from exc

    missing = []
    if not _accepts_keyword(Fitness, "use_jax_vmap"):
        missing.append("Fitness.use_jax_vmap")
    if not _accepts_keyword(Fitness, "batch_size"):
        missing.append("Fitness.batch_size")
    instance_from_vector = getattr(af.Model, "instance_from_vector", None)
    if instance_from_vector is None or not _accepts_keyword(
        instance_from_vector,
        "xp",
    ):
        missing.append("Model.instance_from_vector(xp=...)")
    if missing:
        raise RuntimeError(
            "Unsupported installed JAX backend: "
            f"autofit={_installed_version('autofit')}, "
            f"autolens={_installed_version('autolens')}; "
            "missing API seam: " + ", ".join(missing)
        )


def _metadata_value(metadata: Any, name: str) -> Any:
    """Return one field from dictionary- or attribute-style metadata."""
    if isinstance(metadata, dict):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _native_array(value: Any) -> np.ndarray:
    """Return an object as a contiguous native array."""
    if hasattr(value, "kernel"):
        value = value.kernel
    if hasattr(value, "native"):
        value = value.native
    return np.ascontiguousarray(np.asarray(value))


def _array_hash(value: Any) -> Optional[str]:
    """Return a dtype- and shape-aware SHA-256 for native array bytes."""
    if value is None:
        return None
    array = _native_array(value)
    shape = "x".join(str(size) for size in array.shape)
    prefix = f"{array.dtype}:{shape}:".encode("utf-8")
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def _mask_hash(value: Any) -> Optional[str]:
    """Return the hash of an array's boolean mask, if it carries one."""
    return _array_hash(getattr(value, "mask", None))


def _pixel_scales(value: Any) -> Optional[list]:
    """Return the physical pixel scales bound to an array-like object."""
    if value is None:
        return None
    if hasattr(value, "kernel"):
        value = value.kernel
    scales = getattr(value, "pixel_scales", None)
    if scales is None:
        return None
    return [float(scale) for scale in np.atleast_1d(scales)]


def analysis_key_from(
    dataset: Any,
    dataset_metadata: Any,
    model_metadata: Dict[str, Any],
) -> str:
    """Return a deterministic 16-hex analysis identity key.

    AutoFit model hashes distinguish priors but not the fitted dataset. This
    identity prevents a completed fit from being silently reused for a
    different PSF-bank draw, noise realization, or model context.

    Parameters
    ----------
    dataset : `object`
        PyAutoLens imaging dataset.
    dataset_metadata : `object`
        Dataset provenance fields.
    model_metadata : `dict`
        Resolved fit mode, custom-context hashes, and prior widths.

    Returns
    -------
    analysis_key : `str`
        First 16 hexadecimal characters of the canonical SHA-256.
    """
    prior_widths = model_metadata.get(
        "resolved_prior_widths",
        model_metadata.get("prior_widths", {}),
    )
    prior_repr = json.dumps(
        prior_widths,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    psf = getattr(dataset, "psf", None)
    payload = {
        "dataset_kind": _metadata_value(dataset_metadata, "dataset_kind"),
        "background_treatment": _metadata_value(
            dataset_metadata,
            "background_treatment",
        ),
        "psf_truth_label": _metadata_value(
            dataset_metadata,
            "psf_truth_label",
        ),
        "psf_fit_label": _metadata_value(
            dataset_metadata,
            "psf_fit_label",
        ),
        "data_sha256": _array_hash(getattr(dataset, "data", None)),
        "noise_map_sha256": _array_hash(
            getattr(dataset, "noise_map", None)
        ),
        "psf_sha256": _array_hash(psf),
        "data_mask_sha256": _mask_hash(getattr(dataset, "data", None)),
        "noise_map_mask_sha256": _mask_hash(
            getattr(dataset, "noise_map", None)
        ),
        "data_pixel_scales": _pixel_scales(getattr(dataset, "data", None)),
        "noise_map_pixel_scales": _pixel_scales(
            getattr(dataset, "noise_map", None)
        ),
        "psf_pixel_scales": _pixel_scales(psf),
        "fit_mode": model_metadata.get("fit_mode"),
        "clumpy_fit_parameterization": model_metadata.get(
            "clumpy_fit_parameterization"
        ),
        "mass_context_hash": model_metadata.get("mass_context_hash"),
        "image_source_asset_hash": model_metadata.get(
            "image_source_asset_hash"
        ),
        "prior_widths_sha256": hashlib.sha256(
            prior_repr.encode("utf-8")
        ).hexdigest(),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _get_nested_attr(obj: Any, attr_path: str) -> Any:
    """Get a nested attribute path from an object."""
    value = obj
    for attr in attr_path.split("."):
        value = getattr(value, attr)
    return value


def _coerce_log_likelihood(value: Any) -> float:
    """Extract a float log likelihood from a result-like object."""
    if hasattr(value, "log_likelihood"):
        value = value.log_likelihood
    return float(value)


def extract_max_log_likelihood_with_method(result: Any) -> Tuple[float, str]:
    """Extract maximum log likelihood and record the accessor used.

    Parameters
    ----------
    result : `object`
        PyAutoFit result object.

    Returns
    -------
    log_likelihood : `float`
        Maximum log likelihood.
    method : `str`
        Accessor path used to retrieve the value.

    Raises
    ------
    AttributeError
        Raised when no known accessor is available.
    """
    accessors = (
        "samples.max_log_likelihood",
        "max_log_likelihood_fit.figure_of_merit",
        "samples.max_log_likelihood_sample",
        "best_samples.max_log_likelihood",
    )
    errors = []
    for accessor in accessors:
        try:
            value = _get_nested_attr(result, accessor)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    value = value(as_instance=False)
            return _coerce_log_likelihood(value), accessor
        except Exception as exc:
            errors.append(f"{accessor}: {exc}")

    raise AttributeError(
        "Could not extract max log likelihood from PyAutoFit result. "
        + " | ".join(errors)
    )


def extract_max_log_likelihood(result: Any) -> float:
    """Extract maximum log likelihood from a PyAutoFit result.

    Parameters
    ----------
    result : `object`
        PyAutoFit result object.

    Returns
    -------
    log_likelihood : `float`
        Maximum log likelihood.
    """
    log_likelihood, _ = extract_max_log_likelihood_with_method(result)
    return log_likelihood


def _extract_log_evidence(result: Any) -> Optional[float]:
    """Extract log evidence if present on a result object."""
    accessors = (
        "samples.log_evidence",
        "log_evidence",
    )
    for accessor in accessors:
        try:
            return float(_get_nested_attr(result, accessor))
        except Exception:
            continue
    return None


def _extract_result_path(result: Any) -> Optional[str]:
    """Extract a result path if present."""
    accessors = (
        "paths.output_path",
        "search.paths.output_path",
    )
    for accessor in accessors:
        try:
            return str(_get_nested_attr(result, accessor))
        except Exception:
            continue
    return None


def _model_parameter_count(model: Any) -> Optional[int]:
    """Return a PyAutoFit model parameter count if available."""
    for attr in ("total_free_parameters", "prior_count", "total_free_parameter_count"):
        if hasattr(model, attr):
            try:
                return int(getattr(model, attr))
            except TypeError:
                try:
                    return int(getattr(model, attr)())
                except Exception:
                    continue
    return None


def _n_like_max_reached(
    result: Any,
    n_like_max: Optional[int],
) -> Optional[bool]:
    """Return whether a configured likelihood-call ceiling was reached."""
    if n_like_max is None:
        return None
    samples_info = getattr(getattr(result, "samples", None), "samples_info", None)
    if not isinstance(samples_info, dict) or "total_samples" not in samples_info:
        return None
    try:
        total_samples = float(samples_info["total_samples"])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(total_samples):
        return None
    return total_samples >= float(n_like_max)


def _search_setting(search: Any, name: str) -> Any:
    """Return one sampler setting from a constructed search object."""
    if hasattr(search, name):
        return getattr(search, name)
    kwargs = getattr(search, "kwargs", None)
    if isinstance(kwargs, dict):
        return kwargs.get(name)
    return None


def _effective_sampler_settings(
    search: Any,
) -> Tuple[Optional[float], Optional[int], Optional[bool]]:
    """Return effective convergence settings from a constructed search.

    Nautilus stores its ``n_eff``, ``n_shell``, and
    ``discard_exploration`` keyword arguments as attributes, applying
    backend defaults for keywords that were not passed; unrecognized
    keywords survive on ``search.kwargs``. Undiscoverable or
    unexpectedly typed values are recorded as None.

    Parameters
    ----------
    search : `object`
        Constructed PyAutoFit search object.

    Returns
    -------
    n_eff : `float`, optional
        Effective minimum effective sample size.
    n_shell : `int`, optional
        Effective minimum shell-point count.
    discard_exploration : `bool`, optional
        Effective exploration-phase discard flag.
    """
    n_eff = _search_setting(search, "n_eff")
    n_shell = _search_setting(search, "n_shell")
    discard_exploration = _search_setting(search, "discard_exploration")
    if isinstance(n_eff, bool) or not isinstance(n_eff, (int, float)):
        n_eff = None
    else:
        n_eff = float(n_eff)
    if isinstance(n_shell, bool) or not isinstance(n_shell, int):
        n_shell = None
    if not isinstance(discard_exploration, bool):
        discard_exploration = None
    return n_eff, n_shell, discard_exploration


def _apply_search_internal_retention() -> Tuple[bool, Any]:
    """Enable AutoFit search-internal retention and return prior state.

    AutoFit post-fit cleanup removes the raw sampler state directory
    when the autoconf ``output.search_internal`` setting is false.
    Overriding the cached setting to True keeps the raw Nautilus
    live/dead-point history on disk after the fit.

    Returns
    -------
    existed : `bool`
        Whether the setting existed before the override.
    previous : `object`
        Prior setting value, or None when the setting did not exist.
    """
    from autoconf import conf

    output_config = conf.instance["output"]
    try:
        previous = output_config["search_internal"]
        existed = True
    except KeyError:
        previous = None
        existed = False
    output_config["search_internal"] = True
    return existed, previous


def _restore_search_internal_retention(state: Tuple[bool, Any]) -> None:
    """Restore the autoconf ``output.search_internal`` setting exactly.

    Parameters
    ----------
    state : `tuple`
        Prior state returned by ``_apply_search_internal_retention``.
    """
    from autoconf import conf

    existed, previous = state
    output_config = conf.instance["output"]
    if existed:
        output_config["search_internal"] = previous
    else:
        del output_config["search_internal"]


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter keyword arguments to those accepted by a callable."""
    signature = inspect.signature(callable_obj)
    return {
        key: val
        for key, val in kwargs.items()
        if key in signature.parameters and val is not None
    }


def _patch_analysis_imaging_adapt_images_compat(al: Any) -> None:
    """Patch PyAutoLens fit_from for installed AutoGalaxy adapt-image API.

    PyAutoLens 2026.5.14.2 can call ``adapt_images_via_instance_from`` with a
    ``dataset_model`` keyword, while the paired AutoGalaxy method in this
    environment accepts only ``instance`` and ``galaxies``. The mismatch occurs
    before any science likelihood is evaluated, including on the JAX path.
    """
    analysis_cls = al.AnalysisImaging
    if getattr(analysis_cls, "_hwoslaps_adapt_images_compat", False):
        return
    try:
        signature = inspect.signature(analysis_cls.adapt_images_via_instance_from)
    except (AttributeError, ValueError, TypeError):
        return
    if "dataset_model" in signature.parameters:
        return

    def fit_from_compat(self, instance):
        if getattr(self, "_use_jax", False):
            register_pytrees = getattr(self, "_register_fit_imaging_pytrees", None)
            if register_pytrees is not None:
                register_pytrees()

        tracer = self.tracer_via_instance_from(instance=instance)
        dataset_model = self.dataset_model_via_instance_from(instance=instance)
        adapt_images = self.adapt_images_via_instance_from(
            instance=instance,
            galaxies=tracer.galaxies,
        )

        from autolens.imaging.fit_imaging import FitImaging

        kwargs = _filter_kwargs(
            FitImaging,
            {
                "dataset": self.dataset,
                "tracer": tracer,
                "dataset_model": dataset_model,
                "adapt_images": adapt_images,
                "settings": getattr(self, "settings", None),
                "xp": getattr(self, "_xp", None),
            },
        )
        return FitImaging(**kwargs)

    analysis_cls.fit_from = fit_from_compat
    analysis_cls._hwoslaps_adapt_images_compat = True


class AutoLensFitRunner:
    """Run PyAutoLens validation fits and summarize their outputs.

    Parameters
    ----------
    settings : `NonlinearSearchSettings`
        Search-engine and execution settings.
    output_dir : `str`
        Root directory for AutoFit outputs.
    """

    def __init__(self, settings: NonlinearSearchSettings, output_dir: str):
        if settings.use_jax and settings.number_of_cores != 1:
            raise ValueError(
                "JAX likelihood requires number_of_cores=1; AutoFit "
                "vectorizes parameter batches and ignores the process count; "
                "parallelize cases via CUDA_VISIBLE_DEVICES"
            )
        self.settings = settings
        self.output_dir = str(output_dir)

    def make_analysis(
        self,
        dataset: Any,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Create a PyAutoLens imaging analysis object.

        Parameters
        ----------
        dataset : `autolens.Imaging`
            Dataset to fit.
        model_metadata : `dict`, optional
            Model provenance used to enforce CPU-only custom profiles.

        Returns
        -------
        analysis : `autolens.AnalysisImaging`
            PyAutoLens analysis object.
        """
        if (
            self.settings.use_jax
            and model_metadata
            and model_metadata.get("requires_cpu")
        ):
            raise ValueError(
                "JAX likelihood cannot run a model marked "
                "requires_cpu=True; this model requires CPU execution"
            )
        if self.settings.use_jax:
            ensure_jax_x64()
            ensure_target_jax_backend()

        import autolens as al

        _patch_analysis_imaging_adapt_images_compat(al)
        if self.settings.use_jax:
            try:
                analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)
            except TypeError as exc:
                raise RuntimeError(
                    "JAX likelihood was requested, but AutoLens could not "
                    "construct AnalysisImaging with use_jax=True"
                ) from exc
            if getattr(analysis, "_use_jax", None) is not True:
                raise RuntimeError(
                    "JAX likelihood was requested, but constructed AutoLens "
                    "analysis does not report _use_jax is True"
                )
            return analysis

        try:
            return al.AnalysisImaging(dataset=dataset, use_jax=False)
        except TypeError:
            return al.AnalysisImaging(dataset=dataset)

    def _make_search(
        self,
        case_id: str,
        role: str,
        n_live: int,
        analysis_key: str,
    ) -> Any:
        """Create a PyAutoFit search object."""
        if self.settings.engine != "Nautilus":
            raise ValueError("Only the Nautilus search engine is currently supported")

        import autofit as af

        name = f"{case_id}_{role}_{analysis_key}"
        if self.settings.use_jax:
            name += f"_jax_vmap_b{self.settings.jax_n_batch}"
        kwargs = {
            "path_prefix": str(Path(self.output_dir) / self.settings.path_prefix),
            "name": name,
            "unique_tag": self.settings.unique_tag,
            "n_live": int(n_live),
            "number_of_cores": int(self.settings.number_of_cores),
            "iterations_per_quick_update": (
                self.settings.iterations_per_quick_update
            ),
            "iterations_per_full_update": (
                self.settings.iterations_per_full_update
            ),
            "n_like_max": self.settings.maxcall,
            "seed": self.settings.seed,
            "n_eff": self.settings.n_eff,
            "n_shell": self.settings.n_shell,
            "discard_exploration": self.settings.discard_exploration,
        }
        if self.settings.use_jax:
            kwargs.update(
                {
                    "n_batch": int(self.settings.jax_n_batch),
                    "use_jax_vmap": True,
                }
            )
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        search = af.Nautilus(**kwargs)
        if self.settings.use_jax:
            problems = []
            if getattr(search, "n_batch", None) != self.settings.jax_n_batch:
                problems.append("n_batch")
            if getattr(search, "use_jax_vmap", None) is not True:
                problems.append("use_jax_vmap")
            if problems:
                raise RuntimeError(
                    "Nautilus does not expose the requested effective JAX "
                    "execution seam: "
                    f"autofit={_installed_version('autofit')}, "
                    f"autolens={_installed_version('autolens')}; "
                    "missing effective state: " + ", ".join(problems)
                )
        return search

    def _effective_jax_provenance(
        self,
        analysis: Any,
        search: Any,
    ) -> Tuple[Optional[bool], Optional[int]]:
        """Derive effective execution state from constructed objects."""
        if not self.settings.use_jax:
            return None, None

        problems = []
        if getattr(analysis, "_use_jax", None) is not True:
            problems.append("analysis._use_jax is not True")
        if getattr(search, "use_jax_vmap", None) is not True:
            problems.append("search.use_jax_vmap is not True")
        effective_batch = getattr(search, "n_batch", None)
        if effective_batch != self.settings.jax_n_batch:
            problems.append(
                "search.n_batch does not match requested jax_n_batch"
            )
        if problems:
            raise RuntimeError(
                "Requested JAX execution disagrees with effective JAX "
                "execution: " + "; ".join(problems)
            )
        return True, int(effective_batch)

    def run_model(
        self,
        model: Any,
        analysis: Any,
        role: str,
        fit_mode: str,
        case_id: str,
        n_live: int,
        analysis_key: str,
        result_callback: Optional[Callable] = None,
    ) -> NonlinearFitSummary:
        """Run one nonlinear model fit.

        Parameters
        ----------
        model : `object`
            PyAutoFit model.
        analysis : `object`
            PyAutoLens analysis object.
        role : `str`
            Model role, such as ``"smooth"`` or ``"subhalo"``.
        fit_mode : `str`
            Validation mode.
        case_id : `str`
            Validation case identifier.
        n_live : `int`
            Number of live points requested for the search.
        analysis_key : `str`
            Dataset-and-model identity embedded in the search path.
        result_callback : `callable`, optional
            Callback receiving ``(result, model)`` before result disposal.

        Returns
        -------
        summary : `NonlinearFitSummary`
            Fit summary.
        """
        start = time.time()
        use_jax_effective = None
        jax_n_batch_effective = None
        n_eff_effective = None
        n_shell_effective = None
        discard_exploration_effective = None
        retention_applied = False
        try:
            search = self._make_search(
                case_id=case_id,
                role=role,
                n_live=n_live,
                analysis_key=analysis_key,
            )
            use_jax_effective, jax_n_batch_effective = (
                self._effective_jax_provenance(analysis, search)
            )
            n_eff_effective, n_shell_effective, discard_exploration_effective = (
                _effective_sampler_settings(search)
            )
            saved_visualization = os.environ.get(_VISUALIZATION_ENV)
            os.environ[_VISUALIZATION_ENV] = (
                "1" if self.settings.disable_visualization else "0"
            )
            try:
                if self.settings.retain_search_internal:
                    retention_state = _apply_search_internal_retention()
                    retention_applied = True
                try:
                    result = search.fit(model=model, analysis=analysis)
                finally:
                    if retention_applied:
                        _restore_search_internal_retention(retention_state)
            finally:
                if saved_visualization is None:
                    os.environ.pop(_VISUALIZATION_ENV, None)
                else:
                    os.environ[_VISUALIZATION_ENV] = saved_visualization
            log_likelihood, method = extract_max_log_likelihood_with_method(result)
            warnings = []
            if result_callback is not None:
                try:
                    result_callback(result, model)
                except Exception as exc:
                    warnings.append(f"result_callback failed: {exc}")
            runtime_s = time.time() - start
            return NonlinearFitSummary(
                model_role=role,
                fit_mode=fit_mode,
                status="success",
                log_likelihood_max=log_likelihood,
                figure_of_merit_max=log_likelihood,
                log_evidence=_extract_log_evidence(result),
                n_free_parameters=_model_parameter_count(model),
                result_path=_extract_result_path(result),
                runtime_s=runtime_s,
                warnings=warnings,
                log_likelihood_extraction_method=method,
                use_jax_requested=self.settings.use_jax,
                use_jax_effective=use_jax_effective,
                jax_n_batch_effective=jax_n_batch_effective,
                search_engine=self.settings.engine,
                n_live=n_live,
                analysis_key=analysis_key,
                n_like_max_reached=_n_like_max_reached(
                    result,
                    self.settings.maxcall,
                ),
                visualization_disabled=self.settings.disable_visualization,
                n_eff_requested=self.settings.n_eff,
                n_eff_effective=n_eff_effective,
                n_shell_requested=self.settings.n_shell,
                n_shell_effective=n_shell_effective,
                discard_exploration_requested=(
                    self.settings.discard_exploration
                ),
                discard_exploration_effective=discard_exploration_effective,
                search_internal_retained=retention_applied,
            )
        except Exception as exc:
            runtime_s = time.time() - start
            return NonlinearFitSummary(
                model_role=role,
                fit_mode=fit_mode,
                status="failed",
                n_free_parameters=_model_parameter_count(model),
                runtime_s=runtime_s,
                error=str(exc),
                use_jax_requested=self.settings.use_jax,
                use_jax_effective=use_jax_effective,
                jax_n_batch_effective=jax_n_batch_effective,
                search_engine=self.settings.engine,
                n_live=n_live,
                analysis_key=analysis_key,
                visualization_disabled=self.settings.disable_visualization,
                n_eff_requested=self.settings.n_eff,
                n_eff_effective=n_eff_effective,
                n_shell_requested=self.settings.n_shell,
                n_shell_effective=n_shell_effective,
                discard_exploration_requested=(
                    self.settings.discard_exploration
                ),
                discard_exploration_effective=discard_exploration_effective,
                search_internal_retained=retention_applied,
            )
