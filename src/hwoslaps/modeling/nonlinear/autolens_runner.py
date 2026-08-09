"""Runtime wrapper for PyAutoLens nonlinear validation fits."""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .output_schema import NonlinearFitSummary


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


def _metadata_value(metadata: Any, name: str) -> Any:
    """Return one field from dictionary- or attribute-style metadata."""
    if isinstance(metadata, dict):
        return metadata.get(name)
    return getattr(metadata, name, None)


def _native_array(value: Any) -> np.ndarray:
    """Return an object as a contiguous native float64 array."""
    if hasattr(value, "kernel"):
        value = value.kernel
    if hasattr(value, "native"):
        value = value.native
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


def _array_hash(value: Any) -> Optional[str]:
    """Return a SHA-256 digest for native float64 array bytes."""
    if value is None:
        return None
    array = _native_array(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


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
                "use_jax=True is unsupported for Item 7 custom profiles; "
                "this model requires CPU execution"
            )
        import autolens as al

        _patch_analysis_imaging_adapt_images_compat(al)
        try:
            return al.AnalysisImaging(dataset=dataset, use_jax=self.settings.use_jax)
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
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return af.Nautilus(**kwargs)

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
        try:
            search = self._make_search(
                case_id=case_id,
                role=role,
                n_live=n_live,
                analysis_key=analysis_key,
            )
            result = search.fit(model=model, analysis=analysis)
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
                search_engine=self.settings.engine,
                n_live=n_live,
                analysis_key=analysis_key,
                n_like_max_reached=_n_like_max_reached(
                    result,
                    self.settings.maxcall,
                ),
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
                search_engine=self.settings.engine,
                n_live=n_live,
                analysis_key=analysis_key,
            )
