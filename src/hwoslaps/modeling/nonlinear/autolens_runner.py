"""Runtime wrapper for PyAutoLens nonlinear validation fits."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    iterations_per_update : `int`, optional
        PyAutoFit update cadence.
    maxcall : `int`, optional
        Maximum likelihood calls if supported by the search backend.
    path_prefix : `str`, optional
        PyAutoFit output path prefix.
    unique_tag : `str`, optional
        Optional PyAutoFit unique tag.
    resume : `bool`, optional
        Whether PyAutoFit should resume existing searches when supported.
    use_jax : `bool`, optional
        Whether to request PyAutoLens JAX execution.
    """

    engine: str = "Nautilus"
    n_live_smooth: int = 100
    n_live_subhalo_fixed: int = 100
    n_live_subhalo_search: int = 200
    number_of_cores: int = 1
    iterations_per_update: Optional[int] = None
    maxcall: Optional[int] = None
    path_prefix: str = "nonlinear"
    unique_tag: Optional[str] = None
    resume: bool = True
    use_jax: bool = False


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


def _filter_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter keyword arguments to those accepted by a callable."""
    signature = inspect.signature(callable_obj)
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: val for key, val in kwargs.items() if val is not None}
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
    """Run PyAutoLens validation fits and summarize their outputs."""

    def __init__(self, settings: NonlinearSearchSettings, output_dir: str):
        self.settings = settings
        self.output_dir = str(output_dir)

    def make_analysis(self, dataset: Any) -> Any:
        """Create a PyAutoLens imaging analysis object.

        Parameters
        ----------
        dataset : `autolens.Imaging`
            Dataset to fit.

        Returns
        -------
        analysis : `autolens.AnalysisImaging`
            PyAutoLens analysis object.
        """
        import autolens as al

        _patch_analysis_imaging_adapt_images_compat(al)
        try:
            return al.AnalysisImaging(dataset=dataset, use_jax=self.settings.use_jax)
        except TypeError:
            return al.AnalysisImaging(dataset=dataset)

    def _make_search(self, case_id: str, role: str, n_live: int) -> Any:
        """Create a PyAutoFit search object."""
        if self.settings.engine != "Nautilus":
            raise ValueError("Only the Nautilus search engine is currently supported")

        import autofit as af

        name = f"{case_id}_{role}"
        kwargs = {
            "path_prefix": str(Path(self.output_dir) / self.settings.path_prefix),
            "name": name,
            "unique_tag": self.settings.unique_tag,
            "n_live": int(n_live),
            "number_of_cores": int(self.settings.number_of_cores),
            "iterations_per_update": self.settings.iterations_per_update,
            "n_like_max": self.settings.maxcall,
            "resume": self.settings.resume,
        }
        return af.Nautilus(**_filter_kwargs(af.Nautilus, kwargs))

    def run_model(
        self,
        model: Any,
        analysis: Any,
        role: str,
        fit_mode: str,
        case_id: str,
        n_live: int,
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

        Returns
        -------
        summary : `NonlinearFitSummary`
            Fit summary.
        """
        start = time.time()
        try:
            search = self._make_search(case_id=case_id, role=role, n_live=n_live)
            result = search.fit(model=model, analysis=analysis)
            log_likelihood, method = extract_max_log_likelihood_with_method(result)
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
                log_likelihood_extraction_method=method,
                use_jax_requested=self.settings.use_jax,
                search_engine=self.settings.engine,
                n_live=n_live,
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
            )
