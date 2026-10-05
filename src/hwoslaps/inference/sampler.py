"""One Nautilus search of one role: construction, the settings read back, the fit, retained state.

The search name carries a digest of everything that determines the run (case, role, live
points, fit model, data identity, every sampler setting and the seed), because AutoFit's
Nautilus identifier leaves out ``f_live``, ``n_like_max``, ``discard_exploration``,
``number_of_cores`` and ``n_batch``: two runs differing only there would otherwise share an
output path and AutoFit would resume or reuse the other run. AutoFit is imported inside the
functions that need it.
"""

from __future__ import annotations

import hashlib
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ..identity import file_digest, mapping_digest
from .fit_model import FitModel
from .result import RetentionInventory, SamplerRecord
from .settings import OBJECTIVE_VERSION, SamplerSettings

if TYPE_CHECKING:
    from .backend import BackendSession

__all__ = ["REQUIRED_SEARCH_INTERNAL_FILES", "SearchOutcome", "effective_settings", "inspect_retention",
           "make_search", "run_search", "search_identity"]

REQUIRED_SEARCH_INTERNAL_FILES: tuple[str, ...] = ("search_internal.dill",)
"""Sampler state AutoFit must leave under ``files/search_internal``; the timer files beside it are not state."""

EFFECTIVE_FIELDS = ("n_live", "n_eff", "n_shell", "f_live", "discard_exploration", "n_like_max", "number_of_cores",
                    "seed", "n_batch", "use_jax_vmap")

_BLOCK_BYTES = 1 << 20


@dataclass(frozen=True)
class SearchOutcome:
    """A finished search: its record, the AutoFit result (None when ``search.fit`` raised) and the error."""

    record: SamplerRecord
    result: Any | None
    error: str | None


def search_identity(*, case_id: str, role: str, model: FitModel, data_identity: Mapping[str, Any],
                    settings: SamplerSettings, seed: int, n_live: int) -> str:
    """Digest of everything that determines a role search."""
    return mapping_digest({"case_id": case_id, "role": role, "n_live": n_live, "model": model.digest(),
                           "data": mapping_digest(data_identity), "sampler": settings.to_mapping(), "seed": seed,
                           "objective_version": OBJECTIVE_VERSION})


def _search_keywords(settings: SamplerSettings, *, n_live: int, seed: int) -> dict[str, Any]:
    keywords: dict[str, Any] = {"n_live": n_live, "number_of_cores": settings.number_of_cores, "seed": seed}
    for name in ("n_like_max", "n_eff", "n_shell", "f_live", "discard_exploration"):
        if getattr(settings, name) is not None:
            keywords[name] = getattr(settings, name)
    if settings.use_jax:
        keywords.update(n_batch=settings.jax_n_batch, use_jax_vmap=True)
    return keywords


def make_search(*, model: FitModel, role: Literal["smooth", "subhalo"], n_live: int, settings: SamplerSettings,
                seed: int, case_dir: Path, case_id: str, data_identity: Mapping[str, Any]) -> Any:
    """The ``af.Nautilus`` of one role search, named ``<role>_<identity[:16]>`` under ``case_dir``.

    Nothing here reads ``search.paths.output_path``: AutoFit fixes the identifier on first
    access, and before ``fit`` the model is unset.
    """
    identity = search_identity(case_id=case_id, role=role, model=model, data_identity=data_identity,
                               settings=settings, seed=seed, n_live=n_live)
    return _named_search(identity, role=role, n_live=n_live, settings=settings, seed=seed, case_dir=case_dir)


def _named_search(identity: str, *, role: str, n_live: int, settings: SamplerSettings, seed: int,
                  case_dir: Path) -> Any:
    import autofit as af

    if role not in ("smooth", "subhalo"):
        raise ValueError(f"role must be smooth or subhalo, got {role!r}")
    if isinstance(n_live, bool) or not isinstance(n_live, int) or n_live < 1:
        raise ValueError(f"n_live must be at least 1, got {n_live!r}")
    return af.Nautilus(path_prefix=str(case_dir), name=f"{role}_{identity[:16]}",
                       **_search_keywords(settings, n_live=n_live, seed=seed))


def effective_settings(search: Any) -> dict[str, Any]:
    """The sampler settings as the constructed search holds them, backend defaults included."""
    return {name: getattr(search, name) for name in EFFECTIVE_FIELDS}


def _check_effective(effective: Mapping[str, Any], requested: Mapping[str, Any], analysis: Any,
                     use_jax: bool) -> None:
    mismatched = [f"{name}: requested {value!r}, constructed {effective[name]!r}"
                  for name, value in requested.items()
                  if not (float(effective[name]) == float(value) if name == "n_eff" else effective[name] == value)]
    if use_jax and analysis._use_jax is not True:
        mismatched.append("analysis._use_jax is not True")
    if mismatched:
        raise RuntimeError("the constructed search does not hold the requested settings: " + "; ".join(mismatched))


def _member_digest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with archive.open(info) as stream:
        for block in iter(lambda: stream.read(_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_retention(output_path: Path, *, result_path: Path | None) -> RetentionInventory:
    """Inventory the raw sampler state of the search with output directory ``output_path``.

    The state is under ``files/search_internal`` of the directory, or under a
    ``search_internal`` member directory of ``<output_path>.zip`` when AutoFit zipped the
    output. It is bound to the result only when ``result_path`` (the result's output path) is
    the same directory. Only the required file with nonzero size counts as retained state.
    """
    output_path = Path(output_path)
    internal = output_path / "files" / "search_internal"
    archive_path = Path(f"{output_path}.zip")
    files: dict[str, dict[str, Any]] = {}
    route: Literal["directory", "zip"] | None = None
    if internal.is_dir():
        route = "directory"
        for item in sorted(internal.iterdir()):
            if item.is_file():
                files[item.name] = {"bytes": item.stat().st_size, "sha256": file_digest(item),
                                    "location": item.relative_to(output_path).as_posix()}
    elif archive_path.is_file():
        route = "zip"
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                parts = info.filename.split("/")
                if info.is_dir() or len(parts) < 2 or parts[-2] != "search_internal":
                    continue
                files[parts[-1]] = {"bytes": info.file_size, "sha256": _member_digest(archive, info),
                                    "location": info.filename}
    missing = tuple(name for name in REQUIRED_SEARCH_INTERNAL_FILES
                    if name not in files or files[name]["bytes"] <= 0)
    bound = result_path is not None and Path(result_path).resolve() == output_path.resolve()
    return RetentionInventory(route=route, files=files, missing_required=missing, bound_to_result_path=bound)


def run_search(*, fit_model: FitModel, model: Any, analysis: Any, role: Literal["smooth", "subhalo"], n_live: int,
               settings: SamplerSettings, seed: int, case_dir: Path, case_id: str,
               data_identity: Mapping[str, Any], session: BackendSession) -> SearchOutcome:
    """Construct, check and run one role search of ``model`` (the AutoFit model of ``fit_model``).

    Construction errors and a constructed search that does not hold the requested settings
    raise before any fit. Only an exception raised by ``search.fit`` becomes a failed outcome;
    a backend accessor failing after a successful fit raises.
    """
    case_dir = Path(case_dir)
    if not case_dir.is_absolute():
        raise ValueError(f"case_dir must be absolute, got {case_dir}")
    identity = search_identity(case_id=case_id, role=role, model=fit_model, data_identity=data_identity,
                               settings=settings, seed=seed, n_live=n_live)
    search = _named_search(identity, role=role, n_live=n_live, settings=settings, seed=seed, case_dir=case_dir)
    effective = effective_settings(search)
    _check_effective(effective, _search_keywords(settings, n_live=n_live, seed=seed), analysis, settings.use_jax)
    result = None
    error = None
    start = time.perf_counter()
    with session.search_scope(retain_search_internal=settings.retain_search_internal,
                              number_of_cores=settings.number_of_cores) as training_workers:
        try:
            result = search.fit(model=model, analysis=analysis)
        except Exception as exception:  # a sampler failure is a failed role, recorded with its settings
            error = f"{type(exception).__name__}: {exception}"
    runtime_s = time.perf_counter() - start
    root = case_dir.resolve()
    output_path = Path(search.paths.output_path).resolve()
    if not output_path.is_relative_to(root):
        raise RuntimeError(f"the search output {output_path} lies outside the case directory {root}")
    retention = None
    if settings.retain_search_internal:
        retention = inspect_retention(output_path, result_path=None if result is None
                                      else Path(result.paths.output_path))
    log_likelihood_max = log_evidence = likelihood_calls = None
    if result is not None:
        log_likelihood_max = float(result.max_log_likelihood_fit.figure_of_merit)
        evidence = result.samples.log_evidence
        log_evidence = None if evidence is None else float(evidence)
        likelihood_calls = int(result.samples.total_samples)
    record = SamplerRecord(name=search.paths.name, output_path=output_path.relative_to(root).as_posix(),
                           identity=identity, n_live=n_live, requested=settings.to_mapping(),
                           effective=effective, seed=seed, training_workers=training_workers,
                           log_likelihood_max=log_likelihood_max, log_evidence=log_evidence,
                           likelihood_calls=likelihood_calls, retention=retention, runtime_s=runtime_s)
    return SearchOutcome(record=record, result=result, error=error)
