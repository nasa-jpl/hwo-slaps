"""Prior-sampled PSF nuisance banks for nonlinear model comparison."""

from __future__ import annotations

__all__ = (
    "STRONG_EVIDENCE_DELTA_LOG_Z_THRESHOLD",
    "PsfBankCandidate",
    "PsfBank",
    "PsfBankCandidateFit",
    "PsfBankSummary",
    "PsfBankCaseResult",
    "build_psf_bank",
    "save_psf_bank_npz",
    "load_psf_bank_npz",
    "combine_psf_bank_fits",
    "run_psf_bank_case",
)

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .likelihood_metrics import SCDD_Q_THRESHOLD


STRONG_EVIDENCE_DELTA_LOG_Z_THRESHOLD = 5.0
"""Strong-evidence threshold on marginalized ``Delta log Z`` (`float`)."""

_ARTIFACT_SCHEMA_VERSION = 1
_BANK_VERSION_PACKAGES = ("hwoslaps", "hcipy", "numpy")


@dataclass(frozen=True)
class PsfBankCandidate:
    """One generated PSF-bank candidate.

    Parameters
    ----------
    label : `str`
        Stable candidate label.
    kind : `str`
        Candidate kind: ``"draw"``, ``"perfect"``, ``"truth"``, or
        ``"explicit"``.
    amplitude_rms_nm : `float`, optional
        Requested draw amplitude. Truth and explicit candidates have no
        requested amplitude.
    orthonormal_segment : `dict`, optional
        Per-segment orthonormal-basis coefficients.
    orthonormal_global : `dict`, optional
        Global orthonormal-basis coefficients.
    aberrations : `dict`
        Realized raw aberrations block.
    kernel : `numpy.ndarray`
        Native detector-sampled kernel.
    kernel_pixel_scale : `float`
        Kernel pixel scale in arcseconds per pixel.
    kernel_sha256 : `str`
        SHA-256 of the ``"{rows}x{columns}:"`` shape prefix followed by
        contiguous native float64 kernel bytes.
    measured_total_rms_nm : `float`
        Measured piston-removed aperture wavefront RMS.
    """

    label: str
    kind: str
    amplitude_rms_nm: Optional[float]
    orthonormal_segment: Optional[dict]
    orthonormal_global: Optional[dict]
    aberrations: dict
    kernel: np.ndarray
    kernel_pixel_scale: float
    kernel_sha256: str
    measured_total_rms_nm: float


@dataclass(frozen=True)
class PsfBank:
    """Persistable PSF nuisance bank and generation provenance.

    Parameters
    ----------
    bank_id : `str`
        First 16 hexadecimal characters of the generation-input SHA-256.
    candidates : `tuple` of `PsfBankCandidate`
        Ordered marginalization set.
    anchors : `tuple` of `PsfBankCandidate`
        Optional perfect and truth controls excluded from marginalization.
    seed : `int`, optional
        Root seed for prior draws.
    n_draws : `int`
        Number of prior draws. Explicit banks record zero.
    prior_table_path : `str`, optional
        Resolved prior-table path.
    prior_table_sha256 : `str`, optional
        SHA-256 of prior-table content.
    psf_config_hash : `str`
        Provenance hash of the top-level PSF configuration.
    lensing_pixel_scale : `float`
        Detector pixel scale consumed during kernel generation.
    bank_config : `dict`
        Validated bank configuration block. Explicit aberrations use the
        canonical float-and-list representation.
    versions : `dict`
        HWO-SLAPS, HCIPy, and NumPy versions.

    Notes
    -----
    Kernel hashes pin only the stored shape and float64 bytes. Truth-anchor
    byte identity with an observation PSF requires the same configuration and
    code versions. Execution compares versions as a soft diagnostic.
    """

    bank_id: str
    candidates: tuple
    anchors: tuple
    seed: Optional[int]
    n_draws: int
    prior_table_path: Optional[str]
    prior_table_sha256: Optional[str]
    psf_config_hash: str
    lensing_pixel_scale: float
    bank_config: dict
    versions: dict


@dataclass(frozen=True)
class PsfBankCandidateFit:
    """Smooth and subhalo fit results for one marginalization candidate.

    Parameters
    ----------
    label : `str`
        Stable candidate label.
    amplitude_rms_nm : `float`, optional
        Requested draw amplitude, carried for diagnostics only.
    log_l_smooth : `float`, optional
        Maximum log likelihood under the smooth model.
    log_l_subhalo : `float`, optional
        Maximum log likelihood under the subhalo model.
    log_evidence_smooth : `float`, optional
        Bayesian log evidence under the smooth model.
    log_evidence_subhalo : `float`, optional
        Bayesian log evidence under the subhalo model.
    success : `bool`, optional
        Whether both candidate fits completed successfully.
    """

    label: str
    amplitude_rms_nm: Optional[float]
    log_l_smooth: Optional[float]
    log_l_subhalo: Optional[float]
    log_evidence_smooth: Optional[float]
    log_evidence_subhalo: Optional[float]
    success: bool = True


@dataclass(frozen=True)
class PsfBankSummary:
    """Combined profile and evidence statistics for a PSF bank.

    Parameters
    ----------
    n_candidates : `int`
        Total marginalization-set size.
    n_success : `int`
        Number of candidates with paired finite likelihoods.
    n_evidence : `int`
        Number of likelihood-usable candidates with paired finite evidence.
    log_l_smooth_profile : `float`, optional
        Smooth-model log likelihood profiled over the paired set.
    log_l_subhalo_profile : `float`, optional
        Subhalo-model log likelihood profiled over the paired set.
    q_fit_psf_profile : `float`, optional
        Non-negative profiled likelihood-ratio statistic.
    log_evidence_smooth_psf_marg : `float`, optional
        Equal-weight smooth evidence estimate.
    log_evidence_subhalo_psf_marg : `float`, optional
        Equal-weight subhalo evidence estimate.
    delta_log_evidence_psf_marg : `float`, optional
        Marginalized evidence difference, subhalo minus smooth.
    best_smooth_label : `str`, optional
        Lexically tie-broken smooth profile label.
    best_subhalo_label : `str`, optional
        Lexically tie-broken subhalo profile label.
    ess_evidence_smooth : `float`, optional
        Smooth-hypothesis evidence-weight effective sample size.
    ess_evidence_subhalo : `float`, optional
        Subhalo-hypothesis evidence-weight effective sample size.
    detected_fit_scdd_psf_profile : `bool`, optional
        Whether the profile statistic reaches the SCDD threshold.
    detected_evidence_psf_marg : `bool`, optional
        Whether the marginalized evidence difference is strong.
    """

    n_candidates: int
    n_success: int
    n_evidence: int
    log_l_smooth_profile: Optional[float]
    log_l_subhalo_profile: Optional[float]
    q_fit_psf_profile: Optional[float]
    log_evidence_smooth_psf_marg: Optional[float]
    log_evidence_subhalo_psf_marg: Optional[float]
    delta_log_evidence_psf_marg: Optional[float]
    best_smooth_label: Optional[str]
    best_subhalo_label: Optional[str]
    ess_evidence_smooth: Optional[float]
    ess_evidence_subhalo: Optional[float]
    detected_fit_scdd_psf_profile: Optional[bool]
    detected_evidence_psf_marg: Optional[bool]


def _kernel_sha256(kernel: Any) -> str:
    """Return the canonical SHA-256 for a native detector kernel."""
    array = np.ascontiguousarray(np.asarray(kernel, dtype=np.float64))
    if array.ndim != 2:
        raise ValueError("PSF bank kernels must be two-dimensional")
    prefix = f"{array.shape[0]}x{array.shape[1]}:".encode("utf-8")
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def _usable(value: Any) -> bool:
    """Return whether a statistic is present and finite."""
    return value is not None and math.isfinite(float(value))


def _likelihood_usable(candidate: PsfBankCandidateFit) -> bool:
    """Return whether both likelihood values form a usable pair."""
    return bool(
        candidate.success
        and _usable(candidate.log_l_smooth)
        and _usable(candidate.log_l_subhalo)
    )


def _evidence_usable(candidate: PsfBankCandidateFit) -> bool:
    """Return whether a likelihood-usable fit has paired evidence."""
    return bool(
        _likelihood_usable(candidate)
        and _usable(candidate.log_evidence_smooth)
        and _usable(candidate.log_evidence_subhalo)
    )


def _logsumexp(values: Iterable[float]) -> Optional[float]:
    """Return a stable log-sum-exp or `None` for an empty input."""
    finite = [float(value) for value in values if _usable(value)]
    if not finite:
        return None
    maximum = max(finite)
    return float(
        maximum + math.log(sum(math.exp(value - maximum) for value in finite))
    )


def _effective_sample_size(values: Sequence[float]) -> Optional[float]:
    """Return evidence-weight effective sample size."""
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    weights = np.exp(array - np.max(array))
    weights = weights / np.sum(weights)
    return float(1.0 / np.sum(weights*weights))


def _best_candidate(
    candidates: Sequence[PsfBankCandidateFit],
    field_name: str,
) -> Optional[PsfBankCandidateFit]:
    """Return a profile maximum with lexical label tie-breaking."""
    if not candidates:
        return None
    maximum = max(float(getattr(item, field_name)) for item in candidates)
    tied = [
        item
        for item in candidates
        if float(getattr(item, field_name)) == maximum
    ]
    return min(tied, key=lambda item: item.label)


def combine_psf_bank_fits(
    candidates: Sequence[PsfBankCandidateFit],
    fit_mode: Optional[str] = None,
) -> PsfBankSummary:
    """Combine paired candidate fits into bank profile and evidence results.

    Parameters
    ----------
    candidates : sequence of `PsfBankCandidateFit`
        Fits for the full marginalization set, including failed fits.
    fit_mode : `str`, optional
        Nonlinear fit mode. ``"freed"`` nulls the fixed-calibration SCDD
        detection flag but retains the profile statistic.

    Returns
    -------
    summary : `PsfBankSummary`
        Paired-set profile, evidence, and ESS diagnostics.

    Raises
    ------
    ValueError
        Raised when the marginalization set is empty.

    Notes
    -----
    Equal prior weights make each hypothesis evidence an MC estimate over
    the marginalization set. For an ordered amplitude list, the bank is an
    equal-allocation stratified sample over a uniform prior on that list,
    rather than iid sampling from an amplitude distribution.

    Each per-hypothesis ``logZ_marg`` estimate is biased low when evidence
    weights concentrate because it is the log of an MC average. The two
    biases may partially cancel in their difference, but cancellation is not
    guaranteed. ESS monitors concentration; repeated-bank or grown-bank M
    convergence checks are the quantitative stage-two diagnostic.

    The paired-set identities are: each profile dominates every included
    per-candidate likelihood; profile q is no larger than the largest paired
    candidate q; and the evidence difference lies between the minimum and
    maximum paired per-candidate evidence differences.
    """
    if not candidates:
        raise ValueError(
            "combine_psf_bank_fits requires at least one candidate"
        )
    ordered = sorted(candidates, key=lambda item: item.label)
    likelihood_set = [item for item in ordered if _likelihood_usable(item)]
    evidence_set = [item for item in likelihood_set if _evidence_usable(item)]
    best_smooth = _best_candidate(likelihood_set, "log_l_smooth")
    best_subhalo = _best_candidate(likelihood_set, "log_l_subhalo")
    smooth_profile = (
        None
        if best_smooth is None
        else float(best_smooth.log_l_smooth)
    )
    subhalo_profile = (
        None
        if best_subhalo is None
        else float(best_subhalo.log_l_subhalo)
    )
    q_fit = None
    if smooth_profile is not None and subhalo_profile is not None:
        q_fit = max(0.0, 2.0*(subhalo_profile - smooth_profile))

    log_prior = -math.log(len(candidates))
    smooth_values = [
        float(item.log_evidence_smooth) for item in evidence_set
    ]
    subhalo_values = [
        float(item.log_evidence_subhalo) for item in evidence_set
    ]
    smooth_logz = _logsumexp(
        value + log_prior for value in smooth_values
    )
    subhalo_logz = _logsumexp(
        value + log_prior for value in subhalo_values
    )
    delta_logz = None
    if smooth_logz is not None and subhalo_logz is not None:
        delta_logz = float(subhalo_logz - smooth_logz)

    detected_q = None if q_fit is None else q_fit >= SCDD_Q_THRESHOLD
    if fit_mode == "freed":
        detected_q = None
    return PsfBankSummary(
        n_candidates=len(candidates),
        n_success=len(likelihood_set),
        n_evidence=len(evidence_set),
        log_l_smooth_profile=smooth_profile,
        log_l_subhalo_profile=subhalo_profile,
        q_fit_psf_profile=q_fit,
        log_evidence_smooth_psf_marg=smooth_logz,
        log_evidence_subhalo_psf_marg=subhalo_logz,
        delta_log_evidence_psf_marg=delta_logz,
        best_smooth_label=(
            None if best_smooth is None else best_smooth.label
        ),
        best_subhalo_label=(
            None if best_subhalo is None else best_subhalo.label
        ),
        ess_evidence_smooth=_effective_sample_size(smooth_values),
        ess_evidence_subhalo=_effective_sample_size(subhalo_values),
        detected_fit_scdd_psf_profile=detected_q,
        detected_evidence_psf_marg=(
            None
            if delta_logz is None
            else delta_logz > STRONG_EVIDENCE_DELTA_LOG_Z_THRESHOLD
        ),
    )


def _resolve_prior_table_path(path: Any) -> Path:
    """Resolve a prior table through absolute, CWD, then repository paths."""
    requested = Path(path).expanduser()
    if requested.is_absolute():
        candidates = (requested,)
    else:
        repository_root = Path(__file__).resolve().parents[4]
        candidates = (Path.cwd() / requested, repository_root / requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"PSF bank prior table {path!s} was not found; tried: {attempted}"
    )


def _identity_payload(
    bank_config: dict,
    prior_table_sha256: Optional[str],
    psf_config_hash: str,
    lensing_pixel_scale: float,
) -> dict:
    """Return canonical bank generation inputs."""
    kind = str(bank_config["kind"]).lower()
    if kind == "prior_draws":
        configured_amplitude = bank_config["amplitude_rms_nm"]
        amplitudes = (
            [float(value) for value in configured_amplitude]
            if isinstance(configured_amplitude, list)
            else [float(configured_amplitude)]
        )
        n_draws = int(bank_config["n_draws"])
        seed = int(bank_config["seed"])
        include_perfect = bool(bank_config.get("include_perfect", False))
        include_truth = bool(bank_config.get("include_truth", False))
        explicit_candidates = []
    else:
        amplitudes = []
        n_draws = 0
        seed = None
        include_perfect = False
        include_truth = False
        explicit_candidates = [
            _aberrations_from_wire(_aberrations_to_wire(candidate))
            for candidate in bank_config["candidates"]
        ]
    return {
        "kind": kind,
        "prior_table_sha256": prior_table_sha256,
        "amplitude_rms_nm": amplitudes,
        "n_draws": n_draws,
        "seed": seed,
        "include_perfect": include_perfect,
        "include_truth": include_truth,
        "explicit_candidates": explicit_candidates,
        "psf_config_hash": psf_config_hash,
        "lensing_pixel_scale": float(lensing_pixel_scale),
    }


def _bank_id_from_inputs(
    bank_config: dict,
    prior_table_sha256: Optional[str],
    psf_config_hash: str,
    lensing_pixel_scale: float,
) -> str:
    """Hash canonical bank generation inputs."""
    payload = _identity_payload(
        bank_config,
        prior_table_sha256,
        psf_config_hash,
        lensing_pixel_scale,
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _empty_aberrations() -> dict:
    """Return an all-disabled aberrations block."""
    return {
        "enable_segment_pistons": False,
        "enable_segment_tiptilts": False,
        "enable_segment_hexikes": False,
        "enable_global_zernikes": False,
        "segment_pistons": {},
        "segment_tiptilts": {},
        "segment_hexikes": {},
        "global_zernikes": {},
    }


def _canonical_aberrations(aberrations: dict) -> dict:
    """Return aberrations with canonical float coefficients and pair lists."""
    canonical = deepcopy(aberrations)
    if "segment_pistons" in canonical:
        canonical["segment_pistons"] = {
            key: float(value)
            for key, value in canonical["segment_pistons"].items()
        }
    if "segment_tiptilts" in canonical:
        canonical["segment_tiptilts"] = {
            key: [float(value[0]), float(value[1])]
            for key, value in canonical["segment_tiptilts"].items()
        }
    if "segment_hexikes" in canonical:
        canonical["segment_hexikes"] = {
            segment: {
                mode: float(value) for mode, value in modes.items()
            }
            for segment, modes in canonical["segment_hexikes"].items()
        }
    if "global_zernikes" in canonical:
        canonical["global_zernikes"] = {
            key: float(value)
            for key, value in canonical["global_zernikes"].items()
        }
    return canonical


def _generate_candidate(
    label: str,
    kind: str,
    amplitude_rms_nm: Optional[float],
    orthonormal_segment: Optional[dict],
    orthonormal_global: Optional[dict],
    aberrations: dict,
    psf_config: dict,
    full_config: dict,
) -> PsfBankCandidate:
    """Generate and record one candidate through the canonical PSF path."""
    from ...config.validation import validate_psf_config
    from ...psf.generator import generate_psf_system
    from ...psf.utils import pyauto_kernel_native

    candidate_psf_config = deepcopy(psf_config)
    candidate_psf_config["aberrations"] = deepcopy(aberrations)
    validate_psf_config(candidate_psf_config)
    canonical_aberrations = _canonical_aberrations(aberrations)
    candidate_psf_config["aberrations"] = canonical_aberrations
    psf_data = generate_psf_system(candidate_psf_config, full_config)
    kernel = np.ascontiguousarray(
        pyauto_kernel_native(psf_data.kernel),
        dtype=np.float64,
    )
    return PsfBankCandidate(
        label=label,
        kind=kind,
        amplitude_rms_nm=amplitude_rms_nm,
        orthonormal_segment=deepcopy(orthonormal_segment),
        orthonormal_global=deepcopy(orthonormal_global),
        aberrations=canonical_aberrations,
        kernel=kernel,
        kernel_pixel_scale=float(psf_data.kernel_pixel_scale),
        kernel_sha256=_kernel_sha256(kernel),
        measured_total_rms_nm=float(psf_data.total_rms_nm),
    )


def _current_versions() -> dict:
    """Return the software versions recorded for a generated bank."""
    from ...provenance import _package_version

    return {
        package: _package_version(package)
        for package in _BANK_VERSION_PACKAGES
    }


def build_psf_bank(full_config: dict) -> PsfBank:
    """Build a deterministic PSF nuisance bank from validated configuration.

    Parameters
    ----------
    full_config : `dict`
        Validated full HWO-SLAPS configuration with ``fit_psf.mode=bank``.

    Returns
    -------
    bank : `PsfBank`
        Generated marginalization candidates, optional anchors, and
        provenance.

    Raises
    ------
    ValueError
        Raised when the configured fit-PSF mode is not ``"bank"``.

    Notes
    -----
    A scalar amplitude applies to every prior draw. An ordered amplitude list
    cycles by candidate index and is an equal-allocation stratified sample
    over the listed uniform amplitude prior, not an iid amplitude draw.
    """
    from ...provenance import config_hash

    fit_psf = full_config["modeling"]["fit_psf"]
    if str(fit_psf.get("mode", "")).lower() != "bank":
        raise ValueError(
            "build_psf_bank requires modeling.fit_psf.mode to be 'bank'"
        )
    bank_config = deepcopy(fit_psf["bank"])
    kind = str(bank_config["kind"]).lower()
    if kind == "explicit":
        bank_config["candidates"] = [
            _canonical_aberrations(candidate)
            for candidate in bank_config["candidates"]
        ]
    psf_config = deepcopy(full_config["psf"])
    psf_hash = config_hash(psf_config)
    lensing_pixel_scale = float(
        full_config["lensing"]["grid"]["pixel_scale"]
    )
    candidates = []
    anchors = []
    prior_path = None
    prior_sha256 = None
    seed = None
    n_draws = 0

    if kind == "prior_draws":
        from ...psf.families import (
            draw_weighted_combined_family,
            load_mode_weight_prior,
            realize_weighted_draw,
        )
        from ...psf.opd_basis import ApertureBasisTransform
        from ...psf.telescope_models import create_hcipy_telescope

        prior_path = _resolve_prior_table_path(bank_config["prior_table"])
        prior_sha256 = hashlib.sha256(prior_path.read_bytes()).hexdigest()
        prior = load_mode_weight_prior(prior_path)
        telescope_data = create_hcipy_telescope(psf_config)
        transform = ApertureBasisTransform(
            telescope_data,
            global_mode_nolls=sorted(prior.global_weights),
            segment_mode_nolls=sorted(prior.segment_weights),
        )
        segments = range(len(telescope_data["segments"]))
        seed = int(bank_config["seed"])
        n_draws = int(bank_config["n_draws"])
        configured_amplitudes = bank_config["amplitude_rms_nm"]
        amplitudes = (
            [float(value) for value in configured_amplitudes]
            if isinstance(configured_amplitudes, list)
            else [float(configured_amplitudes)]
        )
        children = np.random.SeedSequence(seed).spawn(n_draws)
        for index, child in enumerate(children):
            amplitude = amplitudes[index % len(amplitudes)]
            rng = np.random.default_rng(child)
            segment_orth, global_orth = draw_weighted_combined_family(
                rng,
                segments,
                prior,
                amplitude,
            )
            segment_raw, global_raw = realize_weighted_draw(
                telescope_data,
                transform,
                amplitude,
                segment_coefficients=segment_orth,
                global_coefficients=global_orth,
            )
            aberrations = _empty_aberrations()
            aberrations.update({
                "enable_segment_hexikes": bool(segment_raw),
                "segment_hexikes": segment_raw,
                "enable_global_zernikes": bool(global_raw),
                "global_zernikes": global_raw,
            })
            candidates.append(_generate_candidate(
                label=f"draw{index:03d}",
                kind="draw",
                amplitude_rms_nm=amplitude,
                orthonormal_segment=segment_orth,
                orthonormal_global=global_orth,
                aberrations=aberrations,
                psf_config=psf_config,
                full_config=full_config,
            ))
        if bank_config.get("include_perfect", False):
            anchors.append(_generate_candidate(
                label="perfect",
                kind="perfect",
                amplitude_rms_nm=0.0,
                orthonormal_segment=None,
                orthonormal_global=None,
                aberrations=_empty_aberrations(),
                psf_config=psf_config,
                full_config=full_config,
            ))
        if bank_config.get("include_truth", False):
            anchors.append(_generate_candidate(
                label="truth",
                kind="truth",
                amplitude_rms_nm=None,
                orthonormal_segment=None,
                orthonormal_global=None,
                aberrations=deepcopy(psf_config["aberrations"]),
                psf_config=psf_config,
                full_config=full_config,
            ))
    elif kind == "explicit":
        for index, aberrations in enumerate(bank_config["candidates"]):
            candidates.append(_generate_candidate(
                label=f"explicit{index:03d}",
                kind="explicit",
                amplitude_rms_nm=None,
                orthonormal_segment=None,
                orthonormal_global=None,
                aberrations=aberrations,
                psf_config=psf_config,
                full_config=full_config,
            ))
    else:
        raise ValueError(f"Unsupported PSF bank kind: {kind!r}")

    bank_id = _bank_id_from_inputs(
        bank_config,
        prior_sha256,
        psf_hash,
        lensing_pixel_scale,
    )
    return PsfBank(
        bank_id=bank_id,
        candidates=tuple(candidates),
        anchors=tuple(anchors),
        seed=seed,
        n_draws=n_draws,
        prior_table_path=(None if prior_path is None else str(prior_path)),
        prior_table_sha256=prior_sha256,
        psf_config_hash=psf_hash,
        lensing_pixel_scale=lensing_pixel_scale,
        bank_config=bank_config,
        versions=_current_versions(),
    )


def _flat_int_map_to_wire(mapping: Optional[dict]) -> Optional[list]:
    """Encode one integer-keyed scalar map as sorted entries."""
    if mapping is None:
        return None
    return [[int(key), float(value)] for key, value in sorted(mapping.items())]


def _nested_int_map_to_wire(mapping: Optional[dict]) -> Optional[list]:
    """Encode nested integer-keyed scalar maps as sorted entries."""
    if mapping is None:
        return None
    return [
        [int(segment), _flat_int_map_to_wire(modes)]
        for segment, modes in sorted(mapping.items())
    ]


def _flat_int_map_from_wire(entries: Optional[list]) -> Optional[dict]:
    """Decode sorted scalar entries to integer-keyed maps."""
    if entries is None:
        return None
    return {int(key): float(value) for key, value in entries}


def _nested_int_map_from_wire(entries: Optional[list]) -> Optional[dict]:
    """Decode nested sorted entries to integer-keyed maps."""
    if entries is None:
        return None
    return {
        int(segment): _flat_int_map_from_wire(modes)
        for segment, modes in entries
    }


def _aberrations_to_wire(aberrations: dict) -> dict:
    """Encode integer-keyed aberration maps without JSON key coercion."""
    wire = deepcopy(aberrations)
    if "segment_pistons" in wire:
        wire["segment_pistons"] = _flat_int_map_to_wire(
            aberrations["segment_pistons"]
        )
    if "segment_tiptilts" in wire:
        wire["segment_tiptilts"] = [
            [int(key), [float(value[0]), float(value[1])]]
            for key, value in sorted(aberrations["segment_tiptilts"].items())
        ]
    if "segment_hexikes" in wire:
        wire["segment_hexikes"] = _nested_int_map_to_wire(
            aberrations["segment_hexikes"]
        )
    if "global_zernikes" in wire:
        wire["global_zernikes"] = _flat_int_map_to_wire(
            aberrations["global_zernikes"]
        )
    return wire


def _aberrations_from_wire(wire: dict) -> dict:
    """Restore integer-keyed aberration maps from typed entries."""
    aberrations = deepcopy(wire)
    if "segment_pistons" in aberrations:
        aberrations["segment_pistons"] = _flat_int_map_from_wire(
            wire["segment_pistons"]
        )
    if "segment_tiptilts" in aberrations:
        aberrations["segment_tiptilts"] = {
            int(key): [float(value[0]), float(value[1])]
            for key, value in wire["segment_tiptilts"]
        }
    if "segment_hexikes" in aberrations:
        aberrations["segment_hexikes"] = _nested_int_map_from_wire(
            wire["segment_hexikes"]
        )
    if "global_zernikes" in aberrations:
        aberrations["global_zernikes"] = _flat_int_map_from_wire(
            wire["global_zernikes"]
        )
    return aberrations


def _bank_config_to_wire(bank_config: dict) -> dict:
    """Encode any explicit candidate maps in a bank configuration."""
    wire = deepcopy(bank_config)
    if str(wire["kind"]).lower() == "explicit":
        wire["candidates"] = [
            _aberrations_to_wire(candidate)
            for candidate in bank_config["candidates"]
        ]
    return wire


def _bank_config_from_wire(wire: dict) -> dict:
    """Restore any explicit candidate maps in a bank configuration."""
    bank_config = deepcopy(wire)
    if str(bank_config["kind"]).lower() == "explicit":
        bank_config["candidates"] = [
            _aberrations_from_wire(candidate)
            for candidate in wire["candidates"]
        ]
    return bank_config


def _candidate_metadata(candidate: PsfBankCandidate) -> dict:
    """Return typed JSON metadata for one candidate without its kernel."""
    return {
        "label": candidate.label,
        "kind": candidate.kind,
        "amplitude_rms_nm": candidate.amplitude_rms_nm,
        "orthonormal_segment": _nested_int_map_to_wire(
            candidate.orthonormal_segment
        ),
        "orthonormal_global": _flat_int_map_to_wire(
            candidate.orthonormal_global
        ),
        "aberrations": _aberrations_to_wire(candidate.aberrations),
        "kernel_pixel_scale": candidate.kernel_pixel_scale,
        "kernel_sha256": candidate.kernel_sha256,
        "measured_total_rms_nm": candidate.measured_total_rms_nm,
    }


def _candidate_from_metadata(metadata: dict, kernel: np.ndarray) -> PsfBankCandidate:
    """Reconstruct one candidate from typed metadata and a kernel array."""
    return PsfBankCandidate(
        label=str(metadata["label"]),
        kind=str(metadata["kind"]),
        amplitude_rms_nm=(
            None
            if metadata["amplitude_rms_nm"] is None
            else float(metadata["amplitude_rms_nm"])
        ),
        orthonormal_segment=_nested_int_map_from_wire(
            metadata["orthonormal_segment"]
        ),
        orthonormal_global=_flat_int_map_from_wire(
            metadata["orthonormal_global"]
        ),
        aberrations=_aberrations_from_wire(metadata["aberrations"]),
        kernel=np.ascontiguousarray(kernel, dtype=np.float64),
        kernel_pixel_scale=float(metadata["kernel_pixel_scale"]),
        kernel_sha256=str(metadata["kernel_sha256"]),
        measured_total_rms_nm=float(metadata["measured_total_rms_nm"]),
    )


def _verify_candidate(candidate: PsfBankCandidate, bank_id: str) -> None:
    """Verify candidate kernel integrity and aberration key types."""
    actual = _kernel_sha256(candidate.kernel)
    if actual != candidate.kernel_sha256:
        raise ValueError(
            f"PSF bank {bank_id} candidate {candidate.label} kernel "
            "sha256 mismatch"
        )
    from ...config.validation import _validate_psf_aberrations

    _validate_psf_aberrations(
        candidate.aberrations,
        f"psf_bank.{candidate.label}.aberrations",
    )


def _verify_bank_structure(bank: PsfBank) -> None:
    """Verify that bank metadata matches its canonical configured manifest."""
    def fail(check: str) -> None:
        raise ValueError(
            f"PSF bank {bank.bank_id} structure check failed: {check}"
        )

    if not isinstance(bank.bank_config, dict):
        fail("bank_config must be a dictionary")
    try:
        kind = str(bank.bank_config["kind"]).lower()
    except KeyError:
        fail("bank_config kind is missing")

    if kind == "prior_draws":
        try:
            configured_n_draws = bank.bank_config["n_draws"]
            configured_seed = bank.bank_config["seed"]
            configured_amplitudes = bank.bank_config["amplitude_rms_nm"]
        except KeyError as exc:
            fail(f"bank_config {exc.args[0]} is missing")
        if bank.n_draws != configured_n_draws:
            fail("n_draws does not match bank_config")
        if bank.seed != configured_seed:
            fail("seed does not match bank_config")

        expected_labels = [
            f"draw{index:03d}" for index in range(configured_n_draws)
        ]
        if [candidate.label for candidate in bank.candidates] != expected_labels:
            fail("candidate labels or order do not match prior draws")
        if any(candidate.kind != "draw" for candidate in bank.candidates):
            fail("prior-draw candidate kind is not 'draw'")

        amplitudes = (
            configured_amplitudes
            if isinstance(configured_amplitudes, list)
            else [configured_amplitudes]
        )
        if not amplitudes:
            fail("configured amplitude sequence is empty")
        for index, candidate in enumerate(bank.candidates):
            expected_amplitude = float(amplitudes[index % len(amplitudes)])
            if candidate.amplitude_rms_nm != expected_amplitude:
                fail(
                    f"candidate {candidate.label} amplitude does not match "
                    "the cyclic assignment"
                )

        expected_anchor_labels = []
        if bank.bank_config.get("include_perfect", False):
            expected_anchor_labels.append("perfect")
        if bank.bank_config.get("include_truth", False):
            expected_anchor_labels.append("truth")
        if [anchor.label for anchor in bank.anchors] != expected_anchor_labels:
            fail("anchor labels or order do not match enabled controls")
        for anchor in bank.anchors:
            if anchor.kind != anchor.label:
                fail(f"anchor {anchor.label} kind does not match its label")
            if anchor.label == "perfect" and anchor.amplitude_rms_nm != 0.0:
                fail("perfect anchor amplitude is not 0.0")
            if anchor.label == "truth" and anchor.amplitude_rms_nm is not None:
                fail("truth anchor amplitude is not None")
        return

    if kind == "explicit":
        if bank.n_draws != 0:
            fail("explicit bank n_draws is not zero")
        if bank.seed is not None:
            fail("explicit bank seed is not None")
        if bank.anchors:
            fail("explicit bank contains anchors")
        try:
            configured_candidates = bank.bank_config["candidates"]
        except KeyError:
            fail("bank_config candidates are missing")
        expected_labels = [
            f"explicit{index:03d}"
            for index in range(len(configured_candidates))
        ]
        if [candidate.label for candidate in bank.candidates] != expected_labels:
            fail("candidate labels, order, or count do not match explicit config")
        if any(candidate.kind != "explicit" for candidate in bank.candidates):
            fail("explicit candidate kind is not 'explicit'")
        for candidate, configured in zip(
            bank.candidates,
            configured_candidates,
        ):
            if candidate.aberrations != configured:
                fail(
                    f"candidate {candidate.label} aberrations do not match "
                    "bank_config"
                )
        return

    fail(f"unsupported bank_config kind {kind!r}")


def save_psf_bank_npz(bank: PsfBank, path: Any) -> None:
    """Save a PSF bank as one verified NPZ artifact.

    Parameters
    ----------
    bank : `PsfBank`
        Bank to persist.
    path : path-like
        Destination NPZ path.

    Raises
    ------
    ValueError
        Raised when the structural manifest, a kernel hash, or a candidate
        label is invalid.

    Notes
    -----
    Kernel hashes pin only the stored shape and float64 bytes. They do not
    prove that another code version would regenerate the same kernel.
    """
    _verify_bank_structure(bank)
    all_candidates = bank.candidates + bank.anchors
    labels = [candidate.label for candidate in all_candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("PSF bank candidate labels must be unique")
    for candidate in all_candidates:
        _verify_candidate(candidate, bank.bank_id)
    metadata = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "bank_id": bank.bank_id,
        "candidates": [
            _candidate_metadata(candidate) for candidate in bank.candidates
        ],
        "anchors": [
            _candidate_metadata(candidate) for candidate in bank.anchors
        ],
        "seed": bank.seed,
        "n_draws": bank.n_draws,
        "prior_table_path": bank.prior_table_path,
        "prior_table_sha256": bank.prior_table_sha256,
        "psf_config_hash": bank.psf_config_hash,
        "lensing_pixel_scale": bank.lensing_pixel_scale,
        "bank_config": _bank_config_to_wire(bank.bank_config),
        "versions": deepcopy(bank.versions),
    }
    arrays = {
        f"kernel_{candidate.label}": np.ascontiguousarray(
            candidate.kernel,
            dtype=np.float64,
        )
        for candidate in all_candidates
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("wb") as handle:
        np.savez(handle, **arrays)


def load_psf_bank_npz(path: Any) -> PsfBank:
    """Load and fully verify a PSF-bank NPZ artifact.

    Parameters
    ----------
    path : path-like
        Source NPZ path.

    Returns
    -------
    bank : `PsfBank`
        Reconstructed bank with integer coefficient keys restored.

    Raises
    ------
    ValueError
        Raised for schema, identity, kernel, or aberration mismatches.
    """
    with np.load(path, allow_pickle=False) as stored:
        if "metadata_json" not in stored.files:
            raise ValueError("PSF bank artifact is missing metadata_json")
        raw_metadata = stored["metadata_json"].item()
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        metadata = json.loads(str(raw_metadata))
        if metadata.get("schema_version") != _ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported PSF bank artifact schema version")
        candidate_metadata = metadata["candidates"]
        anchor_metadata = metadata["anchors"]
        labels = [
            item["label"]
            for item in candidate_metadata + anchor_metadata
        ]
        expected_arrays = {"metadata_json"} | {
            f"kernel_{label}" for label in labels
        }
        if set(stored.files) != expected_arrays:
            raise ValueError("PSF bank artifact kernel array set mismatch")
        candidates = tuple(
            _candidate_from_metadata(
                item,
                stored[f"kernel_{item['label']}"][:],
            )
            for item in candidate_metadata
        )
        anchors = tuple(
            _candidate_from_metadata(
                item,
                stored[f"kernel_{item['label']}"][:],
            )
            for item in anchor_metadata
        )
    bank = PsfBank(
        bank_id=str(metadata["bank_id"]),
        candidates=candidates,
        anchors=anchors,
        seed=(None if metadata["seed"] is None else int(metadata["seed"])),
        n_draws=int(metadata["n_draws"]),
        prior_table_path=metadata["prior_table_path"],
        prior_table_sha256=metadata["prior_table_sha256"],
        psf_config_hash=str(metadata["psf_config_hash"]),
        lensing_pixel_scale=float(metadata["lensing_pixel_scale"]),
        bank_config=_bank_config_from_wire(metadata["bank_config"]),
        versions=dict(metadata["versions"]),
    )
    expected_id = _bank_id_from_inputs(
        bank.bank_config,
        bank.prior_table_sha256,
        bank.psf_config_hash,
        bank.lensing_pixel_scale,
    )
    if expected_id != bank.bank_id:
        raise ValueError(
            f"PSF bank artifact bank_id mismatch: {bank.bank_id} != "
            f"{expected_id}"
        )
    for candidate in bank.candidates + bank.anchors:
        _verify_candidate(candidate, bank.bank_id)
    _verify_bank_structure(bank)
    return bank


def _case_slim_row(case: Any) -> dict:
    """Return the non-kernel case fields persisted in bank JSON."""
    return {
        "label": case.psf_case.rsplit(":", 1)[-1],
        "fit_status_smooth": case.smooth_fit.status,
        "fit_status_subhalo": case.subhalo_fit.status,
        "log_l_smooth": case.smooth_fit.log_likelihood_max,
        "log_l_subhalo": case.subhalo_fit.log_likelihood_max,
        "log_evidence_smooth": case.smooth_fit.log_evidence,
        "log_evidence_subhalo": case.subhalo_fit.log_evidence,
        "analysis_key_smooth": case.smooth_fit.analysis_key,
        "analysis_key_subhalo": case.subhalo_fit.analysis_key,
        "quality_flags": list(case.quality_flags),
    }


@dataclass
class PsfBankCaseResult:
    """Result of executing one nonlinear case over a PSF bank.

    Parameters
    ----------
    bank_id : `str`
        Executed bank identity.
    case_id : `str`
        Stable nonlinear case identifier.
    fit_mode : `str`
        Nonlinear fit mode.
    summary : `PsfBankSummary`
        Marginalization-set combination statistics.
    candidate_results : `list`
        Marginalization-set nonlinear case results.
    anchor_results : `list`
        Optional anchor-control nonlinear case results.
    anchor_diagnostics : `dict`
        Per-anchor q and evidence-difference controls.
    quality_flags : `list`
        Bank-level failure and callback diagnostics.
    bank_provenance : `dict`, optional
        Bank inputs and software versions, excluding kernels.
    """

    bank_id: str
    case_id: str
    fit_mode: str
    summary: PsfBankSummary
    candidate_results: list
    anchor_results: list
    anchor_diagnostics: dict
    quality_flags: list
    bank_provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert the bank result to a compact JSON-compatible dictionary.

        Returns
        -------
        payload : `dict`
            Summary, slim case rows, anchors, flags, and bank provenance.
        """
        from .output_schema import _json_safe

        payload = {
            "bank_id": self.bank_id,
            "case_id": self.case_id,
            "fit_mode": self.fit_mode,
            "summary": asdict(self.summary),
            "candidate_results": [
                _case_slim_row(case) for case in self.candidate_results
            ],
            "anchor_results": [
                _case_slim_row(case) for case in self.anchor_results
            ],
            "anchor_diagnostics": deepcopy(self.anchor_diagnostics),
            "quality_flags": list(self.quality_flags),
            "bank_provenance": deepcopy(self.bank_provenance),
        }
        return _json_safe(payload)

    def write_json(self, path: Any) -> None:
        """Write the compact bank result as formatted JSON.

        Parameters
        ----------
        path : path-like
            Destination JSON path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


def _bank_provenance(
    bank: PsfBank,
    version_mismatches: Optional[dict] = None,
) -> dict:
    """Return non-kernel provenance for result JSON."""
    provenance = {
        "seed": bank.seed,
        "n_draws": bank.n_draws,
        "prior_table_path": bank.prior_table_path,
        "prior_table_sha256": bank.prior_table_sha256,
        "psf_config_hash": bank.psf_config_hash,
        "lensing_pixel_scale": bank.lensing_pixel_scale,
        "bank_config": _bank_config_to_wire(bank.bank_config),
        "versions": deepcopy(bank.versions),
    }
    if version_mismatches:
        provenance["version_mismatches"] = deepcopy(version_mismatches)
    return provenance


def _validate_execution_inputs(
    observation: Any,
    full_config: dict,
    bank: PsfBank,
) -> dict:
    """Validate hard inputs and return soft software-version mismatches."""
    from ...provenance import config_hash

    _verify_bank_structure(bank)
    expected_psf_hash = config_hash(full_config["psf"])
    if bank.psf_config_hash != expected_psf_hash:
        raise ValueError(
            f"PSF bank {bank.bank_id} psf_config_hash is incompatible with "
            "the requested configuration"
        )
    expected_scale = float(full_config["lensing"]["grid"]["pixel_scale"])
    if bank.lensing_pixel_scale != expected_scale:
        raise ValueError(
            f"PSF bank {bank.bank_id} lensing pixel scale is incompatible "
            "with the requested configuration"
        )
    for candidate in bank.candidates + bank.anchors:
        if not np.isclose(
            candidate.kernel_pixel_scale,
            observation.pixel_scale,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"PSF bank {bank.bank_id} candidate {candidate.label} "
                "kernel pixel scale is incompatible with the observation"
            )
        _verify_candidate(candidate, bank.bank_id)
    current_versions = _current_versions()
    return {
        package: {
            "bank": bank.versions.get(package),
            "current": current_versions[package],
        }
        for package in _BANK_VERSION_PACKAGES
        if bank.versions.get(package) != current_versions[package]
    }


def _wrapped_candidate_kernel(candidate: PsfBankCandidate) -> Any:
    """Wrap stored native bytes through the observation convolution path."""
    from ...observation.generator import _ensure_odd_kernel
    from ...psf.utils import make_pyauto_convolver, make_pyauto_kernel

    kernel = make_pyauto_kernel(
        values=candidate.kernel,
        pixel_scales=candidate.kernel_pixel_scale,
        normalize=False,
    )
    return make_pyauto_convolver(_ensure_odd_kernel(kernel))


def _anchor_diagnostic(case: Any) -> dict:
    """Return null-safe q and evidence controls for one anchor."""
    smooth_log_l = case.smooth_fit.log_likelihood_max
    subhalo_log_l = case.subhalo_fit.log_likelihood_max
    q_fit = None
    if _usable(smooth_log_l) and _usable(subhalo_log_l):
        q_fit = max(0.0, 2.0*(float(subhalo_log_l) - float(smooth_log_l)))
    smooth_logz = case.smooth_fit.log_evidence
    subhalo_logz = case.subhalo_fit.log_evidence
    delta_logz = None
    if _usable(smooth_logz) and _usable(subhalo_logz):
        delta_logz = float(subhalo_logz) - float(smooth_logz)
    return {"q_fit": q_fit, "delta_log_evidence": delta_logz}


def run_psf_bank_case(
    validator: Any,
    observation: Any,
    full_config: dict,
    trial: Any,
    bank: PsfBank,
    fit_mode: str = "freed",
    dataset_kind: str = "asimov",
    background_treatment: str = "subtract_known",
    mask_bool_use: Optional[np.ndarray] = None,
    psf_truth_label: str = "observation",
    priors_config: Optional[dict] = None,
    mass_context: Any = None,
    clumpy_fit_parameterization: str = "host_free",
    include_anchors: bool = True,
    on_candidate: Any = None,
) -> PsfBankCaseResult:
    """Run one nonlinear validation case over a compatible PSF bank.

    Parameters
    ----------
    validator : `NonlinearMetricValidator`
        Nonlinear case executor.
    observation : `ObservationData`
        Observation being analyzed.
    full_config : `dict`
        Validated configuration compatible with ``bank``.
    trial : `SubhaloTrial`
        Physical subhalo trial.
    bank : `PsfBank`
        Marginalization candidates and optional anchor controls.
    fit_mode : `str`, optional
        Nonlinear fit mode.
    dataset_kind : `str`, optional
        Validation dataset kind.
    background_treatment : `str`, optional
        Dataset background treatment.
    mask_bool_use : `numpy.ndarray`, optional
        Boolean pixel-include mask.
    psf_truth_label : `str`, optional
        Observation PSF provenance label.
    priors_config : `dict`, optional
        Nonlinear prior-width overrides.
    mass_context : `MassMappingContext`, optional
        Required explicit mass context for freed fits.
    clumpy_fit_parameterization : `str`, optional
        Clumpy-source fit parameterization.
    include_anchors : `bool`, optional
        Whether to fit stored anchors as controls.
    on_candidate : callable, optional
        Callback receiving each completed nonlinear case result.

    Returns
    -------
    result : `PsfBankCaseResult`
        Combined bank summary, individual cases, controls, and flags.

    Raises
    ------
    ValueError
        Raised before fitting for a missing freed mass context, incompatible
        bank, pixel-scale mismatch, or corrupted kernel.

    Notes
    -----
    Kernel hashes pin only the stored shape and float64 bytes. Truth-anchor
    byte identity with the observation PSF requires the same configuration and
    code versions. A version mismatch is a soft diagnostic recorded as
    ``"bank_version_mismatch"`` rather than an execution error.
    """
    if fit_mode == "freed" and mass_context is None:
        raise ValueError(
            "freed mode requires mass_context from "
            "build_mass_mapping_context or "
            "build_mass_mapping_context_explicit"
        )
    version_mismatches = _validate_execution_inputs(
        observation,
        full_config,
        bank,
    )
    from .dataset_builder import imaging_from_observation

    candidate_results = []
    anchor_results = []
    fits = []
    callback_failed = False
    selected = list(bank.candidates)
    if include_anchors:
        selected.extend(bank.anchors)
    for index, candidate in enumerate(selected):
        psf_case = f"bank:{bank.bank_id}:{candidate.label}"
        dataset, metadata = imaging_from_observation(
            observation,
            psf_for_fit=_wrapped_candidate_kernel(candidate),
            dataset_kind=dataset_kind,
            background_treatment=background_treatment,
            mask_bool_use=mask_bool_use,
            psf_truth_label=psf_truth_label,
            psf_fit_label=psf_case,
        )
        case = validator.validate_case(
            dataset,
            metadata,
            full_config,
            trial,
            fit_mode=fit_mode,
            psf_case=psf_case,
            priors_config=priors_config,
            mass_context=mass_context,
            clumpy_fit_parameterization=clumpy_fit_parameterization,
            smooth_result=None,
        )
        if on_candidate is not None:
            try:
                on_candidate(case)
            except Exception:
                callback_failed = True
        if index < len(bank.candidates):
            candidate_results.append(case)
            fits.append(PsfBankCandidateFit(
                label=candidate.label,
                amplitude_rms_nm=candidate.amplitude_rms_nm,
                log_l_smooth=case.smooth_fit.log_likelihood_max,
                log_l_subhalo=case.subhalo_fit.log_likelihood_max,
                log_evidence_smooth=case.smooth_fit.log_evidence,
                log_evidence_subhalo=case.subhalo_fit.log_evidence,
                success=(
                    case.smooth_fit.status == "success"
                    and case.subhalo_fit.status == "success"
                ),
            ))
        else:
            anchor_results.append(case)

    summary = combine_psf_bank_fits(fits, fit_mode=fit_mode)
    anchor_diagnostics = {
        case.psf_case.rsplit(":", 1)[-1]: _anchor_diagnostic(case)
        for case in anchor_results
    }
    quality_flags = []
    if any(not _likelihood_usable(candidate) for candidate in fits):
        quality_flags.append("bank_candidate_failed")
    if any(
        _likelihood_usable(candidate) and not _evidence_usable(candidate)
        for candidate in fits
    ):
        quality_flags.append("bank_missing_evidence")
    if any(
        case.smooth_fit.status != "success"
        or case.subhalo_fit.status != "success"
        or not _usable(case.smooth_fit.log_likelihood_max)
        or not _usable(case.subhalo_fit.log_likelihood_max)
        for case in anchor_results
    ):
        quality_flags.append("bank_anchor_failed")
    if callback_failed:
        quality_flags.append("bank_on_candidate_callback_failed")
    if version_mismatches:
        quality_flags.append("bank_version_mismatch")
    return PsfBankCaseResult(
        bank_id=bank.bank_id,
        case_id=trial.case_id,
        fit_mode=fit_mode,
        summary=summary,
        candidate_results=candidate_results,
        anchor_results=anchor_results,
        anchor_diagnostics=anchor_diagnostics,
        quality_flags=quality_flags,
        bank_provenance=_bank_provenance(bank, version_mismatches),
    )
