"""Execute one nonlinear fit under an explicit PSF mismatch."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...psf.mismatch import (
    PsfMismatchSpec,
    _aberrations_to_wire,
    _flat_int_map_to_wire,
    _kernel_sha256,
    _nested_int_map_to_wire,
    build_psf_mismatch_spec,
    generate_fit_psf,
)
from .likelihood_metrics import (
    SCDD_Q_THRESHOLD,
    STRONG_EVIDENCE_DELTA_LOG_Z_THRESHOLD,
)


__all__ = (
    "PsfMismatchSpec",
    "PsfMismatchCaseResult",
    "build_psf_mismatch_spec",
    "run_psf_mismatch_case",
)


def _usable(value: Any) -> bool:
    """Return whether a statistic is present and finite."""
    return value is not None and math.isfinite(float(value))


@dataclass
class PsfMismatchCaseResult:
    """Result of one nonlinear fit-side PSF mismatch case.

    Parameters
    ----------
    case_id : `str`
        Stable nonlinear case identifier.
    delta_id : `str`
        Stable fit-PSF mismatch identity.
    mode : `str`
        Fit-PSF mismatch mode.
    fit_mode : `str`
        Nonlinear subhalo fit mode.
    psf_truth_label : `str`
        Observation PSF label.
    psf_fit_label : `str`
        Fit-side mismatch label.
    q_fit : `float`, optional
        Non-negative likelihood-ratio statistic.
    delta_log_evidence : `float`, optional
        Subhalo-minus-smooth log evidence.
    log_l_smooth : `float`, optional
        Smooth maximum log likelihood.
    log_l_subhalo : `float`, optional
        Subhalo maximum log likelihood.
    log_evidence_smooth : `float`, optional
        Smooth log evidence.
    log_evidence_subhalo : `float`, optional
        Subhalo log evidence.
    smooth_status : `str`
        Smooth fit status.
    subhalo_status : `str`
        Subhalo fit status.
    detected_fit : `bool`, optional
        Fixed-calibration SCDD detection state.
    detected_evidence : `bool`, optional
        Strong-evidence detection state.
    requested_amplitude_rms_nm : `float`, optional
        Requested additive mismatch RMS.
    measured_draw_rms_nm : `float`, optional
        Measured additive mismatch RMS.
    family : `str`, optional
        Delta draw family.
    seed : `int`, optional
        Delta draw seed.
    prior_table_path : `str`, optional
        Resolved prior-table path.
    prior_table_sha256 : `str`, optional
        Prior-table content digest.
    draw_aberrations_wire : `dict`, optional
        Typed JSON representation of the raw delta draw.
    orthonormal_segment_wire : `list`, optional
        Typed JSON representation of segment orthonormal coefficients.
    orthonormal_global_wire : `list`, optional
        Typed JSON representation of global orthonormal coefficients.
    truth_psf_config_hash : `str`
        Canonical truth PSF configuration hash.
    fit_psf_config_hash : `str`
        Canonical fit PSF configuration hash.
    kernel_sha256 : `str`
        Shape-aware fit-kernel digest.
    truth_kernel_sha256 : `str`
        Shape-aware observation-kernel digest.
    kernel_pixel_scale : `float`
        Fit-kernel pixel scale.
    fit_psf_total_rms_nm : `float`
        Measured total fit-side wavefront RMS.
    quality_flags : `list`
        Fit failure and unusable-statistic flags.
    provenance : `dict`
        Version, label, and execution inputs.
    case : `NonlinearCaseResult`
        Embedded full nonlinear result, excluded from slim serialization.

    Notes
    -----
    Result-level provenance records package versions only. Git revision and
    dirty-worktree provenance are captured by the run-level provenance module
    for production executions.
    """

    case_id: str
    delta_id: str
    mode: str
    fit_mode: str
    psf_truth_label: str
    psf_fit_label: str
    q_fit: Optional[float]
    delta_log_evidence: Optional[float]
    log_l_smooth: Optional[float]
    log_l_subhalo: Optional[float]
    log_evidence_smooth: Optional[float]
    log_evidence_subhalo: Optional[float]
    smooth_status: str
    subhalo_status: str
    detected_fit: Optional[bool]
    detected_evidence: Optional[bool]
    requested_amplitude_rms_nm: Optional[float]
    measured_draw_rms_nm: Optional[float]
    family: Optional[str]
    seed: Optional[int]
    prior_table_path: Optional[str]
    prior_table_sha256: Optional[str]
    draw_aberrations_wire: Optional[dict]
    orthonormal_segment_wire: Optional[list]
    orthonormal_global_wire: Optional[list]
    truth_psf_config_hash: str
    fit_psf_config_hash: str
    kernel_sha256: str
    truth_kernel_sha256: str
    kernel_pixel_scale: float
    fit_psf_total_rms_nm: float
    quality_flags: list
    provenance: dict
    case: Any

    def to_dict(self) -> dict:
        """Convert the result to a compact JSON-compatible dictionary.

        Returns
        -------
        payload : `dict`
            Mismatch statistics and provenance without the embedded case.
        """
        payload = {
            key: deepcopy(value)
            for key, value in self.__dict__.items()
            if key != "case"
        }
        from .output_schema import _json_safe

        return _json_safe(payload)

    def write_json(self, path: Any) -> None:
        """Write the compact mismatch result as formatted JSON.

        Parameters
        ----------
        path : path-like
            Destination JSON path.
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)


def _quality_flags(case: Any) -> list:
    """Return status and statistic usability diagnostics for one case."""
    fits = (case.smooth_fit, case.subhalo_fit)
    flags = []
    if any(fit.status != "success" for fit in fits):
        flags.append("fit_failed")
    if any(
        fit.status == "success" and not _usable(fit.log_likelihood_max)
        for fit in fits
    ):
        flags.append("likelihood_unusable")
    if any(
        fit.status == "success" and not _usable(fit.log_evidence)
        for fit in fits
    ):
        flags.append("evidence_unusable")
    return flags


def run_psf_mismatch_case(
    validator: Any,
    observation: Any,
    full_config: dict,
    trial: Any,
    *,
    fit_mode: str = "freed",
    dataset_kind: str = "asimov",
    background_treatment: str = "subtract_known",
    mask_bool_use: Optional[np.ndarray] = None,
    psf_truth_label: str = "observation",
    priors_config: Optional[dict] = None,
    mass_context: Any = None,
) -> PsfMismatchCaseResult:
    """Run one nonlinear case with a deterministic mismatched fit PSF.

    Parameters
    ----------
    validator : `NonlinearMetricValidator`
        Nonlinear case executor.
    observation : `ObservationData`
        Observation being analyzed.
    full_config : `dict`
        Validated delta or explicit fit-PSF configuration.
    trial : `SubhaloTrial`
        Physical subhalo trial.
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
    Returns
    -------
    result : `PsfMismatchCaseResult`
        Mismatch metadata, paired-fit statistics, and the embedded case.

    Raises
    ------
    ValueError
        Raised before fitting for missing freed context, truth-kernel
        mismatch, or kernel pixel-scale mismatch.
    """
    if fit_mode == "freed" and mass_context is None:
        raise ValueError(
            "freed mode requires mass_context from "
            "build_mass_mapping_context or "
            "build_mass_mapping_context_explicit"
        )
    from ...provenance import config_hash, revision_provenance

    # Captured at entry, before spec construction and fitting, so the
    # record describes the source and configuration the case ran with.
    entry_revision = revision_provenance()
    entry_config_hash = config_hash(full_config)
    spec = build_psf_mismatch_spec(full_config)
    truth_kernel, truth_scale, truth_total_rms = generate_fit_psf(
        full_config["psf"],
        full_config,
    )
    from ...psf.utils import pyauto_kernel_native

    observation_kernel = np.ascontiguousarray(
        pyauto_kernel_native(observation.psf),
        dtype=np.float64,
    )
    truth_kernel_sha256 = _kernel_sha256(truth_kernel)
    observation_kernel_sha256 = _kernel_sha256(observation_kernel)
    if truth_kernel_sha256 != observation_kernel_sha256:
        raise ValueError(
            "observation was not generated by full_config['psf']: "
            f"regenerated truth kernel sha256 {truth_kernel_sha256}, "
            f"observation kernel sha256 {observation_kernel_sha256}"
        )

    if (
        spec.mode == "delta"
        and spec.requested_amplitude_rms_nm == 0.0
    ):
        fit_kernel = truth_kernel
        kernel_pixel_scale = truth_scale
        fit_total_rms = truth_total_rms
    else:
        fit_kernel, kernel_pixel_scale, fit_total_rms = generate_fit_psf(
            spec.fit_psf_config,
            full_config,
        )
    if not np.isclose(
        kernel_pixel_scale,
        observation.pixel_scale,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "PSF mismatch fit kernel pixel scale is incompatible with the "
            "observation"
        )

    from ...observation.generator import _ensure_odd_kernel
    from ...psf.utils import make_pyauto_convolver, make_pyauto_kernel
    from .dataset_builder import (
        fitted_kernel_sha256,
        imaging_from_observation,
    )

    wrapped = make_pyauto_kernel(
        values=fit_kernel,
        pixel_scales=kernel_pixel_scale,
        normalize=False,
    )
    wrapped = make_pyauto_convolver(_ensure_odd_kernel(wrapped))
    psf_case = f"{spec.mode}:{spec.delta_id}"
    dataset, metadata = imaging_from_observation(
        observation,
        psf_for_fit=wrapped,
        dataset_kind=dataset_kind,
        background_treatment=background_treatment,
        mask_bool_use=mask_bool_use,
        psf_truth_label=psf_truth_label,
        psf_fit_label=psf_case,
    )
    wrapped_kernel_sha256 = fitted_kernel_sha256(
        dataset,
        wrapped,
        kernel_pixel_scale,
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
        smooth_result=None,
        expected_psf_fit_sha256=wrapped_kernel_sha256,
    )

    smooth_log_l = case.smooth_fit.log_likelihood_max
    subhalo_log_l = case.subhalo_fit.log_likelihood_max
    paired_success = (
        case.smooth_fit.status == "success"
        and case.subhalo_fit.status == "success"
    )
    q_fit = None
    if (
        paired_success
        and _usable(smooth_log_l)
        and _usable(subhalo_log_l)
    ):
        q_fit = max(
            0.0,
            2.0*(float(subhalo_log_l) - float(smooth_log_l)),
        )
    smooth_logz = case.smooth_fit.log_evidence
    subhalo_logz = case.subhalo_fit.log_evidence
    delta_logz = None
    if paired_success and _usable(smooth_logz) and _usable(subhalo_logz):
        delta_logz = float(subhalo_logz) - float(smooth_logz)
    detected_fit = (
        None if q_fit is None else q_fit >= SCDD_Q_THRESHOLD
    )
    if fit_mode == "freed":
        detected_fit = None
    detected_evidence = (
        None
        if delta_logz is None
        else delta_logz > STRONG_EVIDENCE_DELTA_LOG_Z_THRESHOLD
    )
    provenance = {
        "versions": deepcopy(spec.versions),
        "revision": entry_revision,
        "config_hash": entry_config_hash,
        "psf_truth_label": psf_truth_label,
        "psf_fit_label": psf_case,
        "fit_mode": fit_mode,
        "dataset_kind": dataset_kind,
        "background_treatment": background_treatment,
    }
    return PsfMismatchCaseResult(
        case_id=str(case.case_id),
        delta_id=spec.delta_id,
        mode=spec.mode,
        fit_mode=fit_mode,
        psf_truth_label=psf_truth_label,
        psf_fit_label=psf_case,
        q_fit=q_fit,
        delta_log_evidence=delta_logz,
        log_l_smooth=smooth_log_l,
        log_l_subhalo=subhalo_log_l,
        log_evidence_smooth=smooth_logz,
        log_evidence_subhalo=subhalo_logz,
        smooth_status=case.smooth_fit.status,
        subhalo_status=case.subhalo_fit.status,
        detected_fit=detected_fit,
        detected_evidence=detected_evidence,
        requested_amplitude_rms_nm=spec.requested_amplitude_rms_nm,
        measured_draw_rms_nm=spec.measured_draw_rms_nm,
        family=spec.family,
        seed=spec.seed,
        prior_table_path=spec.prior_table_path,
        prior_table_sha256=spec.prior_table_sha256,
        draw_aberrations_wire=(
            None
            if spec.draw_aberrations is None
            else _aberrations_to_wire(spec.draw_aberrations)
        ),
        orthonormal_segment_wire=_nested_int_map_to_wire(
            spec.orthonormal_segment
        ),
        orthonormal_global_wire=_flat_int_map_to_wire(
            spec.orthonormal_global
        ),
        truth_psf_config_hash=spec.truth_psf_config_hash,
        fit_psf_config_hash=spec.fit_psf_config_hash,
        kernel_sha256=wrapped_kernel_sha256,
        truth_kernel_sha256=truth_kernel_sha256,
        kernel_pixel_scale=float(kernel_pixel_scale),
        fit_psf_total_rms_nm=float(fit_total_rms),
        quality_flags=_quality_flags(case),
        provenance=provenance,
        case=case,
    )
