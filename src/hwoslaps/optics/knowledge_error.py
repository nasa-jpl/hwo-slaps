"""Wavefront draws at an exact aperture RMS, for PSF truth states and for PSF knowledge errors.

A draw takes a direction from a mode-weight prior in the orthonormal aperture bases,
converts it to raw coefficients of the pupil, and scales them so that the piston-removed
OPD RMS over the illuminated pupil equals the requested amplitude. The scale is measured
through the phase at the provider's reference wavelength, so a draw is a function of the
sampled pupil and that wavelength; its coefficients are recorded in every identity.

A knowledge error is a draw added to the truth coefficients: the model PSF has
``truth + draw``. The addition is checked to keep the requested amplitude (adding a small
draw to huge truth coefficients can erase it in floating point).

On a pupil with dark segments only the active segments are drawn; the dark ones are
listed in the draw record and get no coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..config.checks import Integer, Key, Real, Table, Text
from ..identity import mapping_digest
from .aperture_basis import ApertureBasisTransform
from .kernels import deterministic_blas
from .mode_priors import (
    PRIOR_TABLE, ModeWeightPriorSpec, draw_combined_orthonormal, draw_global_orthonormal, draw_segment_orthonormal,
    load_prior, parse_prior,
)
from .wavefront import WavefrontBasis, WavefrontCoefficients

if TYPE_CHECKING:
    from .optical_psf import OpticalPSF

__all__ = [
    "DRAW_FAMILIES", "DRAW_TABLE", "RMS_RELATIVE_TOLERANCE", "KnowledgeErrorDraw", "WavefrontDraw",
    "WavefrontDrawSpec", "draw_knowledge_error", "draw_wavefront", "parse_wavefront_draw",
]

DRAW_FAMILIES: tuple[str, ...] = ("combined", "global", "segment")
"""``combined``: segment hexikes and global Zernikes; ``global`` and ``segment``: one side only."""

RMS_RELATIVE_TOLERANCE: float = 1.0e-9
"""A draw's measured RMS must equal the amplitude within ``RMS_RELATIVE_TOLERANCE * max(1, amplitude)`` nm."""


@dataclass(frozen=True)
class WavefrontDrawSpec:
    """A draw of ``family`` modes from ``prior`` at an aperture RMS of ``amplitude_rms_nm``, from ``seed``."""

    prior: ModeWeightPriorSpec
    amplitude_rms_nm: float
    seed: int
    family: Literal["combined", "global", "segment"]


DRAW_TABLE = Table((
    Key("prior", PRIOR_TABLE, "mode-weight prior: exactly one of packaged, path, power_law"),
    Key("amplitude_rms_nm", Real(min=0.0), "piston-removed OPD RMS of the draw over the illuminated pupil",
        unit="nm"),
    Key("seed", Integer(min=0), "seed of the draw's random numbers"),
    Key("family", Text(choices=DRAW_FAMILIES),
        "combined (segment hexikes and global Zernikes), global or segment", default="combined"),
))


def parse_wavefront_draw(mapping: Mapping[str, Any], path: str) -> WavefrontDrawSpec:
    """The draw specification of a ``draw`` mapping, read strictly through ``DRAW_TABLE``."""
    values = DRAW_TABLE.read(mapping, path)
    return WavefrontDrawSpec(parse_prior(values["prior"], f"{path}.prior" if path else "prior"),
                             values["amplitude_rms_nm"], values["seed"], values["family"])


@dataclass(frozen=True)
class WavefrontDraw:
    """A realized draw: its orthonormal direction, raw coefficients and measured RMS.

    ``orthonormal_segment`` is keyed by the pupil's active segments; ``dark_segments`` are
    the segments left out.
    """

    spec: WavefrontDrawSpec
    prior_digest: str
    orthonormal_segment: Mapping[int, Mapping[int, float]]
    orthonormal_global: Mapping[int, float]
    coefficients: WavefrontCoefficients
    measured_rms_nm: float
    dark_segments: tuple[int, ...]

    def to_mapping(self) -> dict[str, Any]:
        """The record: prior content digest (not its location), parameters, direction, coefficients."""
        return {
            "prior_digest": self.prior_digest,
            "amplitude_rms_nm": self.spec.amplitude_rms_nm,
            "seed": self.spec.seed,
            "family": self.spec.family,
            "orthonormal_segment": {s: dict(modes) for s, modes in self.orthonormal_segment.items()},
            "orthonormal_global": dict(self.orthonormal_global),
            "coefficients": self.coefficients.to_mapping(),
            "measured_rms_nm": self.measured_rms_nm,
            "dark_segments": list(self.dark_segments),
        }


def _tolerance(amplitude: float) -> float:
    return RMS_RELATIVE_TOLERANCE * max(1.0, amplitude)


def draw_wavefront(basis: WavefrontBasis, spec: WavefrontDrawSpec) -> WavefrontDraw:
    """Draw raw coefficients on ``basis.pupil`` whose aperture RMS equals ``spec.amplitude_rms_nm``."""
    prior, prior_digest = load_prior(spec.prior)
    pupil = basis.pupil
    amplitude = float(spec.amplitude_rms_nm)
    if spec.family not in DRAW_FAMILIES:
        raise ValueError(f"draw family must be one of {DRAW_FAMILIES}, got {spec.family!r}")
    if amplitude == 0.0:
        return WavefrontDraw(spec, prior_digest, {}, {}, WavefrontCoefficients.empty(), 0.0, pupil.dark_segments)
    uses_segments = spec.family in ("combined", "segment")
    uses_global = spec.family in ("combined", "global")
    if uses_segments and "segment_hexikes" not in basis.families:
        raise ValueError(f"a {spec.family} draw needs segment hexikes, which a {pupil.spec.kind} pupil does not have")
    if uses_segments and not pupil.active_segments:
        raise ValueError(f"a {spec.family} draw needs at least one active segment; every segment is dark")
    if uses_segments and not prior.segment_weights:
        raise ValueError(f"a {spec.family} draw needs segment weights; prior {prior.name!r} has none")
    if uses_global and not prior.global_weights:
        raise ValueError(f"a {spec.family} draw needs global weights; prior {prior.name!r} has none")
    with deterministic_blas():
        transform = ApertureBasisTransform(
            basis, global_nolls=sorted(prior.global_weights) if uses_global else (),
            segment_nolls=sorted(prior.segment_weights) if uses_segments else ())
        rng = np.random.default_rng(np.random.SeedSequence(spec.seed))
        segments = pupil.active_segments
        if spec.family == "combined":
            orthonormal_segment, orthonormal_global = draw_combined_orthonormal(rng, segments, prior, amplitude)
        elif spec.family == "global":
            orthonormal_segment, orthonormal_global = {}, draw_global_orthonormal(rng, prior, amplitude)
        else:
            orthonormal_segment, orthonormal_global = draw_segment_orthonormal(rng, segments, prior, amplitude), {}
        raw = transform.to_raw(segment=orthonormal_segment, global_=orthonormal_global)
        raw_rms = basis.aperture_rms_nm(raw)
        if raw_rms == 0.0:
            raise ValueError("the drawn wavefront has zero aperture RMS and cannot be scaled to the amplitude")
        coefficients = raw.scaled(amplitude / raw_rms)
        measured = basis.aperture_rms_nm(coefficients)
    if abs(measured - amplitude) > _tolerance(amplitude):
        raise ValueError(f"the scaled draw measures {measured!r} nm, not the requested {amplitude!r} nm")
    return WavefrontDraw(spec, prior_digest, orthonormal_segment, orthonormal_global, coefficients, measured,
                         pupil.dark_segments)


@dataclass(frozen=True)
class KnowledgeErrorDraw:
    """A PSF knowledge error: the draw, the truth and model coefficients, and the realized RMS.

    ``truth_provider_digest`` identifies the truth optics (pupil, sampling, wavelength,
    coefficients); ``digest()`` identifies the knowledge error by that and the draw
    parameters, with the prior by content.
    """

    draw: WavefrontDraw
    truth: WavefrontCoefficients
    model: WavefrontCoefficients
    effective_rms_nm: float
    truth_provider_digest: str

    def digest(self) -> str:
        spec = self.draw.spec
        return mapping_digest({
            "schema": "hwoslaps.knowledge_error.v1",
            "prior_digest": self.draw.prior_digest,
            "amplitude_rms_nm": spec.amplitude_rms_nm,
            "seed": spec.seed,
            "family": spec.family,
            "truth": self.truth_provider_digest,
        })

    def to_mapping(self) -> dict[str, Any]:
        return {
            "digest": self.digest(),
            "draw": self.draw.to_mapping(),
            "truth_provider_digest": self.truth_provider_digest,
            "model_coefficients_digest": self.model.digest(),
            "effective_rms_nm": self.effective_rms_nm,
        }


def draw_knowledge_error(truth: OpticalPSF, spec: WavefrontDrawSpec) -> KnowledgeErrorDraw:
    """The model coefficients ``truth + draw`` and the RMS the addition realizes."""
    basis = truth.basis
    draw = draw_wavefront(basis, spec)
    model = truth.coefficients.plus(draw.coefficients)
    amplitude = float(spec.amplitude_rms_nm)
    with deterministic_blas():
        effective = basis.aperture_rms_nm(model.minus(truth.coefficients))
    if amplitude > 0.0:
        if effective == 0.0:
            raise ValueError(f"adding the draw to the truth coefficients erased it in floating point: effective "
                             f"aperture RMS 0 for the requested {amplitude!r} nm")
        if abs(effective - amplitude) > _tolerance(amplitude):
            raise ValueError(f"adding the draw to the truth coefficients destroyed it in floating point: effective "
                             f"aperture RMS {effective!r} nm for the requested {amplitude!r} nm")
    return KnowledgeErrorDraw(draw, truth.coefficients, model, effective, mapping_digest(truth.to_mapping()))
