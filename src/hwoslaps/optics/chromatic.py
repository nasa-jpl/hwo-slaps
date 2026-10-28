"""Photon-bin weights and effective detector kernels for a captured spectrum.

Rates share one whole-band logarithmic scale: the mathematical bin integrals
are ``rates * exp(log_rate_scale)``. Their normalization never needs that
potentially unrepresentable common factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..spectra.bandpass import Bandpass, bin_integrals
from ..spectra.sed import SED
from .kernels import DetectorPSF, PIXEL_SCALE_ATOL_ARCSEC

__all__ = ["NoSpectralResponse", "SpectralWeights", "effective_kernel", "sed_weights"]


class NoSpectralResponse(ValueError):
    """The declared spectral shape has no photons through the captured band."""


@dataclass(frozen=True, eq=False)
class SpectralWeights:
    """Finite common-scale photon integrals in the provider's wavelength bins."""

    wavelengths_m: np.ndarray
    bin_edges_m: np.ndarray
    rates: np.ndarray
    log_rate_scale: float

    def __post_init__(self) -> None:
        if any(np.iscomplexobj(value) for value in (self.wavelengths_m, self.bin_edges_m, self.rates)):
            raise ValueError("spectral nodes, edges and rates must be real")
        if any(np.asarray(value).dtype.kind == "b" for value in (self.wavelengths_m, self.bin_edges_m)):
            raise ValueError("spectral wavelengths must be real numbers, not booleans")
        nodes, edges, rates = (np.array(value, dtype=float, copy=True)
                               for value in (self.wavelengths_m, self.bin_edges_m, self.rates))
        if (nodes.ndim != 1 or nodes.size == 0 or not np.all(np.isfinite(nodes))
                or np.any(nodes <= 0.0) or np.any(np.diff(nodes) <= 0.0)):
            raise ValueError("spectral nodes must be a positive finite strictly increasing vector")
        if (edges.shape != (nodes.size + 1,) or not np.all(np.isfinite(edges))
                or np.any(edges <= 0.0) or np.any(np.diff(edges) < 0.0)):
            raise ValueError("spectral edges must be positive finite and non-decreasing, one more than nodes")
        if (rates.shape != nodes.shape or not np.all(np.isfinite(rates)) or np.any(rates < 0.0)
                or not np.isfinite(np.sum(rates)) or np.sum(rates) <= 0.0):
            raise ValueError("spectral rates must be finite non-negative bins with a positive finite sum")
        if (isinstance(self.log_rate_scale, (bool, np.bool_)) or not isinstance(self.log_rate_scale, Real)
                or not np.isfinite(self.log_rate_scale)):
            raise ValueError("log_rate_scale must be finite")
        for name, value in (("wavelengths_m", nodes), ("bin_edges_m", edges), ("rates", rates)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "log_rate_scale", float(self.log_rate_scale))

    @property
    def normalized(self) -> np.ndarray:
        normalized = self.rates / self.rates.sum()
        normalized.setflags(write=False)
        return normalized

    def to_mapping(self) -> dict[str, Any]:
        return {"wavelengths_m": self.wavelengths_m.tolist(), "bin_edges_m": self.bin_edges_m.tolist(),
                "rates": self.rates.tolist(), "log_rate_scale": self.log_rate_scale,
                "normalized": self.normalized.tolist(), "rate_measure": "throughput*fnu*dlnlambda",
                "rate_definition": "mathematical bin rates = rates * exp(log_rate_scale)"}


def sed_weights(bandpass: Bandpass, sed: SED, wavelengths_m: ArrayLike) -> SpectralWeights:
    if np.iscomplexobj(wavelengths_m) or np.asarray(wavelengths_m).dtype.kind == "b":
        raise ValueError("spectral wavelengths must be real numbers")
    nodes = np.asarray(wavelengths_m, dtype=float)
    edges = bandpass.bin_edges(nodes)
    wavelengths, throughput = bandpass.integration_grid(sed)
    log_shape = sed.log_fnu(wavelengths)
    if np.any(np.isnan(log_shape) | np.isposinf(log_shape)):
        raise ValueError("the spectral log measure must be finite; -inf denotes zero response")
    positive = np.isfinite(log_shape) & (throughput > 0.0)
    if not np.any(positive):
        raise NoSpectralResponse("the spectrum has no photons through the bandpass")
    logarithms = log_shape[positive] + np.log(throughput[positive])
    offset = float(np.max(logarithms))
    integrand = np.zeros_like(wavelengths)
    integrand[positive] = np.exp(logarithms - offset)
    rates = bin_integrals(integrand, wavelengths, edges)
    return SpectralWeights(nodes, edges, rates, offset)


def effective_kernel(kernels: Sequence[DetectorPSF], weights: SpectralWeights, pixel_scale_arcsec: float, *,
                     source: Mapping[str, Any]) -> DetectorPSF:
    """Combine the node kernels in order, preserving the exact one-node object."""
    nodes = tuple(kernels)
    if len(nodes) != weights.rates.size or not nodes or not all(isinstance(node, DetectorPSF) for node in nodes):
        raise ValueError("one DetectorPSF is required per spectral node")
    if (isinstance(pixel_scale_arcsec, (bool, np.bool_)) or not isinstance(pixel_scale_arcsec, Real)
            or not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0.0):
        raise ValueError("pixel_scale_arcsec must be positive and finite")
    for node in nodes:
        if node.shape != nodes[0].shape or abs(node.pixel_scale_arcsec - pixel_scale_arcsec) > PIXEL_SCALE_ATOL_ARCSEC:
            raise ValueError("node kernels must share their support and requested angular sampling")
    if len(nodes) == 1:
        return nodes[0]
    normalized = weights.normalized
    combined = normalized[0] * nodes[0].kernel
    for weight, node in zip(normalized[1:], nodes[1:], strict=True):
        combined = combined + weight * node.kernel
    provenance = dict(source)
    provenance.update(kind="effective", captured_power_fraction=None)
    return DetectorPSF.from_array(combined, pixel_scale_arcsec, normalize=False, source=provenance)
