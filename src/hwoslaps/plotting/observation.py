"""Current Observation images and expected source signal-to-noise on caller Axes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ._axes import axes_or_new, pixel_extent

if TYPE_CHECKING:
    from ..observation.observation import Observation

__all__ = ["plot_observation"]

_LABELS = {"data": "Observed data (ADU)", "expected": "Expected data (ADU)",
           "noise": "Noise standard deviation (ADU)", "snr": "Expected source S/N"}


def plot_observation(observation: Observation, quantity: Literal["data", "expected", "noise", "snr"], *, ax=None) -> "Axes":
    """Show one current image; S/N is expected source-plane signal over all-light noise.

    S/N uses exposure.signal_adu(source plane) / noise_map_adu for both expected
    and noisy observations. It is not noisy data divided by the noise map.
    Native detector rows start at positive y, so images use origin upper.
    """
    if quantity not in _LABELS:
        raise ValueError(f"quantity must be one of {tuple(_LABELS)}, got {quantity!r}")
    if quantity == "snr":
        values = observation.exposure.signal_adu(observation.light_rate_by_plane_e_per_s["source"]) \
            / observation.noise_map_adu
    else:
        values = {"data": observation.data_adu, "expected": observation.expected_adu,
                  "noise": observation.noise_map_adu}[quantity]
    ax = axes_or_new(ax)
    ax.imshow(values, origin="upper", extent=pixel_extent(observation.grid.shape,
                                                         observation.grid.pixel_scale_arcsec),
              interpolation="nearest")
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title(_LABELS[quantity])
    return ax
