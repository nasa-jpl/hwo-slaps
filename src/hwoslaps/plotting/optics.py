"""Detector kernels and pupil transmission; plotting never changes their values."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .axes import axes_or_new, pixel_extent

if TYPE_CHECKING:
    from ..optics.kernels import DetectorPSF
    from ..optics.pupils import Pupil

__all__ = ["plot_kernel", "plot_pupil"]


def plot_kernel(psf: DetectorPSF, *, log: bool, ax=None) -> "Axes":
    """Show the detector kernel with linear or logarithmic colour normalization.

    Kernel values are preserved, with no peak normalization or added floor.
    Zero kernel values are masked in logarithmic colour space. Native row zero
    has positive y, matching the convolved point image.
    """
    if not isinstance(log, (bool, np.bool_)):
        raise ValueError("log must be boolean")
    values = psf.kernel
    options = {}
    if log:
        from matplotlib.colors import LogNorm
        positive = values[values > 0]
        if positive.size == 0:
            raise ValueError("a logarithmic kernel plot needs positive kernel values")
        options["norm"] = LogNorm(vmin=float(positive.min()), vmax=float(positive.max()))
        values = np.ma.masked_less_equal(values, 0)
    ax = axes_or_new(ax)
    ax.imshow(values, origin="upper", extent=pixel_extent(psf.kernel.shape, psf.pixel_scale_arcsec),
              interpolation="nearest", **options)
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Detector kernel")
    return ax


def plot_pupil(pupil: Pupil, *, ax=None) -> "Axes":
    """Show grey amplitude transmission on the physical pupil grid in metres."""
    ax = axes_or_new(ax)
    ax.imshow(pupil.transmission.shaped, origin="lower",
              extent=pixel_extent((pupil.spec.pixels, pupil.spec.pixels),
                                  pupil.spec.diameter_m / pupil.spec.pixels),
              interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Pupil amplitude transmission")
    return ax
