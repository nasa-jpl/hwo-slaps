"""Geometry conversions in AutoGalaxy's conventions, without importing a backend.

Angles are degrees counter-clockwise from +x; pairs are returned in AutoGalaxy's component
order. The first three functions repeat the operation order of AutoGalaxy's ``convert``
module (``ell_comps_from``, ``shear_gamma_1_2_from``, ``multipole_comps_from``), so their
values are bitwise those of the backend.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ell_comps_from", "multipole_components_from", "polar_offset", "shear_components_from"]


def ell_comps_from(axis_ratio: float, angle_deg: float) -> tuple[float, float]:
    """(e1, e2) = f (sin 2 phi, cos 2 phi) with f = (1 - q) / (1 + q), phi the major-axis angle."""
    angle = angle_deg * (np.pi / 180.0)
    factor = (1 - axis_ratio) / (1 + axis_ratio)
    return float(factor * np.sin(2 * angle)), float(factor * np.cos(2 * angle))


def shear_components_from(magnitude: float, angle_deg: float) -> tuple[float, float]:
    """(gamma_1, gamma_2) = g (cos 2 phi, sin 2 phi): the opposite component order to ``ell_comps_from``."""
    return (float(magnitude * np.cos(2 * angle_deg * np.pi / 180.0)),
            float(magnitude * np.sin(2 * angle_deg * np.pi / 180.0)))


def multipole_components_from(strength: float, angle_deg: float, order: int) -> tuple[float, float]:
    """(c1, c2) = k_m (sin m phi_m, cos m phi_m) of a multipole of order ``m`` with angle ``phi_m``."""
    angle = angle_deg * float(order) * (np.pi / 180.0)
    return float(strength * np.sin(angle)), float(strength * np.cos(angle))


def polar_offset(radius: float, angle_deg: float, centre_yx: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    """(y, x) at ``radius`` (arcsec) and ``angle_deg`` (from +x toward +y) about ``centre_yx``."""
    angle = np.deg2rad(angle_deg)
    return float(centre_yx[0] + radius * np.sin(angle)), float(centre_yx[1] + radius * np.cos(angle))
