"""The AutoGalaxy light profile of an image asset (backend glue: importing it loads AutoGalaxy).

Numerics moved unchanged from 8fa6209 ``lensing/image_source.ImageSource``: the numpy path
interpolates with a linear ``RectBivariateSpline`` over the one-pixel zero pad, the array
namespace path (JAX fits) with the explicit bilinear expression. Both are paper paths.
"""

from __future__ import annotations

from typing import Any

import autoarray as aa
import autogalaxy as ag
import numpy as np
from autogalaxy.profiles.light.decorators import check_operated_only
from scipy.interpolate import RectBivariateSpline

from .image_source import ImageAsset

__all__ = ["ImageLightProfile"]


def _as_float(value: Any) -> Any:
    """Concrete scalars as float; traced values unchanged."""
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    return value


class ImageLightProfile(ag.LightProfile):
    """Pixel-grid light profile with zero-padded bilinear evaluation.

    The image is rotated counter-clockwise by ``rotation_deg`` about ``centre``, magnified by
    ``size_scale`` at fixed surface brightness, and scaled by ``total_flux * flux_scale``;
    its integral is ``total_flux * flux_scale * size_scale**2``.
    """

    def __init__(self, centre: tuple[float, float], rotation_deg: float, pixel_scale_arcsec: float, sb: np.ndarray,
                 total_flux: float, flux_scale: float, size_scale: float):
        total_flux = _as_float(total_flux)
        flux_scale = _as_float(flux_scale)
        size_scale = _as_float(size_scale)
        super().__init__(centre=tuple(_as_float(value) for value in centre), ell_comps=(0.0, 0.0),
                         intensity=total_flux * flux_scale)
        self.rotation_deg = _as_float(rotation_deg)
        self.pixel_scale_arcsec = _as_float(pixel_scale_arcsec)
        self.sb = np.asarray(sb, dtype=np.float64)
        if self.sb.ndim != 2:
            raise ValueError("ImageLightProfile sb must be a 2D array")
        self.total_flux = total_flux
        self.flux_scale = flux_scale
        self.size_scale = size_scale
        self._padded = np.pad(self.sb, 1, mode="constant")
        self._spline = None

    @classmethod
    def from_asset(cls, asset: ImageAsset, centre: tuple[float, float], rotation_deg: float, total_flux: float,
                   flux_scale: float, size_scale: float) -> ImageLightProfile:
        """The profile of a loaded asset."""
        return cls(centre=centre, rotation_deg=rotation_deg, pixel_scale_arcsec=asset.pixel_scale_arcsec, sb=asset.sb,
                   total_flux=total_flux, flux_scale=flux_scale, size_scale=size_scale)

    def _spline_from_samples(self) -> RectBivariateSpline:
        if self._spline is None:
            row_coords = np.arange(-1, self.sb.shape[0] + 1, dtype=float)
            col_coords = np.arange(-1, self.sb.shape[1] + 1, dtype=float)
            self._spline = RectBivariateSpline(row_coords, col_coords, self._padded, kx=1, ky=1)
        return self._spline

    @aa.over_sample
    @aa.decorators.to_array
    @check_operated_only
    @aa.decorators.transform
    def image_2d_from(self, grid: aa.type.Grid2DLike, xp=np, operated_only=None, **kwargs) -> aa.Array2D:
        """Zero-padded bilinear surface brightness at sky (y, x) coordinates (translated to ``centre``)."""
        if xp is np:
            relative = np.asarray(grid, dtype=float)
            dy = relative[:, 0]
            dx = relative[:, 1]
            theta = np.deg2rad(self.rotation_deg)
            cosine = np.cos(theta)
            sine = np.sin(theta)
            u = dx * cosine + dy * sine
            v = -dx * sine + dy * cosine
            row_c = (self.sb.shape[0] - 1) / 2.0
            col_c = (self.sb.shape[1] - 1) / 2.0
            scale = self.pixel_scale_arcsec * self.size_scale
            rows = v / scale + row_c
            cols = u / scale + col_c
            in_bounds = (rows >= -1.0) & (rows <= self.sb.shape[0]) & (cols >= -1.0) & (cols <= self.sb.shape[1])
            brightness = np.zeros(rows.shape, dtype=float)
            if np.any(in_bounds):
                brightness[in_bounds] = self._spline_from_samples().ev(rows[in_bounds], cols[in_bounds])
            return self.total_flux * self.flux_scale * brightness

        grid_values = grid.array if hasattr(grid, "array") else grid
        relative = xp.asarray(grid_values, dtype=float)
        dy = relative[:, 0]
        dx = relative[:, 1]
        theta = xp.deg2rad(self.rotation_deg)
        cosine = xp.cos(theta)
        sine = xp.sin(theta)
        u = dx * cosine + dy * sine
        v = -dx * sine + dy * cosine
        row_c = (self.sb.shape[0] - 1) / 2.0
        col_c = (self.sb.shape[1] - 1) / 2.0
        scale = self.pixel_scale_arcsec * self.size_scale
        rows = v / scale + row_c
        cols = u / scale + col_c
        in_bounds = (rows >= -1.0) & (rows <= self.sb.shape[0]) & (cols >= -1.0) & (cols <= self.sb.shape[1])

        padded = xp.asarray(self._padded)
        padded_rows = rows + 1.0
        padded_cols = cols + 1.0
        row_lower = xp.floor(padded_rows).astype(int)
        col_lower = xp.floor(padded_cols).astype(int)
        row_upper = row_lower + 1
        col_upper = col_lower + 1
        row_lower_safe = xp.clip(row_lower, 0, padded.shape[0] - 1)
        row_upper_safe = xp.clip(row_upper, 0, padded.shape[0] - 1)
        col_lower_safe = xp.clip(col_lower, 0, padded.shape[1] - 1)
        col_upper_safe = xp.clip(col_upper, 0, padded.shape[1] - 1)
        row_weight = padded_rows - row_lower
        col_weight = padded_cols - col_lower
        lower = ((1.0 - col_weight) * padded[row_lower_safe, col_lower_safe]
                 + col_weight * padded[row_lower_safe, col_upper_safe])
        upper = ((1.0 - col_weight) * padded[row_upper_safe, col_lower_safe]
                 + col_weight * padded[row_upper_safe, col_upper_safe])
        brightness = (1.0 - row_weight) * lower + row_weight * upper
        brightness = xp.where(in_bounds, brightness, 0.0)
        return self.total_flux * self.flux_scale * brightness

    def image_2d_via_radii_from(self, grid_radii, xp=np, **kwargs):
        """Refused: image morphology is not radially symmetric."""
        raise NotImplementedError("ImageLightProfile is not radially symmetric; evaluate it on a 2D grid.")

    def __getstate__(self):
        """Pickle state without the spline, which is rebuilt on first evaluation."""
        state = self.__dict__.copy()
        state["_spline"] = None
        return state
