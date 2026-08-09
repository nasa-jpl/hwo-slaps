"""Prepared image assets and their AutoGalaxy light-profile evaluator.

Editing an asset file in place after it has been loaded is unsupported.  The
absolute-path loader cache deliberately keeps one immutable in-process view;
the recorded SHA-256 prefix makes the loaded content visible in provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from typing import Any, Dict, Tuple

import autoarray as aa
import autogalaxy as ag
import numpy as np
from scipy.interpolate import RectBivariateSpline

from autogalaxy.profiles.light.decorators import check_operated_only


_ASSET_KEYS = {'sb', 'pixel_scale_arcsec', 'metadata_json'}
_FORMAT_VERSION = 1
_MIN_ASSET_DIMENSION = 8
_MAX_ASSET_DIMENSION = 4096


def _as_float(value: Any) -> Any:
    """Convert concrete scalars to float while preserving traced values."""
    if isinstance(value, (int, float, np.generic)):
        return float(value)
    return value


@dataclass(frozen=True)
class SourceImageAsset:
    """Validated source-image asset and its content hash.

    Parameters
    ----------
    sb : `numpy.ndarray`
        Unit-integral surface-brightness samples with shape ``(ny, nx)``.
    pixel_scale_arcsec : `float`
        Pixel-centre spacing in arcseconds.
    metadata : `dict`
        Parsed asset metadata.
    sha256_16 : `str`
        First 16 hexadecimal characters of the asset file SHA-256.
    """

    sb: np.ndarray
    pixel_scale_arcsec: float
    metadata: Dict[str, Any]
    sha256_16: str


def _file_sha256_16(path: str) -> str:
    """Return the first 16 hexadecimal characters of a file SHA-256."""
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()[:16]


def _metadata_from_array(value: np.ndarray, path: str) -> Dict[str, Any]:
    """Parse and validate the scalar JSON metadata array."""
    metadata_array = np.asarray(value)
    if metadata_array.ndim != 0 or metadata_array.dtype.kind not in {'U', 'S'}:
        raise ValueError(
            f"Source image asset {path} metadata_json must be a "
            "zero-dimensional numpy string array"
        )
    encoded = metadata_array.item()
    if isinstance(encoded, bytes):
        encoded = encoded.decode('utf-8')
    try:
        metadata = json.loads(encoded)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Source image asset {path} metadata_json must contain valid JSON"
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Source image asset {path} metadata_json must contain a JSON dict"
        )
    format_version = metadata.get('format_version')
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version != _FORMAT_VERSION
    ):
        raise ValueError(
            f"Source image asset {path} metadata format_version must be the "
            f"integer {_FORMAT_VERSION}"
        )
    if not isinstance(metadata.get('provenance'), dict):
        raise ValueError(
            f"Source image asset {path} metadata provenance must be a dict"
        )
    return metadata


@lru_cache(maxsize=None)
def _load_source_image_asset_absolute(path: str) -> SourceImageAsset:
    """Load one source-image asset using an already absolute cache key."""
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if keys != _ASSET_KEYS:
                raise ValueError(
                    f"Source image asset {path} must contain exactly these keys: "
                    "metadata_json, pixel_scale_arcsec, sb; "
                    f"found {sorted(keys)}"
                )
            try:
                sb = np.asarray(data['sb'], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Source image asset {path} sb must be coercible to float64"
                ) from exc
            pixel_scale_array = np.asarray(data['pixel_scale_arcsec'])
            if (
                pixel_scale_array.ndim != 0
                or pixel_scale_array.dtype != np.dtype(np.float64)
            ):
                raise ValueError(
                    f"Source image asset {path} pixel_scale_arcsec must be "
                    "a float64 scalar"
                )
            pixel_scale = float(pixel_scale_array)
            metadata = _metadata_from_array(data['metadata_json'], path)
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith('Source image asset'):
            raise
        raise ValueError(f"Could not load source image asset {path}: {exc}") from exc

    if sb.ndim != 2:
        raise ValueError(f"Source image asset {path} sb must be a 2D array")
    if any(
        dimension < _MIN_ASSET_DIMENSION or dimension > _MAX_ASSET_DIMENSION
        for dimension in sb.shape
    ):
        raise ValueError(
            f"Source image asset {path} dimensions must be between "
            f"{_MIN_ASSET_DIMENSION} and {_MAX_ASSET_DIMENSION} pixels"
        )
    if not np.all(np.isfinite(sb)):
        raise ValueError(f"Source image asset {path} sb values must be finite")
    if np.any(sb < 0.0):
        raise ValueError(f"Source image asset {path} sb values must be non-negative")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise ValueError(
            f"Source image asset {path} pixel_scale_arcsec must be positive and finite"
        )
    integral = pixel_scale**2 * float(sb.sum())
    if not np.isclose(integral, 1.0, rtol=1.0e-8, atol=0.0):
        raise ValueError(
            f"Source image asset {path} sb must be normalized so "
            "pixel_scale_arcsec**2 * sb.sum() == 1"
        )
    sb.setflags(write=False)
    return SourceImageAsset(
        sb=sb,
        pixel_scale_arcsec=pixel_scale,
        metadata=metadata,
        sha256_16=_file_sha256_16(path),
    )


def load_source_image_asset(path) -> SourceImageAsset:
    """Load and validate a prepared source-image asset.

    Parameters
    ----------
    path : `str` or `os.PathLike`
        Path to the ``.npz`` source-image asset.

    Returns
    -------
    asset : `SourceImageAsset`
        Validated samples, scale, metadata, and file hash.  Repeated loads of
        the same absolute path return the same cached object.

    Notes
    -----
    Editing an asset in place during a process's lifetime is unsupported.
    """
    absolute_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    return _load_source_image_asset_absolute(absolute_path)


class ImageSource(ag.LightProfile):
    """Pixel-grid source light profile with zero-padded bilinear evaluation.

    Parameters
    ----------
    centre : `tuple` of `float`
        Sky ``(y, x)`` centre in arcseconds.
    rotation_deg : `float`
        Counterclockwise rotation of the image on the sky in degrees.
    pixel_scale_arcsec : `float`
        Unscaled pixel-centre spacing in arcseconds.
    sb : `numpy.ndarray`
        Unit-integral surface-brightness samples in origin-lower orientation.
    total_flux : `float`
        Intrinsic flux normalization at unit flux and size scales.
    flux_scale : `float`
        Multiplicative brightness scale.
    size_scale : `float`
        Similarity scale applied at fixed surface brightness.
    """

    def __init__(
        self,
        centre: Tuple[float, float],
        rotation_deg: float,
        pixel_scale_arcsec: float,
        sb: np.ndarray,
        total_flux: float,
        flux_scale: float,
        size_scale: float,
    ):
        total_flux = _as_float(total_flux)
        flux_scale = _as_float(flux_scale)
        size_scale = _as_float(size_scale)
        super().__init__(
            centre=tuple(_as_float(value) for value in centre),
            ell_comps=(0.0, 0.0),
            intensity=total_flux * flux_scale,
        )
        self.rotation_deg = _as_float(rotation_deg)
        self.pixel_scale_arcsec = _as_float(pixel_scale_arcsec)
        self.sb = np.asarray(sb, dtype=np.float64)
        if self.sb.ndim != 2:
            raise ValueError('ImageSource sb must be a 2D array')
        self.total_flux = total_flux
        self.flux_scale = flux_scale
        self.size_scale = size_scale
        self._padded = np.pad(self.sb, 1, mode='constant')
        self._spline = None

    @classmethod
    def from_asset(
        cls,
        asset: SourceImageAsset,
        centre: Tuple[float, float],
        rotation_deg: float,
        total_flux: float,
        flux_scale: float,
        size_scale: float,
    ) -> 'ImageSource':
        """Construct a profile from a validated asset.

        Parameters
        ----------
        asset : `SourceImageAsset`
            Prepared unit-integral source-image asset.
        centre : `tuple` of `float`
            Sky ``(y, x)`` centre in arcseconds.
        rotation_deg : `float`
            Counterclockwise image rotation in degrees.
        total_flux : `float`
            Intrinsic flux normalization.
        flux_scale : `float`
            Multiplicative brightness scale.
        size_scale : `float`
            Similarity scale at fixed brightness.

        Returns
        -------
        profile : `ImageSource`
            Image-source profile using the asset samples and pixel scale.
        """
        return cls(
            centre=centre,
            rotation_deg=rotation_deg,
            pixel_scale_arcsec=asset.pixel_scale_arcsec,
            sb=asset.sb,
            total_flux=total_flux,
            flux_scale=flux_scale,
            size_scale=size_scale,
        )

    def _spline_from_samples(self) -> RectBivariateSpline:
        """Return the lazily built spline over the one-pixel zero pad."""
        if self._spline is None:
            row_coords = np.arange(-1, self.sb.shape[0] + 1, dtype=float)
            col_coords = np.arange(-1, self.sb.shape[1] + 1, dtype=float)
            self._spline = RectBivariateSpline(
                row_coords,
                col_coords,
                self._padded,
                kx=1,
                ky=1,
            )
        return self._spline

    @aa.over_sample
    @aa.decorators.to_array
    @check_operated_only
    @aa.decorators.transform
    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only=None,
        **kwargs,
    ) -> aa.Array2D:
        """Evaluate zero-padded bilinear surface brightness on a sky grid.

        Parameters
        ----------
        grid : `autoarray.type.Grid2DLike`
            Sky ``(y, x)`` coordinates.  The standard profile decorators
            translate these coordinates to ``centre`` before evaluation.

        Returns
        -------
        image : `autoarray.Array2D`
            Surface brightness at every input coordinate.
        """
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
            in_bounds = (
                (rows >= -1.0)
                & (rows <= self.sb.shape[0])
                & (cols >= -1.0)
                & (cols <= self.sb.shape[1])
            )
            brightness = np.zeros(rows.shape, dtype=float)
            if np.any(in_bounds):
                brightness[in_bounds] = self._spline_from_samples().ev(
                    rows[in_bounds], cols[in_bounds]
                )
            return self.total_flux * self.flux_scale * brightness

        grid_values = grid.array if hasattr(grid, 'array') else grid
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
        in_bounds = (
            (rows >= -1.0)
            & (rows <= self.sb.shape[0])
            & (cols >= -1.0)
            & (cols <= self.sb.shape[1])
        )

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
        lower = (
            (1.0 - col_weight) * padded[row_lower_safe, col_lower_safe]
            + col_weight * padded[row_lower_safe, col_upper_safe]
        )
        upper = (
            (1.0 - col_weight) * padded[row_upper_safe, col_lower_safe]
            + col_weight * padded[row_upper_safe, col_upper_safe]
        )
        brightness = (1.0 - row_weight) * lower + row_weight * upper
        brightness = xp.where(in_bounds, brightness, 0.0)
        return self.total_flux * self.flux_scale * brightness

    def image_2d_via_radii_from(self, grid_radii, xp=np, **kwargs):
        """Reject radial evaluation of non-radially-symmetric morphology."""
        raise NotImplementedError(
            'ImageSource is not radially symmetric; evaluate it on a 2D grid.'
        )

    def __getstate__(self):
        """Return pickle state without the reconstructible spline object."""
        state = self.__dict__.copy()
        state['_spline'] = None
        return state

    def __setstate__(self, state):
        """Restore pickle state with lazy spline construction enabled."""
        self.__dict__.update(state)
        if '_padded' not in self.__dict__:
            self._padded = np.pad(self.sb, 1, mode='constant')


__all__ = ['ImageSource', 'SourceImageAsset', 'load_source_image_asset']
