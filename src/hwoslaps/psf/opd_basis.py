"""Sampled OPD bases and coefficient transforms for segmented pupils.

The JWST-derived mode-weight tables are defined in sequentially
QR-orthonormalized aperture bases. This module contains both the pure
sampled-basis functions used by the offline derivation and the runtime
change of basis needed before HCIPy applies raw Noll coefficients.
"""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np


def orthonormalize_basis(raw_basis, mask):
    """Orthonormalize a sampled basis over selected aperture pixels.

    Parameters
    ----------
    raw_basis : `numpy.ndarray`
        Basis with shape ``(n_modes, *map_shape)``.
    mask : `numpy.ndarray`
        Boolean aperture mask with shape ``map_shape``.

    Returns
    -------
    basis : `numpy.ndarray`
        Basis with the same shape as ``raw_basis`` and unit mean square per
        mode over ``mask``.
    """
    raw_basis = np.asarray(raw_basis, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if raw_basis.ndim != mask.ndim + 1 or raw_basis.shape[1:] != mask.shape:
        raise ValueError('raw_basis shape must be (n_modes, *mask.shape).')
    if raw_basis.shape[0] == 0:
        raise ValueError('raw_basis must contain at least one mode.')
    values = raw_basis[:, mask].T
    if values.shape[0] < values.shape[1]:
        raise ValueError('mask must contain at least as many pixels as modes.')
    if not np.all(np.isfinite(values)):
        raise ValueError('raw_basis must be finite on mask.')

    q_matrix, r_matrix = np.linalg.qr(values, mode='reduced')
    diagonal = np.abs(np.diag(r_matrix))
    tolerance = np.finfo(float).eps * max(values.shape) * np.linalg.norm(
        r_matrix, ord=np.inf
    )
    if np.any(diagonal <= tolerance):
        raise ValueError('raw_basis is rank-deficient on mask.')
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    q_matrix = q_matrix * signs[np.newaxis, :]

    result = np.zeros_like(raw_basis, dtype=float)
    result[:, mask] = (q_matrix * np.sqrt(values.shape[0])).T
    return result


def fit_orthonormal_basis(opd_nm, mask, basis):
    """Least-squares fit an orthonormal sampled basis to one OPD map.

    Parameters
    ----------
    opd_nm : `numpy.ndarray`
        Wavefront OPD map in nanometers.
    mask : `numpy.ndarray`
        Boolean fit mask.
    basis : `numpy.ndarray`
        Orthonormal basis returned by :func:`orthonormalize_basis`.

    Returns
    -------
    coefficients : `numpy.ndarray`
        Mode coefficients in nanometers RMS.
    model : `numpy.ndarray`
        Fitted OPD map, zero outside ``mask``.
    """
    opd_nm = np.asarray(opd_nm, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    basis = np.asarray(basis, dtype=float)
    if opd_nm.shape != mask.shape:
        raise ValueError('opd_nm and mask must have identical shapes.')
    if basis.ndim != mask.ndim + 1 or basis.shape[1:] != mask.shape:
        raise ValueError('basis shape must be (n_modes, *mask.shape).')
    if not np.all(np.isfinite(opd_nm[mask])):
        raise ValueError('opd_nm must be finite on mask.')

    coefficients = np.mean(basis[:, mask] * opd_nm[mask], axis=1)
    model = np.zeros_like(opd_nm, dtype=float)
    model[mask] = coefficients @ basis[:, mask]
    return coefficients, model


def noll_to_zernike(noll):
    """Convert one Noll index to signed radial and azimuthal orders."""
    radial = int(np.sqrt(2 * noll - 1) + 0.5) - 1
    if radial % 2:
        azimuthal = 2 * int((2 * (noll + 1) - radial * (radial + 1)) // 4) - 1
    else:
        azimuthal = 2 * int((2 * noll + 1 - radial * (radial + 1)) // 4)
    return radial, azimuthal * (-1)**(noll % 2)


def _zernike_radial(radial_order, azimuthal_order, radius):
    """Evaluate the radial component of a Zernike polynomial."""
    azimuthal_order = abs(azimuthal_order)
    result = np.zeros_like(radius, dtype=float)
    half_difference = (radial_order - azimuthal_order) // 2
    for index in range(half_difference + 1):
        coefficient = (
            (-1)**index
            * math.factorial(radial_order - index)
            / (
                math.factorial(index)
                * math.factorial((radial_order + azimuthal_order) // 2 - index)
                * math.factorial((radial_order - azimuthal_order) // 2 - index)
            )
        )
        result += coefficient * radius**(radial_order - 2 * index)
    return result


def build_raw_noll_basis(x_coordinates, y_coordinates, mask, mode_nolls):
    """Evaluate raw Noll-ordered Zernikes on a masked sampled aperture.

    Coordinates are centered on the selected pixels and scaled by their
    maximum radius. Applied to one segment at a time, subsequent QR
    orthonormalization produces the corresponding sampled hexike span.
    """
    x_coordinates = np.asarray(x_coordinates, dtype=float)
    y_coordinates = np.asarray(y_coordinates, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if x_coordinates.shape != mask.shape or y_coordinates.shape != mask.shape:
        raise ValueError('coordinate arrays and mask must have identical shapes.')
    if not np.any(mask):
        raise ValueError('mask must contain illuminated pixels.')
    x_local = x_coordinates - np.mean(x_coordinates[mask])
    y_local = y_coordinates - np.mean(y_coordinates[mask])
    scale = np.max(np.hypot(x_local[mask], y_local[mask]))
    if scale == 0.0:
        raise ValueError('mask coordinates must span a nonzero radius.')
    radius = np.hypot(x_local, y_local) / scale
    angle = np.arctan2(y_local, x_local)
    modes = []
    for noll in mode_nolls:
        radial_order, azimuthal_order = noll_to_zernike(int(noll))
        radial = _zernike_radial(radial_order, azimuthal_order, radius)
        if azimuthal_order == 0:
            mode = radial
        elif azimuthal_order > 0:
            mode = radial * np.cos(azimuthal_order * angle)
        else:
            mode = radial * np.sin(abs(azimuthal_order) * angle)
        modes.append(np.where(mask, mode, 0.0))
    return np.asarray(modes)


def _validate_modes(mode_nolls, name, minimum):
    """Validate a strictly increasing runtime mode sequence."""
    modes = tuple(mode_nolls)
    if any(
        isinstance(mode, (bool, np.bool_)) or not isinstance(mode, Integral)
        for mode in modes
    ):
        raise ValueError(f'{name} must contain integer Noll indices.')
    modes = tuple(int(mode) for mode in modes)
    if any(mode < minimum for mode in modes):
        raise ValueError(f'{name} modes must be >= {minimum}.')
    if tuple(sorted(set(modes))) != modes:
        raise ValueError(f'{name} must be strictly increasing.')
    return modes


def _triangular_factor(raw_values):
    """Return the positive-diagonal QR factor for sampled raw modes."""
    values = np.asarray(raw_values, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('raw mode values must have shape (n_pixels, n_modes).')
    if values.shape[0] < values.shape[1]:
        raise ValueError('basis must have at least as many pixels as modes.')
    if not np.all(np.isfinite(values)):
        raise ValueError('raw mode values must be finite.')
    _, r_matrix = np.linalg.qr(values, mode='reduced')
    tolerance = np.finfo(float).eps * max(values.shape) * np.linalg.norm(
        r_matrix, ord=np.inf
    )
    if np.any(np.abs(np.diag(r_matrix)) <= tolerance):
        raise ValueError('raw mode values are rank-deficient.')
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return r_matrix * signs[:, np.newaxis]


def _convert_coefficients(coefficients, modes, factor, pixel_count, name):
    """Convert one ordered orthonormal coefficient dictionary to raw."""
    if not coefficients:
        return {}
    if set(coefficients) != set(modes) or len(coefficients) != len(modes):
        raise ValueError(
            f'{name} keys must exactly match the transform modes {modes}.'
        )
    values = np.array([coefficients[mode] for mode in modes], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} coefficients must be finite.')
    raw = np.sqrt(pixel_count) * np.linalg.solve(factor, values)
    return {mode: float(raw[index]) for index, mode in enumerate(modes)}


class ApertureBasisTransform:
    """Convert orthonormal-aperture coefficients to raw HCIPy coefficients.

    The transform samples the exact global Zernike and segmented-hexike
    modes used by :mod:`hwoslaps.psf.aberration_models` on the illuminated
    EAC1 pupil. Its QR factors are constructed once and retained on the
    instance for all subsequent draws.

    Parameters
    ----------
    telescope_data : `dict`
        Telescope dictionary returned by
        :func:`hwoslaps.psf.telescope_models.create_hcipy_telescope`.
    global_mode_nolls : sequence of `int`
        Strictly increasing global Noll modes. May be empty.
    segment_mode_nolls : sequence of `int`
        Strictly increasing segment hexike Noll modes. May be empty.
    """

    def __init__(self, telescope_data, global_mode_nolls=(),
                 segment_mode_nolls=()):
        """Construct and cache factors from one HCIPy optical configuration."""
        global_modes = _validate_modes(
            global_mode_nolls, 'global_mode_nolls', 4
        )
        segment_modes = _validate_modes(
            segment_mode_nolls, 'segment_mode_nolls', 1
        )
        if not global_modes and not segment_modes:
            raise ValueError('At least one global or segment mode is required.')

        from .aberration_models import (
            _make_global_zernike_basis,
            _make_segmented_hexike_surface,
        )

        aperture_mask = np.asarray(telescope_data['aper']) > 0.5
        global_values = None
        if global_modes:
            global_basis = _make_global_zernike_basis(
                telescope_data, max(global_modes)
            )
            global_values = np.column_stack([
                np.asarray(global_basis[mode - 1])[aperture_mask]
                for mode in global_modes
            ])

        segment_values = {}
        if segment_modes:
            surface = _make_segmented_hexike_surface(
                telescope_data, max(segment_modes)
            )
            for segment, segment_field in enumerate(telescope_data['segments']):
                segment_mask = aperture_mask & (np.asarray(segment_field) > 0.5)
                columns = []
                for mode in segment_modes:
                    surface.flatten()
                    surface.set_segment_coefficients(
                        segment, {mode: 0.5}, indexing='noll'
                    )
                    columns.append(np.asarray(surface.opd)[segment_mask])
                segment_values[segment] = np.column_stack(columns)
            surface.flatten()

        self._initialize(global_modes, global_values, segment_modes,
                         segment_values)

    @classmethod
    def from_sampled_values(cls, global_mode_nolls=(), global_values=None,
                            segment_mode_nolls=(), segment_values=None):
        """Construct a transform directly from aperture-sampled raw modes.

        Parameters
        ----------
        global_mode_nolls : sequence of `int`, optional
            Strictly increasing global mode identifiers.
        global_values : `numpy.ndarray`, optional
            Raw global values with shape ``(n_pixels, n_modes)``.
        segment_mode_nolls : sequence of `int`, optional
            Strictly increasing segment mode identifiers.
        segment_values : `dict`, optional
            Mapping from segment identifier to arrays with shape
            ``(n_segment_pixels, n_modes)``.

        Returns
        -------
        transform : `ApertureBasisTransform`
            Cached transform built from the supplied sampled values.
        """
        instance = cls.__new__(cls)
        global_modes = _validate_modes(
            global_mode_nolls, 'global_mode_nolls', 1
        )
        segment_modes = _validate_modes(
            segment_mode_nolls, 'segment_mode_nolls', 1
        )
        if not global_modes and not segment_modes:
            raise ValueError('At least one global or segment mode is required.')
        instance._initialize(
            global_modes,
            global_values,
            segment_modes,
            {} if segment_values is None else segment_values,
        )
        return instance

    def _initialize(self, global_modes, global_values, segment_modes,
                    segment_values):
        """Initialize factors from already aperture-sampled mode values."""
        self.global_mode_nolls = global_modes
        self.segment_mode_nolls = segment_modes
        self.global_triangular_factor = None
        self.global_pixel_count = 0
        if global_modes:
            values = np.asarray(global_values, dtype=float)
            if values.ndim != 2 or values.shape[1] != len(global_modes):
                raise ValueError(
                    'global_values must have shape (n_pixels, n_global_modes).'
                )
            self.global_triangular_factor = _triangular_factor(values)
            self.global_pixel_count = values.shape[0]

        self.segment_triangular_factors = {}
        self.segment_pixel_counts = {}
        if segment_modes:
            if not isinstance(segment_values, dict) or not segment_values:
                raise ValueError('segment_values must be a non-empty dictionary.')
            for segment, raw_values in segment_values.items():
                values = np.asarray(raw_values, dtype=float)
                if values.ndim != 2 or values.shape[1] != len(segment_modes):
                    raise ValueError(
                        'Each segment_values array must have shape '
                        '(n_pixels, n_segment_modes).'
                    )
                segment = int(segment)
                self.segment_triangular_factors[segment] = _triangular_factor(
                    values
                )
                self.segment_pixel_counts[segment] = values.shape[0]

    def global_to_raw(self, coefficients):
        """Convert a global orthonormal-basis coefficient dictionary to raw."""
        if self.global_triangular_factor is None:
            if coefficients:
                raise ValueError('This transform has no global modes.')
            return {}
        return _convert_coefficients(
            coefficients,
            self.global_mode_nolls,
            self.global_triangular_factor,
            self.global_pixel_count,
            'global coefficients',
        )

    def segment_to_raw(self, coefficients):
        """Convert per-segment orthonormal-basis dictionaries to raw."""
        if not coefficients:
            return {}
        if not self.segment_triangular_factors:
            raise ValueError('This transform has no segment modes.')
        unknown = sorted(set(coefficients) - set(self.segment_triangular_factors))
        if unknown:
            raise ValueError(f'Unknown segment index: {unknown[0]}')
        return {
            segment: _convert_coefficients(
                modes,
                self.segment_mode_nolls,
                self.segment_triangular_factors[segment],
                self.segment_pixel_counts[segment],
                f'segment {segment} coefficients',
            )
            for segment, modes in coefficients.items()
        }

    def to_raw(self, segment_coefficients=None, global_coefficients=None):
        """Convert segment and global coefficient dictionaries together.

        Returns
        -------
        segment_raw : `dict`
            Raw per-segment HCIPy coefficients.
        global_raw : `dict`
            Raw global HCIPy coefficients.
        """
        return (
            self.segment_to_raw(segment_coefficients or {}),
            self.global_to_raw(global_coefficients or {}),
        )
