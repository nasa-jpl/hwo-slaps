#!/usr/bin/env python
"""Derive shape-only mode-weight priors from the JWST WSS OPD series.

The numerical decomposition functions in this module depend only on NumPy.
STPSF and its MAST dependencies are imported lazily by the thin data-access
layer used by :func:`main`.
"""

from __future__ import annotations

import argparse
from datetime import date
import inspect
import math
from pathlib import Path

import numpy as np
import yaml


DECOMPOSITION_METHOD = 'sequential_global_then_segment_qr_orthonormal'
SCRIPT_NAME = 'derive_jwst_mode_weight_tables.py'
SCRIPT_VERSION = '1'


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


def decompose_opd_map(opd_nm, aperture_mask, global_raw_basis,
                      segment_masks, segment_raw_bases):
    """Sequentially decompose one OPD map into global and segment modes.

    The global basis is fitted first over the full illuminated aperture.
    Per-segment bases are then fitted to the global-fit residual.

    Parameters
    ----------
    opd_nm : `numpy.ndarray`
        Wavefront OPD map in nanometers.
    aperture_mask : `numpy.ndarray`
        Full illuminated-aperture mask.
    global_raw_basis : `numpy.ndarray`
        Raw global basis with shape ``(n_global, *map_shape)``.
    segment_masks : `numpy.ndarray`
        Segment masks with shape ``(n_segments, *map_shape)``.
    segment_raw_bases : `numpy.ndarray`
        Raw segment bases with shape
        ``(n_segments, n_segment_modes, *map_shape)``.

    Returns
    -------
    global_coefficients : `numpy.ndarray`
        Orthonormal global coefficients in nanometers RMS.
    segment_coefficients : `numpy.ndarray`
        Orthonormal per-segment coefficients in nanometers RMS.
    residual : `numpy.ndarray`
        Residual after both sequential fits.
    """
    opd_nm = np.asarray(opd_nm, dtype=float)
    aperture_mask = np.asarray(aperture_mask, dtype=bool)
    segment_masks = np.asarray(segment_masks, dtype=bool)
    segment_raw_bases = np.asarray(segment_raw_bases, dtype=float)
    if opd_nm.shape != aperture_mask.shape:
        raise ValueError('opd_nm and aperture_mask must have identical shapes.')
    if segment_masks.ndim != aperture_mask.ndim + 1:
        raise ValueError('segment_masks must have shape (n_segments, *map_shape).')
    if segment_masks.shape[1:] != aperture_mask.shape:
        raise ValueError('segment_masks map shape must match aperture_mask.')
    expected = (segment_masks.shape[0],) + segment_raw_bases.shape[1:2] + aperture_mask.shape
    if segment_raw_bases.shape != expected:
        raise ValueError(
            'segment_raw_bases must have shape '
            '(n_segments, n_modes, *map_shape).'
        )
    if np.any(np.sum(segment_masks, axis=0) > 1):
        raise ValueError('segment_masks must not overlap.')
    if np.any(segment_masks & ~aperture_mask[np.newaxis, ...]):
        raise ValueError('segment_masks must be contained in aperture_mask.')

    global_basis = orthonormalize_basis(global_raw_basis, aperture_mask)
    global_coefficients, global_model = fit_orthonormal_basis(
        opd_nm, aperture_mask, global_basis
    )
    residual = np.array(opd_nm - global_model, copy=True)
    segment_coefficients = np.empty(
        (segment_masks.shape[0], segment_raw_bases.shape[1]), dtype=float
    )
    for segment_index, segment_mask in enumerate(segment_masks):
        segment_basis = orthonormalize_basis(
            segment_raw_bases[segment_index], segment_mask
        )
        coefficients, model = fit_orthonormal_basis(
            residual, segment_mask, segment_basis
        )
        segment_coefficients[segment_index] = coefficients
        residual[segment_mask] -= model[segment_mask]
    return global_coefficients, segment_coefficients, residual


def decompose_opd_series(opd_maps_nm, aperture_mask, global_raw_basis,
                         segment_masks, segment_raw_bases):
    """Decompose a series of OPD maps with one fixed aperture geometry."""
    opd_maps_nm = np.asarray(opd_maps_nm, dtype=float)
    if opd_maps_nm.ndim != np.asarray(aperture_mask).ndim + 1:
        raise ValueError('opd_maps_nm must have shape (n_maps, *map_shape).')
    global_series = []
    segment_series = []
    for opd_nm in opd_maps_nm:
        global_coefficients, segment_coefficients, _ = decompose_opd_map(
            opd_nm,
            aperture_mask,
            global_raw_basis,
            segment_masks,
            segment_raw_bases,
        )
        global_series.append(global_coefficients)
        segment_series.append(segment_coefficients)
    return np.asarray(global_series), np.asarray(segment_series)


def difference_opd_series(opd_maps_nm, step=1):
    """Return step-separated differences and their source-index pairs.

    Parameters
    ----------
    opd_maps_nm : `numpy.ndarray`
        OPD series with shape ``(n_maps, *map_shape)``.
    step : `int`, optional
        Positive separation between paired maps.

    Returns
    -------
    differences : `numpy.ndarray`
        Maps ``opd[k + step] - opd[k]``.
    pairs : `numpy.ndarray`
        Integer ``(k, k + step)`` pairs with shape ``(n_differences, 2)``.
    """
    opd_maps_nm = np.asarray(opd_maps_nm, dtype=float)
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
        raise ValueError('step must be a positive integer.')
    step = int(step)
    if step < 1:
        raise ValueError('step must be a positive integer.')
    if opd_maps_nm.ndim < 2:
        raise ValueError('opd_maps_nm must contain a map axis.')
    if opd_maps_nm.shape[0] <= step:
        raise ValueError('step must be smaller than the number of maps.')
    starts = np.arange(opd_maps_nm.shape[0] - step, dtype=int)
    pairs = np.column_stack((starts, starts + step))
    return opd_maps_nm[step:] - opd_maps_nm[:-step], pairs


def aggregate_mode_statistics(global_coefficients, segment_coefficients,
                              segment_area_fractions):
    """Aggregate coefficient RMS weights and the segment variance fraction.

    Parameters
    ----------
    global_coefficients : `numpy.ndarray`
        Coefficients with shape ``(n_samples, n_global_modes)``.
    segment_coefficients : `numpy.ndarray`
        Coefficients with shape
        ``(n_samples, n_segments, n_segment_modes)``.
    segment_area_fractions : `numpy.ndarray`
        Segment areas divided by total illuminated aperture area.

    Returns
    -------
    global_weights : `numpy.ndarray`
        Per-global-mode root mean square coefficients.
    segment_weights : `numpy.ndarray`
        Per-segment-mode RMS coefficients, pooled over samples and segments.
    segment_variance_fraction : `float`
        Mean per-sample segment fraction of represented variance.
    """
    global_coefficients = np.asarray(global_coefficients, dtype=float)
    segment_coefficients = np.asarray(segment_coefficients, dtype=float)
    area_fractions = np.asarray(segment_area_fractions, dtype=float)
    if global_coefficients.ndim != 2:
        raise ValueError('global_coefficients must be two-dimensional.')
    if segment_coefficients.ndim != 3:
        raise ValueError('segment_coefficients must be three-dimensional.')
    if global_coefficients.shape[0] != segment_coefficients.shape[0]:
        raise ValueError('coefficient arrays must have the same sample count.')
    if area_fractions.shape != (segment_coefficients.shape[1],):
        raise ValueError('segment_area_fractions must have one value per segment.')
    if np.any(~np.isfinite(area_fractions)) or np.any(area_fractions < 0.0):
        raise ValueError('segment_area_fractions must be finite and non-negative.')
    if not np.isclose(np.sum(area_fractions), 1.0):
        raise ValueError('segment_area_fractions must sum to one.')
    if not np.all(np.isfinite(global_coefficients)) or not np.all(
        np.isfinite(segment_coefficients)
    ):
        raise ValueError('coefficient arrays must be finite.')

    global_weights = np.sqrt(np.mean(global_coefficients**2, axis=0))
    segment_weights = np.sqrt(np.mean(segment_coefficients**2, axis=(0, 1)))
    global_variance = np.sum(global_coefficients**2, axis=1)
    per_segment_variance = np.sum(segment_coefficients**2, axis=2)
    segment_variance = per_segment_variance @ area_fractions
    total_variance = segment_variance + global_variance
    if np.any(total_variance <= 0.0):
        raise ValueError('every sample must contain positive represented variance.')
    fraction = float(np.mean(segment_variance / total_variance))
    return global_weights, segment_weights, fraction


def normalize_weights(weights):
    """Normalize a non-negative weight vector to unit squared sum."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError('weights must be a non-empty vector.')
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError('weights must be finite and non-negative.')
    norm = float(np.linalg.norm(weights))
    if norm == 0.0:
        raise ValueError('weights must have a positive sum of squares.')
    return weights / norm


def cosine_similarity(first, second):
    """Return the cosine similarity of two nonzero vectors."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError('vectors must be one-dimensional with equal shape.')
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0.0:
        raise ValueError('vectors must be nonzero.')
    return float(np.dot(first, second) / denominator)


def make_weight_document(name, global_mode_nolls, global_weights,
                         segment_mode_nolls, segment_weights,
                         segment_variance_fraction, metadata):
    """Build a serializable normalized mode-weight table document."""
    global_mode_nolls = [int(mode) for mode in global_mode_nolls]
    segment_mode_nolls = [int(mode) for mode in segment_mode_nolls]
    global_weights = normalize_weights(global_weights)
    segment_weights = normalize_weights(segment_weights)
    if len(global_mode_nolls) != len(global_weights):
        raise ValueError('global mode and weight counts must match.')
    if len(segment_mode_nolls) != len(segment_weights):
        raise ValueError('segment mode and weight counts must match.')
    return {
        'name': str(name),
        'segment_variance_fraction': float(segment_variance_fraction),
        'global_weights': {
            mode: float(weight)
            for mode, weight in zip(global_mode_nolls, global_weights)
        },
        'segment_weights': {
            mode: float(weight)
            for mode, weight in zip(segment_mode_nolls, segment_weights)
        },
        'metadata': dict(metadata),
    }


def write_weight_table(path, document):
    """Write one mode-weight table as portable safe YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(document, stream, sort_keys=False)


def _noll_to_zernike(noll):
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
        radial_order, azimuthal_order = _noll_to_zernike(int(noll))
        radial = _zernike_radial(radial_order, azimuthal_order, radius)
        if azimuthal_order == 0:
            mode = radial
        elif azimuthal_order > 0:
            mode = radial * np.cos(azimuthal_order * angle)
        else:
            mode = radial * np.sin(abs(azimuthal_order) * angle)
        modes.append(np.where(mask, mode, 0.0))
    return np.asarray(modes)


def connected_component_masks(mask):
    """Split a pixel mask into four-connected component masks."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('mask must be two-dimensional.')
    unseen = np.array(mask, copy=True)
    components = []
    for row, column in np.argwhere(mask):
        if not unseen[row, column]:
            continue
        component = np.zeros_like(mask)
        stack = [(int(row), int(column))]
        unseen[row, column] = False
        while stack:
            current_row, current_column = stack.pop()
            component[current_row, current_column] = True
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < mask.shape[0]
                    and 0 <= next_column < mask.shape[1]
                    and unseen[next_row, next_column]
                ):
                    unseen[next_row, next_column] = False
                    stack.append((next_row, next_column))
        components.append(component)
    components.sort(key=lambda component: tuple(np.mean(np.argwhere(component), axis=0)))
    return np.asarray(components, dtype=bool)


def _import_stpsf():
    """Import optional STPSF dependencies with an actionable error."""
    try:
        import stpsf
    except ImportError as exc:
        raise ImportError(
            'STPSF is required to download JWST WSS OPDs; install the '
            'optional "stpsf" package and its MAST/astroquery dependencies.'
        ) from exc
    return stpsf


def _resolve_stpsf_api(stpsf):
    """Resolve and validate the installed STPSF WSS query/load API."""
    mast_wss = getattr(stpsf, 'mast_wss', None)
    if mast_wss is None:
        raise RuntimeError('Installed stpsf has no mast_wss module.')
    query = getattr(mast_wss, 'query_wss_opds', None)
    loader = getattr(stpsf, 'load_wss_opd_by_date', None)
    if loader is None:
        loader = getattr(mast_wss, 'load_wss_opd_by_date', None)
    if query is None or loader is None:
        version = getattr(stpsf, '__version__', 'unknown')
        public = sorted(name for name in dir(mast_wss) if 'opd' in name.lower())
        raise RuntimeError(
            'Unsupported stpsf WSS API. Inspect the installed documentation '
            f'and adapt only _resolve_stpsf_api (version={version}, OPD '
            f'callables={public}). Required semantics are a date-range query '
            'and load_wss_opd_by_date.'
        )
    query_parameters = inspect.signature(query).parameters
    if not {'start_date', 'end_date'} <= set(query_parameters):
        raise RuntimeError(
            'stpsf.mast_wss.query_wss_opds does not expose documented '
            'start_date/end_date parameters; adapt the thin IO layer.'
        )
    return query, loader


def _table_dates(table):
    """Extract an ordered date list from an STPSF WSS query table."""
    names = getattr(table, 'colnames', ())
    candidates = ('date', 'date_obs', 'observation_date', 'datetime')
    column = next((name for name in candidates if name in names), None)
    if column is None:
        raise RuntimeError(
            f'Cannot identify a date column in WSS query result: {list(names)}'
        )
    return [str(value) for value in table[column]]


def _extract_opd_nm(opd_product):
    """Extract an OPD array, mask, and unit from a loaded STPSF product."""
    if hasattr(opd_product, '__iter__') and hasattr(opd_product, '__getitem__'):
        candidate_hdus = list(opd_product)
    else:
        candidate_hdus = [opd_product]
    data_hdu = next(
        (hdu for hdu in candidate_hdus if getattr(hdu, 'data', None) is not None),
        None,
    )
    if data_hdu is None:
        raise RuntimeError('STPSF WSS loader returned no array data.')
    opd = np.asarray(data_hdu.data, dtype=float).squeeze()
    if opd.ndim != 2:
        raise RuntimeError(f'Expected a two-dimensional OPD map, got {opd.shape}.')
    unit = str(getattr(data_hdu, 'header', {}).get('BUNIT', '')).strip().lower()
    unit_scales = {
        'm': 1e9,
        'meter': 1e9,
        'meters': 1e9,
        'um': 1e3,
        'micron': 1e3,
        'microns': 1e3,
        'nm': 1.0,
        'nanometer': 1.0,
        'nanometers': 1.0,
    }
    if unit not in unit_scales:
        raise RuntimeError(f'Unsupported or missing OPD BUNIT: {unit!r}.')
    mask = np.isfinite(opd)
    if np.all(mask):
        raise RuntimeError(
            'OPD product does not mark the aperture outside with nonfinite '
            'pixels; adapt _extract_opd_nm to the installed STPSF product.'
        )
    return np.where(mask, opd * unit_scales[unit], 0.0), mask


def load_wss_opd_series(start_date, end_date, opd_cache_dir=None,
                        max_maps=None):
    """Query and load a bounded JWST WSS OPD series through STPSF."""
    stpsf = _import_stpsf()
    query, loader = _resolve_stpsf_api(stpsf)
    table = query(start_date=start_date, end_date=end_date)
    measurement_dates = _table_dates(table)
    if max_maps is not None and len(measurement_dates) > max_maps:
        indices = np.linspace(
            0, len(measurement_dates) - 1, max_maps, dtype=int
        )
        measurement_dates = [measurement_dates[index] for index in indices]

    loader_parameters = inspect.signature(loader).parameters
    cache_keyword = None
    for candidate in ('output_path', 'opd_cache_dir', 'cache_dir'):
        if candidate in loader_parameters:
            cache_keyword = candidate
            break
    if opd_cache_dir is not None and cache_keyword is None:
        raise RuntimeError(
            'Installed load_wss_opd_by_date has no documented cache-path '
            'parameter; adapt the thin IO layer.'
        )
    maps = []
    aperture_mask = None
    for measurement_date in measurement_dates:
        keywords = {}
        if cache_keyword is not None and opd_cache_dir is not None:
            keywords[cache_keyword] = str(opd_cache_dir)
        product = loader(measurement_date, **keywords)
        opd_nm, current_mask = _extract_opd_nm(product)
        if aperture_mask is None:
            aperture_mask = current_mask
        elif not np.array_equal(aperture_mask, current_mask):
            raise RuntimeError('WSS OPD aperture masks differ across the series.')
        maps.append(opd_nm)
    if not maps:
        raise RuntimeError('No WSS OPD maps matched the requested date range.')
    return np.asarray(maps), aperture_mask, measurement_dates, stpsf.__version__


def _prepare_geometry(aperture_mask, global_max_noll, segment_max_noll):
    """Build raw sampled bases and connected segment masks."""
    segment_masks = connected_component_masks(aperture_mask)
    if len(segment_masks) != 18:
        raise RuntimeError(
            f'Expected 18 disconnected JWST segments, found {len(segment_masks)}.'
        )
    rows, columns = np.indices(aperture_mask.shape, dtype=float)
    global_modes = list(range(4, global_max_noll + 1))
    segment_modes = list(range(1, segment_max_noll + 1))
    global_basis = build_raw_noll_basis(
        columns, rows, aperture_mask, global_modes
    )
    segment_bases = np.asarray([
        build_raw_noll_basis(columns, rows, mask, segment_modes)
        for mask in segment_masks
    ])
    areas = np.sum(segment_masks, axis=tuple(range(1, segment_masks.ndim)))
    area_fractions = areas / np.sum(areas)
    return (
        global_modes,
        segment_modes,
        segment_masks,
        global_basis,
        segment_bases,
        area_fractions,
    )


def _derive_document(opd_maps_nm, aperture_mask, statistic, step,
                     global_max_noll, segment_max_noll, metadata):
    """Derive one static or drift weight document from loaded OPD maps."""
    if statistic == 'drift':
        maps_to_fit, _ = difference_opd_series(opd_maps_nm, step=step)
    elif statistic == 'static':
        maps_to_fit = opd_maps_nm
    else:
        raise ValueError("statistic must be 'static' or 'drift'.")
    geometry = _prepare_geometry(
        aperture_mask, global_max_noll, segment_max_noll
    )
    (global_modes, segment_modes, segment_masks, global_basis,
     segment_bases, area_fractions) = geometry
    global_coefficients, segment_coefficients = decompose_opd_series(
        maps_to_fit,
        aperture_mask,
        global_basis,
        segment_masks,
        segment_bases,
    )
    global_weights, segment_weights, fraction = aggregate_mode_statistics(
        global_coefficients, segment_coefficients, area_fractions
    )
    table_metadata = dict(metadata)
    count_field = (
        'number_of_differences_used'
        if statistic == 'drift'
        else 'number_of_maps_used'
    )
    table_metadata.update({
        'statistic': statistic,
        count_field: int(len(maps_to_fit)),
        'step': int(step),
        'global_max_noll': int(global_max_noll),
        'segment_max_noll': int(segment_max_noll),
        'decomposition_method': DECOMPOSITION_METHOD,
        'script': SCRIPT_NAME,
        'script_version': SCRIPT_VERSION,
        'generation_date': date.today().isoformat(),
    })
    return make_weight_document(
        f'jwst_wss_{statistic}_v1',
        global_modes,
        global_weights,
        segment_modes,
        segment_weights,
        fraction,
        table_metadata,
    )


def _print_summary(document):
    """Print one human-readable derived-weight summary table."""
    print(f"\n{document['name']}")
    print(
        'segment_variance_fraction: '
        f"{document['segment_variance_fraction']:.6f}"
    )
    print('side     mode    normalized_weight')
    for side in ('global_weights', 'segment_weights'):
        label = 'global' if side == 'global_weights' else 'segment'
        for mode, weight in document[side].items():
            print(f'{label:7s}  {mode:4d}    {weight:.8f}')


def _build_parser():
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--statistic', choices=('static', 'drift', 'both'), default='both'
    )
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--baseline-check', action='store_true')
    parser.add_argument('--max-maps', type=int)
    parser.add_argument('--global-max-noll', type=int, default=55)
    parser.add_argument('--segment-max-noll', type=int, default=10)
    parser.add_argument('--output-dir', default='configs/psf_priors')
    parser.add_argument('--opd-cache-dir')
    return parser


def main(argv=None):
    """Run the JWST WSS mode-weight derivation CLI."""
    args = _build_parser().parse_args(argv)
    if args.step < 1:
        raise ValueError('--step must be a positive integer.')
    if args.max_maps is not None and args.max_maps < 2:
        raise ValueError('--max-maps must be at least two.')
    if args.global_max_noll < 4:
        raise ValueError('--global-max-noll must be at least four.')
    if args.segment_max_noll < 1:
        raise ValueError('--segment-max-noll must be at least one.')

    maps, aperture_mask, measurement_dates, stpsf_version = load_wss_opd_series(
        args.start_date,
        args.end_date,
        opd_cache_dir=args.opd_cache_dir,
        max_maps=args.max_maps,
    )
    metadata = {
        'date_range': [args.start_date, args.end_date],
        'number_of_source_maps': int(len(maps)),
        'first_measurement': measurement_dates[0],
        'last_measurement': measurement_dates[-1],
        'stpsf_version': str(stpsf_version),
    }
    statistics = ('static', 'drift') if args.statistic == 'both' else (args.statistic,)
    documents = {}
    for statistic in statistics:
        document = _derive_document(
            maps,
            aperture_mask,
            statistic,
            args.step,
            args.global_max_noll,
            args.segment_max_noll,
            metadata,
        )
        documents[statistic] = document
        output_path = Path(args.output_dir) / f'jwst_wss_{statistic}_v1.yaml'
        write_weight_table(output_path, document)
        _print_summary(document)

    if args.baseline_check:
        baseline = _derive_document(
            maps,
            aperture_mask,
            'drift',
            1,
            args.global_max_noll,
            args.segment_max_noll,
            metadata,
        )
        print('\nDrift baseline cosine similarities against step 1')
        print('step     global     segment')
        for step in (2, 4, 8):
            comparison = _derive_document(
                maps,
                aperture_mask,
                'drift',
                step,
                args.global_max_noll,
                args.segment_max_noll,
                metadata,
            )
            global_similarity = cosine_similarity(
                list(baseline['global_weights'].values()),
                list(comparison['global_weights'].values()),
            )
            segment_similarity = cosine_similarity(
                list(baseline['segment_weights'].values()),
                list(comparison['segment_weights'].values()),
            )
            print(f'{step:4d}     {global_similarity:.6f}    {segment_similarity:.6f}')


if __name__ == '__main__':
    main()
