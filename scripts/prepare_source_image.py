#!/usr/bin/env python
"""Prepare local galaxy images as deterministic source-light assets."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import ndimage


SCRIPT_VERSION = 2

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OBSERVING_REFERENCE_RELPATH = 'configs/observing/hwo_eac1_hri_reference_v1.yaml'
"""Committed physical observing reference carrying the detected source rate."""

PRODUCTION_SCENE_RELPATH = 'configs/scenes/scene4_cosmos.yaml'
"""Production Image-source scene supplying the contract render geometry."""

PIXEL_SCALE_ABS_TOLERANCE = 1.0e-12
"""Accepted departure between the scene and reference pixel scales."""

DISCRETE_MAPPING_TOLERANCE = 1.0e-2
"""Accepted ``pixel_area * discrete_sum`` departure from the unit integral.

An asset integrates to one at its own pixel scale, so a unit-``total_flux``
render on the production grid must sum to ``1 / pixel_area``. A wider miss
means the asset spills off the production grid or the interpolation loses
flux, not sub-pixel sampling.
"""

RATE_CONTRACT_RELATIVE_TOLERANCE = 1.0e-12
"""Accepted ``|realized / target - 1|`` for a solved detected rate."""


def _finite_2d_image(image, name="image"):
    """Return a finite float64 two-dimensional image."""
    try:
        result = np.asarray(image, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be coercible to a float64 array") from exc
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def load_input_image(path):
    """Load a local ``.npy`` or first-two-dimensional-HDU FITS image.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Input ``.npy`` or FITS file.

    Returns
    -------
    image : `numpy.ndarray`
        Finite float64 image.
    """
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == '.npy':
        try:
            image = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not load NumPy image {path}: {exc}") from exc
        return _finite_2d_image(image, f"NumPy image {path}")
    if suffix in {'.fits', '.fit', '.fts'}:
        from astropy.io import fits

        try:
            with fits.open(path, memmap=False) as hdus:
                descriptions = []
                for index, hdu in enumerate(hdus):
                    data = hdu.data
                    shape = None if data is None else np.shape(data)
                    descriptions.append(
                        f"{index}:{type(hdu).__name__} shape={shape}"
                    )
                    if data is not None and np.ndim(data) == 2:
                        return _finite_2d_image(data, f"FITS image {path}")
        except OSError as exc:
            raise ValueError(f"Could not load FITS image {path}: {exc}") from exc
        raise ValueError(
            f"No 2D image HDU found in {path}; HDUs: "
            + ", ".join(descriptions)
        )
    raise ValueError(f"Input image must be .npy or FITS, got: {path}")


def bin_image(image, n):
    """Block-mean an image after cropping bottom/right remainders.

    Parameters
    ----------
    image : `numpy.ndarray`
        Input surface-brightness image.
    n : `int`
        Positive integer block width.

    Returns
    -------
    binned : `numpy.ndarray`
        Block-mean image.
    crop : `dict`
        Numbers of removed bottom rows and right columns.
    """
    image = _finite_2d_image(image)
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError('bin factor n must be a positive integer')
    n = int(n)
    bottom_rows = image.shape[0] % n
    right_columns = image.shape[1] % n
    kept_y = image.shape[0] - bottom_rows
    kept_x = image.shape[1] - right_columns
    if kept_y == 0 or kept_x == 0:
        raise ValueError('bin factor is larger than an input-image dimension')
    cropped = image[:kept_y, :kept_x]
    if n == 1:
        binned = np.array(cropped, copy=True)
    else:
        binned = cropped.reshape(kept_y // n, n, kept_x // n, n).mean(
            axis=(1, 3)
        )
    return binned, {
        'bottom_rows': int(bottom_rows),
        'right_columns': int(right_columns),
    }


def _border_values(image, border_frac):
    """Return a non-duplicated flattened border frame."""
    if not isinstance(border_frac, (int, float)) or isinstance(border_frac, bool):
        raise ValueError('border_frac must be a finite number')
    border_frac = float(border_frac)
    if not math.isfinite(border_frac) or border_frac <= 0.0 or border_frac > 0.5:
        raise ValueError('border_frac must be greater than 0 and at most 0.5')
    width = max(1, int(math.ceil(border_frac * min(image.shape))))
    border_mask = np.zeros(image.shape, dtype=bool)
    border_mask[:width, :] = True
    border_mask[-width:, :] = True
    border_mask[:, :width] = True
    border_mask[:, -width:] = True
    return image[border_mask]


def subtract_background(image, border_frac=0.1):
    """Subtract the sigma-clipped median of the image border.

    Parameters
    ----------
    image : `numpy.ndarray`
        Input image.
    border_frac : `float`, optional
        Fractional border width relative to the shorter image dimension.

    Returns
    -------
    subtracted : `numpy.ndarray`
        Background-subtracted image.
    background : `float`
        Sigma-clipped border median.
    """
    image = _finite_2d_image(image)
    border = _border_values(image, border_frac)
    _, median, _ = sigma_clipped_stats(border, sigma=3.0, maxiters=10)
    background = float(median)
    if not math.isfinite(background):
        raise ValueError('Sigma-clipped border background is not finite')
    return image - background, background


def footprint_mask(image, k_sigma=2.0):
    """Keep the largest dilated 8-connected source footprint.

    Parameters
    ----------
    image : `numpy.ndarray`
        Background-subtracted image.
    k_sigma : `float`, optional
        Detection threshold in sigma-clipped border RMS units.

    Returns
    -------
    masked : `numpy.ndarray`
        Non-negative image, zero outside the dilated largest component.
    threshold : `float`
        Applied surface-brightness threshold.
    component_size : `int`
        Pixel count of the undilated largest component.
    """
    image = _finite_2d_image(image)
    if isinstance(k_sigma, bool) or not isinstance(k_sigma, (int, float)):
        raise ValueError('k_sigma must be a finite non-negative number')
    k_sigma = float(k_sigma)
    if not math.isfinite(k_sigma) or k_sigma < 0.0:
        raise ValueError('k_sigma must be a finite non-negative number')
    border = _border_values(image, 0.1)
    _, _, border_rms = sigma_clipped_stats(border, sigma=3.0, maxiters=10)
    border_rms = float(border_rms)
    if not math.isfinite(border_rms) or border_rms < 0.0:
        raise ValueError('Sigma-clipped border RMS is invalid')
    threshold = k_sigma * border_rms
    labels, count = ndimage.label(
        image > threshold,
        structure=np.ones((3, 3), dtype=int),
    )
    if count == 0:
        raise ValueError('No positive source footprint found above the mask threshold')
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label_index = int(np.argmax(sizes))
    component = labels == label_index
    if (
        np.any(component[0, :])
        or np.any(component[-1, :])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    ):
        raise ValueError('Largest source component touches the array edge')
    dilated = ndimage.binary_dilation(component, iterations=2)
    masked = np.where(dilated, np.maximum(image, 0.0), 0.0)
    if not float(masked.sum()) > 0.0:
        raise ValueError('Masked source flux must be positive')
    return masked, float(threshold), int(sizes[label_index])


def centre_on_centroid(image):
    """Centre a source by integer crop and symmetric zero padding.

    Parameters
    ----------
    image : `numpy.ndarray`
        Non-negative source image.

    Returns
    -------
    centred : `numpy.ndarray`
        Square image whose flux centroid is within half a pixel of centre.
    shift : `tuple` of `int`
        Integer ``(y, x)`` shift from the selected input centroid pixel to
        the output centre.
    """
    image = _finite_2d_image(image)
    if np.any(image < 0.0) or not float(image.sum()) > 0.0:
        raise ValueError('Image for centroid centering must have positive non-negative flux')
    rows, cols = np.indices(image.shape, dtype=float)
    total = float(image.sum())
    centroid_y = float((rows * image).sum() / total)
    centroid_x = float((cols * image).sum() / total)
    centre_index_y = int(math.floor(centroid_y + 0.5))
    centre_index_x = int(math.floor(centroid_x + 0.5))
    half_y = min(centre_index_y, image.shape[0] - 1 - centre_index_y)
    half_x = min(centre_index_x, image.shape[1] - 1 - centre_index_x)
    cropped = image[
        centre_index_y - half_y:centre_index_y + half_y + 1,
        centre_index_x - half_x:centre_index_x + half_x + 1,
    ]
    side = max(cropped.shape)
    pad_y = side - cropped.shape[0]
    pad_x = side - cropped.shape[1]
    centred = np.pad(
        cropped,
        (
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        ),
        mode='constant',
    )
    shift = (
        side // 2 - centre_index_y,
        side // 2 - centre_index_x,
    )
    out_rows, out_cols = np.indices(centred.shape, dtype=float)
    out_total = float(centred.sum())
    out_centroid_y = float((out_rows * centred).sum() / out_total)
    out_centroid_x = float((out_cols * centred).sum() / out_total)
    out_centre = (side - 1) / 2.0
    if abs(out_centroid_y - out_centre) > 0.5 or abs(out_centroid_x - out_centre) > 0.5:
        raise ValueError('Integer centering could not place the centroid within half a pixel')
    return centred, shift


def half_light_radius_pixels(image):
    """Return the circular half-light radius about the array centre.

    Parameters
    ----------
    image : `numpy.ndarray`
        Non-negative centred source image.

    Returns
    -------
    radius : `float`
        Interpolated half-light radius in pixels.
    """
    image = _finite_2d_image(image)
    total = float(image.sum())
    if not total > 0.0:
        raise ValueError('Image flux must be positive for a half-light radius')
    rows, cols = np.indices(image.shape, dtype=float)
    centre_y = (image.shape[0] - 1) / 2.0
    centre_x = (image.shape[1] - 1) / 2.0
    radii = np.hypot(rows - centre_y, cols - centre_x).ravel()
    flux = image.ravel()
    order = np.argsort(radii, kind='stable')
    radii = radii[order]
    cumulative = np.cumsum(flux[order])
    target = 0.5 * total
    index = int(np.searchsorted(cumulative, target, side='left'))
    if index == 0:
        raise ValueError(
            'The central pixel contains at least half the flux; the source '
            'is unresolved at this sampling. Reduce --bin or use a higher '
            'resolution input.'
        )
    radius = float(
        np.interp(
            target,
            cumulative[index - 1:index + 1],
            radii[index - 1:index + 1],
        )
    )
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError('Half-light radius must be positive and finite')
    return radius


def rescale_to_half_light(target_arcsec, r_half_pixels):
    """Convert a target half-light radius to an asset pixel scale.

    Parameters
    ----------
    target_arcsec : `float`
        Target half-light radius in arcseconds.
    r_half_pixels : `float`
        Measured half-light radius in pixels.

    Returns
    -------
    pixel_scale_arcsec : `float`
        Pixel scale that places the half-light radius at the target.
    """
    values = (target_arcsec, r_half_pixels)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in values
    ):
        raise ValueError('target_arcsec and r_half_pixels must be positive and finite')
    return float(target_arcsec) / float(r_half_pixels)


def normalize_unit_flux(image, pixel_scale_arcsec):
    """Normalize sampled surface brightness to unit bilinear integral.

    Parameters
    ----------
    image : `numpy.ndarray`
        Non-negative surface-brightness image.
    pixel_scale_arcsec : `float`
        Pixel scale in arcseconds.

    Returns
    -------
    normalized : `numpy.ndarray`
        Image scaled so ``pixel_scale_arcsec**2 * normalized.sum() == 1``.
    """
    image = _finite_2d_image(image)
    if np.any(image < 0.0):
        raise ValueError('Image surface brightness must be non-negative')
    if (
        isinstance(pixel_scale_arcsec, bool)
        or not isinstance(pixel_scale_arcsec, (int, float))
        or not math.isfinite(float(pixel_scale_arcsec))
        or float(pixel_scale_arcsec) <= 0.0
    ):
        raise ValueError('pixel_scale_arcsec must be positive and finite')
    normalization = float(pixel_scale_arcsec) ** 2 * float(image.sum())
    if not normalization > 0.0 or not math.isfinite(normalization):
        raise ValueError('Image flux must be positive and finite for normalization')
    return image / normalization


def _read_yaml(path):
    """Parse one committed YAML configuration or fail with its path."""
    import yaml

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Configuration {path} does not exist')
    with path.open('r', encoding='utf-8') as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError(f'Configuration {path} must parse to a mapping')
    return document


def _positive_float(value, name):
    """Return one strictly positive finite float or fail loudly."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a positive finite number, got {value!r}')
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f'{name} must be a positive finite number, got {value!r}')
    return number


def detected_rate_reference(reference_path):
    """Read the committed unlensed detected source rate and pixel scale.

    The rate is the physical photometry of the observing reference, in
    detected electrons per second for the unlensed intrinsic source. It
    is never the qualification profile angular integral.

    Parameters
    ----------
    reference_path : `str` or `pathlib.Path`
        Committed observing-reference YAML.

    Returns
    -------
    reference : `dict`
        Target rate, reference pixel scale, source magnitude, and the
        reference file identity.
    """
    reference_path = Path(reference_path).expanduser().resolve()
    document = _read_yaml(reference_path)
    metadata = document.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError(f'{reference_path} carries no metadata block')
    photometry = metadata.get('source_photometry')
    detector = metadata.get('detector')
    if not isinstance(photometry, dict) or not isinstance(detector, dict):
        raise ValueError(
            f'{reference_path} must carry metadata.source_photometry and '
            'metadata.detector'
        )
    return {
        'reference_path': _provenance_path(reference_path),
        'reference_sha256': _sha256(reference_path),
        'reference_name': metadata.get('reference_name'),
        'target_rate_e_per_s': _positive_float(
            photometry.get('detected_rate_e_per_s'),
            'metadata.source_photometry.detected_rate_e_per_s',
        ),
        'source_magnitude_ab': photometry.get('derived_magnitude_ab'),
        'source_band': photometry.get('derived_band'),
        'pixel_scale_arcsec': _positive_float(
            detector.get('pixel_scale_arcsec'),
            'metadata.detector.pixel_scale_arcsec',
        ),
    }


def production_render_config(scene_path, reference_pixel_scale_arcsec):
    """Read the production grid and Image-source geometry of one scene.

    Parameters
    ----------
    scene_path : `str` or `pathlib.Path`
        Committed production scene carrying an ``Image`` source.
    reference_pixel_scale_arcsec : `float`
        Reference detector pixel scale the scene grid must match.

    Returns
    -------
    grid_config : `dict`
        ``lensing.grid`` block of the scene.
    source_config : `dict`
        ``lensing.source_galaxy`` block of the scene.
    """
    scene_path = Path(scene_path).expanduser().resolve()
    document = _read_yaml(scene_path)
    lensing = document.get('lensing')
    if not isinstance(lensing, dict):
        raise ValueError(f'{scene_path} carries no lensing block')
    grid_config = deepcopy(lensing.get('grid'))
    source_config = deepcopy(lensing.get('source_galaxy'))
    if not isinstance(grid_config, dict) or not isinstance(source_config, dict):
        raise ValueError(
            f'{scene_path} must carry lensing.grid and lensing.source_galaxy'
        )
    light = source_config.get('light')
    if not isinstance(light, dict) or light.get('type') != 'Image':
        raise ValueError(
            f'{scene_path} must carry an Image source light block, got '
            f'{None if not isinstance(light, dict) else light.get("type")!r}'
        )
    for field in ('flux_scale', 'size_scale'):
        if float(light.get(field, 0.0)) != 1.0:
            raise ValueError(
                f'{scene_path} source light {field} must be 1 for the rate '
                f'contract, got {light.get(field)!r}; the contract normalizes '
                'total_flux at unit flux and size scales'
            )
    scene_pixel_scale = _positive_float(
        grid_config.get('pixel_scale'), f'{scene_path} lensing.grid.pixel_scale'
    )
    if abs(scene_pixel_scale - reference_pixel_scale_arcsec) > PIXEL_SCALE_ABS_TOLERANCE:
        raise ValueError(
            f'{scene_path} samples at {scene_pixel_scale} arcsec while the '
            f'observing reference declares {reference_pixel_scale_arcsec} '
            'arcsec; the detected per-pixel rates would not apply to it'
        )
    return grid_config, source_config


def render_unlensed_asset(asset_path, grid_config, source_config, total_flux):
    """Render one asset unlensed on the production grid.

    The grid and the light profile are built by the production
    constructors in ``hwoslaps.lensing.generator``, so the samples carry
    the exact production geometry, asset loader, and sub-pixel
    oversampling. No lensing and therefore no magnification is applied.

    Parameters
    ----------
    asset_path : `str` or `pathlib.Path`
        Prepared unit-integral source-image asset.
    grid_config : `dict`
        ``lensing.grid`` block.
    source_config : `dict`
        ``lensing.source_galaxy`` block carrying an ``Image`` light type.
    total_flux : `float`
        Intrinsic flux normalization to render at.

    Returns
    -------
    image : `numpy.ndarray`
        Unlensed source samples in the per-pixel convention the
        observation layer reads as detected electrons per second.
    """
    from hwoslaps.lensing.generator import _create_grid, _create_source_galaxy

    source_config = deepcopy(source_config)
    light = source_config['light']
    light['asset_path'] = str(Path(asset_path).expanduser().resolve())
    light['total_flux'] = _positive_float(total_flux, 'total_flux')
    galaxy = _create_source_galaxy(source_config)
    image = galaxy.image_2d_from(grid=_create_grid(grid_config))
    return np.asarray(image, dtype=float)


def solve_detected_rate_normalization(asset_path, reference, grid_config,
                                      source_config, scene_path):
    """Solve one asset's ``total_flux`` for the target detected rate.

    The ``Image`` source is exactly linear in ``total_flux`` at unit flux
    and size scales, so a single unit render fixes the solution; the
    solved normalization is then rendered again and its discrete pixel
    sum is recorded as the realized rate rather than assumed.

    Parameters
    ----------
    asset_path : `str` or `pathlib.Path`
        Prepared unit-integral source-image asset.
    reference : `dict`
        Record from :func:`detected_rate_reference`.
    grid_config : `dict`
        ``lensing.grid`` block of the production scene.
    source_config : `dict`
        ``lensing.source_galaxy`` block of the production scene.
    scene_path : `str` or `pathlib.Path`
        Production scene the grid and geometry were read from.

    Returns
    -------
    contract : `dict`
        Target rate, solved ``total_flux``, realized rate, and the render
        that maps between them.
    """
    scene_path = Path(scene_path).expanduser().resolve()
    target_rate = _positive_float(
        reference['target_rate_e_per_s'], 'target_rate_e_per_s'
    )
    pixel_scale = _positive_float(
        grid_config['pixel_scale'], 'lensing.grid.pixel_scale'
    )
    pixel_area = pixel_scale ** 2
    unit_sum = float(
        np.sum(render_unlensed_asset(asset_path, grid_config, source_config, 1.0))
    )
    if not math.isfinite(unit_sum) or unit_sum <= 0.0:
        raise ValueError(
            f'Asset {asset_path} renders a unit-total_flux discrete sum of '
            f'{unit_sum}; the prepared source carries no light on the '
            'production grid'
        )
    mapping_ratio = pixel_area * unit_sum
    if abs(mapping_ratio - 1.0) > DISCRETE_MAPPING_TOLERANCE:
        raise ValueError(
            f'Asset {asset_path} renders a unit-total_flux discrete sum whose '
            f'pixel-area integral is {mapping_ratio} instead of 1; the asset '
            'spills off the production grid or loses flux in interpolation'
        )
    total_flux = target_rate / unit_sum
    realized_rate = float(
        np.sum(
            render_unlensed_asset(
                asset_path, grid_config, source_config, total_flux
            )
        )
    )
    if abs(realized_rate / target_rate - 1.0) > RATE_CONTRACT_RELATIVE_TOLERANCE:
        raise ValueError(
            f'Asset {asset_path} realizes {realized_rate} e-/s against the '
            f'target {target_rate} e-/s; the Image source is not linear in '
            'total_flux as the solve assumes'
        )
    light = source_config['light']
    return {
        'target_rate_e_per_s': target_rate,
        'realized_rate_e_per_s': realized_rate,
        'total_flux': total_flux,
        'units': 'detected electrons per second, unlensed intrinsic source total',
        'unit_total_flux_discrete_sum': unit_sum,
        'discrete_mapping_ratio': mapping_ratio,
        'grid_shape': [int(value) for value in grid_config['shape']],
        'pixel_scale_arcsec': pixel_scale,
        'pixel_area_arcsec2': pixel_area,
        'render_geometry': {
            'centre': [float(value) for value in light['centre']],
            'rotation_deg': float(light['rotation_deg']),
            'flux_scale': float(light['flux_scale']),
            'size_scale': float(light['size_scale']),
            'source_redshift': float(source_config['redshift']),
        },
        'render_method': (
            'unlensed source galaxy from '
            'hwoslaps.lensing.generator._create_source_galaxy evaluated on '
            'hwoslaps.lensing.generator._create_grid: the exact production '
            'constructors, asset loader, grid geometry, and sub-pixel '
            'oversampling'
        ),
        'magnification_note': (
            'This contract is unlensed. The lensed pipeline applies '
            'magnification exactly once, emergently, by evaluating this same '
            'surface brightness at ray-traced source-plane positions; no '
            'magnification factor multiplies the normalization anywhere.'
        ),
        'qualification_convention_note': (
            'The 0.289151264 qualification value is a profile angular '
            'integral in profile units and is never a detected electron rate.'
        ),
        'observing_reference': {
            'path': reference['reference_path'],
            'sha256': reference['reference_sha256'],
            'reference_name': reference['reference_name'],
            'source_magnitude_ab': reference['source_magnitude_ab'],
            'source_band': reference['source_band'],
            'note': (
                'recorded at preparation time; the contract is gated on '
                'target_rate_e_per_s, not on this file hash'
            ),
        },
        'production_scene': {
            'path': _provenance_path(scene_path),
            'sha256': _sha256(scene_path),
        },
    }


def _clear_asset_loader_cache():
    """Drop the in-process source-image asset cache.

    The production loader keeps one immutable view per absolute path, so
    a freshly written asset at a path already read in this process would
    otherwise be re-rendered from the stale view.
    """
    from hwoslaps.lensing.image_source import _load_source_image_asset_absolute

    _load_source_image_asset_absolute.cache_clear()


def _solved_rate_contract(sb, pixel_scale_arcsec, reference_path, scene_path):
    """Solve the rate contract for samples not yet written to their asset.

    The solve needs the production asset loader, which reads a file, so
    the samples are staged into a temporary asset first. The staged
    samples and pixel scale are the ones written to the final asset, so
    the render is identical; :func:`verify_asset_rate_contract` proves
    that against the written file afterwards.
    """
    import tempfile

    reference = detected_rate_reference(reference_path)
    grid_config, source_config = production_render_config(
        scene_path, reference['pixel_scale_arcsec']
    )
    with tempfile.TemporaryDirectory() as staging:
        staged_path = Path(staging) / 'rate_contract_solve.npz'
        write_asset(
            staged_path,
            sb,
            pixel_scale_arcsec,
            {'note': 'temporary staging asset for the rate-contract solve'},
        )
        _clear_asset_loader_cache()
        contract = solve_detected_rate_normalization(
            staged_path, reference, grid_config, source_config, scene_path
        )
    _clear_asset_loader_cache()
    return contract


def verify_asset_rate_contract(asset_path, scene_path=None,
                               reference_path=None):
    """Re-render one written asset and check its stored rate contract.

    Parameters
    ----------
    asset_path : `str` or `pathlib.Path`
        Prepared asset carrying ``provenance.rate_contract``.
    scene_path : `str` or `pathlib.Path`, optional
        Production scene to render on. Defaults to the committed
        production Image scene.
    reference_path : `str` or `pathlib.Path`, optional
        Observing reference the stored target rate must still match.
        Defaults to the committed observing reference.

    Returns
    -------
    contract : `dict`
        The stored contract, once re-rendering has reproduced it.
    """
    from hwoslaps.lensing import load_source_image_asset

    asset_path = Path(asset_path).expanduser().resolve()
    scene_path = Path(
        scene_path
        if scene_path is not None
        else PROJECT_ROOT / PRODUCTION_SCENE_RELPATH
    ).expanduser().resolve()
    reference_path = Path(
        reference_path
        if reference_path is not None
        else PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH
    ).expanduser().resolve()

    _clear_asset_loader_cache()
    asset = load_source_image_asset(asset_path)
    provenance = asset.metadata.get('provenance')
    if not isinstance(provenance, dict) or 'rate_contract' not in provenance:
        raise ValueError(
            f'Asset {asset_path} carries no provenance.rate_contract block'
        )
    contract = provenance['rate_contract']
    reference = detected_rate_reference(reference_path)
    target_rate = _positive_float(
        contract['target_rate_e_per_s'], 'rate_contract.target_rate_e_per_s'
    )
    if target_rate != reference['target_rate_e_per_s']:
        raise ValueError(
            f'Asset {asset_path} stores target rate {target_rate} e-/s while '
            f'{reference_path} now declares '
            f'{reference["target_rate_e_per_s"]} e-/s'
        )
    grid_config, source_config = production_render_config(
        scene_path, reference['pixel_scale_arcsec']
    )
    realized_rate = float(
        np.sum(
            render_unlensed_asset(
                asset_path,
                grid_config,
                source_config,
                _positive_float(
                    contract['total_flux'], 'rate_contract.total_flux'
                ),
            )
        )
    )
    for name, expected in (
        ('target_rate_e_per_s', target_rate),
        ('realized_rate_e_per_s', float(contract['realized_rate_e_per_s'])),
    ):
        if abs(realized_rate / expected - 1.0) > RATE_CONTRACT_RELATIVE_TOLERANCE:
            raise ValueError(
                f'Asset {asset_path} re-renders {realized_rate} e-/s against '
                f'its stored {name} of {expected} e-/s'
            )
    return contract


def write_asset(path, sb, pixel_scale_arcsec, provenance):
    """Write the version-one source-image NPZ asset.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        Output path ending in ``.npz``.
    sb : `numpy.ndarray`
        Unit-integral surface-brightness image.
    pixel_scale_arcsec : `float`
        Asset pixel scale in arcseconds.
    provenance : `dict`
        Free-form preparation provenance.

    Returns
    -------
    path : `pathlib.Path`
        Resolved output path.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != '.npz':
        raise ValueError(f"Output asset path must end in .npz: {path}")
    sb = _finite_2d_image(sb, 'sb')
    if np.any(sb < 0.0):
        raise ValueError('sb must be non-negative')
    if any(dimension < 8 or dimension > 4096 for dimension in sb.shape):
        raise ValueError('sb dimensions must be between 8 and 4096 pixels')
    pixel_scale_arcsec = float(pixel_scale_arcsec)
    if not math.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0.0:
        raise ValueError('pixel_scale_arcsec must be positive and finite')
    if not np.isclose(
        pixel_scale_arcsec**2 * float(sb.sum()),
        1.0,
        rtol=1.0e-8,
        atol=0.0,
    ):
        raise ValueError('sb must have unit integral at pixel_scale_arcsec')
    if not isinstance(provenance, dict):
        raise ValueError('provenance must be a dict')
    metadata = {'format_version': 1, 'provenance': provenance}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        sb=np.asarray(sb, dtype=np.float64),
        pixel_scale_arcsec=np.asarray(pixel_scale_arcsec, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def _provenance_path(path):
    """Return a repository-relative path where one exists.

    Paths inside the repository are recorded relative to its root so that
    an asset prepared from repository inputs is byte-identical on every
    machine; paths outside it are recorded absolute.
    """
    path = Path(path).expanduser().resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256(path):
    """Return the full SHA-256 hex digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _argument_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--target-half-light-arcsec', type=float, required=True)
    parser.add_argument('--bin', type=int, default=1)
    parser.add_argument('--flip-y', action='store_true')
    parser.add_argument('--border-frac', type=float, default=0.1)
    parser.add_argument('--k-sigma', type=float, default=2.0)
    parser.add_argument('--catalog-id', default='')
    parser.add_argument('--note', default='')
    parser.add_argument(
        '--rate-contract',
        action='store_true',
        help=(
            'solve and store the discrete detected-rate contract: render the '
            'asset unlensed on the production grid and normalize total_flux '
            'so the discrete pixel sum equals the committed detected rate'
        ),
    )
    parser.add_argument(
        '--rate-contract-reference',
        default=str(PROJECT_ROOT / OBSERVING_REFERENCE_RELPATH),
        help='observing reference supplying the target detected rate',
    )
    parser.add_argument(
        '--rate-contract-scene',
        default=str(PROJECT_ROOT / PRODUCTION_SCENE_RELPATH),
        help='production Image-source scene supplying the contract render',
    )
    return parser


def main(argv=None):
    """Run the local source-image preparation CLI.

    Parameters
    ----------
    argv : sequence of `str`, optional
        Arguments excluding the program name. Defaults to ``sys.argv``.

    Returns
    -------
    status : `int`
        Zero on success.
    """
    args = _argument_parser().parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    image = load_input_image(input_path)
    if args.flip_y:
        image = np.flipud(image)
    image, crop = bin_image(image, args.bin)
    image, background = subtract_background(image, args.border_frac)
    image, mask_threshold, component_size = footprint_mask(image, args.k_sigma)
    image, centroid_shift = centre_on_centroid(image)
    r_half_pixels = half_light_radius_pixels(image)
    pixel_scale_arcsec = rescale_to_half_light(
        args.target_half_light_arcsec,
        r_half_pixels,
    )
    sb = normalize_unit_flux(image, pixel_scale_arcsec)
    provenance = {
        'input_path': _provenance_path(input_path),
        'output_path': _provenance_path(output_path),
        'input_sha256': _sha256(input_path),
        'target_half_light_arcsec': float(args.target_half_light_arcsec),
        'bin': int(args.bin),
        'flip_y': bool(args.flip_y),
        'border_frac': float(args.border_frac),
        'k_sigma': float(args.k_sigma),
        'catalog_id': str(args.catalog_id),
        'note': str(args.note),
        'background': background,
        'mask_threshold': mask_threshold,
        'mask_component_size': component_size,
        'crop': crop,
        'centroid_shift': list(centroid_shift),
        'r_half_pixels': r_half_pixels,
        'pixel_scale_arcsec': pixel_scale_arcsec,
        'script_version': SCRIPT_VERSION,
    }
    if args.rate_contract:
        provenance['rate_contract'] = _solved_rate_contract(
            sb,
            pixel_scale_arcsec,
            args.rate_contract_reference,
            args.rate_contract_scene,
        )
    write_asset(output_path, sb, pixel_scale_arcsec, provenance)
    print(f"{'background':<24} {background:.12g}")
    print(f"{'r_half_pixels':<24} {r_half_pixels:.12g}")
    print(f"{'pixel_scale_arcsec':<24} {pixel_scale_arcsec:.12g}")
    print(f"{'pixel_scale^2 * sb.sum':<24} {pixel_scale_arcsec**2 * sb.sum():.12g}")
    if args.rate_contract:
        contract = verify_asset_rate_contract(
            output_path,
            args.rate_contract_scene,
            args.rate_contract_reference,
        )
        print(f"{'target_rate_e_per_s':<24} {contract['target_rate_e_per_s']:.12g}")
        print(f"{'realized_rate_e_per_s':<24} {contract['realized_rate_e_per_s']:.12g}")
        print(f"{'contract_total_flux':<24} {contract['total_flux']:.12g}")
        print(
            f"{'discrete_mapping_ratio':<24} "
            f"{contract['discrete_mapping_ratio']:.12g}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
