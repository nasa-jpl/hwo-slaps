"""Image assets: the storage format of pixelized source light, its loader and its preparation.

An asset is an ``.npz`` file with exactly the members ``sb`` (2-D float64, each side in
[8, 4096], finite, non-negative, rows toward +y and columns toward +x),
``pixel_scale_arcsec`` (a float64 scalar) and ``metadata_json`` (a 0-d string array holding
a JSON object with integer ``format_version`` 1 and a ``provenance`` object). The surface
brightness integrates to one: ``pixel_scale_arcsec**2 * sb.sum() == 1`` within 1e-8.

``load_image_asset`` keeps a bounded memo keyed by the file's SHA-256, so a file rewritten in
place is read again by the next load. Inside one preparation renders use the assets that
preparation loaded (``build_scene(..., assets=)``).
"""

from __future__ import annotations

import json
import math
import types
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..identity import array_digest, file_digest

__all__ = ["ASSET_FORMAT_VERSION", "ImageAsset", "frozen_value", "load_image_asset", "prepare_image_asset"]

ASSET_FORMAT_VERSION = 1
_ASSET_MEMBERS = {"sb", "pixel_scale_arcsec", "metadata_json"}
_MIN_SIDE, _MAX_SIDE = 8, 4096
_NORMALIZATION_RTOL = 1.0e-8
_MEMO_LIMIT = 8
_MEMO: OrderedDict[str, ImageAsset] = OrderedDict()


def frozen_value(value: Any) -> Any:
    """A read-only copy of a JSON-like value, at every depth.

    Mappings become read-only mapping proxies, lists and tuples become tuples. Scene component
    values and asset metadata are frozen this way.
    """
    if isinstance(value, Mapping):
        return types.MappingProxyType({key: frozen_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(frozen_value(item) for item in value)
    return value


@dataclass(frozen=True, eq=False)
class ImageAsset:
    """A validated asset: read-only samples, pixel scale, metadata and the digest of its content.

    ``digest`` is the file's SHA-256 for a loaded asset and ``identity.array_digest`` of ``sb``
    for a prepared one. Assets compare and hash by identity (``digest`` names their content).
    """

    sb: np.ndarray
    pixel_scale_arcsec: float
    metadata: Mapping[str, Any]
    digest: str


def _validated(sb: np.ndarray, pixel_scale: float, metadata: Any, digest: str, source: str) -> ImageAsset:
    if sb.ndim != 2:
        raise ValueError(f"{source}: sb must be a 2-D array, got shape {sb.shape}")
    if not all(_MIN_SIDE <= side <= _MAX_SIDE for side in sb.shape):
        raise ValueError(f"{source}: sb sides must lie between {_MIN_SIDE} and {_MAX_SIDE} pixels, got {sb.shape}")
    if not np.all(np.isfinite(sb)):
        raise ValueError(f"{source}: sb values must be finite")
    if np.any(sb < 0.0):
        raise ValueError(f"{source}: sb values must be non-negative")
    if not (math.isfinite(pixel_scale) and pixel_scale > 0.0):
        raise ValueError(f"{source}: pixel_scale_arcsec must be positive and finite, got {pixel_scale!r}")
    integral = pixel_scale**2 * float(sb.sum())
    if not abs(integral - 1.0) <= _NORMALIZATION_RTOL:
        raise ValueError(f"{source}: sb must be normalized so that pixel_scale_arcsec**2 * sb.sum() == 1, got {integral!r}")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: metadata must be a JSON object")
    version = metadata.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != ASSET_FORMAT_VERSION:
        raise ValueError(f"{source}: metadata format_version must be the integer {ASSET_FORMAT_VERSION}, got {version!r}")
    if not isinstance(metadata.get("provenance"), Mapping):
        raise ValueError(f"{source}: metadata provenance must be a JSON object")
    sb.setflags(write=False)
    return ImageAsset(sb=sb, pixel_scale_arcsec=pixel_scale, metadata=frozen_value(metadata), digest=digest)


def _read(path: str, digest: str) -> ImageAsset:
    try:
        with np.load(path, allow_pickle=False) as data:
            members = {name: np.array(data[name]) for name in data.files}
    except (OSError, EOFError, ValueError) as error:
        raise ValueError(f"{path}: not a readable .npz image asset ({error})") from error
    if set(members) != _ASSET_MEMBERS:
        raise ValueError(f"{path}: an image asset holds exactly {sorted(_ASSET_MEMBERS)}, found {sorted(members)}")
    sb, scale, text = members["sb"], members["pixel_scale_arcsec"], members["metadata_json"]
    if sb.dtype != np.float64:
        raise ValueError(f"{path}: sb must be float64, got {sb.dtype}")
    if scale.ndim != 0 or scale.dtype != np.float64:
        raise ValueError(f"{path}: pixel_scale_arcsec must be a float64 scalar, got {scale.dtype} of shape {scale.shape}")
    if text.ndim != 0 or text.dtype.kind not in "US":
        raise ValueError(f"{path}: metadata_json must be a 0-d string array")
    raw = text.item()
    try:
        metadata = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: metadata_json is not valid JSON ({error})") from error
    return _validated(sb, float(scale), metadata, digest, path)


def load_image_asset(path: Any) -> ImageAsset:
    """Load and validate an asset file; repeated loads of unchanged content return one object."""
    location = str(path)
    digest = file_digest(location)
    if digest in _MEMO:
        _MEMO.move_to_end(digest)
        return _MEMO[digest]
    asset = _read(location, digest)
    _MEMO[digest] = asset
    while len(_MEMO) > _MEMO_LIMIT:
        _MEMO.popitem(last=False)
    return asset


# ------------------------------------------------------------------ preparation


def _border_values(image: np.ndarray, border_fraction: float) -> np.ndarray:
    width = max(1, int(math.ceil(border_fraction * min(image.shape))))
    border = np.zeros(image.shape, dtype=bool)
    border[:width, :] = True
    border[-width:, :] = True
    border[:, :width] = True
    border[:, -width:] = True
    return image[border]


def _binned(image: np.ndarray, factor: int) -> tuple[np.ndarray, dict[str, int]]:
    last_rows, last_columns = image.shape[0] % factor, image.shape[1] % factor
    kept_y, kept_x = image.shape[0] - last_rows, image.shape[1] - last_columns
    if kept_y == 0 or kept_x == 0:
        raise ValueError(f"bin_factor {factor} exceeds an image side {image.shape}")
    cropped = image[:kept_y, :kept_x]
    binned = (np.array(cropped, copy=True) if factor == 1
              else cropped.reshape(kept_y // factor, factor, kept_x // factor, factor).mean(axis=(1, 3)))
    return binned, {"last_rows": int(last_rows), "last_columns": int(last_columns)}


def _footprint(image: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    labels, count = ndimage.label(image > threshold, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        raise ValueError("no source footprint lies above the detection threshold")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    component = labels == int(np.argmax(sizes))
    if component[0, :].any() or component[-1, :].any() or component[:, 0].any() or component[:, -1].any():
        raise ValueError("the largest source footprint touches the image edge; supply a larger cutout")
    dilated = ndimage.binary_dilation(component, iterations=2)
    return np.where(dilated, np.maximum(image, 0.0), 0.0), int(sizes[int(np.argmax(sizes))])


def _centred(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = np.indices(image.shape, dtype=float)
    total = float(image.sum())
    centre_y = int(math.floor(float((rows * image).sum() / total) + 0.5))
    centre_x = int(math.floor(float((cols * image).sum() / total) + 0.5))
    half_y = min(centre_y, image.shape[0] - 1 - centre_y)
    half_x = min(centre_x, image.shape[1] - 1 - centre_x)
    cropped = image[centre_y - half_y:centre_y + half_y + 1, centre_x - half_x:centre_x + half_x + 1]
    side = max(cropped.shape)
    pad_y, pad_x = side - cropped.shape[0], side - cropped.shape[1]
    centred = np.pad(cropped, ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)), mode="constant")
    out_rows, out_cols = np.indices(centred.shape, dtype=float)
    out_total = float(centred.sum())
    middle = (side - 1) / 2.0
    if (abs(float((out_rows * centred).sum() / out_total) - middle) > 0.5
            or abs(float((out_cols * centred).sum() / out_total) - middle) > 0.5):
        raise ValueError("integer centring could not place the flux centroid within half a pixel of the centre")
    return centred, (side // 2 - centre_y, side // 2 - centre_x)


def _half_light_radius_pixels(image: np.ndarray) -> float:
    total = float(image.sum())
    rows, cols = np.indices(image.shape, dtype=float)
    radii = np.hypot(rows - (image.shape[0] - 1) / 2.0, cols - (image.shape[1] - 1) / 2.0).ravel()
    order = np.argsort(radii, kind="stable")
    radii = radii[order]
    cumulative = np.cumsum(image.ravel()[order])
    target = 0.5 * total
    index = int(np.searchsorted(cumulative, target, side="left"))
    if index == 0:
        raise ValueError("the central pixel holds at least half the flux: the source is unresolved at this sampling; "
                         "use a smaller bin_factor or a finer image")
    return float(np.interp(target, cumulative[index - 1:index + 1], radii[index - 1:index + 1]))


def prepare_image_asset(image: ArrayLike, *, half_light_radius_arcsec: float | None = None,
                        pixel_scale_arcsec: float | None = None, bin_factor: int = 1, border_fraction: float = 0.1,
                        footprint_sigma: float = 2.0, provenance: Mapping[str, Any] | None = None) -> ImageAsset:
    """An asset from a galaxy image: bin, subtract the border background, keep the main footprint,
    centre on the flux centroid, set the scale, normalize to unit integral.

    Row 0 of ``image`` is its bottom row (+y upward, the FITS convention). Binning first removes
    the last ``shape % bin_factor`` rows and columns (the top rows and right columns), recorded
    as ``bin_crop`` in the provenance. Exactly one of ``half_light_radius_arcsec`` (the scale
    then puts the circular half-light radius there) and ``pixel_scale_arcsec`` is given. The
    background is the 3-sigma-clipped median of the border frame of width
    ``ceil(border_fraction * min(shape))``, and the footprint is the largest 8-connected region
    above ``footprint_sigma`` times the clipped border RMS, dilated by two pixels.
    """
    from astropy.stats import sigma_clipped_stats

    if (half_light_radius_arcsec is None) == (pixel_scale_arcsec is None):
        raise ValueError("give exactly one of half_light_radius_arcsec and pixel_scale_arcsec")
    for name, value in (("half_light_radius_arcsec", half_light_radius_arcsec), ("pixel_scale_arcsec", pixel_scale_arcsec)):
        if value is not None and not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be positive and finite, got {value!r}")
    if isinstance(bin_factor, bool) or not isinstance(bin_factor, int) or bin_factor < 1:
        raise ValueError(f"bin_factor must be a positive integer, got {bin_factor!r}")
    if not 0.0 < border_fraction <= 0.5:
        raise ValueError(f"border_fraction must lie in (0, 0.5], got {border_fraction!r}")
    if not (math.isfinite(footprint_sigma) and footprint_sigma >= 0.0):
        raise ValueError(f"footprint_sigma must be finite and non-negative, got {footprint_sigma!r}")
    original = np.asarray(image, dtype=np.float64)
    if original.ndim != 2 or not np.all(np.isfinite(original)):
        raise ValueError(f"image must be a finite 2-D array, got shape {original.shape}")

    binned, crop = _binned(original, bin_factor)
    _, background, _ = sigma_clipped_stats(_border_values(binned, border_fraction), sigma=3.0, maxiters=10)
    subtracted = binned - float(background)
    _, _, border_rms = sigma_clipped_stats(_border_values(subtracted, border_fraction), sigma=3.0, maxiters=10)
    threshold = footprint_sigma * float(border_rms)
    footprint, footprint_pixels = _footprint(subtracted, threshold)
    centred, centroid_shift = _centred(footprint)
    r_half_pixels = _half_light_radius_pixels(centred)
    scale = half_light_radius_arcsec / r_half_pixels if pixel_scale_arcsec is None else float(pixel_scale_arcsec)
    sb = centred / (scale**2 * float(centred.sum()))
    metadata = {
        "format_version": ASSET_FORMAT_VERSION,
        "provenance": {
            "input_shape": list(original.shape),
            "input_digest": array_digest(original),
            "bin_factor": bin_factor,
            "bin_crop": crop,
            "border_fraction": border_fraction,
            "background": float(background),
            "footprint_sigma": footprint_sigma,
            "footprint_threshold": threshold,
            "footprint_pixels": footprint_pixels,
            "centroid_shift": list(centroid_shift),
            "r_half_pixels": r_half_pixels,
            "pixel_scale_arcsec": scale,
            "caller": dict(provenance or {}),
        },
    }
    return _validated(sb, scale, json.loads(json.dumps(metadata)), array_digest(sb), "prepared image")
