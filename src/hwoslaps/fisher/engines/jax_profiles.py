"""Verified JAX light evaluators using the paper's analytic and bilinear expressions."""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence
import jax.numpy as jnp
import numpy as np

_SOURCE_VERIFY_POINTS = 128
_IMAGE_SUPPORT_GRID_SIZE = 16
LightEvaluator = Callable[[Any], Any]

def _sersic_constant(sersic_index: float) -> float:
    n = float(sersic_index)
    return (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n**2)
        + 131.0 / (1148175.0 * n**3)
        - 2194697.0 / (30690717750.0 * n**4)
    )
def _sersic_params_from_profile(light_profile) -> Dict[str, float]:
    centre = tuple(float(v) for v in light_profile.centre)
    ell = tuple(float(v) for v in light_profile.ell_comps)
    sersic_index = float(getattr(light_profile, "sersic_index", 1.0))
    return {
        "centre_y": centre[0],
        "centre_x": centre[1],
        "e1": ell[0],
        "e2": ell[1],
        "intensity": float(light_profile.intensity),
        "effective_radius": float(light_profile.effective_radius),
        "sersic_index": sersic_index,
        "sersic_b": float(
            getattr(light_profile, "sersic_constant", _sersic_constant(sersic_index))
        ),
    }

def _sersic_brightness_np(params: Dict[str, float], points: np.ndarray) -> np.ndarray:
    y = points[:, 0] - params["centre_y"]
    x = points[:, 1] - params["centre_x"]
    fac = np.hypot(params["e1"], params["e2"])
    axis_ratio = (1.0 - fac) / (1.0 + fac)
    angle = 0.5 * np.arctan2(params["e1"], params["e2"])
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)
    radius = np.sqrt(axis_ratio) * np.sqrt(x_rot**2 + (y_rot / axis_ratio) ** 2)
    return params["intensity"] * np.exp(
        -params["sersic_b"]
        * (
            (radius / params["effective_radius"]) ** (1.0 / params["sersic_index"])
            - 1.0
        )
    )

def _image_params_from_profile(light_profile) -> Dict[str, Any]:
    """Extract one ImageSource into immutable JAX-kernel parameters."""
    sb = np.asarray(light_profile.sb, dtype=float)
    theta = np.deg2rad(float(light_profile.rotation_deg))
    return {
        "sb_padded": np.pad(sb, 1, mode="constant"),
        "pixel_scale_arcsec": float(light_profile.pixel_scale_arcsec),
        "size_scale": float(light_profile.size_scale),
        "amplitude": float(light_profile.total_flux * light_profile.flux_scale),
        "centre_y": float(light_profile.centre[0]),
        "centre_x": float(light_profile.centre[1]),
        "rotation_cos": float(np.cos(theta)),
        "rotation_sin": float(np.sin(theta)),
        "row_c": (sb.shape[0] - 1) / 2.0,
        "col_c": (sb.shape[1] - 1) / 2.0,
    }

def _bilinear_gather_np(
    array: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Evaluate a padded array bilinearly with exact zero outside."""
    in_bounds = (
        (rows >= 0.0)
        & (rows <= array.shape[0] - 1)
        & (cols >= 0.0)
        & (cols <= array.shape[1] - 1)
    )
    row0 = np.floor(rows).astype(int)
    col0 = np.floor(cols).astype(int)
    row1 = row0 + 1
    col1 = col0 + 1
    row0_clip = np.clip(row0, 0, array.shape[0] - 1)
    row1_clip = np.clip(row1, 0, array.shape[0] - 1)
    col0_clip = np.clip(col0, 0, array.shape[1] - 1)
    col1_clip = np.clip(col1, 0, array.shape[1] - 1)
    row_weight = rows - row0
    col_weight = cols - col0
    values = (
        (1.0 - row_weight)
        * (1.0 - col_weight)
        * array[row0_clip, col0_clip]
        + (1.0 - row_weight)
        * col_weight
        * array[row0_clip, col1_clip]
        + row_weight
        * (1.0 - col_weight)
        * array[row1_clip, col0_clip]
        + row_weight
        * col_weight
        * array[row1_clip, col1_clip]
    )
    return np.where(in_bounds, values, 0.0)

def _image_brightness_np(
    params: Dict[str, Any],
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate the image-source convention in pure NumPy."""
    dy = points[:, 0] - params["centre_y"]
    dx = points[:, 1] - params["centre_x"]
    u = dx * params["rotation_cos"] + dy * params["rotation_sin"]
    v = -dx * params["rotation_sin"] + dy * params["rotation_cos"]
    scale = params["pixel_scale_arcsec"] * params["size_scale"]
    cols = u / scale + params["col_c"] + 1.0
    rows = v / scale + params["row_c"] + 1.0
    return params["amplitude"] * _bilinear_gather_np(
        params["sb_padded"], rows, cols
    )

def _image_support_points(params: Dict[str, Any]) -> np.ndarray:
    """Build deterministic sky-plane probes for an image-source asset."""
    sb = np.asarray(params["sb_padded"])[1:-1, 1:-1]
    row_grid = np.linspace(
        0,
        sb.shape[0] - 1,
        min(sb.shape[0], _IMAGE_SUPPORT_GRID_SIZE),
        dtype=int,
    )
    col_grid = np.linspace(
        0,
        sb.shape[1] - 1,
        min(sb.shape[1], _IMAGE_SUPPORT_GRID_SIZE),
        dtype=int,
    )
    grid_rows, grid_cols = np.meshgrid(row_grid, col_grid, indexing="ij")
    pixel_indices = np.column_stack((grid_rows.ravel(), grid_cols.ravel()))
    grid_values = sb[pixel_indices[:, 0], pixel_indices[:, 1]]
    supported_grid = pixel_indices[grid_values != 0.0]

    first_nonzero = None
    for row, values in enumerate(sb):
        nonzero = np.flatnonzero(values != 0.0)
        if nonzero.size:
            first_nonzero = (row, int(nonzero[0]))
            break
    if first_nonzero is not None:
        pixel_indices = np.vstack((supported_grid, first_nonzero))
    elif supported_grid.size:
        pixel_indices = supported_grid
    pixel_indices = np.unique(pixel_indices, axis=0)

    offsets = np.array(
        [[0.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [0.0, -0.5], [0.0, 0.5]],
        dtype=float,
    )
    row_col = pixel_indices[:, None, :] + offsets[None, :, :]
    row_col = row_col.reshape(-1, 2)
    scale = params["pixel_scale_arcsec"] * params["size_scale"]
    u = (row_col[:, 1] - params["col_c"]) * scale
    v = (row_col[:, 0] - params["row_c"]) * scale
    dx = u * params["rotation_cos"] - v * params["rotation_sin"]
    dy = u * params["rotation_sin"] + v * params["rotation_cos"]
    return np.column_stack(
        (params["centre_y"] + dy, params["centre_x"] + dx)
    )


def sersic_evaluator(profile: Any) -> LightEvaluator:
    params = _sersic_params_from_profile(profile)
    def evaluate(traced):
        y = traced[:, 0] - params["centre_y"]
        x = traced[:, 1] - params["centre_x"]
        fac = np.hypot(params["e1"], params["e2"])
        axis_ratio = (1.0 - fac) / (1.0 + fac)
        angle = 0.5 * np.arctan2(params["e1"], params["e2"])
        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)
        radius_ell = jnp.sqrt(axis_ratio) * jnp.sqrt(x_rot**2 + (y_rot / axis_ratio) ** 2)
        return params["intensity"] * jnp.exp(
            -params["sersic_b"] * ((radius_ell / params["effective_radius"]) ** (1.0 / params["sersic_index"]) - 1.0))
    return evaluate


def image_evaluator(profile: Any) -> LightEvaluator:
    params = _image_params_from_profile(profile)
    params["sb_padded"] = jnp.asarray(params["sb_padded"])
    def evaluate(traced):
        dy = traced[:, 0] - params["centre_y"]
        dx = traced[:, 1] - params["centre_x"]
        u = dx * params["rotation_cos"] + dy * params["rotation_sin"]
        v = -dx * params["rotation_sin"] + dy * params["rotation_cos"]
        scale = params["pixel_scale_arcsec"] * params["size_scale"]
        cols = u / scale + params["col_c"] + 1.0
        rows = v / scale + params["row_c"] + 1.0
        array = params["sb_padded"]
        in_bounds = ((rows >= 0.0) & (rows <= array.shape[0] - 1) &
                     (cols >= 0.0) & (cols <= array.shape[1] - 1))
        row0 = jnp.floor(rows).astype(jnp.int32)
        col0 = jnp.floor(cols).astype(jnp.int32)
        row1 = row0 + 1
        col1 = col0 + 1
        row0 = jnp.clip(row0, 0, array.shape[0] - 1)
        row1 = jnp.clip(row1, 0, array.shape[0] - 1)
        col0 = jnp.clip(col0, 0, array.shape[1] - 1)
        col1 = jnp.clip(col1, 0, array.shape[1] - 1)
        row_weight = rows - jnp.floor(rows)
        col_weight = cols - jnp.floor(cols)
        interpolated = ((1.0 - row_weight) * (1.0 - col_weight) * array[row0, col0]
                        + (1.0 - row_weight) * col_weight * array[row0, col1]
                        + row_weight * (1.0 - col_weight) * array[row1, col0]
                        + row_weight * col_weight * array[row1, col1])
        return params["amplitude"] * jnp.where(in_bounds, interpolated, 0.0)
    return evaluate


def verification_points(traced_macro: np.ndarray, image_profiles: Sequence[Any]) -> np.ndarray:
    rng = np.random.default_rng(0)
    low, high = traced_macro.min(axis=0), traced_macro.max(axis=0)
    box = rng.uniform(low, high, size=(_SOURCE_VERIFY_POINTS, 2))
    step = max(1, int(np.ceil(traced_macro.shape[0] / 256)))
    parts = [box, traced_macro[::step]]
    parts.extend(_image_support_points(_image_params_from_profile(profile)) for profile in image_profiles)
    return np.vstack(parts)


def build_light_evaluator(profile: Any, points: np.ndarray) -> LightEvaluator:
    import autolens as al
    from ...scene.image_profile import ImageLightProfile

    reference = np.asarray(profile.image_2d_from(grid=al.Grid2DIrregular(values=points)), dtype=float)
    if isinstance(profile, ImageLightProfile):
        if not np.any(reference != 0.0):
            raise ValueError("JAX image evaluator verification sampled no nonzero reference brightness")
        mirror = _image_brightness_np(_image_params_from_profile(profile), points)
        evaluator, tolerance = image_evaluator(profile), 1.0e-9
    elif type(profile) in {al.lp.Exponential, al.lp.Sersic, al.lp.ExponentialSph, al.lp.SersicSph}:
        mirror = _sersic_brightness_np(_sersic_params_from_profile(profile), points)
        evaluator, tolerance = sersic_evaluator(profile), 1.0e-10
    else:
        raise ValueError(f"no JAX light evaluator for {type(profile).__name__}")
    if not np.allclose(mirror, reference, rtol=tolerance, atol=0.0):
        raise ValueError(f"JAX light evaluator cannot reproduce {type(profile).__name__}: convention drift")
    return evaluator
