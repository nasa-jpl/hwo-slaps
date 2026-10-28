"""Radial deflection tables and exact affine log-grid interpolation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

R_MIN_ARCSEC = 1.0e-6
MATCHED_SAMPLES = 8192
MISMATCHED_SAMPLES = 32768
MARGIN_FRACTION = 1.0e-6
EXTENSION_SAMPLE_FACTOR = 4

def affine_log_grid_parameters(
    log_radii: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Return affine lookup parameters when one correction is sufficient.

    The stored ``log(logspace())`` knots are not bitwise affine.  Let the
    endpoint-defined affine knot ``k`` be ``origin + k * step``.  If every
    stored knot is less than half a step from that location (including a
    conservative floating-point arithmetic allowance), a query in stored
    interval ``k`` can have an affine floor of only ``k - 1``, ``k``, or
    ``k + 1``.  The hot lookup can therefore recover the exact right-sided
    interval with one comparison against each actual neighbour.

    Unsupported, non-finite, non-monotonic, or too-irregular grids return
    ``None`` so callers can retain the general ``jnp.interp`` path.
    """
    knots = np.asarray(log_radii)
    if knots.dtype != np.dtype(np.float64):
        return None
    if (
        knots.ndim != 1
        or knots.size < 2
        or knots.size > np.iinfo(np.int32).max
        or not np.all(np.isfinite(knots))
    ):
        return None
    knot_steps = np.diff(knots)
    interp_epsilon = np.spacing(np.finfo(np.float64).eps)
    if np.any(knot_steps <= interp_epsilon):
        return None

    origin = float(knots[0])
    step = float((knots[-1] - knots[0]) / (knots.size - 1))
    if not np.isfinite(step) or step <= 0.0:
        return None
    inverse_step = 1.0 / step
    if not np.isfinite(inverse_step) or inverse_step <= 0.0:
        return None

    affine_knots = origin + step * np.arange(knots.size, dtype=np.float64)
    displacement_bins = float(
        np.max(np.abs(knots - affine_knots)) * inverse_step
    )
    # Bound the subtraction and multiplication used for an in-range affine
    # coordinate.  The factor eight is deliberately conservative for the two
    # rounded operations; production tables leave many orders of magnitude
    # more margin than this guard requires.
    eps = np.finfo(np.float64).eps
    arithmetic_bins = 8.0 * eps * (
        (float(np.max(np.abs(knots))) + abs(origin)) * inverse_step
        + float(knots.size)
    )
    total_error_bins = displacement_bins + arithmetic_bins
    if not np.isfinite(total_error_bins) or total_error_bins >= 0.5:
        return None

    # Directly cover the right-sided cases most sensitive to rounding.  This
    # is a construction-time check only; it never scans the table in a rung.
    probes = np.concatenate(
        (
            knots,
            np.nextafter(knots, -np.inf),
            np.nextafter(knots, np.inf),
        )
    )
    probes = probes[np.isfinite(probes)]
    affine_indices = np.floor((probes - origin) * inverse_step).astype(np.int64)
    reference_indices = np.clip(
        np.searchsorted(knots, probes, side="right") - 1,
        0,
        knots.size - 2,
    )
    affine_indices = np.clip(affine_indices, 0, knots.size - 2)
    if np.any(np.abs(affine_indices - reference_indices) > 1):
        return None
    return origin, inverse_step


def _affine_log_grid_interval_index(
    query,
    log_radii,
    origin: float,
    inverse_step: float,
):
    """Locate intervals on a validated float64 near-affine log grid.

    This internal helper requires parameters returned for the same
    ``log_radii`` by :func:`affine_log_grid_parameters`.  Queries are cast to
    the knot dtype before lookup, matching the engine's float64 contract.
    """
    import jax.numpy as jnp

    query = jnp.asarray(query, dtype=log_radii.dtype)
    last_interval = log_radii.shape[0] - 2
    affine_coordinate = (query - origin) * inverse_step
    # Avoid implementation-defined float-to-int conversion for NaN/inf.  The
    # final interpolation still propagates NaN and clamps infinities exactly
    # like jnp.interp; choosing the last bin for NaN matches searchsorted.
    affine_coordinate = jnp.where(
        jnp.isnan(affine_coordinate),
        float(last_interval + 1),
        jnp.clip(affine_coordinate, -1.0, float(last_interval + 1)),
    )
    estimate = jnp.clip(
        jnp.floor(affine_coordinate).astype(jnp.int32),
        0,
        last_interval,
    )

    lower = log_radii[estimate]
    upper = log_radii[estimate + 1]
    corrected = estimate + (query >= upper).astype(jnp.int32)
    corrected = corrected - (query < lower).astype(jnp.int32)
    return jnp.clip(corrected, 0, last_interval)


def _interp_on_affine_log_grid(
    query,
    log_radii,
    values,
    origin: float,
    inverse_step: float,
):
    """Interpolate float64 values using exact stored endpoints.

    ``log_radii`` and ``values`` are the validated float64 production arrays;
    queries are cast to their knot dtype before the right-sided bin lookup.
    """
    import jax.numpy as jnp

    query = jnp.asarray(query, dtype=log_radii.dtype)
    interval = _affine_log_grid_interval_index(
        query,
        log_radii,
        origin,
        inverse_step,
    )
    x_lo = log_radii[interval]
    x_hi = log_radii[interval + 1]
    y_lo = values[interval]
    y_hi = values[interval + 1]
    value_delta = y_hi - y_lo
    knot_delta = x_hi - x_lo
    query_delta = query - x_lo
    interp_epsilon = np.spacing(np.finfo(np.float64).eps)
    zero_delta = jnp.abs(knot_delta) <= interp_epsilon
    interpolated = jnp.where(
        zero_delta,
        y_lo,
        y_lo
        + (query_delta / jnp.where(zero_delta, 1.0, knot_delta)) * value_delta,
    )
    interpolated = jnp.where(query < log_radii[0], values[0], interpolated)
    return jnp.where(query > log_radii[-1], values[-1], interpolated)


@dataclass(frozen=True)
class RadialGrid:
    radii: np.ndarray
    log_radii: np.ndarray
    affine: tuple[float, float] | None
    image_radius_arcsec: float

    @property
    def r_max(self) -> float:
        return min(float(self.radii[-1]), float(np.exp(self.log_radii[-1])))


def radial_grid(over_sampled_yx: np.ndarray, centre_yx: tuple[float, float],
                domain_radius_arcsec: float, *, samples: int) -> RadialGrid:
    coordinates = np.asarray(over_sampled_yx, dtype=float)
    offsets = coordinates - np.asarray(centre_yx)[None, :]
    image_radius = float(np.max(np.hypot(offsets[:, 0], offsets[:, 1])))
    span_y, span_x = float(np.ptp(coordinates[:, 0])), float(np.ptp(coordinates[:, 1]))
    base_r_max = 4.0 * float(np.hypot(span_y, span_x))
    minimum_r_max = R_MIN_ARCSEC * (1.0 + MARGIN_FRACTION)
    base_r_max = max(base_r_max, minimum_r_max)
    required_r_max = image_radius + domain_radius_arcsec
    required_r_max += MARGIN_FRACTION * max(required_r_max, 1.0)
    r_max = max(base_r_max, required_r_max, minimum_r_max)
    sample_count = int(samples)
    if r_max > base_r_max:
        base_log_span = np.log(base_r_max / R_MIN_ARCSEC)
        required_log_span = np.log(r_max / R_MIN_ARCSEC)
        sample_count = int(np.ceil(sample_count * required_log_span / base_log_span))
        sample_count *= EXTENSION_SAMPLE_FACTOR
    radii = np.logspace(np.log10(R_MIN_ARCSEC), np.log10(r_max), sample_count)
    logs = np.log(radii)
    return RadialGrid(radii, logs, affine_log_grid_parameters(logs), image_radius)


def radial_deflection(profile, radii: np.ndarray) -> np.ndarray:
    import autolens as al
    grid = al.Grid2DIrregular(values=np.column_stack([np.zeros_like(radii), radii]))
    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid), dtype=float)
    radial = deflections[:, 1]
    if not np.all(np.isfinite(radial)):
        raise ValueError("subhalo radial deflection table contains non-finite values")
    return radial


def interpolate_log_grid(query, log_radii, values, affine):
    import jax.numpy as jnp
    if affine is None:
        return jnp.interp(query, log_radii, values)
    return _interp_on_affine_log_grid(query, log_radii, values, *affine)


def check_coverage(grid: RadialGrid, positions_yx: np.ndarray, centre_yx) -> None:
    offsets = np.asarray(positions_yx, dtype=float) - np.asarray(centre_yx)[None, :]
    possible_radius = float(np.max(np.hypot(offsets[:, 0], offsets[:, 1]))) + grid.image_radius_arcsec
    if possible_radius > grid.r_max:
        raise ValueError(f"JAX radial table is too small: possible query radius {possible_radius:.6g} "
                         f"exceeds table maximum {grid.r_max:.6g}; prepare with the full position domain")
