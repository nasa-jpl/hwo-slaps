"""Explicit uniform subpixel sampling shared by generation and fitting."""

from __future__ import annotations

import numpy as np

LEGACY_SUB_SIZE = 4


def positive_sub_size(value):
    """Validate a uniform number of subpixels per pixel axis."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("over_sample_size must be a positive integer")
    return int(value)


def configured_sub_size(grid_config):
    """Preserve the historical LP4 default, with an explicit override."""
    return positive_sub_size(grid_config.get("over_sample_size", LEGACY_SUB_SIZE))


def actual_sub_size(grid):
    """Read uniform sampling from a constructed grid; reject adaptive grids."""
    if grid is None:
        return None
    value = grid.over_sample_size
    value = getattr(value, "array", value)
    values = np.asarray(value)
    if values.size == 0:
        return None
    if not np.all(np.isfinite(values)) or not np.all(values == values.flat[0]):
        raise ValueError("A uniform finite over_sample_size is required")
    size = float(values.flat[0])
    if not size.is_integer() or size < 1:
        raise ValueError("Invalid actual over_sample_size")
    return int(size)
