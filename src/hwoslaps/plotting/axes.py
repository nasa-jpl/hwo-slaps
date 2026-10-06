"""Axes creation and pixel geometry shared by the optional plotting consumers."""


__all__ = ["axes_or_new", "pixel_extent"]


def axes_or_new(ax):
    """Return caller Axes, or create one with an on-demand pyplot import."""
    if ax is None:
        from matplotlib import pyplot as plt
        return plt.subplots()[1]
    return ax


def pixel_extent(shape, pixel_scale):
    """Centred (left, right, bottom, top) bounds for (ny,nx) pixels at one physical scale."""
    ny, nx = shape
    return (-nx * pixel_scale / 2, nx * pixel_scale / 2,
            -ny * pixel_scale / 2, ny * pixel_scale / 2)
