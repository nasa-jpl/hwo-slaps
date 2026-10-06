"""Axes creation and pixel geometry shared by the optional plotting consumers."""


def axes_or_new(ax):
    if ax is None:
        from matplotlib import pyplot as plt
        return plt.subplots()[1]
    return ax


def pixel_extent(shape, pixel_scale):
    ny, nx = shape
    return (-nx * pixel_scale / 2, nx * pixel_scale / 2,
            -ny * pixel_scale / 2, ny * pixel_scale / 2)
