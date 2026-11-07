"""Seven-smooth transform sizes for zero-padded linear convolution."""
from __future__ import annotations

def next_fast_length(target: int) -> int:
    """Return the smallest 7-smooth integer greater than or equal to ``target``.

    A 7-smooth integer factors entirely into the radices {2, 3, 5, 7} that
    the FFT backends implement directly; other factors fall back on generic
    or Bluestein transforms that are far slower.  The linear-convolution
    minimum length is a property of the geometry and is frequently a poor
    transform length, so the convolution is zero-padded up to the next fast
    length.  Padding beyond the linear-convolution minimum is exact: the
    extra samples only extend the zero tail of the result.
    """
    target = int(target)
    if target <= 1:
        return 1
    # A power of two is 7-smooth, so this is always a valid upper bound.
    best = 1 << (target - 1).bit_length()
    power_7 = 1
    while power_7 < best:
        power_5 = power_7
        while power_5 < best:
            power_3 = power_5
            while power_3 < best:
                candidate = power_3
                while candidate < target:
                    candidate *= 2
                if candidate < best:
                    best = candidate
                power_3 *= 3
            power_5 *= 5
        power_7 *= 7
    return best


def convolution_fft_shape(
    shape_native: tuple[int, int],
    kernel_shape: tuple[int, int],
) -> tuple[int, int]:
    """Return the zero-padded FFT shape used for one PSF convolution.

    The minimum is the linear-convolution length ``n + k - 1`` per axis;
    each axis is then padded up to the next fast transform length.  The
    crop that extracts the ``same``-mode region depends only on the kernel
    half-size and the native shape, so it is unaffected by the padding.
    """
    return (
        next_fast_length(shape_native[0] + kernel_shape[0] - 1),
        next_fast_length(shape_native[1] + kernel_shape[1] - 1),
    )
