"""The one random element of the forward model: the detector noise draw."""

from __future__ import annotations

import numpy as np

from ..seeding import root_rng
from .expected import Exposure

__all__ = ["draw_noisy_adu"]


def draw_noisy_adu(counts_e: np.ndarray, exposure: Exposure, seed: int) -> np.ndarray:
    """One detector realization in ADU of the expected counts ``counts_e`` (electrons).

    The generator is ``root_rng(seed)`` (``numpy.random.default_rng(seed)``), used for one
    Poisson draw over the whole array and then one normal read-noise draw of the same shape,
    always made; this is the paper's byte sequence for a given seed.
    """
    rng = root_rng(seed)
    detected = rng.poisson(counts_e).astype(float)
    final = detected + rng.normal(0.0, exposure.read_sigma_e, size=detected.shape)
    return final / exposure.detector.gain_e_per_adu
