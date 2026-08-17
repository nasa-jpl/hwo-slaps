"""Figures for one dark-matter subhalo mass-function forecast."""

from pathlib import Path
from typing import Any

import numpy as np


def _thermal_from_half_mode(values):
    """Convert half-mode masses to thermal-relic masses for an axis."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 3.3*np.power(values/3.0e8, -1.0/3.33)


def _half_mode_from_thermal(values):
    """Convert thermal-relic masses to half-mode masses for an axis."""
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 3.0e8*np.power(values/3.3, -3.33)


def plot_expected_detections_vs_mhm(data: Any, path: Any) -> Path:
    """Plot expected detections per lens against half-mode mass.

    Parameters
    ----------
    data : `SubhaloForecastData`
        Forecast containing CDM and WDM expected counts.
    path : path-like
        Explicit figure destination.

    Returns
    -------
    path : `pathlib.Path`
        Written figure path.
    """
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.plot(
        data.mhm_grid_msun,
        data.mu_wdm,
        color="#376996",
        linewidth=2.0,
        label="WDM",
    )
    axis.axhline(
        data.mu_cdm,
        color="#a3531f",
        linestyle="--",
        linewidth=1.5,
        label="CDM",
    )
    axis.set_xscale("log")
    axis.set_xlabel(r"Half-mode mass $M_{\rm hm}$ ($M_\odot$)")
    axis.set_ylabel("Expected detectable subhaloes per lens")
    axis.set_title("Subhalo Detection Forecast")
    axis.grid(alpha=0.3)
    axis.legend()
    secondary = axis.secondary_xaxis(
        "top",
        functions=(_thermal_from_half_mode, _half_mode_from_thermal),
    )
    secondary.set_xlabel(r"Thermal-relic mass $m_{\rm WDM}$ (keV)")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def plot_lenses_to_discriminate(data: Any, path: Any) -> Path:
    """Plot mass-binned and total-count required-lens curves.

    Parameters
    ----------
    data : `SubhaloForecastData`
        Forecast containing required-lens counts.
    path : path-like
        Explicit figure destination.

    Returns
    -------
    path : `pathlib.Path`
        Written figure path.
    """
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    axis.loglog(
        data.mhm_grid_msun,
        data.N_req,
        color="#376996",
        linewidth=2.0,
        label="Mass-binned Poisson",
    )
    axis.loglog(
        data.mhm_grid_msun,
        data.N_req_single_bin,
        color="#a3531f",
        linestyle="--",
        linewidth=1.5,
        label="Total counts only",
    )
    axis.set_xlabel(r"Half-mode mass $M_{\rm hm}$ ($M_\odot$)")
    axis.set_ylabel("Lenses required")
    axis.set_title("CDM--WDM Discrimination Forecast")
    axis.grid(alpha=0.3)
    axis.legend()
    secondary = axis.secondary_xaxis(
        "top",
        functions=(_thermal_from_half_mode, _half_mode_from_thermal),
    )
    secondary.set_xlabel(r"Thermal-relic mass $m_{\rm WDM}$ (keV)")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination
