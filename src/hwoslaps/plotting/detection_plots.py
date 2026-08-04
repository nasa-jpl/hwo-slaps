"""Detection plotting functions for HWO-SLAPS Module 4.

This module provides visualization functions for subhalo detection results.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from ..modeling.utils_fisher import FisherDetectionData
from .registry import plot_function

SCDD_Q_THRESHOLD = 10.0
SCDD_Z_THRESHOLD = np.sqrt(SCDD_Q_THRESHOLD)


def _modeling_output_dir(plot_config: Dict[str, Any], run_name: str | None) -> Path:
    if run_name is None:
        run_name = plot_config.get("run_name", "detection")
    output_dir = Path(plot_config["output_dir"]) / run_name / "modeling"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _fisher_detection_guard(detection_data: FisherDetectionData, label: str) -> bool:
    if not isinstance(detection_data, FisherDetectionData):
        print(f"Skipping {label}: detection_data is not FisherDetectionData.")
        return False
    return True


def _maybe_format(value: float | None, fmt: str) -> str:
    if value is None:
        return "N/A"
    return format(value, fmt)


def _angle_from_positions(positions_yx: np.ndarray) -> np.ndarray:
    angles = np.degrees(np.arctan2(positions_yx[:, 0], positions_yx[:, 1]))
    return np.mod(angles, 360.0)


@plot_function(
    module="detection",
    detection_mode_only=True,
    description="Compact local Fisher summary panel for local runs",
)
def plot_fisher_local_summary(
    detection_data: FisherDetectionData,
    plot_config: Dict[str, Any],
    run_name: str = None,
) -> None:
    """Create a compact Fisher summary for local and both modes."""
    if not _fisher_detection_guard(detection_data, "Fisher local summary plot"):
        return
    if detection_data.local is None:
        print("Skipping Fisher local summary plot: Fisher output has no local payload.")
        return

    local = detection_data.local
    output_dir = _modeling_output_dir(plot_config, run_name)

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(111)
    ax.axis("off")

    rows = [
        ("Mode", detection_data.mode),
        ("Pixels analyzed", f"{detection_data.pixels_unmasked}"),
        ("Nuisance directions", f"{detection_data.n_nuisance}"),
        ("Condition number", f"{detection_data.gram_condition_number:.3e}"),
        ("SNR mask threshold", f"{detection_data.snr_threshold:.3f}"),
        ("SNR_asimov", f"{local.snr_asimov:.4f}"),
        ("SCDD threshold", f"q_F > {SCDD_Q_THRESHOLD:.1f}; Z_F > {SCDD_Z_THRESHOLD:.3f}"),
        ("DeltaChi2 raw", f"{local.delta_chi2_raw:.4f}"),
        ("DeltaChi2 profiled", f"{local.delta_chi2_profiled:.4f}"),
        ("Profiling degradation", f"{local.degradation:.4f}"),
        ("Sigma amplitude profiled", _maybe_format(local.sigma_amplitude_profiled, ".4g")),
        ("Local one-sided p-value", _maybe_format(local.local_p_one_sided, ".4g")),
        ("Absorbed fraction", _maybe_format(local.absorbed_fraction, ".4f")),
        ("Whitened residual norm", _maybe_format(local.residual_norm_whitened, ".4g")),
        ("Subhalo mass (M_sun)", _maybe_format(local.true_subhalo_mass, ".3e")),
        (
            "Subhalo position (y, x)",
            "N/A" if local.true_subhalo_position is None else str(local.true_subhalo_position),
        ),
    ]

    ax.text(
        0.02,
        0.97,
        "Fisher Local Summary",
        fontsize=17,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.02,
        0.91,
        "\n".join(f"{label}: {value}" for label, value in rows),
        family="monospace",
        fontsize=11,
        va="top",
        transform=ax.transAxes,
    )

    if local.psf_mode_scan is not None and len(local.psf_mode_scan.couplings) > 0:
        couplings = sorted(
            list(local.psf_mode_scan.couplings),
            key=lambda item: abs(item.one_sigma_z) if item.one_sigma_z is not None else abs(item.z_per_unit),
            reverse=True,
        )[:5]
        coupling_lines = []
        for coupling in couplings:
            line = f"{coupling.mode_name}: z/unit={coupling.z_per_unit:.4g}"
            if coupling.one_sigma_z is not None:
                line += f", z(1sigma)={coupling.one_sigma_z:.4g}"
            if coupling.tolerance_for_zmax is not None:
                line += f", tol={coupling.tolerance_for_zmax:.4g}"
            coupling_lines.append(line)
        if local.psf_mode_scan.rms_spurious_z is not None:
            coupling_lines.append(f"RMS spurious z={local.psf_mode_scan.rms_spurious_z:.4g}")
        ax.text(
            0.52,
            0.91,
            "Leading PSF/Systematic Couplings\n\n" + "\n".join(coupling_lines),
            family="monospace",
            fontsize=10.5,
            va="top",
            transform=ax.transAxes,
        )

    save_path = output_dir / "fisher_local_summary.png"
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Fisher local summary plot: {save_path}")


@plot_function(
    module="detection",
    detection_mode_only=True,
    description="PSF/systematic mode scan bar chart for Fisher runs",
)
def plot_fisher_psf_mode_scan(
    detection_data: FisherDetectionData,
    plot_config: Dict[str, Any],
    run_name: str = None,
) -> None:
    """Plot PSF-mode scan outputs for local runs."""
    if not _fisher_detection_guard(detection_data, "Fisher PSF mode scan plot"):
        return
    if detection_data.local is None or detection_data.local.psf_mode_scan is None:
        print("Skipping Fisher PSF mode scan plot: Fisher output has no PSF mode scan payload.")
        return

    mode_scan = detection_data.local.psf_mode_scan
    couplings = list(mode_scan.couplings)
    if not couplings:
        print("Skipping Fisher PSF mode scan plot: no mode couplings are available.")
        return

    couplings = sorted(
        couplings,
        key=lambda item: abs(item.one_sigma_z) if item.one_sigma_z is not None else abs(item.z_per_unit),
    )
    mode_names = [coupling.mode_name for coupling in couplings]
    z_per_unit = np.asarray([coupling.z_per_unit for coupling in couplings], dtype=float)
    one_sigma_z = np.asarray(
        [np.nan if coupling.one_sigma_z is None else coupling.one_sigma_z for coupling in couplings],
        dtype=float,
    )
    tolerance = np.asarray(
        [np.nan if coupling.tolerance_for_zmax is None else coupling.tolerance_for_zmax
         for coupling in couplings],
        dtype=float,
    )

    output_dir = _modeling_output_dir(plot_config, run_name)
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4.5, 0.6 * len(couplings) + 2.5)), sharey=True)
    ypos = np.arange(len(couplings))

    axes[0].barh(ypos, z_per_unit, color="#376996")
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Spurious z per unit")
    axes[0].set_xlabel("z / unit")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(mode_names)

    finite_one_sigma = np.isfinite(one_sigma_z)
    if np.any(finite_one_sigma):
        axes[1].barh(ypos[finite_one_sigma], one_sigma_z[finite_one_sigma], color="#a3531f")
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Spurious z at 1 sigma")
    axes[1].set_xlabel("z(1sigma)")

    finite_tol = np.isfinite(tolerance)
    if np.any(finite_tol):
        axes[2].barh(ypos[finite_tol], tolerance[finite_tol], color="#5a8f29")
    axes[2].set_title("Tolerance for z budget")
    axes[2].set_xlabel("Mode amplitude")
    if mode_scan.z_tolerance is not None:
        axes[2].axvline(0.0, color="black", linewidth=0.8)

    title = "Fisher PSF Mode Scan"
    if mode_scan.z_tolerance is not None:
        title += f" (z_max={mode_scan.z_tolerance:.3g})"
    fig.suptitle(title, fontsize=12)
    fig.text(
        0.5,
        0.01,
        "Mode amplitudes use the units configured for each PSF family; "
        "tolerance is the amplitude for the stated z budget.",
        ha="center",
        fontsize=8.5,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.98))

    save_path = output_dir / "fisher_psf_mode_scan.png"
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Fisher PSF mode scan plot: {save_path}")


@plot_function(
    module='detection',
    detection_mode_only=True,
    description="Fisher map detectability summary for ring-position scans",
)
def plot_fisher_detection_map_summary(
    detection_data: FisherDetectionData,
    plot_config: Dict[str, Any],
    run_name: str = None,
) -> None:
    """Create a compact Fisher map summary when map output is available."""
    if not _fisher_detection_guard(detection_data, "Fisher map plot"):
        return
    if detection_data.map is None:
        print("Skipping Fisher map plot: Fisher output has no map payload.")
        return

    output_dir = _modeling_output_dir(plot_config, run_name)

    map_data = detection_data.map
    positions = map_data.positions_yx
    snr = map_data.snr_asimov_by_position
    angles = _angle_from_positions(positions)
    order = np.argsort(angles)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    scatter = ax.scatter(
        positions[:, 1],
        positions[:, 0],
        c=snr,
        cmap='viridis',
        s=50,
        edgecolors='k',
        linewidths=0.4,
    )
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    ax.set_title('Fisher Map: Candidate Positions')
    ax.set_aspect('equal')
    plt.colorbar(scatter, ax=ax, fraction=0.046, label='SNR_asimov')

    ax = axes[1]
    ax.plot(angles[order], snr[order], marker='o', linestyle='-', linewidth=1.5, markersize=4)
    ax.axhline(SCDD_Z_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.2, label="SCDD sqrt(10)")
    ax.set_xlabel('Ring angle (deg)')
    ax.set_ylabel('SNR_asimov')
    ax.set_title('Fisher Map: SNR vs Angle')
    ax.set_xlim(0.0, 360.0)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Fisher Map Summary: "
        f"median={map_data.median_snr_asimov:.3f}, "
        f"p25={map_data.p25_snr_asimov:.3f}, "
        f"p75={map_data.p75_snr_asimov:.3f}",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.01,
        r"Detection threshold shown as $Z_F=\sqrt{10}$, equivalent to $q_F=10$.",
        ha="center",
        fontsize=8.5,
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.96))

    save_path = output_dir / 'fisher_map_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Fisher map summary plot: {save_path}")


@plot_function(
    module="detection",
    detection_mode_only=True,
    description="Fisher map degradation summary for profiled runs",
)
def plot_fisher_map_degradation(
    detection_data: FisherDetectionData,
    plot_config: Dict[str, Any],
    run_name: str = None,
) -> None:
    """Visualize where nuisance profiling degrades map-level detectability."""
    if not _fisher_detection_guard(detection_data, "Fisher map degradation plot"):
        return
    if detection_data.map is None:
        print("Skipping Fisher map degradation plot: Fisher output has no map payload.")
        return

    map_data = detection_data.map
    if map_data.degradation_by_position is None:
        print("Skipping Fisher map degradation plot: degradation data are unavailable.")
        return

    output_dir = _modeling_output_dir(plot_config, run_name)
    positions = map_data.positions_yx
    degradation = map_data.degradation_by_position
    angles = _angle_from_positions(positions)
    order = np.argsort(angles)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    scatter = ax.scatter(
        positions[:, 1],
        positions[:, 0],
        c=degradation,
        cmap="magma",
        s=55,
        edgecolors="k",
        linewidths=0.4,
    )
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_title("Map Degradation by Position")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, fraction=0.046, label="degradation")

    ax = axes[1]
    ax.plot(angles[order], degradation[order], marker="o", linestyle="-", linewidth=1.5, markersize=4)
    ax.set_xlabel("Ring angle (deg)")
    ax.set_ylabel("degradation")
    ax.set_title("Profiling Degradation vs Angle")
    ax.set_xlim(0.0, 360.0)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Fisher Map Degradation: "
        f"min={np.min(degradation):.3f}, "
        f"median={np.median(degradation):.3f}, "
        f"max={np.max(degradation):.3f}",
        fontsize=10,
    )
    plt.tight_layout()

    save_path = output_dir / "fisher_map_degradation.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Fisher map degradation plot: {save_path}")


@plot_function(
    module="detection",
    detection_mode_only=True,
    description="2D Fisher sensitivity grid map with detectable-area contour",
)
def plot_fisher_detection_grid_map(
    detection_data: FisherDetectionData,
    plot_config: Dict[str, Any],
    run_name: str = None,
) -> None:
    """Plot the 2D sensitivity grid map when grid output is available."""
    if not _fisher_detection_guard(detection_data, "Fisher grid map plot"):
        return
    if detection_data.grid_map is None:
        print("Skipping Fisher grid map plot: Fisher output has no grid map payload.")
        return

    output_dir = _modeling_output_dir(plot_config, run_name)
    grid = detection_data.grid_map

    half_cell = 0.5 * grid.spacing_arcsec
    extent = (
        grid.x_coords[0] - half_cell,
        grid.x_coords[-1] + half_cell,
        grid.y_coords[0] - half_cell,
        grid.y_coords[-1] + half_cell,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    z_map = np.ma.masked_invalid(grid.z_asimov_2d)
    image = ax.imshow(
        z_map,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    plt.colorbar(image, ax=ax, fraction=0.046, label=r"$Z_F$")

    if np.any(grid.detectable_mask_2d):
        ax.contour(
            grid.x_coords,
            grid.y_coords,
            np.ma.masked_invalid(grid.q_asimov_2d),
            levels=[grid.detection_q_threshold],
            colors="white",
            linewidths=1.4,
        )

    if grid.lens_einstein_radius is not None:
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        centre_y, centre_x = grid.centre_yx
        ax.plot(
            centre_x + grid.lens_einstein_radius * np.cos(theta),
            centre_y + grid.lens_einstein_radius * np.sin(theta),
            color="#d95f02",
            linestyle="--",
            linewidth=1.2,
            label="Einstein ring",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_aspect("equal")
    mass_label = (
        "" if grid.subhalo_mass is None else f", M={grid.subhalo_mass:.2e} M_sun"
    )
    ax.set_title(
        "Fisher Sensitivity Grid Map\n"
        f"detectable area {grid.detectable_area_arcsec2:.3f} arcsec$^2$ "
        f"(q_F >= {grid.detection_q_threshold:.1f}){mass_label}",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.01,
        "White contour bounds the detectable region; NaN nodes were excluded by the annulus.",
        ha="center",
        fontsize=8.5,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    save_path = output_dir / "fisher_grid_map.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Fisher grid map plot: {save_path}")

    if grid.z_spurious_2d is None:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    z_spurious = np.ma.masked_invalid(grid.z_spurious_2d)
    image = ax.imshow(
        z_spurious,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
    )
    plt.colorbar(image, ax=ax, fraction=0.046, label=r"$Z_{\rm spur}$")

    if (
        grid.false_positive_mask_2d is not None
        and np.any(grid.false_positive_mask_2d)
    ):
        ax.contour(
            grid.x_coords,
            grid.y_coords,
            grid.false_positive_mask_2d.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=1.4,
        )

    if grid.lens_einstein_radius is not None:
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        centre_y, centre_x = grid.centre_yx
        ax.plot(
            centre_x + grid.lens_einstein_radius * np.cos(theta),
            centre_y + grid.lens_einstein_radius * np.sin(theta),
            color="#d95f02",
            linestyle="--",
            linewidth=1.2,
            label="Einstein ring",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.set_aspect("equal")
    mass_label = (
        "" if grid.subhalo_mass is None else f", M={grid.subhalo_mass:.2e} M_sun"
    )
    ax.set_title(
        "PSF-Mismatch Spurious Significance\n"
        f"false-positive area {grid.false_positive_area_arcsec2:.3f} arcsec$^2$ "
        f"(q >= {grid.detection_q_threshold:.1f}){mass_label}",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.01,
        "White contour bounds the one-sided false-positive region; "
        "NaN nodes were excluded by the annulus.",
        ha="center",
        fontsize=8.5,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))

    spurious_path = output_dir / "fisher_grid_map_spurious.png"
    plt.savefig(spurious_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Fisher spurious grid map plot: {spurious_path}")
