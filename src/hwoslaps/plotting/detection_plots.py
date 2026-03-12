"""Detection plotting functions for HWO-SLAPS Module 4.

This module provides visualization functions for subhalo detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from ..modeling.utils_fisher import FisherDetectionData
from .registry import plot_function

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
    """Create a compact Fisher map summary plot when map output is available."""
    if not isinstance(detection_data, FisherDetectionData):
        print("Skipping Fisher map plot: detection_data is not FisherDetectionData.")
        return
    if detection_data.map is None:
        print("Skipping Fisher map plot: Fisher output has no map payload.")
        return

    if run_name is None:
        run_name = plot_config.get('run_name', 'detection')
    output_dir = Path(plot_config['output_dir']) / run_name / 'modeling'
    output_dir.mkdir(parents=True, exist_ok=True)

    map_data = detection_data.map
    positions = map_data.positions_yx
    snr = map_data.snr_asimov_by_position
    angles = np.degrees(np.arctan2(positions[:, 0], positions[:, 1]))
    angles = np.mod(angles, 360.0)
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
    ax.set_xlabel('Ring angle (deg)')
    ax.set_ylabel('SNR_asimov')
    ax.set_title('Fisher Map: SNR vs Angle')
    ax.set_xlim(0.0, 360.0)
    ax.grid(alpha=0.3)

    fig.suptitle(
        "Fisher Map Summary: "
        f"median={map_data.median_snr_asimov:.3f}, "
        f"p25={map_data.p25_snr_asimov:.3f}, "
        f"p75={map_data.p75_snr_asimov:.3f}",
        fontsize=10,
    )
    plt.tight_layout()

    save_path = output_dir / 'fisher_map_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Fisher map summary plot: {save_path}")
