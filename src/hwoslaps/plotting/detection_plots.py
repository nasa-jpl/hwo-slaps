"""Detection plotting functions for HWO-SLAPS Module 4.

This module provides visualization functions for subhalo detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any

from ..observation.utils import ObservationData
from ..modeling.utils import DetectionData
from ..modeling.chernoff_detector import ChernoffDetectionData
from .registry import plot_function


@plot_function(module='detection', detection_mode_only=True,
               description="6-panel detection analysis: original, with subhalo, difference, SNR, mask, residual")
def plot_detection_comparison(
    detection_data: DetectionData,
    plot_config: Dict[str, Any],
    obs_baseline: ObservationData,
    obs_test: ObservationData,
) -> None:
    """Create detection visualization using the real observed images.
    
    Panels:
      1) Original Image (No Subhalo)      -> obs_baseline.data
      2) Image with Subhalo               -> obs_test.data
      3) Difference Image (test - base)
      4) SNR Map                          -> detection_data.snr_array_2d
      5) Detection Mask                   -> detection_data.snr_mask_2d
      6) Residual in Detection Region     -> (test - base) masked by detection mask
    """
    # Create output directory - get run_name from detection_data.config if available
    run_name = 'detection'
    if hasattr(detection_data, 'config') and detection_data.config and 'run_name' in detection_data.config:
        run_name = detection_data.config['run_name']
    elif 'run_name' in plot_config:
        run_name = plot_config['run_name']
    
    output_dir = Path(plot_config['output_dir']) / run_name / 'modeling'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization identical to prototype lines 373-432
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Subhalo pixel marker (if truth available)
    if detection_data.has_subhalo_truth:
        subhalo_x, subhalo_y = detection_data.true_subhalo_position
        h, w = detection_data.image_shape
        pixel_x = subhalo_x / detection_data.pixel_scale + w/2
        pixel_y = subhalo_y / detection_data.pixel_scale + h/2
    else:
        pixel_x = pixel_y = None

    # 1) Original image (No Subhalo)
    ax = axes[0, 0]
    base = obs_baseline.data.native
    im1 = ax.imshow(base, origin='lower', cmap='viridis')
    ax.set_title('Original Image (No Subhalo)')
    plt.colorbar(im1, ax=ax, fraction=0.046)
    
    # 2) Image with Subhalo
    ax = axes[0, 1]
    test = obs_test.data.native
    im2 = ax.imshow(test, origin='lower', cmap='viridis')
    if pixel_x is not None:
        ax.scatter(pixel_x, pixel_y, c='red', s=100, marker='x')
    ax.set_title('Image with Subhalo')
    plt.colorbar(im2, ax=ax, fraction=0.046)
    
    # 3) Difference image (test - base)
    ax = axes[0, 2]
    diff_2d = test - base
    max_abs_diff = np.max(np.abs(diff_2d))
    im3 = ax.imshow(diff_2d, origin='lower', cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff)
    if pixel_x is not None:
        ax.scatter(pixel_x, pixel_y, c='black', s=100, marker='x')
    ax.set_title('Difference Image')
    plt.colorbar(im3, ax=ax, fraction=0.046)
    
    # 4. SNR map
    ax = axes[1, 0]
    snr_2d = detection_data.snr_array_2d
    im4 = ax.imshow(snr_2d, origin='lower', cmap='viridis')
    ax.set_title('SNR Map')
    plt.colorbar(im4, ax=ax, fraction=0.046)
    
    # 5. Detection mask
    ax = axes[1, 1]
    mask_2d = detection_data.snr_mask_2d
    im5 = ax.imshow(mask_2d, origin='lower', cmap='binary')
    ax.set_title(f'Detection Mask (SNR > {detection_data.snr_threshold})')
    plt.colorbar(im5, ax=ax, fraction=0.046)
    
    # 6) Residual in detection region (difference masked)
    ax = axes[1, 2]
    masked_residual = np.ma.masked_where(~mask_2d, diff_2d)
    max_abs_residual = np.max(np.abs(diff_2d))
    im6 = ax.imshow(masked_residual, origin='lower', cmap='RdBu_r', vmin=-max_abs_residual, vmax=max_abs_residual)
    ax.set_title('Residual in Detection Region')
    plt.colorbar(im6, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    # Save plot
    save_path = output_dir / 'detection_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detection comparison plot: {save_path}")
    
    # Print summary statistics (lines 435-440 from prototype)
    print("\\nSummary Statistics:")
    print(f"Number of SNR regions: {detection_data.num_regions}")
    print(f"Maximum regional SNR: {detection_data.max_region_snr:.2f}")
    print(f"Pixels in detection mask: {detection_data.pixels_unmasked}")
    print(f"Total pixels: {detection_data.snr_array.size}")
    print(f"Fraction of pixels used: {detection_data.detection_mask_fraction:.3f}")


@plot_function(module='detection', detection_mode_only=True,
               description="6-panel Chernoff detection analysis with fixed-position detection")
def plot_chernoff_detection_comparison(
    detection_data: ChernoffDetectionData,
    plot_config: Dict[str, Any],
    obs_baseline: ObservationData,
    obs_test: ObservationData,
    run_name: str = None,
) -> None:
    """Create detection visualization for Chernoff detection results.
    
    Panels:
      1) Original Image (No Subhalo)      -> obs_baseline.data
      2) Image with Subhalo               -> obs_test.data
      3) Difference Image (test - base)
      4) SNR Map                          -> detection_data.snr_array
      5) Detection Mask                   -> detection_data.result.snr_mask
      6) Residual in Detection Region     -> (test - base) masked by detection mask
    """
    # Create output directory using the same pattern as other plots
    if run_name is None:
        run_name = plot_config.get('run_name', 'detection')
    
    output_dir = Path(plot_config['output_dir']) / run_name / 'modeling'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Subhalo pixel marker (from lensing truth)
    subhalo_x, subhalo_y = detection_data.result.position
    h, w = obs_baseline.data.native.shape
    pixel_scale = 0.05  # Default pixel scale, could be extracted from obs data
    if hasattr(obs_baseline, 'pixel_scale'):
        pixel_scale = obs_baseline.pixel_scale
    pixel_x = subhalo_x / pixel_scale + w/2
    pixel_y = subhalo_y / pixel_scale + h/2

    # 1) Original image (No Subhalo)
    ax = axes[0, 0]
    base = obs_baseline.data.native
    im1 = ax.imshow(base, origin='lower', cmap='viridis')
    ax.set_title('Original Image (No Subhalo)')
    plt.colorbar(im1, ax=ax, fraction=0.046)
    
    # 2) Image with Subhalo
    ax = axes[0, 1]
    test = obs_test.data.native
    im2 = ax.imshow(test, origin='lower', cmap='viridis')
    ax.scatter(pixel_x, pixel_y, c='red', s=100, marker='x')
    ax.set_title('Image with Subhalo')
    plt.colorbar(im2, ax=ax, fraction=0.046)
    
    # 3) Difference image (test - base)
    ax = axes[0, 2]
    diff_2d = test - base
    max_abs_diff = np.max(np.abs(diff_2d))
    im3 = ax.imshow(diff_2d, origin='lower', cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff)
    ax.scatter(pixel_x, pixel_y, c='black', s=100, marker='x')
    ax.set_title('Difference Image')
    plt.colorbar(im3, ax=ax, fraction=0.046)
    
    # 4. SNR map - reshape from 1D to 2D
    ax = axes[1, 0]
    side_length = int(np.sqrt(detection_data.snr_array.size))
    snr_2d = detection_data.snr_array.reshape(side_length, side_length)
    im4 = ax.imshow(snr_2d, origin='lower', cmap='viridis')
    ax.set_title('SNR Map')
    plt.colorbar(im4, ax=ax, fraction=0.046)
    
    # 5. Detection mask - reshape from 1D to 2D
    ax = axes[1, 1]
    mask_2d = detection_data.result.snr_mask.reshape(side_length, side_length)
    im5 = ax.imshow(mask_2d, origin='lower', cmap='binary')
    ax.set_title(f'Detection Mask (SNR > {detection_data.snr_threshold})')
    plt.colorbar(im5, ax=ax, fraction=0.046)
    
    # 6) Residual in detection region (difference masked)
    ax = axes[1, 2]
    masked_residual = np.ma.masked_where(~mask_2d, diff_2d)
    max_abs_residual = np.max(np.abs(diff_2d))
    im6 = ax.imshow(masked_residual, origin='lower', cmap='RdBu_r', vmin=-max_abs_residual, vmax=max_abs_residual)
    ax.set_title('Residual in Detection Region')
    plt.colorbar(im6, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    # Save plot
    save_path = output_dir / 'detection_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detection comparison plot: {save_path}")
    
    # Print summary statistics adapted for Chernoff
    print("\\nSummary Statistics:")
    print(f"Number of SNR regions: {detection_data.num_regions}")
    print(f"Maximum regional SNR: {detection_data.max_region_snr:.2f}")
    print(f"Pixels in detection mask: {detection_data.result.pixels_unmasked}")
    print(f"Total pixels: {detection_data.snr_array.size}")
    frac = detection_data.result.pixels_unmasked / detection_data.snr_array.size if detection_data.snr_array.size > 0 else 0.0
    print(f"Fraction of pixels used: {frac:.3f}")