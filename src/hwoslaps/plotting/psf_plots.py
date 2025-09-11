"""PSF plotting functions for visualization.

This module contains plotting functions for visualizing PSF systems,
aberrations, and phase screens.

The plotting functions support the diverging-path PSF architecture by
providing visualization tools for both high-resolution PSFs used for
optical quality metrics and detector-sampled kernels used for science
applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from hcipy.plotting import imshow_field, imshow_psf
from matplotlib.colors import LogNorm
from ..constants import ARCSEC_PER_RAD
from .registry import plot_function


def _create_output_directory(base_output_dir, run_name, module_name):
    """Create structured output directory following the convention.
    
    Creates output directories with the structure:
    {output_folder}/{run_name}/{module}
    
    Parameters
    ----------
    base_output_dir : `str` or `Path`
        Base output directory.
    run_name : `str`
        Run identifier name.
    module_name : `str`
        Module name (e.g., 'psf', 'lensing').
        
    Returns
    -------
    output_dir : `Path`
        The created output directory path.
    """
    output_dir = Path(base_output_dir) / run_name / module_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@plot_function(module='psf', description="3-panel PSF comparison: phase screen, wavefront, PSF")
def plot_psf_comparison(psf_data, plot_config):
    """Plot PSF system comparison.
    
    Creates a horizontal 3x1 plot showing:
    1. Phase screen (if available)
    2. Total wavefront phase  
    3. PSF
    
    Parameters
    ----------
    psf_data : `PSFData`
        Complete PSF system data from the pipeline.
    plot_config : `dict`
        Plotting configuration including output directory and run name.
        
    Notes
    -----
    The function generates a 3-panel comparison plot showing phase screens,
    total wavefront phase, and the final PSF with aberration information
    automatically extracted from the unified PSFData structure.
    
    Examples
    --------
    Plot a PSF comparison:
    
    >>> plot_config = {'output_dir': '/path/to/output'}
    >>> plot_psf_comparison(psf_data, plot_config)
    """
    # Get run name from config.
    config = psf_data.config
    run_name = config['run_name']
    
    # Create structured output directory.
    output_dir = _create_output_directory(
        plot_config['output_dir'], 
        run_name, 
        'psf'
    )
    
    # Extract data using unified structure.
    telescope_data = psf_data.telescope_data
    aper = telescope_data['aper']
    wavelength_nm = psf_data.wavelength_nm
    
    # Set up the plotting style.
    plt.style.use('default')
    
    # Create the main comparison figure.
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Phase screen (if available).
    plt.subplot(131)
    if psf_data.phase_screens:
        # Get the first available phase screen.
        phase_screen_name = list(psf_data.phase_screens.keys())[0]
        phase_screen = psf_data.phase_screens[phase_screen_name]
        
        if 'api' in phase_screen_name.lower():
            plt.title('Hexike Phase Screen (API)')
        elif 'segment' in phase_screen_name.lower():
            plt.title('Phase Screen (Segment-Level)')
        elif 'global' in phase_screen_name.lower():
            plt.title('Phase Screen (Global Zernikes)')
        else:
            plt.title('Phase Screen')
            
        imshow_field(phase_screen, cmap='RdBu_r')
        plt.colorbar(label='Phase [rad]')
    else:
        plt.title('No Phase Screen Available')
        plt.text(0.5, 0.5, 'No phase screens\navailable', 
                transform=plt.gca().transAxes, ha='center', va='center')
    
    # Plot 2: Total wavefront phase (focal plane).
    plt.subplot(132)
    plt.title('Focal Plane Wavefront Phase')
    imshow_field(np.angle(psf_data.wavefront.electric_field), cmap='RdBu_r')
    plt.colorbar(label='Phase [rad]')
    
    # Plot 3: PSF.
    plt.subplot(133)
    aberration_types = []
    if psf_data.has_segment_pistons:
        aberration_types.append('pistons')
    if psf_data.has_segment_tiptilts:
        aberration_types.append('tiptilts')
    if psf_data.has_segment_hexikes:
        aberration_types.append('hexikes')
    if psf_data.has_global_zernikes:
        aberration_types.append('global')
    
    if aberration_types:
        title = f"PSF with {', '.join(aberration_types)}"
    else:
        title = "Perfect PSF"
    
    plt.title(title)
    imshow_psf(psf_data.psf, normalization='peak')
    
    plt.tight_layout()
    
    # Create meaningful filename.
    filename = f"psf_comparison_{wavelength_nm:.0f}nm.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PSF comparison plot: {filepath}")


@plot_function(module='psf', description="Zoomed PSF phase view for detailed analysis")
def plot_psf_zoom(psf_data, plot_config, zoom_mode='full'):
    """Plot zoomed view of PSF phase.
    
    Parameters
    ----------
    psf_data : `PSFData`
        Complete PSF system data.
    plot_config : `dict`
        Plotting configuration.
    zoom_mode : `str`, optional
        Zoom mode: 'full' for full aperture, 'corner' for corner detail.
        
    Notes
    -----
    Creates a detailed zoom view of the wavefront phase with aperture
    boundaries clearly marked. Uses the unified PSFData structure for
    direct access to system parameters.
    
    Examples
    --------
    Create zoom plots for different regions:
    
    >>> plot_psf_zoom(psf_data, plot_config, zoom_mode='full')
    >>> plot_psf_zoom(psf_data, plot_config, zoom_mode='corner')
    """
    # Get run name from config.
    config = psf_data.config
    run_name = config['run_name']
    
    # Create structured output directory.
    output_dir = _create_output_directory(
        plot_config['output_dir'], 
        run_name, 
        'psf'
    )
    
    telescope_data = psf_data.telescope_data
    pupil_grid = telescope_data['pupil_grid']
    aper = telescope_data['aper']
    wavelength_nm = psf_data.wavelength_nm
    
    # Create zoom plot.
    plt.figure(figsize=(8, 6))
    
    # Get the focal plane grid coordinates (adapted for new architecture).
    focal_grid = psf_data.wavefront.grid
    x = focal_grid.coords[0]
    y = focal_grid.coords[1]
    
    # Define zoom region in focal plane coordinates (adapted for focal plane scale).
    if zoom_mode == 'full':
        # Use the full focal plane extent.
        zoom_xmin, zoom_xmax = x.min(), x.max()
        zoom_ymin, zoom_ymax = y.min(), y.max()
    elif zoom_mode == 'corner':
        # Use a corner region of the focal plane.
        extent_x = x.max() - x.min()
        extent_y = y.max() - y.min()
        zoom_xmin, zoom_xmax = x.min(), x.min() + extent_x * 0.4
        zoom_ymin, zoom_ymax = y.min(), y.min() + extent_y * 0.4
    else:
        raise ValueError(f"Invalid zoom mode: {zoom_mode}")
    
    # Mask for the zoomed region (adapted for focal plane).
    zoom_mask = (x >= zoom_xmin) & (x <= zoom_xmax) & (y >= zoom_ymin) & (y <= zoom_ymax)
    
    # Create a zoomed field for pixel-based visualization (adapted for focal plane).
    zoom_indices = np.where(zoom_mask.reshape(focal_grid.shape))
    y_indices, x_indices = zoom_indices
    
    # Get the bounds of the zoom region in grid indices (adapted for focal plane).
    y_min, y_max = y_indices.min(), y_indices.max() + 1
    x_min, x_max = x_indices.min(), x_indices.max() + 1
    
    # Extract the zoomed phase data as 2D arrays (adapted for focal plane).
    zoom_total_phase_2d = np.angle(psf_data.wavefront.electric_field).shaped[y_min:y_max, x_min:x_max]
    
    # Create extent for proper axis scaling (adapted for focal plane).
    extent = (zoom_xmin, zoom_xmax, zoom_ymin, zoom_ymax)
    
    # For focal plane, we don't need aperture masking since the PSF already contains the aperture effects.
    zoom_phase_masked = zoom_total_phase_2d
    
    # Plot: Focal plane wavefront phase (adapted for new architecture).
    plt.title('Focal Plane Wavefront Phase\n(Zoomed)')
    im = plt.imshow(zoom_phase_masked, cmap='RdBu_r', interpolation='nearest', 
                    extent=extent, origin='lower', aspect='equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    
    # Add colorbar with error handling for edge cases.
    try:
        plt.colorbar(im, label='Phase [rad]')
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not create colorbar for zoom plot: {e}")
        # Create a simple text label instead.
        valid_data = zoom_phase_masked[~np.isnan(zoom_phase_masked)]
        if len(valid_data) > 0:
            plt.text(0.02, 0.98, f'Phase range: [{valid_data.min():.2f}, {valid_data.max():.2f}] rad', 
                    transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot.
    filename = f"psf_zoom_{zoom_mode}_{wavelength_nm:.0f}nm.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PSF zoom plot: {filepath}")
    
    # Print zoom info (adapted for focal plane).
    print(f"Zoom region: x=[{zoom_xmin:.2f}, {zoom_xmax:.2f}], y=[{zoom_ymin:.2f}, {zoom_ymax:.2f}]")
    print(f"Zoomed array shape: {zoom_total_phase_2d.shape}")
    total_phase_pixels = np.count_nonzero(~np.isnan(zoom_phase_masked))
    print(f"Total valid phase pixels in zoom: {total_phase_pixels}")


@plot_function(module='psf', description="Complete PSF system overview with comparison and zoom")
def plot_psf_system_overview(psf_data, plot_config):
    """Create comprehensive PSF system plots including the main 3x1 plot and zoom.
    
    This is the main plotting function that creates both the comparison plot
    and the zoomed view.
    
    Parameters
    ----------
    psf_data : `PSFData`
        Complete PSF system data.
    plot_config : `dict`
        Plotting configuration including output directory and run name.
        
    Notes
    -----
    This function creates a complete set of PSF visualization plots and
    prints a comprehensive summary of PSF quality metrics using the
    unified data structure.
    
    Examples
    --------
    Generate complete PSF analysis plots:
    
    >>> plot_config = {'output_dir': '/path/to/output'}
    >>> plot_psf_system_overview(psf_data, plot_config)
    """
    # Create the main 3x1 comparison plot.
    plot_psf_comparison(psf_data, plot_config)
    
    # Create the zoomed central section plot.
    plot_psf_zoom(psf_data, plot_config, zoom_mode='full')
    
    # Optionally create corner zoom as well.
    plot_psf_zoom(psf_data, plot_config, zoom_mode='corner')

        
@plot_function(module='psf', description="Diverging path validation: high-res PSF vs detector kernel")
def plot_diverging_path_comparison(psf_data, plot_config):
    """Visualize the diverging-path architecture showing both PSF products.
    
    This function creates a comparison plot showing:
    1. High-resolution PSF for metrics
    2. Detector-sampled kernel for science
    
    Parameters
    ----------
    psf_data : `PSFData`
        Complete PSF system data with diverging-path products.
    plot_config : `dict`
        Plotting configuration including output directory.
        
    Notes
    -----
    This visualization helps verify that the detector downsampling is
    working correctly and energy is conserved.
    """
    # Get run name and create output directory.
    config = psf_data.config
    run_name = config['run_name']
    output_dir = Path(plot_config['output_dir']) / run_name / 'psf'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a compact 1x2 figure (avoid unused subplot whitespace).
    # Use constrained_layout so colorbars and titles fit without extra margins.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    # High-resolution PSF.
    ax = axes[0]
    psf_intensity = psf_data.psf.intensity
    
    # Calculate extent for high-res PSF.
    psf_grid = psf_data.psf.grid
    extent_rad = [psf_grid.x.min(), psf_grid.x.max(), 
                  psf_grid.y.min(), psf_grid.y.max()]
    extent_arcsec = [x * ARCSEC_PER_RAD for x in extent_rad]
    
    im = ax.imshow(psf_intensity.shaped, extent=extent_arcsec,
                   origin='lower', cmap='hot',
                   norm=LogNorm(vmin=psf_intensity.max()*1e-5,
                               vmax=psf_intensity.max()))
    ax.set_title('High-Res PSF', fontsize=11)
    ax.set_xlabel('arcsec')
    ax.set_ylabel('arcsec')
    plt.colorbar(im, ax=ax, label='Log(Intensity)')
    
    # Add text showing pixel scale.
    ax.text(0.05, 0.95, f'Pixel scale: {psf_data.pixel_scale_arcsec*1000:.3f} mas',
            transform=ax.transAxes, color='white', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    # Detector-sampled kernel.
    ax = axes[1]
    kernel = psf_data.kernel
    kernel_extent = np.array(kernel.shape_native) * psf_data.kernel_pixel_scale / 2
    kernel_extent = [-kernel_extent[1], kernel_extent[1], 
                     -kernel_extent[0], kernel_extent[0]]
    
    im = ax.imshow(kernel.native, extent=kernel_extent,
                   origin='lower', cmap='hot',
                   norm=LogNorm(vmin=kernel.native.max()*1e-5,
                               vmax=kernel.native.max()))
    ax.set_title(f'Detector Kernel',
                 fontsize=11)
    ax.set_xlabel('arcsec')
    ax.set_ylabel('arcsec')
    plt.colorbar(im, ax=ax, label='Log(Intensity)')
    
    # Add text showing pixel scale.
    ax.text(0.05, 0.95, f'Pixel scale: {psf_data.kernel_pixel_scale*1000:.3f} mas',
            transform=ax.transAxes, color='white', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    fig.suptitle('PSF Downsampling', fontsize=14)
    
    # Save.
    filepath = output_dir / 'diverging_path_comparison.png'
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved diverging path comparison: {filepath}")


def _calculate_radial_profile(image, pixel_scale):
    """Calculate normalized radial profile of an image.
    
    Parameters
    ----------
    image : `numpy.ndarray`
        2D image array.
    pixel_scale : `float`
        Pixel scale in arcseconds.
        
    Returns
    -------
    profile : `dict`
        Dictionary with 'radius' and 'intensity' arrays.
    """
    # Find center.
    center = np.array(image.shape) // 2
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2) * pixel_scale
    
    # Create radial bins.
    max_radius = min(center) * pixel_scale * 0.8  # Use 80% to avoid edge effects.
    bins = np.linspace(0, max_radius, 100)
    
    # Calculate mean intensity in each bin.
    intensity = []
    radius = []
    
    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if np.any(mask):
            mean_val = np.mean(image[mask])
            if mean_val > 0:  # Only include positive values.
                intensity.append(mean_val)
                radius.append((bins[i] + bins[i+1]) / 2)
    
    # Convert to arrays and normalize.
    intensity = np.array(intensity)
    radius = np.array(radius)
    
    if len(intensity) > 0 and intensity[0] > 0:
        intensity /= intensity[0]  # Normalize to peak.
    
    return {'radius': radius, 'intensity': intensity}


@plot_function(module='psf', description="Complete PSF analysis suite including all diagnostic plots")
def plot_psf_complete_analysis(psf_data, plot_config):
    """Create complete PSF analysis including all diagnostic plots.
    
    Parameters
    ----------
    psf_data : `PSFData`
        Complete PSF system data.
    plot_config : `dict`
        Plotting configuration.
        
    Examples
    --------
    >>> plot_psf_complete_analysis(psf_data, plot_config)
    """
    # Original plots.
    from .psf_plots import plot_psf_system_overview
    plot_psf_system_overview(psf_data, plot_config)
    
    # New diagnostic plots.
    plot_diverging_path_comparison(psf_data, plot_config)
    
    print("\n=== Complete PSF analysis plots generated ===")


@plot_function(module='psf', description="Side-by-side segmented pupil and diffraction-limited PSF for presentations")
def plot_psf_segmented_pupil_baseline(psf_data, plot_config):
    """Plot segmented pupil and baseline PSF for presentation slides.
    
    Creates a clean side-by-side layout showing:
    1. Segmented aperture mask with labeled segments
    2. Diffraction-limited PSF (log stretch)
    
    Uses actual telescope configuration parameters from master_config.yaml.
    
    Parameters
    ----------
    psf_data : `PSFData`
        Complete PSF system data from the pipeline.
    plot_config : `dict`
        Plotting configuration including output directory.
        
    Notes
    -----
    This function demonstrates the PSF engine baseline:
    - HCIPy segmented aperture (HWO-like)
    - Actual wavelength and sampling from config
    - High-res PSF generation
    - Baseline (no aberrations) reference
    
    Examples
    --------
    Create segmented pupil baseline figure:
    
    >>> plot_config = {'output_dir': '/path/to/output'}
    >>> plot_psf_segmented_pupil_baseline(psf_data, plot_config)
    """
    # Get run name from config
    config = psf_data.config
    run_name = config['run_name']
    
    # Create structured output directory
    output_dir = _create_output_directory(
        plot_config['output_dir'], 
        run_name, 
        'psf'
    )
    
    # Extract telescope data and configuration parameters
    telescope_data = psf_data.telescope_data
    aperture = telescope_data['aper']
    segments = telescope_data['segments']
    pupil_grid = telescope_data['pupil_grid']
    
    # Get actual config parameters (not defaults)
    psf_config = config['psf']
    telescope_config = psf_config['telescope']
    sim_config = psf_config['hres_psf']
    
    # Extract actual values from config
    pupil_diameter_m = telescope_config['pupil_diameter']
    wavelength_m = sim_config['wavelength']
    wavelength_nm = wavelength_m * 1e9
    num_rings = telescope_config['num_rings']
    gap_size_m = telescope_config['gap_size']
    segment_point_to_point_m = telescope_config['segment_point_to_point']
    focal_length_m = telescope_config['focal_length']
    sampling = psf_data.used_sampling_factor  # Use the auto-adjusted sampling
    
    # Calculate derived parameters
    segment_flat_to_flat_m = segment_point_to_point_m * np.sqrt(3) / 2
    f_number = focal_length_m / pupil_diameter_m
    
    # Create the presentation figure - 1x2 side-by-side layout
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Panel 1: Segmented aperture using HCIPy plotting (actual aperture)
    ax1 = axes[0]
    
    # Use HCIPy's imshow_field to plot the actual aperture
    # Switch to 'gray' so segments are white on a black background for better contrast with red labels
    imshow_field(aperture, ax=ax1, cmap='gray')
    ax1.set_title('Segmented Aperture', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position [m]', fontsize=12)
    ax1.set_ylabel('Position [m]', fontsize=12)
    
    # Add faint segment labels using HCIPy coordinate system
    for i, segment in enumerate(segments):
        # Calculate centroid of each segment using HCIPy approach
        segment_coords = segment.grid.coords
        segment_values = segment.shaped
        
        # Find weighted centroid
        total_weight = np.sum(segment_values)
        if total_weight > 0:
            x_center = np.sum(segment_coords[0] * segment_values.ravel()) / total_weight
            y_center = np.sum(segment_coords[1] * segment_values.ravel()) / total_weight
            
            # Add faint segment number labels
            ax1.text(x_center, y_center, str(i), 
                    fontsize=8, color='red', alpha=0.7,
                    ha='center', va='center', fontweight='bold')
    
    # Panel 2: Diffraction-limited PSF (log stretch)
    ax2 = axes[1]
    
    # Get the PSF intensity (assuming this is the baseline perfect PSF)
    psf_intensity = psf_data.psf.intensity.shaped
    
    # Calculate PSF extent in arcseconds
    psf_grid = psf_data.psf.grid
    extent_rad = [psf_grid.x.min(), psf_grid.x.max(), 
                  psf_grid.y.min(), psf_grid.y.max()]
    extent_arcsec = [x * ARCSEC_PER_RAD for x in extent_rad]
    
    # Log stretch with proper normalization
    psf_log = np.log10(psf_intensity / np.max(psf_intensity) + 1e-6)
    
    im2 = ax2.imshow(psf_log, extent=extent_arcsec, origin='lower', 
                     cmap='hot', vmin=-5, vmax=0)
    
    ax2.set_xlabel('arcsec', fontsize=12)
    ax2.set_ylabel('arcsec', fontsize=12)
    ax2.set_title('Diffraction-Limited PSF (EAC 1)', fontsize=14, fontweight='bold')
    
    # Add colorbar for PSF
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Log₁₀(Normalized Intensity)', fontsize=10)
    
    plt.tight_layout()
    
    # Create filename
    filename = f"psf_segmented_pupil_baseline_{wavelength_nm:.0f}nm.png"
    filepath = output_dir / filename
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved PSF segmented pupil baseline plot: {filepath}")
    
    # Print configuration summary
    print(f"\nPSF Engine Configuration Summary:")
    print(f"Telescope: {pupil_diameter_m:.2f} m diameter, f/{f_number:.1f}")
    print(f"Segmentation: {len(segments)} segments in {num_rings} rings")
    print(f"Wavelength: {wavelength_nm:.0f} nm")
    print(f"Sampling: {sampling:.2f} pixels/λ/D (auto-adjusted)")
    print(f"Gap size: {gap_size_m*1000:.1f} mm")
    print(f"Segment flat-to-flat: {segment_flat_to_flat_m*1000:.1f} mm")