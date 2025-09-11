"""Lensing system plotting functions.

This module contains plotting functions for visualizing lensing systems
and subhalo effects.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import autolens as al
from pathlib import Path
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


@plot_function(module='lensing', requires_subhalo=True, 
               description="2x2 comparison showing subhalo effects with log/linear scaling")
def plot_lensing_comparison(lensing_data, plot_config):
    """Plot lensing system comparison showing subhalo effects.
    
    This function creates a detailed comparison plot showing the lensing system
    with and without the subhalo, highlighting the difference caused by the
    subhalo's gravitational influence.
    
    Parameters
    ----------
    lensing_data : `LensingData`
        Complete lensing system data from the pipeline.
    plot_config : `dict`
        Plotting configuration including output directory.
        
    Notes
    -----
    The function generates a 2x2 comparison plot showing the full lensing
    system, baseline without subhalo, and difference images in both linear
    and logarithmic scaling. Quantitative metrics are printed to console.
    
    Examples
    --------
    Plot a lensing comparison:
    
    >>> plot_config = {'output_dir': '/path/to/output'}
    >>> plot_lensing_comparison(lensing_data, plot_config)
    """
    # Get run name from config
    config = lensing_data.config
    run_name = config['run_name']
    
    # Create structured output directory
    output_dir = _create_output_directory(
        plot_config['output_dir'], 
        run_name, 
        'lensing'
    )
    
    # Extract data from lensing_data using unified structure
    grid = lensing_data.grid
    pixel_scale = lensing_data.pixel_scale
    
    # Check if subhalo is present
    if not lensing_data.has_subhalo:
        print("Warning: No subhalo present in lensing system. Cannot create comparison plot.")
        return
    
    # Get subhalo information from unified structure
    subhalo_position = lensing_data.subhalo_position
    subhalo_model = lensing_data.subhalo_model
    subhalo_mass = lensing_data.subhalo_mass
    
    # Recreate lens galaxy without subhalo for baseline comparison
    # Create lens mass profile (without subhalo)
    lens_mass = al.mp.Isothermal(
        centre=lensing_data.lens_centre,
        einstein_radius=lensing_data.lens_einstein_radius,
        ell_comps=lensing_data.lens_ellipticity
    )
    
    # Create source light profile
    source_light = al.lp.Exponential(
        centre=lensing_data.source_centre,
        ell_comps=lensing_data.source_ellipticity,
        intensity=lensing_data.source_intensity,
        effective_radius=lensing_data.source_effective_radius
    )
    
    # Create galaxies without subhalo
    lens_galaxy_no_subhalo = al.Galaxy(
        redshift=lensing_data.lens_redshift,
        mass=lens_mass
    )
    
    source_galaxy = al.Galaxy(
        redshift=lensing_data.source_redshift,
        light=source_light
    )
    
    # Create tracer without subhalo for baseline
    # Use the same cosmology as used in the lensing run (from config)
    if lensing_data.cosmology_name == 'Planck15':
        cosmo = al.cosmo.Planck15()
    else:
        raise ValueError(f"Unsupported cosmology in plotting: {lensing_data.cosmology_name}")

    tracer_no_subhalo = al.Tracer(
        galaxies=[lens_galaxy_no_subhalo, source_galaxy],
        cosmology=cosmo
    )
    
    # Generate images
    image_with_subhalo = lensing_data.image
    image_no_subhalo = tracer_no_subhalo.image_2d_from(grid=grid)
    difference_image = image_with_subhalo - image_no_subhalo.native
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create the comparison figure - 2x2 layout
    fig = plt.figure(figsize=(12, 10))
    
    # Calculate field of view for extent
    fov_arcsec = grid.shape_native[0] * pixel_scale
    extent = (-fov_arcsec/2, fov_arcsec/2, -fov_arcsec/2, fov_arcsec/2)
    
    # Top left: Original image (no subhalo)
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(image_no_subhalo.native, extent=extent, origin='lower', cmap='viridis')
    ax1.set_title('Original Scene (No Subhalo)', fontsize=16)
    ax1.set_xlabel('arcsec')
    ax1.set_ylabel('arcsec')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Top right: Full lensing system with subhalo
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(image_with_subhalo, extent=extent, origin='lower', cmap='viridis')
    ax2.set_title(f'Scene with Subhalo ({subhalo_model})', fontsize=16)
    ax2.scatter(*subhalo_position[::-1], c='red', s=100, marker='x')
    ax2.set_xlabel('arcsec')
    ax2.set_ylabel('arcsec')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Calculate difference scaling
    max_diff = np.max(np.abs(difference_image))
    linthresh = max_diff * 1e-3 if max_diff > 0 else 1e-6
    
    # Bottom left: Difference image with log scale
    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.imshow(difference_image, extent=extent, origin='lower', cmap='RdBu_r',
                     norm=SymLogNorm(linthresh=linthresh, vmin=-max_diff, vmax=max_diff))
    ax3.set_title('Subhalo Difference (Log Scale)', fontsize=16)
    ax3.scatter(*subhalo_position[::-1], c='black', s=100, marker='x')
    ax3.set_xlabel('arcsec')
    ax3.set_ylabel('arcsec')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Difference (log scale)')
    
    # Bottom right: Difference image with absolute scale
    ax4 = plt.subplot(2, 2, 4)
    im4 = ax4.imshow(difference_image, extent=extent, origin='lower', cmap='RdBu_r',
                     vmin=-max_diff, vmax=max_diff)
    ax4.set_title('Subhalo Difference (Absolute Scale)', fontsize=16)
    ax4.scatter(*subhalo_position[::-1], c='black', s=100, marker='x')
    ax4.set_xlabel('arcsec')
    ax4.set_ylabel('arcsec')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04, label='Difference (absolute scale)')
    
    plt.tight_layout()
    
    # Create meaningful filename
    filename = "lensing_comparison.png"
    filepath = output_dir / filename
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved lensing comparison plot: {filepath}")
    
    # Print quantitative comparison
    peak_signal = np.max(np.abs(difference_image))
    total_signal = np.sum(np.abs(difference_image))
    signal_rms = np.sqrt(np.mean(difference_image**2))
    
    print(f"\nSubhalo Effect Analysis:")
    print(f"Model: {subhalo_model}")
    print(f"Mass: {subhalo_mass:.1e} M_sun")
    print(f"Einstein radius: {lensing_data.subhalo_einstein_radius:.6f} arcsec")
    print(f"Peak signal: {peak_signal:.6e}")
    print(f"Total |signal|: {total_signal:.6e}")
    print(f"Signal RMS: {signal_rms:.6e}")


@plot_function(module='lensing', requires_subhalo=True,
               description="1x3 horizontal layout for presentation slides showing baseline scene")
def plot_lensing_baseline_scene(lensing_data, plot_config):
    """Plot lensing baseline scene for presentation slides.
    
    Creates a clean 1x3 horizontal layout showing:
    1. No-subhalo ring (smooth baseline)
    2. With subhalo (tiny kink visible)  
    3. Residual (difference image)
    
    All panels use the same intensity stretch for direct comparison.
    
    Parameters
    ----------
    lensing_data : `LensingData`
        Complete lensing system data from the pipeline.
    plot_config : `dict`
        Plotting configuration including output directory.
        
    Notes
    -----
    This function is specifically designed for presentation slides showing
    the baseline lensing scene concept. It generates a clean, horizontal
    3-panel figure suitable for slide inclusion.
    
    Examples
    --------
    Create presentation baseline scene figure:
    
    >>> plot_config = {'output_dir': '/path/to/output'}
    >>> plot_lensing_baseline_scene(lensing_data, plot_config)
    """
    # Get run name from config
    config = lensing_data.config
    run_name = config['run_name']
    
    # Create structured output directory
    output_dir = _create_output_directory(
        plot_config['output_dir'], 
        run_name, 
        'lensing'
    )
    
    # Extract data from lensing_data using unified structure
    grid = lensing_data.grid
    pixel_scale = lensing_data.pixel_scale
    
    # Check if subhalo is present
    if not lensing_data.has_subhalo:
        print("Warning: No subhalo present in lensing system. Cannot create baseline scene plot.")
        return
    
    # Get subhalo information
    subhalo_position = lensing_data.subhalo_position
    subhalo_mass = lensing_data.subhalo_mass
    
    # Recreate lens galaxy without subhalo for baseline comparison
    lens_mass = al.mp.Isothermal(
        centre=lensing_data.lens_centre,
        einstein_radius=lensing_data.lens_einstein_radius,
        ell_comps=lensing_data.lens_ellipticity
    )
    
    source_light = al.lp.Exponential(
        centre=lensing_data.source_centre,
        ell_comps=lensing_data.source_ellipticity,
        intensity=lensing_data.source_intensity,
        effective_radius=lensing_data.source_effective_radius
    )
    
    # Create galaxies without subhalo
    lens_galaxy_no_subhalo = al.Galaxy(
        redshift=lensing_data.lens_redshift,
        mass=lens_mass
    )
    
    source_galaxy = al.Galaxy(
        redshift=lensing_data.source_redshift,
        light=source_light
    )
    
    # Create tracer without subhalo for baseline
    if lensing_data.cosmology_name == 'Planck15':
        cosmo = al.cosmo.Planck15()
    else:
        raise ValueError(f"Unsupported cosmology in plotting: {lensing_data.cosmology_name}")

    tracer_no_subhalo = al.Tracer(
        galaxies=[lens_galaxy_no_subhalo, source_galaxy],
        cosmology=cosmo
    )
    
    # Generate images
    image_with_subhalo = lensing_data.image
    image_no_subhalo = tracer_no_subhalo.image_2d_from(grid=grid)
    difference_image = image_with_subhalo - image_no_subhalo.native
    
    # Calculate field of view for extent
    fov_arcsec = grid.shape_native[0] * pixel_scale
    extent = (-fov_arcsec/2, fov_arcsec/2, -fov_arcsec/2, fov_arcsec/2)
    
    # Set up consistent intensity scaling across all panels
    # Use the with-subhalo image for reference scaling
    vmin = np.min(image_with_subhalo) 
    vmax = np.max(image_with_subhalo)
    
    # Create the presentation figure - 1x3 horizontal layout
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: No-subhalo ring (baseline)
    ax1 = axes[0]
    im1 = ax1.imshow(image_no_subhalo.native, extent=extent, origin='lower', 
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax1.set_title('No Subhalo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('arcsec', fontsize=12)
    ax1.set_ylabel('arcsec', fontsize=12)
    # Mark subhalo position for reference
    ax1.scatter(*subhalo_position[::-1], c='red', s=80, marker='x', alpha=0.7)
    
    # Panel 2: With subhalo (tiny kink)
    ax2 = axes[1] 
    im2 = ax2.imshow(image_with_subhalo, extent=extent, origin='lower',
                     cmap='viridis', vmin=vmin, vmax=vmax)
    ax2.set_title('With Subhalo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('arcsec', fontsize=12)
    ax2.set_ylabel('arcsec', fontsize=12)
    # Mark subhalo position
    ax2.scatter(*subhalo_position[::-1], c='red', s=80, marker='x', alpha=0.7)
    
    # Panel 3: Residual (difference)
    ax3 = axes[2]
    max_diff = np.max(np.abs(difference_image))
    im3 = ax3.imshow(difference_image, extent=extent, origin='lower', 
                     cmap='RdBu_r', vmin=-max_diff, vmax=max_diff)
    ax3.set_title('Residual', fontsize=14, fontweight='bold')
    ax3.set_xlabel('arcsec', fontsize=12)
    ax3.set_ylabel('arcsec', fontsize=12)
    # Mark subhalo position in black for contrast
    ax3.scatter(*subhalo_position[::-1], c='black', s=80, marker='x', alpha=0.8)
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04) 
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Create filename for presentation figure
    mass_str = f"{subhalo_mass:.1e}".replace('+', '').replace('e0', 'e')
    filename = f"lensing_baseline_scene_{mass_str}Msun.png"
    filepath = output_dir / filename
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved lensing baseline scene plot: {filepath}")
    
    # Print summary for presentation context
    print(f"\nBaseline Scene Summary:")
    print(f"Subhalo mass: {subhalo_mass:.1e} M_sun")
    print(f"Peak residual signal: {max_diff:.6e}")
    print(f"Position: ({subhalo_position[0]:.3f}, {subhalo_position[1]:.3f}) arcsec")


@plot_function(module='lensing', 
               description="Subhalo placement methodology with allowable band and sample positions")
def plot_subhalo_placement_methodology(lensing_data, plot_config):
    """Plot subhalo placement methodology for presentation slides.
    
    Shows the Einstein ring with allowable placement band shaded and 
    several sample positions dotted, plus an inset residual zoom box.
    
    Parameters
    ----------
    lensing_data : `LensingData`
        Complete lensing system data from the pipeline.
    plot_config : `dict`
        Plotting configuration including output directory.
        
    Notes
    -----
    This function demonstrates the subhalo placement methodology:
    - Random azimuth around Einstein ring
    - Small radial jitter (scatter_pixels)
    - Seeded RNG for reproducibility
    - Sample positions shown as dots
    
    Examples
    --------
    Create placement methodology figure:
    
    >>> plot_config = {'output_dir': '/path/to/output'}
    >>> plot_subhalo_placement_methodology(lensing_data, plot_config)
    """
    # Get run name from config
    config = lensing_data.config
    run_name = config['run_name']
    
    # Create structured output directory
    output_dir = _create_output_directory(
        plot_config['output_dir'], 
        run_name, 
        'lensing'
    )
    
    # Extract data from lensing_data using unified structure
    grid = lensing_data.grid
    pixel_scale = lensing_data.pixel_scale
    
    # Get Einstein radius for placement band
    einstein_radius = lensing_data.lens_einstein_radius
    
    # Get placement parameters from config (with defaults if not specified)
    # Typical scatter_pixels value from pipeline configurations
    scatter_pixels = 3.0  # Default scatter for placement methodology
    if (lensing_data.config and 'lensing' in lensing_data.config and 
        'subhalo' in lensing_data.config['lensing'] and 
        'position' in lensing_data.config['lensing']['subhalo']):
        scatter_pixels = lensing_data.config['lensing']['subhalo']['position'].get('scatter_pixels', 3.0)
    
    # Create baseline Einstein ring (no subhalo) for reference
    lens_mass = al.mp.Isothermal(
        centre=lensing_data.lens_centre,
        einstein_radius=lensing_data.lens_einstein_radius,
        ell_comps=lensing_data.lens_ellipticity
    )
    
    source_light = al.lp.Exponential(
        centre=lensing_data.source_centre,
        ell_comps=lensing_data.source_ellipticity,
        intensity=lensing_data.source_intensity,
        effective_radius=lensing_data.source_effective_radius
    )
    
    lens_galaxy_no_subhalo = al.Galaxy(
        redshift=lensing_data.lens_redshift,
        mass=lens_mass
    )
    
    source_galaxy = al.Galaxy(
        redshift=lensing_data.source_redshift,
        light=source_light
    )
    
    # Create cosmology
    if lensing_data.cosmology_name == 'Planck15':
        cosmo = al.cosmo.Planck15()
    else:
        raise ValueError(f"Unsupported cosmology in plotting: {lensing_data.cosmology_name}")

    tracer_baseline = al.Tracer(
        galaxies=[lens_galaxy_no_subhalo, source_galaxy],
        cosmology=cosmo
    )
    
    # Generate baseline image
    baseline_image = tracer_baseline.image_2d_from(grid=grid)
    
    # Generate sample subhalo positions using same methodology as pipeline
    np.random.seed(12345)  # Fixed seed for reproducible sample positions
    sample_angles = [45, 135, 225, 315, 30, 150]  # Sample azimuthal positions
    sample_positions = []
    
    for angle_deg in sample_angles:
        # Small random offset within scatter range
        offset_pixels = np.random.uniform(-scatter_pixels, scatter_pixels)
        
        # Calculate position using same logic as pipeline
        from ..lensing.utils import get_einstein_ring_position
        position = get_einstein_ring_position(
            angle_deg=angle_deg,
            einstein_radius=einstein_radius,
            offset_pixels=offset_pixels,
            pixel_scale=pixel_scale
        )
        sample_positions.append(position)
    
    # Calculate field of view for extent
    fov_arcsec = grid.shape_native[0] * pixel_scale
    extent = (-fov_arcsec/2, fov_arcsec/2, -fov_arcsec/2, fov_arcsec/2)
    
    # Create the methodology figure
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 8))
    
    # Main plot showing Einstein ring with placement methodology
    ax_main = plt.subplot(1, 1, 1)
    
    # Show baseline Einstein ring
    im = ax_main.imshow(baseline_image.native, extent=extent, origin='lower', 
                       cmap='viridis', alpha=0.8)
    
    # Create allowable placement band (annular region)
    # Inner and outer radii based on scatter
    scatter_arcsec = scatter_pixels * pixel_scale
    inner_radius = einstein_radius - scatter_arcsec
    outer_radius = einstein_radius + scatter_arcsec
    
    # Create circular patches for the allowable band
    from matplotlib.patches import Circle
    import matplotlib.patches as patches
    
    # Outer circle (transparent)
    outer_circle = Circle((0, 0), outer_radius, fill=True, 
                         facecolor='yellow', alpha=0.2, 
                         edgecolor='orange', linewidth=2, linestyle='--')
    ax_main.add_patch(outer_circle)
    
    # Inner circle (to create annular region by overlay)
    inner_circle = Circle((0, 0), inner_radius, fill=True, 
                         facecolor='white', alpha=0.3)
    ax_main.add_patch(inner_circle)
    
    # Show Einstein ring itself
    einstein_circle = Circle((0, 0), einstein_radius, fill=False,
                           edgecolor='red', linewidth=2, alpha=0.8)
    ax_main.add_patch(einstein_circle)
    
    # Plot sample subhalo positions
    for i, (y, x) in enumerate(sample_positions):
        ax_main.scatter(x, y, c='red', s=60, marker='o', 
                       edgecolors='white', linewidth=1.5, 
                       zorder=10, alpha=0.9)
        # Optionally number the sample positions
        ax_main.annotate(f'{i+1}', (x, y), xytext=(3, 3), 
                        textcoords='offset points', fontsize=8, 
                        color='white', fontweight='bold')
    
    # Set up main plot
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])
    ax_main.set_xlabel('arcsec', fontsize=12)
    ax_main.set_ylabel('arcsec', fontsize=12)
    ax_main.set_title('Subhalo Placement Methodology', fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04, label='Intensity')
    
    # Create inset for residual zoom
    # Choose one sample position for the inset (first one)
    sample_y, sample_x = sample_positions[0]
    
    # Generate a test image with subhalo at this position for residual demonstration
    # Create a small subhalo for demonstration
    test_subhalo = al.mp.PointMass(
        centre=(sample_y, sample_x),
        einstein_radius=0.01  # Small Einstein radius for 5e7 Msun subhalo
    )
    
    lens_galaxy_with_subhalo = al.Galaxy(
        redshift=lensing_data.lens_redshift,
        mass=lens_mass,
        subhalo=test_subhalo
    )
    
    tracer_with_subhalo = al.Tracer(
        galaxies=[lens_galaxy_with_subhalo, source_galaxy],
        cosmology=cosmo
    )
    
    test_image = tracer_with_subhalo.image_2d_from(grid=grid)
    residual_image = test_image.native - baseline_image.native
    
    # Create inset axes for residual zoom
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    zoom_size = 0.8  # arcsec
    axins = inset_axes(ax_main, width="30%", height="30%", loc='upper right')
    
    # Show residual in inset
    zoom_extent = [sample_x - zoom_size/2, sample_x + zoom_size/2,
                   sample_y - zoom_size/2, sample_y + zoom_size/2]
    
    max_residual = np.max(np.abs(residual_image))
    axins.imshow(residual_image, extent=extent, origin='lower', 
                cmap='RdBu_r', vmin=-max_residual, vmax=max_residual)
    axins.set_xlim(zoom_extent[0], zoom_extent[1])
    axins.set_ylim(zoom_extent[2], zoom_extent[3])
    axins.set_title('Residual Zoom', fontsize=10)
    
    # Mark the zoom region on main plot
    zoom_box = patches.Rectangle((zoom_extent[0], zoom_extent[2]), 
                                zoom_size, zoom_size,
                                linewidth=1.5, edgecolor='cyan', 
                                facecolor='none', linestyle='-')
    ax_main.add_patch(zoom_box)
    
    # Add methodology text box
    methodology_text = (f"Placement Parameters:\n"
                       f"• Einstein radius: {einstein_radius:.3f}\"\n"
                       f"• Scatter: ±{scatter_pixels} pixels\n"
                       f"• Band width: ±{scatter_arcsec:.3f}\"\n"
                       f"• Sample mass: 5.0×10⁷ M☉")
    
    ax_main.text(0.02, 0.98, methodology_text, transform=ax_main.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Create filename
    filename = "subhalo_placement_methodology.png"
    filepath = output_dir / filename
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved subhalo placement methodology plot: {filepath}")
    
    # Print methodology summary
    print(f"\nPlacement Methodology Summary:")
    print(f"Einstein radius: {einstein_radius:.3f} arcsec")
    print(f"Scatter range: ±{scatter_pixels} pixels (±{scatter_arcsec:.3f} arcsec)")
    print(f"Generated {len(sample_positions)} sample positions")
    print(f"Band coverage: {inner_radius:.3f} to {outer_radius:.3f} arcsec")
    print(f"Seeded RNG ensures reproducible placement across sweeps")