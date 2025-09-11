# HWO-SLAPS Technical Implementation Plan

## Executive Summary

The HWO-SLAPS pipeline aims to determine the PSF quality and stability requirements necessary for detecting 10^7 M☉ dark matter subhalos with the Habitable Worlds Observatory. This technical plan outlines the implementation of a five-module system that simulates end-to-end observations from galaxy generation through performance analysis.

The pipeline follows a linear flow: generate lensing systems with embedded subhalos, create time-varying PSFs with realistic aberrations, simulate observations through convolution and noise addition, attempt to recover the subhalos through lens modeling, and finally analyze the detection performance to derive telescope requirements.

## System Architecture

The pipeline consists of five sequential processing modules, each with well-defined inputs and outputs. The modular design allows independent development and testing while maintaining clear data flow between components.

### Directory Structure
```
hwo-slaps/src/hwoslaps/
├── lensing/          # ✅ Galaxy & subhalo system generation (COMPLETE)
│   ├── generator.py  # Main API: generate_lensing_system()
│   ├── utils.py      # LensingData class and utilities
│   └── mass_models.py # Einstein radius calculations for all models
├── psf/              # ✅ PSF generation with aberrations (COMPLETE)
│   ├── generator.py  # Main API: generate_psf_system() with diverging-path
│   ├── utils.py      # PSFData class (no conversion utilities needed)
│   ├── telescope_models.py # HCIPy telescope setup
│   ├── aberration_models.py # All aberration application functions
│   └── psf_metrics.py # Quality metrics and analysis
├── plotting/         # ✅ Visualization (COMPLETE)
│   ├── lensing_plots.py # Lensing system visualization
│   ├── psf_plots.py     # PSF system visualization with diverging-path
│   ├── observation_plots.py # Observation result visualization
│   └── detection_plots.py # Detection result visualization
├── observation/      # ✅ Observation simulation (MOSTLY COMPLETE)
│   ├── generator.py  # Main API: generate_observation() 
│   ├── utils.py      # ObservationData class and utilities
│   └── noise_models.py # Detector noise modeling functions
├── modeling/         # ✅ Subhalo detection via χ² analysis (COMPLETE)
│   ├── generator.py  # Main API: perform_subhalo_detection()
│   ├── utils.py      # DetectionData class and validation
│   └── chi_square_detector.py # Core χ² detection logic
├── analysis/         # 🚧 Performance metrics & requirements (PLANNED)
└── pipeline.py       # ✅ Simple pipeline orchestration (BASIC)
```

### Data Flow
1. **Lensing** → Produces: `LensingData` object (unified structure with image, subhalo truth, all system parameters)
2. **PSF** → Produces: `PSFData` object (unified structure with PSF, detector kernel, quality metrics, aberration summary)  
3. **Observation** → Produces: `ObservationData` object (PyAutoLens imaging dataset + noise components + metadata)
4. **Modeling** → Produces: `DetectionData` object (χ² results, multi-significance detection, comprehensive diagnostics, truth validation)
5. **Analysis** → Produces: requirements, performance metrics (saved to disk)

**Key Features**: Each data object provides direct property access to all parameters, automatic computation of derived quantities, rich metadata, and seamless format conversion capabilities.

---

## Data Management Strategy

### Memory-First Architecture

The pipeline operates primarily in memory, passing Python objects between modules rather than writing intermediate files. This approach dramatically improves performance and reduces storage requirements during parameter studies.

### Core Data Structures

The pipeline uses comprehensive unified data structures that provide direct access to all system parameters and derived quantities, eliminating the need for nested dictionary navigation.

```python
@dataclass
class LensingData:
    """Complete lensing system data structure with unified access."""
    # === PRIMARY DATA ===
    image: np.ndarray               # Lensed source image as 2D array
    grid: al.Grid2D                # PyAutoLens grid object for ray-tracing
    tracer: al.Tracer              # Complete lensing system tracer
    
    # === SYSTEM PARAMETERS ===
    pixel_scale: float              # Pixel scale in arcsec/pixel
    lens_redshift: float            # Lens galaxy redshift
    source_redshift: float          # Source galaxy redshift
    lens_einstein_radius: float     # Main lens Einstein radius (arcsec)
    cosmology_name: str            # Cosmology model name
    
    # === SUBHALO INFORMATION ===
    subhalo_mass: Optional[float] = None           # Mass in M_sun
    subhalo_model: Optional[str] = None            # 'PointMass', 'SIS', 'NFW'
    subhalo_position: Optional[Tuple[float, float]] = None  # (y, x) in arcsec
    subhalo_einstein_radius: Optional[float] = None        # Einstein radius
    subhalo_concentration: Optional[float] = None          # NFW concentration
    
    # === GALAXY PARAMETERS ===
    lens_centre: Tuple[float, float] = (0.0, 0.0)         # Lens center (y, x)
    lens_ellipticity: Tuple[float, float] = (0.0, 0.0)    # Ellipticity (e1, e2)
    source_centre: Tuple[float, float] = (0.0, 0.0)       # Source center
    source_ellipticity: Tuple[float, float] = (0.0, 0.0)  # Source ellipticity
    source_intensity: float = 1.0                          # Source intensity
    source_effective_radius: float = 1.0                   # Source size (arcsec)
    
    # === DERIVED PROPERTIES (computed automatically) ===
    @property
    def has_subhalo(self) -> bool                  # Whether subhalo present
    @property
    def grid_shape(self) -> Tuple[int, int]        # Grid dimensions
    @property
    def field_of_view_arcsec(self) -> Tuple[float, float]  # FOV in arcsec
    @property
    def total_flux(self) -> float                  # Total image flux
    @property
    def peak_intensity(self) -> float              # Peak pixel value

@dataclass
class PSFData:
    """Complete PSF system data structure with unified access."""
    # === PRIMARY DATA ===
    psf: hcipy.Field                # HCIPy PSF field with intensity/coordinates
    wavefront: hcipy.Wavefront      # Wavefront with electric field/phase
    telescope_data: Dict            # Complex HCIPy objects (grids, propagators)
    kernel: al.Kernel2D             # Detector-sampled kernel via diverging-path
    kernel_pixel_scale: float       # Detector pixel scale (matches lensing grid)
    
    # === SYSTEM PARAMETERS ===
    wavelength_nm: float            # Wavelength in nanometers
    pupil_diameter_m: float         # Telescope diameter in meters
    focal_length_m: float           # Telescope focal length in meters
    pixel_scale_arcsec: float       # High-res PSF pixel scale in arcsec/pixel
    sampling_factor: float          # Auto-adjusted pixels per lambda/D
    requested_sampling_factor: float # User-requested sampling from config
    used_sampling_factor: float     # Auto-calculated for integer subsampling
    integer_subsampling_factor: int # N for detector downsampling
    num_segments: int               # Number of telescope segments
    
    # === TELESCOPE GEOMETRY ===
    segment_flat_to_flat_m: float   # Segment flat-to-flat distance
    segment_point_to_point_m: float # Segment point-to-point distance
    gap_size_m: float               # Gap between segments
    num_rings: int                  # Number of segment rings
    
    # === PSF QUALITY METRICS (auto-computed) ===
    fwhm_arcsec: Optional[float] = None      # FWHM in arcseconds
    fwhm_mas: Optional[float] = None         # FWHM in milliarcseconds
    strehl_ratio: Optional[float] = None     # Strehl ratio vs perfect PSF
    peak_intensity: float = 0.0              # Peak PSF intensity
    total_flux: float = 0.0                  # Total PSF flux
    encircled_energy_50_arcsec: Optional[float] = None  # 50% EE radius
    
    # === ABERRATION SUMMARY ===
    total_rms_nm: float = 0.0                # Total RMS wavefront error (nm)
    segment_piston_rms_nm: float = 0.0       # Segment piston RMS (nm)
    segment_tiptilt_rms_urad: float = 0.0    # Segment tip/tilt RMS (μrad)
    global_zernike_rms_nm: float = 0.0       # Global Zernike RMS (nm)
    
    # === ABERRATION FLAGS ===
    has_segment_pistons: bool = False        # Segment piston aberrations present
    has_segment_tiptilts: bool = False       # Segment tip/tilt aberrations present
    has_segment_hexikes: bool = False        # Segment hexike aberrations present
    has_global_zernikes: bool = False        # Global Zernike aberrations present
    
    # === COMPLEX DATA & DIAGNOSTICS ===
    phase_screens: Optional[Dict] = None     # Generated phase screens
    phase_screen_types: Optional[List[str]] = None  # Types of phase screens
    aberrations: Optional[Dict] = None       # Complete aberration config
    
    # === DERIVED PROPERTIES (computed automatically) ===
    @property
    def wavelength_m(self) -> float                    # Wavelength in meters
    @property
    def diffraction_limit_arcsec(self) -> float       # Lambda/D in arcsec
    @property
    def airy_disk_diameter_arcsec(self) -> float      # 2.44*lambda/D
    @property
    def f_number(self) -> float                       # Telescope F-number
    @property
    def angular_resolution_mas(self) -> float         # Resolution in mas
    @property
    def is_diffraction_limited(self) -> Optional[bool] # Strehl > 0.8
    @property
    def quality_grade(self) -> str                     # 'Excellent'/'Good'/etc
    @property
    def aberration_budget_breakdown(self) -> Dict     # Aberration presence by type
    @property
    def has_aberrations(self) -> bool                 # Any aberrations present

@dataclass
class ObservationData:
    """Complete observation data structure with unified access."""
    # === PRIMARY DATA ===
    imaging: al.Imaging                    # PyAutoLens imaging dataset (data + noise_map + PSF)
    noiseless_source_eps: np.ndarray       # Noiseless PSF-convolved source in e-/s
    noise_components: Dict[str, np.ndarray] # Breakdown of noise sources
    config: Dict[str, Any]                 # Observation configuration
    metadata: Dict[str, Any]               # Observation metadata
    
    # === DERIVED PROPERTIES (computed automatically) ===
    @property
    def data(self) -> al.Array2D                    # Observed data array in ADU
    @property
    def noise_map(self) -> al.Array2D              # Noise map array in ADU
    @property
    def psf(self) -> al.Kernel2D                   # PSF kernel used for convolution
    @property
    def signal_to_noise_map(self) -> al.Array2D    # Signal-to-noise ratio map
    @property
    def exposure_time(self) -> float               # Exposure time in seconds
    @property
    def detector_config(self) -> Dict[str, float]  # Detector configuration parameters
    @property
    def gain(self) -> float                        # Detector gain in e-/ADU
    @property
    def peak_snr(self) -> float                    # Peak signal-to-noise ratio
    @property
    def total_flux_adu(self) -> float              # Total flux in ADU
    @property
    def total_flux_electrons(self) -> float        # Total flux in electrons

@dataclass
class DetectionData:
    """Complete subhalo detection results with unified access."""
    # === PRIMARY RESULTS ===
    detection_results: Dict[float, DetectionResult]  # By significance level
    chi2_value: float                               # Chi-square statistic
    degrees_of_freedom: int                         # Degrees of freedom
    
    # === DETECTION PARAMETERS ===
    snr_threshold: float                            # SNR threshold for masking
    significance_levels: List[float]                # Significance levels tested
    pixels_unmasked: int                            # Number of pixels analyzed
    num_regions: int                                # Number of connected regions
    max_region_snr: float                           # Maximum regional SNR
    
    # === MASKS AND ARRAYS ===
    snr_mask: np.ndarray                            # Boolean mask for analysis
    snr_array: np.ndarray                           # SNR values per pixel
    labeled_regions: np.ndarray                     # Connected region labels
    residual_map: np.ndarray                        # Detection residual map
    variance_2d: Optional[np.ndarray] = None        # Pixel variance map
    
    # === SUBHALO TRUTH ===
    true_subhalo_position: Optional[Tuple[float, float]] = None  # (x, y) position
    true_subhalo_mass: Optional[float] = None                    # Mass in M_sun
    true_subhalo_model: Optional[str] = None                     # Mass model type
    
    # === OBSERVATION METADATA ===
    baseline_exposure_time: float = 1000.0          # Exposure time in seconds
    pixel_scale: float = 0.05                       # Pixel scale in arcsec/pixel
    detector_config: Dict[str, float] = field(default_factory=dict)  # Detector parameters
    
    # === PROVENANCE ===
    config: Optional[Dict] = None                    # Full pipeline configuration
    generation_timestamp: Optional[str] = None      # Creation timestamp
    
    # === DERIVED PROPERTIES (computed automatically) ===
    @property
    def max_significance_detected(self) -> Optional[str]         # Highest significance achieved
    @property
    def detection_summary(self) -> Dict                          # Summary of all detection results
    @property
    def is_detected_3sigma(self) -> bool                        # 3σ detection flag
    @property
    def is_detected_4sigma(self) -> bool                        # 4σ detection flag  
    @property
    def is_detected_5sigma(self) -> bool                        # 5σ detection flag
    @property
    def chi2_p_value(self) -> float                             # Chi-square p-value
    @property
    def detection_mask_fraction(self) -> float                  # Fraction of pixels analyzed
    @property
    def has_subhalo_truth(self) -> bool                         # Truth information available
    @property
    def field_of_view_arcsec(self) -> Tuple[float, float]       # Field of view
    @property
    def image_shape(self) -> Tuple[int, int]                    # Detection array shape
```

### Key Implementation Features

**Unified Access Pattern**: All data structures provide direct property access to all parameters, eliminating nested dictionary navigation:
```python
# Direct access to all key parameters
print(f"Lens at z={lensing_data.lens_redshift}")
print(f"PSF FWHM: {psf_data.fwhm_arcsec:.3f} arcsec")
print(f"Quality: {psf_data.quality_grade}")
print(f"Integer subsampling: {psf_data.integer_subsampling_factor}")
print(f"Observation SNR: {obs_data.peak_snr:.1f}")
print(f"Exposure time: {obs_data.exposure_time} s")
print(f"Total flux: {obs_data.total_flux_electrons:.1e} e-")
print(f"Detection: {detection_data.max_significance_detected}")
print(f"Chi² = {detection_data.chi2_value:.2f}, p = {detection_data.chi2_p_value:.2e}")
print(f"5σ detected: {detection_data.is_detected_5sigma}")
if lensing_data.has_subhalo:
    print(f"Subhalo: {lensing_data.subhalo_mass:.1e} M_sun")
```

**Auto-computed Metrics**: Quality metrics and derived quantities are computed automatically during object initialization, providing immediate access to FWHM, Strehl ratio, aberration statistics, and more.

**Rich Property Interface**: Both objects provide extensive computed properties for derived quantities like diffraction limits, angular scales, aberration breakdowns, and quality assessments.

**Diverging-Path PSF Generation**: The PSFData object contains both high-resolution PSF for metrics and properly downsampled kernel for science, ensuring physically accurate detector modeling.

### Execution Modes

**Production Mode (Default)**
- Full in-memory pipeline execution
- No intermediate file I/O
- Only save final metrics and requirements
- Maximum performance for parameter sweeps

**Debug Mode**
- Save intermediate products for inspection
- Enable visualization at each stage
- Keep failed runs for debugging
- Performance penalty acceptable

**Cache Mode (Development)**
- Cache expensive computations (PSF generation)
- Reuse previous results when iterating
- Automatic cache invalidation on config changes
- Balance between speed and reproducibility

### Implementation Pattern
```python
class Pipeline:
    def __init__(self, mode='production'):
        self.mode = mode
        self.save_intermediate = (mode == 'debug')
        
    def run(self, config):
        # All operations in memory
        lensing_data = self.generate_lensing(config)
        psf_data = self.generate_psf(config)
        baseline_obs = self.simulate_observation(lensing_data, psf_data) # No subhalo
        # Create a second lensing system and observation with a subhalo
        lensing_with_subhalo = self.generate_lensing(config, with_subhalo=True) 
        test_obs = self.simulate_observation(lensing_with_subhalo, psf_data) # With subhalo
        
        detection_data = self.detect_subhalo(baseline_obs, test_obs, lensing_data, lensing_with_subhalo) # ✅ IMPLEMENTED
        metrics = self.analyze_performance(detection_data, lensing_with_subhalo)  # 🚧 PLANNED
        
        # Only save when necessary
        if self.save_intermediate:
            self.save_debug_outputs(lensing_data, psf_data, baseline_obs, test_obs, detection_data)
            
        # Always save final results
        self.save_metrics(metrics, config)
        
        return metrics
```

### Storage Guidelines

**Always Persist:**
- Configuration files (YAML/JSON) - for reproducibility
- Final metrics and requirements (HDF5/CSV) - compact results
- Performance summaries (plots/tables) - for reports
- Failed run logs - for debugging

**Optionally Persist (Debug/Development):**
- Example systems for visualization
- Interesting edge cases
- Validation datasets
- Cached PSF libraries

**Never Persist During Production:**
- Intermediate images for every parameter
- Full PSF time series for every run
- Individual noise realizations
- Raw convolution outputs

This strategy prevents the generation of terabytes of unnecessary data while maintaining full debugging capability when needed.

---

## Implementation Status

### Completed Modules

**Module 1: Lensing System Generation** ✅ **COMPLETE**
- Full implementation with unified `LensingData` structure
- Support for all three mass models (PointMass, SIS, NFW) with proper Einstein radius calculations
- PyAutoLens integration for ray-tracing and galaxy modeling
- Comprehensive parameter extraction and metadata tracking
- Rich property interface with derived quantities

**Module 2: PSF Generation with Diverging-Path Architecture** ✅ **COMPLETE**  
- Full implementation with unified `PSFData` structure
- Diverging-path architecture: single high-res PSF → two products (metrics + detector kernel)
- Automatic sampling adjustment for integer subsampling factors
- Physical detector modeling using HCIPy's NoisyDetector
- Complete aberration framework (segment pistons, tip/tilts, hexikes, global Zernikes)
- HCIPy telescope modeling with segmented aperture support
- Automatic PSF quality metrics (FWHM, Strehl ratio, RMS wavefront error)
- Phase screen diagnostics and aberration budget analysis
- Direct generation of properly downsampled PyAutoLens kernels

**Plotting and Visualization** ✅ **COMPLETE**
- Comprehensive plotting functions for both lensing and PSF systems
- Subhalo effect visualization and quantitative analysis
- PSF quality assessment plots with zoom capabilities
- Diverging-path validation plots (high-res vs detector comparison)
- Pupil plane diagnostics showing aberration application
- Structured output organization by run and module

### In Development

**Module 3: Observation Simulation** ✅ **MOSTLY COMPLETE** (Missing: dithered observations, cosmic ray rejection, bad pixel masks)
**Module 5: Performance Analysis** 🚧 **PLANNED**

---

## Module 1: Lensing System Generation

### Purpose
Generate the astrophysical ground truth - realistic galaxy-galaxy strong lensing systems with precisely known subhalo populations that we will attempt to recover.

### Implementation Requirements

The module must create diverse lensing configurations while maintaining full control over the subhalo population. Each generated system needs complete metadata for later performance analysis. We implement both individual subhalo detection and statistical population approaches.

### Subhalo Mass Models
The module implements three mass models for subhalos, each with different complexity and realism:

- **PointMass**: Simplest model, good for initial tests and comparison
  - Single parameter: Einstein radius
  - Computational efficiency: Fastest
  - Use case: Baseline comparisons, quick parameter studies
  
- **SIS (Singular Isothermal Sphere)**: Standard choice for analytical work
  - Parameters: Velocity dispersion or Einstein radius
  - Well-understood analytical properties
  - Use case: Comparison with literature, moderate realism
  
- **NFW (Navarro-Frenk-White)**: Most realistic for dark matter halos
  - Parameters: M200, concentration (from mass-concentration relation)
  - Includes realistic density profile with scale radius
  - More computationally expensive but physically motivated
  - Use case: Final science requirements, realistic simulations

### Action Items

**Step 1: Implement Core Galaxy Classes**
- Create `MassProfile` base class with deflection angle methods
- Implement `SingularIsothermalSphere` (SIS) profile for all mass components
- Create `LightProfile` class hierarchy for source galaxies
- Implement `Sersic` profile with variable complexity options
- Add multi-component source capability for complexity studies

**Step 2: Build Subhalo Injector**
- Single subhalo injection per system (initial approach)
- Implement all three mass models (PointMass, SIS, NFW)
- Use Einstein radius calculation appropriate for each model:
  ```python
  # Point Mass
  theta_E = sqrt(4GM/c² × D_ls/(D_l × D_s))
  
  # SIS from M200
  sigma_v = sqrt(GM200/(2r200))
  theta_E = 4π(σ_v/c)² × (D_ls/D_s)
  
  # NFW (numerical calculation)
  theta_E = f(M200, c200, z_lens, z_source)
  ```
- Place subhalos in annulus around Einstein ring (~10 pixel width)
- Mass range: 10^7 - 10^8 M☉ 
- Store truth position and mass for detection analysis

**Step 3: Create Ray-Tracing Engine**
```python
def generate_lensing_system(config, full_config=None):
    """
    Generate complete lensing system from configuration.
    
    Returns LensingData object with unified structure providing
    direct access to all system parameters and derived quantities.
    """
    # Create coordinate grid
    grid = al.Grid2D.uniform(
        shape_native=config['grid']['shape'],
        pixel_scales=config['grid']['pixel_scale']
    )
    
    # Create lens and source galaxies
    lens_galaxy = _create_lens_galaxy(config['lens_galaxy'])
    source_galaxy = _create_source_galaxy(config['source_galaxy'])
    
    # Add subhalo if specified
    if 'subhalo' in config:
        subhalo, subhalo_info = _create_subhalo(config['subhalo'], ...)
        lens_galaxy = al.Galaxy(
            redshift=lens_config['redshift'],
            mass=lens_galaxy.mass,
            subhalo=subhalo
        )
    
    # Create tracer and generate image
    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    lensed_image = tracer.image_2d_from(grid=grid)
    
    return LensingData(
        image=lensed_image.native,
        grid=grid,
        tracer=tracer,
        # ... all system parameters extracted automatically
    )
```

**Step 4: Source Complexity Control**
```python
class SourceComplexity:
    def __init__(self, mode='simple'):
        self.mode = mode  # 'simple', 'moderate', 'complex'
        
    def generate_source(self):
        if self.mode == 'simple':
            # Single smooth Sersic
            return Sersic(n=1, effective_radius=0.2)
        elif self.mode == 'moderate':
            # Multiple components
            return [Sersic(n=1), Sersic(n=4, offset=True)]
        elif self.mode == 'complex':
            # Irregular, clumpy structure
            return generate_clumpy_source()
```

**Step 5: Parameter Ranges**
- **Lens galaxy**: 
  - Redshift: 0.3-0.5
  - Mass: SIS with Einstein radius 1.0-1.5"
  - No lens light (initially)
- **Source galaxy**:
  - Redshift: 0.8-2.0
  - Size: 0.1-0.5" effective radius
  - Complexity: switchable parameter
- **Subhalo**:
  - Mass: 10^7, 10^7.5, 10^8 M☉
  - Position: Within Einstein ring ± 10 pixels
  - Model: Configurable (PointMass/SIS/NFW)

### Expected Outputs
- `LensingData` objects with unified structure providing direct access to:
  - Lensed image array and PyAutoLens grid/tracer objects
  - Complete system parameters (redshifts, Einstein radii, pixel scale)
  - Subhalo truth information (mass, model, position, Einstein radius)
  - Galaxy parameters (positions, ellipticities, light profiles)
  - Derived properties (field of view, flux statistics, subhalo presence flag)
- Automatic timestamp and provenance tracking
- Einstein radius calculations for all three mass models (PointMass, SIS, NFW)
- Seamless integration with PyAutoLens modeling workflows
- Rich property interface for analysis and visualization

---

## Module 2: PSF Generation with Diverging-Path Architecture

### Purpose
Create realistic PSFs using a physically accurate "diverging-path" architecture where a single high-fidelity pupil wavefront generates two products: (1) a high-resolution PSF for optical performance metrics, and (2) a correctly downsampled PSF kernel for lensing simulations.

### Implementation Requirements

The module generates PSFs for the segmented HWO aperture with various aberration modes, automatically adjusts sampling to ensure integer subsampling factors, and uses HCIPy's NoisyDetector to physically model how real detectors integrate light over pixel areas.

### Diverging-Path Architecture

The key innovation is propagating the wavefront only once from pupil to focal plane, then using that high-resolution PSF for two purposes:

1. **Branch A (Metrics)**: Direct analysis of the high-resolution PSF for quality metrics (FWHM, Strehl ratio, etc.)
2. **Branch B (Kernel)**: Physical downsampling using detector integration to create the science kernel

This approach is physically accurate because light propagates once in reality - the detector simply integrates whatever light falls on each pixel.

### Automatic Sampling Adjustment

To ensure proper detector downsampling, the module automatically calculates the ideal sampling value:

```python
# Given:
# - target_pixel_scale: desired detector pixel scale (from lensing config)
# - wavelength, pupil_diameter: telescope parameters
# - requested_sampling: user's desired oversampling

# Calculate:
res_element_arcsec = (wavelength / pupil_diameter) * 206264.8062471
N = round(target_pixel_scale / (res_element_arcsec / requested_sampling))

# Adjust sampling to ensure integer N:
used_sampling = (N * res_element_arcsec) / target_pixel_scale
```

This ensures the detector can downsample by exactly N pixels in each direction without interpolation artifacts.

### PSF Aberration Architecture

The module implements four types of aberrations, applied in specific order:

1. **Segment Piston Errors**: Individual segment height offsets
   - Applied via `SegmentedDeformableMirror`
   - Units: nm of wavefront OPD (not surface)
   - Reflection logic: For a mirror, OPD = 2 × surface height, so the code
     divides the requested OPD by 2 before writing the piston actuator value.

2. **Segment Tip/Tilt**: Individual segment angle errors
   - Applied via `SegmentedDeformableMirror`  
   - Units: microradians
   - Affects pointing of each segment

3. **Segment-Level Zernikes**: Per-segment wavefront aberrations
   - Uses HCIPy's hexike basis (hexagonal Zernike polynomials)
   - Applied as phase screen
   - Enables complex per-segment aberrations

4. **Global Zernikes**: Full-aperture wavefront aberrations
   - Standard Zernike basis over full pupil
   - Applied as phase screen
   - Represents telescope-level aberrations

### Implementation Flow

```python
def generate_psf_system(config, full_config=None):
    """
    Generate PSF system using diverging-path architecture.
    """
    # 1. Auto-adjust sampling for integer subsampling
    target_pixel_scale = full_config['lensing']['grid']['pixel_scale']
    N, used_sampling = calculate_integer_subsampling(...)
    
    # 2. Create telescope model
    telescope_data = create_hcipy_telescope(config)
    
    # 3. Apply aberrations to pupil wavefront (single source of truth)
    wf_pupil = apply_all_aberrations(telescope_data, aberrations)
    
    # 4. Single propagation to high-res focal plane
    focal_grid_hres = hcipy.make_focal_grid(
        q=used_sampling,  # Auto-adjusted value
        num_airy=config['hres_psf']['num_airy'],
        spatial_resolution=wavelength/pupil_diameter
    )
    prop_hres = hcipy.FraunhoferPropagator(pupil_grid, focal_grid_hres)
    wf_psf_hres = prop_hres(wf_pupil)
    
    # 5. Branch A: Calculate metrics from high-res PSF
    quality_metrics = analyze_psf_quality(wf_psf_hres, telescope_data)
    
    # 6. Branch B: Detector downsampling
    detector_grid = hcipy.make_uniform_grid(
        dims=kernel_shape_native,  # Enforced odd dimensions
        extent=kernel_shape_native * target_pixel_scale * (π/180/3600)  # rad
    )
    detector = hcipy.NoiselessDetector(detector_grid, subsampling=N)
    detector.integrate(wf_psf_hres, dt=1)
    psf_downsampled = detector.read_out()
    
    # 7. Create PyAutoLens kernel
    kernel = al.Kernel2D.no_mask(
        values=psf_downsampled.shaped / psf_downsampled.sum(),
        pixel_scales=target_pixel_scale
    )
    
    return PSFData(
        psf=wf_psf_hres,
        kernel=kernel,
        requested_sampling_factor=requested_sampling,
        used_sampling_factor=used_sampling,
        integer_subsampling_factor=N,
        # ... all metrics and parameters
    )
```

### PSF Quality Metrics

The module calculates standard optical quality metrics from the high-resolution PSF:

- **FWHM (Full Width at Half Maximum)**:
  - 2D Gaussian fitting to PSF core
  - Geometric mean of x and y FWHM
  - Units: arcseconds or milliarcseconds

- **Strehl Ratio**:
  - Ratio of peak intensity to perfect PSF peak
  - Measures overall aberration impact
  - Range: 0 to 1 (1 = perfect)

- **RMS Wavefront Error**:
  - Root mean square of OPD over pupil
  - Units: nanometers
  - Direct measure of aberration amplitude

### Expected Outputs
- `PSFData` objects with unified structure providing direct access to:
  - High-resolution HCIPy PSF field for quality analysis
  - Properly downsampled PyAutoLens kernel for convolution
  - Complete telescope parameters (diameter, focal length, segment geometry)
  - Auto-computed PSF quality metrics (FWHM, Strehl ratio, peak intensity)
  - Comprehensive aberration summary (RMS values, breakdown by type)
  - Sampling metadata (requested, used, integer subsampling factor)
  - Rich property interface (diffraction limits, F-number, quality grades)
- Energy conservation through detector downsampling process
- Phase screen diagnostics for aberration analysis
- Automatic quality assessment and aberration budget breakdown
- Integration with plotting and analysis workflows

---

## Module 3: Observation Simulation

### Purpose
Apply realistic observation effects including PSF convolution, detector noise, and observing strategy to create data that matches what HWO would actually observe.

### Implementation Status: ✅ **MOSTLY COMPLETE**

The module accurately simulates the complete observation process from perfect lensed images through all instrumental effects to final noisy data. Core functionality is implemented with realistic detector physics and comprehensive noise modeling.

### Implemented Features ✅

**PSF Convolution Engine**
```python
def generate_observation(
    lensing_data: LensingData,
    psf_data: PSFData, 
    observation_config: Optional[Dict] = None,
    full_config: Optional[Dict] = None
) -> ObservationData:
    """Generate realistic observation from lensing and PSF data.
    
    Two-step process:
    1. PSF convolution using PyAutoLens SimulatorImaging (noiseless)
    2. Application of realistic detector noise model
    """
    # Step 1: Noiseless PSF convolution
    simulator_noiseless = al.SimulatorImaging(
        exposure_time=exposure_time,
        psf=psf_kernel,
        background_sky_level=0.0,
        normalize_psf=True,
        add_poisson_noise_to_data=False
    )
    noiseless_dataset = simulator_noiseless.via_image_from(image=lensed_image)
    
    # Step 2: Apply realistic detector noise
    final_image_adu, components = apply_detector_noise(...)
    noise_map_adu = create_noise_map(...)
    
    # Create final PyAutoLens imaging dataset
    imaging_dataset = al.Imaging(data=data, noise_map=noise_map, psf=psf_kernel)
    
    return ObservationData(
        imaging=imaging_dataset,
        noiseless_source_eps=source_only_eps,
        noise_components=components,
        config=observation_config,
        metadata=metadata
    )
```

**Comprehensive Noise Modeling**
- **Poisson noise**: Photon shot noise on source + sky + dark current
- **Read noise**: Gaussian, ~5 e-/pixel (configurable)
- **Dark current**: ~0.001 e-/pixel/s (configurable)
- **Sky background**: ~1.0 e-/pixel/s (configurable)
- **Proper unit conversions**: e-/s → e- → ADU via gain
- **Realistic detector parameters**: Gain, read noise, dark current, sky background

**Detector Physics Implementation**
```python
def apply_detector_noise(source_eps, exposure_time, detector_config, seed=None):
    """Apply complete detector noise model.
    
    Physics:
    1. Convert all components to electrons
    2. Apply Poisson statistics to total expected counts  
    3. Add Gaussian read noise
    4. Convert to ADU using gain
    """
    # Convert to electrons
    source_e = source_eps * exposure_time
    dark_e = dark_current * exposure_time
    sky_e = sky_background * exposure_time
    
    # Total expected electrons per pixel
    expected_e = source_e + dark_e + sky_e
    
    # Apply Poisson noise
    detected_e = np.random.poisson(expected_e)
    
    # Add read noise
    final_e = detected_e + np.random.normal(0, read_noise, size=detected_e.shape)
    
    # Convert to ADU
    final_image_adu = final_e / gain
    
    return final_image_adu, components
```

**Noise Map Generation**
```python
def create_noise_map(source_eps, exposure_time, detector_config):
    """Create proper noise map representing total uncertainty.
    
    Total variance = Poisson variance + read noise variance
    """
    expected_e = source_e + dark_e + sky_e
    total_variance_e2 = expected_e + read_noise**2
    noise_map_adu = np.sqrt(total_variance_e2) / gain
    return noise_map_adu
```

**ObservationData Features**
- **Complete PyAutoLens integration**: Ready-to-use `al.Imaging` dataset
- **Rich property interface**: Direct access to all observation parameters
- **Noise component breakdown**: Individual noise sources stored for analysis
- **Metadata tracking**: Complete provenance and observation parameters
- **SNR analysis**: Built-in signal-to-noise calculations
- **Unit conversions**: Automatic conversions between e-, ADU, and flux units

### Remaining Planned Features 🚧

**Advanced Observing Strategies**
```python
class ObservingStrategy:
    def __init__(self, mode='single'):
        self.mode = mode  # 'single', 'dithered', 'temporal'
        
    def observe(self, true_image, psf, noise_model):
        if self.mode == 'dithered':
            # Multiple dithered exposures - PLANNED
            return combine_dithers(true_image, psf, noise_model)
        elif self.mode == 'temporal':
            # Time series for PSF monitoring - PLANNED
            return temporal_sequence(true_image, psf, noise_model)
```

**Additional Detector Realism**
- **Cosmic ray rejection**: Space-based detector effects
- **Bad pixel masks**: Dead/hot pixel simulation
- **Wavelength-dependent sky background**: More realistic background model
- **Time-varying PSF convolution**: For PSF drift studies

### Current Outputs ✅
- `ObservationData` objects ready for lens modeling
- Realistic S/N ratios matching HWO expectations
- Proper detector physics with comprehensive noise modeling
- Complete PyAutoLens integration for seamless modeling workflow
- Rich metadata for analysis and debugging
- Global seed support for reproducible noise realizations

### Scientific Impact
The implemented observation simulation provides physically accurate detector modeling that enables:
- Realistic signal-to-noise calculations for subhalo detection
- Proper noise correlations for statistical analysis
- Accurate representation of space-based detector characteristics
- Seamless integration with PyAutoLens modeling pipeline

---

## Module 4: Subhalo Detection via Chi-Square Analysis

### Purpose
Quantify subhalo detectability by comparing a simulated observation containing a subhalo against a baseline (null hypothesis) observation without one. This module uses a rapid, statistically robust chi-square analysis rather than full lens modeling.

### Implementation Status: ✅ **COMPLETE**

The module is fully implemented with a robust chi-square detection system that exactly follows the validated prototype methodology. It provides computationally efficient, statistically rigorous detection ideal for large parameter sweeps.

**Core Classes Implemented:**
- `ChiSquareSubhaloDetector`: Main detection engine with SNR masking and statistical testing
- `DetectionData`: Unified data structure with rich computed properties  
- `DetectionResult`: Individual detection results storage
- `perform_subhalo_detection()`: Main API function following HWO-SLAPS pattern

### Key Implementation Features

**Statistical Detection Methodology**
- **Relative Comparison Approach**: Direct statistical hypothesis test comparing baseline (no subhalo) vs test (with subhalo) observations
- **Null Hypothesis (H₀)**: Observed data consistent with smooth lens model
- **Alternative Hypothesis (H₁)**: Observed data contains subhalo perturbations

**SNR-Based Pixel Masking**
```python
# Absolute SNR threshold pixel selection (default: 3.0)
snr_array_2d = source_adu_2d / noise_map_adu_2d  
snr_mask_2d = snr_array_2d > snr_threshold

# Cross-shaped connectivity for region identification
structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
labeled_regions_2d, num_regions = ndimage.label(snr_mask_2d, structure=structure)
```

**Physically Accurate Variance Modeling**
- **Best Practice**: Reuses baseline observation's noise map for both SNR calculation and chi-square weighting
- **Complete Physics**: Includes Poisson shot noise (source + sky + dark) and detector read noise
- **Consistent Units**: All calculations performed in ADU space using observation's gain factor

**Pearson's Chi-Square Implementation**
```python
# Statistical test with proper degrees of freedom
chi2_value = np.sum((observed - expected)**2 / variance_adu)
dof = pixels_unmasked - 3  # Account for subhalo parameters (mass, x, y)

# Multi-significance testing with proper p-values
significance_levels = [1.349898e-3, 3.167124e-5, 2.866516e-7]  # 3σ, 4σ, 5σ
threshold = chi2(dof).isf(sig_level)
detected = chi2_value > threshold
```

**DetectionData Unified Structure**
- **Primary Results**: Chi-square values, degrees of freedom, detection flags
- **Rich Properties**: `max_significance_detected`, `detection_summary`, `is_detected_5sigma`
- **Complete Diagnostics**: SNR masks, residual maps, regional analysis
- **Truth Integration**: Subhalo mass, model, position from lensing data
- **Validation**: Comprehensive validation functions with mathematical consistency checks

**Advanced Features**
- **Regional SNR Analysis**: Connected region identification with maximum regional SNR calculation  
- **Comprehensive Validation**: `validate_detection_results()` with consistency checks
- **Rich Reporting**: `print_detection_summary()` with multi-significance breakdown
- **Exact Prototype Fidelity**: Implementation preserves validated methodology from notebooks

### Current Outputs ✅
- `DetectionData` objects with complete unified access to:
  - Detection results at 3σ, 4σ, 5σ significance levels with proper p-values
  - Chi-square statistics (χ² value, DOF, global p-value, thresholds)
  - High-quality diagnostics (SNR masks, residual maps, regional analysis)
  - Ground truth integration (subhalo mass, model, position validation)
  - Comprehensive metadata (detection parameters, observation config, provenance)
  - Rich computed properties (`max_significance_detected`, `detection_mask_fraction`)
  - Mathematical validation with consistency checks
  - Direct property access following HWO-SLAPS unified pattern

### Scientific Impact
The implemented detection system provides:
- **Statistical Rigor**: Proper chi-square testing with validated variance modeling
- **Computational Efficiency**: ~1000x faster than full lens modeling for parameter sweeps  
- **Physical Accuracy**: Complete noise physics matching observation module
- **Prototype Validation**: Exactly preserves validated methodology from prototype studies
- **Integration Ready**: Seamless data flow to Module 5 performance analysis

---

## Module 5: Performance Analysis & Requirements

### Purpose
Quantify detection performance across parameter space and derive specific PSF requirements for successful 10^7 M☉ subhalo detection at various significance levels.

### Implementation Requirements

The analysis must provide statistically robust results that account for both astrophysical and instrumental uncertainties. Requirements should be specific and actionable for telescope designers.

### Action Items

**Step 1: Implement Performance Metrics**
```python
class PerformanceAnalyzer:
    def calculate_detection_rate(self, significance_threshold):
        # Fraction detected at 3σ, 4σ, 5σ
        # As function of PSF quality
        
    def correlate_with_source_complexity(self):
        # Detection rate vs source structure
        # Key for PSF requirements
        
    def calculate_false_positive_rate(self):
        # Run on fields without subhalos
        # Calibrate detection thresholds
```

**Step 2: Map PSF Dependencies**
- For each Zernike mode Z4-Z50:
  - Run detection at amplitudes: 0, 10, 25, 50, 100, 200 nm RMS
  - Track detection rates at 3σ, 4σ, 5σ
  - Find critical amplitude where 5σ detections drop below threshold
- Test combined aberration effects
- Identify most harmful mode combinations

**Step 3: Analyze Temporal Requirements**
- Test drift rates: 0.1, 0.5, 1, 5, 10 nm/hour
- Test vibration frequencies: 0.01, 0.1, 1 Hz
- Determine critical timescales for:
  - PSF calibration validity
  - Maximum exposure time
  - Required stability period

**Step 4: Derive Requirements**
1. Process all simulation results
2. Generate requirement curves:
   - Detection efficiency vs RMS wavefront error
   - Critical modes ranked by impact
   - Stability timescale requirements
   - Source complexity dependencies
3. Create summary tables:
   - Maximum tolerable aberration per mode for 5σ detection
   - Required stability for different observation strategies
   - Recommendations based on source type

**Step 5: Statistical Validation**
- Bootstrap resample to estimate uncertainties
- Run null tests (no subhalos) to calibrate false positives
- Verify requirements are robust across lens configurations

### Expected Outputs
- Performance metric database (HDF5 format) - **saved to disk**
- Requirement curves (publication-quality plots) - **saved to disk**
- Technical requirements document - **saved to disk**
- All intermediate calculations in memory until final aggregation

---

## Integration and Testing Plan

### Phase 1: Module Validation (Weeks 1-2)
Each module needs standalone testing before integration:
- Unit tests for core functions
- Validation against analytical cases
- Performance benchmarking

### Module-Specific Validation
**Module 1 (Lensing):**
- Compare SIS deflections to analytical formula: α = 4π(σ_v/c)² r/|r|
- Verify Einstein radii for each mass model match predictions
- Check image flux conservation in ray-tracing
- Validate cosmological distance calculations

**Module 2 (PSF):**
- Check PSF normalization (total flux = 1)
- Verify Strehl ratio ≤ 1 for all aberrated PSFs
- Compare FWHM to λ/D for perfect aperture
- Validate integer subsampling factors
- Verify energy conservation through detector

**Integration Tests:**
- Consistent pixel scales between modules
- Proper coordinate system alignment
- Memory usage within limits
- No unit conversion errors

### Phase 2: Pipeline Integration (Week 3)
Connect modules with standardized interfaces:
- Define data formats between modules
- Implement pipeline orchestrator
- Test end-to-end data flow

### Phase 3: Science Validation (Week 4)
Verify scientific accuracy:
- Run known test cases
- Check detection of obvious subhalos
- Verify null detection rates

### Phase 4: Production Runs (Weeks 5-8)
Execute parameter studies:
- Start with subset for debugging
- Scale to full parameter space
- Monitor for failures and edge cases

---

## Computational Considerations

### Memory Management
- Typical run uses ~150 MB in memory (very manageable)
- PSF time series are the largest objects (~100 MB)
- Clear references promptly to enable garbage collection
- Use generators for parameter sweeps to avoid loading all at once

### Parallel Processing
- Each parameter point is independent
- Use multiprocessing pool for sweeps
- Share read-only data (PSF libraries) between processes
- Collect only metrics, not full data

### Storage Strategy
- Production runs: ~1 GB total output for 1000 parameters
- Debug mode: ~150 GB if saving all intermediates
- Use hierarchical directories: `results/{date}/{run_id}/`
- Compress final metric files

### Performance Optimization
```python
# Bad: Saves 150 GB unnecessarily
for params in parameter_grid:
    result = pipeline.run(params)
    save_everything_to_disk(result)  # Don't do this!

# Good: Saves only 1 GB of metrics
results = []
for params in parameter_grid:
    metrics = pipeline.run(params)  # Memory only
    results.append(metrics)
save_aggregated_metrics(results)  # Save once at end
```

---

## Module Dependencies

### External Package Requirements

**Module 1 (Lensing System Generation):**
- PyAutoLens >= 2025.5.10.1
- NumPy >= 1.20
- Matplotlib >= 3.3
- Astropy >= 5.0 (for units and cosmology)

**Module 2 (PSF Generation):**
- HCIPy >= 0.5
- PyAutoLens >= 2025.5.10.1 (for Kernel2D output)
- NumPy >= 1.20
- SciPy >= 1.7 (for curve fitting in FWHM calculation)
- Astropy >= 5.0 (for units)

**Module 3 (Observation Simulation):**
- PyAutoLens >= 2025.5.10.1 (for SimulatorImaging and Imaging dataset)
- NumPy >= 1.20
- Astropy >= 5.0 (for units)
- Dataclasses (for ObservationData structure)

**Module 4 (Subhalo Detection):**
- NumPy >= 1.20
- SciPy >= 1.7 (for ndimage and chi2 distribution)
- Dataclasses (for DetectionData and DetectionResult structures)
- ObservationData and LensingData from other modules

**Module 5 (Performance Analysis):**
- NumPy >= 1.20
- Matplotlib >= 3.3
- Pandas >= 1.3
- h5py >= 3.0 (for HDF5 storage)

---

## Success Criteria

The pipeline is successful when it can:
1. Generate realistic lensing systems with proper mass model options
2. Create physically accurate PSFs using diverging-path architecture with proper detector modeling
3. Achieve detection rates: ~90% at 3σ, ~50% at 4σ, ~10% at 5σ
4. Correlate detection efficiency with source complexity
5. Provide specific PSF stability requirements as function of:
   - Individual Zernike mode amplitudes
   - Temporal drift rates and frequencies
   - Source galaxy morphology
   - Subhalo mass and position

The ultimate deliverable is a technical requirements document that tells the HWO team exactly what PSF quality and stability is needed for dark matter science, with explicit consideration of how source galaxy complexity affects these requirements.