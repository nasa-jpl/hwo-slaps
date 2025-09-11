"""Main pipeline orchestration for HWO-SLAPS.

This module provides high-level functions to run the complete
strong lensing analysis pipeline, including both standard simulation
mode and subhalo detection mode.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from copy import deepcopy

from .lensing import generate_lensing_system
from .psf import generate_psf_system  
from .observation import generate_observation
from .lensing.utils import print_lensing_data_summary, LensingData
from .psf.utils import print_psf_data_summary, PSFData
from .observation.utils import print_observation_summary, ObservationData
from .plotting import generate_all_plots
from .modeling.utils import print_detection_summary, DetectionData
from .config.validation import validate_or_raise


class Pipeline:
    """Enhanced HWO-SLAPS pipeline with detection mode support.
    
    This class provides automatic mode detection and handles both:
    - Standard mode: Single observation generation
    - Detection mode: Paired observation generation + subhalo detection
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize pipeline.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress information.
        """
        self.verbose = verbose
    
    def run(self, config: Dict) -> Union[ObservationData, 'DetectionData']:
        """Main pipeline entry point with automatic mode detection.
        
        Parameters
        ----------
        config : dict
            Full pipeline configuration dictionary.
            
        Returns
        -------
        result : ObservationData or DetectionData
            - ObservationData in standard mode
            - DetectionData in detection mode (when modeling.enabled: true)
        """
        # Validate configuration (strict, fail-fast)
        validate_or_raise(config)
        
        # Route to appropriate pipeline based on configuration
        if config['modeling']['enabled']:
            if self.verbose:
                print("🔍 Detection mode enabled - running paired observation analysis")
            return self._run_detection_pipeline(config)
        else:
            if self.verbose:
                print("📊 Standard mode - running single observation pipeline")
            return self._run_standard_pipeline(config)
    
    def _run_detection_pipeline(self, config: Dict) -> 'DetectionData':
        """Generate paired observations and perform detection analysis.
        
        This method replicates the exact workflow from the prototype notebook
        (lines 75-361) within the pipeline architecture.
        
        Parameters
        ----------
        config : dict
            Full pipeline configuration.
            
        Returns
        -------
        detection_data : DetectionData
            Complete detection results with unified structure.
        """
        # Strict validation already applied at entry
        
        if self.verbose:
            print("\n" + "="*50)
            print("DETECTION PIPELINE EXECUTION")  
            print("="*50)
        
        # Generate PSF once (shared between both observations for efficiency)
        if self.verbose:
            print("Generating shared PSF system...")
        psf_data = generate_psf_system(config['psf'], full_config=config)
        if self.verbose:
            print_psf_data_summary(psf_data)
        
        # Generate baseline observation (no subhalo)
        if self.verbose:
            print("\nGenerating baseline lensing system (no subhalo)...")
        config_baseline = self._create_baseline_config(config)
        lensing_baseline = generate_lensing_system(
            config_baseline['lensing'], full_config=config_baseline
        )
        if self.verbose:
            print_lensing_data_summary(lensing_baseline)
        
        if self.verbose:
            print("Generating baseline observation (no subhalo)...")
        obs_baseline = generate_observation(
            lensing_data=lensing_baseline,
            psf_data=psf_data,
            observation_config=config_baseline['observation'],
            full_config=config_baseline
        )
        if self.verbose:
            print_observation_summary(obs_baseline)
        
        # Generate test observation (with subhalo)
        if self.verbose:
            print("\nGenerating test lensing system (with subhalo)...")
        config_test = self._create_test_config(config)
        lensing_test = generate_lensing_system(
            config_test['lensing'], full_config=config_test
        )
        if self.verbose:
            print_lensing_data_summary(lensing_test)
        
        if self.verbose:
            print("Generating test observation (with subhalo)...")
        obs_test = generate_observation(
            lensing_data=lensing_test,
            psf_data=psf_data,
            observation_config=config_test['observation'],
            full_config=config_test
        )
        if self.verbose:
            print_observation_summary(obs_test)
        
        # Perform detection (Module 4), routing by modeling.detection
        detection_method = config['modeling'].get('detection', 'gof').lower()
        if detection_method == 'chernoff':
            if self.verbose:
                print("\nPerforming Chernoff minimal-fit detection (fixed position)...")
            from .modeling.generator_chernoff import perform_chernoff_detection
            from .modeling.utils_chernoff import print_chernoff_summary
            # Use the test observation for both reference (template) and test (noisy)
            chernoff_data = perform_chernoff_detection(
                observation_baseline=obs_baseline,
                observation_ref_with_subhalo=obs_test,
                observation_test=obs_test,
                lensing_test=lensing_test,
                detection_config=config['modeling'],
            )
            if self.verbose:
                print("\n🎯 Chernoff detection analysis complete!")
                print_chernoff_summary(chernoff_data)
            detection_data = chernoff_data
        elif detection_method == 'mejiro':
            if self.verbose:
                print("\nPerforming Mejiro detectability (paper-exact)...")
            from .modeling.generator_mejiro import perform_mejiro_detection
            from .modeling.utils_mejiro import print_mejiro_summary
            mejiro_data = perform_mejiro_detection(
                observation_baseline=obs_baseline,
                observation_test=obs_test,
                lensing_test=lensing_test,
                detection_config=config['modeling'],
                full_config=config,
            )
            if self.verbose:
                print("\n🎯 Mejiro detectability analysis complete!")
                print_mejiro_summary(mejiro_data)
            detection_data = mejiro_data
        else:
            if self.verbose:
                print("\nPerforming chi-square subhalo detection (goodness-of-fit)...")
            from .modeling import perform_subhalo_detection
            detection_data = perform_subhalo_detection(
                observation_baseline=obs_baseline,
                observation_test=obs_test,
                lensing_baseline=lensing_baseline,
                lensing_test=lensing_test,
                detection_config=config['modeling'],
                full_config=config
            )
            if self.verbose:
                print("\n🎯 Detection analysis complete!")
                # Comprehensive significance summary (includes 3σ/4σ/5σ with exact sigma and p)
                print_detection_summary(detection_data)
        
        # Generate plots if enabled
        if config['plotting']['enabled']:
            if self.verbose:
                print("\nGenerating plots...")
            
            # Create context for automatic plot generation
            context = {
                'mode': 'detection',
                'has_subhalo': lensing_test.has_subhalo,
                'lensing_data': lensing_test,  # Use test lensing (with subhalo) for plots
                'psf_data': psf_data,
                'obs_data': obs_baseline,  # Use baseline for observation plots
                'detection_data': detection_data,
                'obs_baseline': obs_baseline,
                'obs_test': obs_test,
                'run_name': config['run_name']
            }
            
            # Generate all applicable plots automatically
            generate_all_plots(context, config['plotting'], verbose=self.verbose)
        
        return detection_data
    
    def _run_standard_pipeline(self, config: Dict) -> ObservationData:
        """Standard single observation pipeline (unchanged for backward compatibility).
        
        Parameters
        ----------
        config : dict
            Full pipeline configuration.
            
        Returns
        -------
        observation_data : ObservationData
            Generated observation data.
        """
        if self.verbose:
            print("\n" + "="*50)
            print("STANDARD PIPELINE EXECUTION")
            print("="*50)
        
        # Generate lensing system
        if self.verbose:
            print("Generating lensing system...")
        lensing_data = generate_lensing_system(config['lensing'], full_config=config)
        if self.verbose:
            print_lensing_data_summary(lensing_data)
        
        # Generate PSF system
        if self.verbose:
            print("\nGenerating PSF system...")
        psf_data = generate_psf_system(config['psf'], full_config=config)
        if self.verbose:
            print_psf_data_summary(psf_data)
        
        # Generate observation
        if self.verbose:
            print("\nSimulating observation...")
        obs_data = generate_observation(
            lensing_data=lensing_data,
            psf_data=psf_data,
            observation_config=config['observation'],
            full_config=config
        )
        if self.verbose:
            print_observation_summary(obs_data)
        
        # Generate plots if enabled
        if config['plotting']['enabled']:
            if self.verbose:
                print("\nGenerating plots...")
            
            # Create context for automatic plot generation
            context = {
                'mode': 'standard',
                'has_subhalo': lensing_data.has_subhalo,
                'lensing_data': lensing_data,
                'psf_data': psf_data,
                'obs_data': obs_data,
                'run_name': config['run_name']
            }
            
            # Generate all applicable plots automatically
            generate_all_plots(context, config['plotting'], verbose=self.verbose)
        
        return obs_data
    
    def _create_baseline_config(self, config: Dict) -> Dict:
        """Create configuration for baseline observation (no subhalo).
        
        Replicates prototype lines 82-84:
        config_no_subhalo = deepcopy(config)
        config_no_subhalo['lensing']['subhalo']['enabled'] = False
        
        Parameters
        ----------
        config : dict
            Original configuration.
            
        Returns
        -------
        baseline_config : dict
            Configuration with subhalo disabled.
        """
        baseline_config = deepcopy(config)
        if 'lensing' in baseline_config and 'subhalo' in baseline_config['lensing']:
            baseline_config['lensing']['subhalo']['enabled'] = False
        return baseline_config
    
    def _create_test_config(self, config: Dict) -> Dict:
        """Create configuration for test observation (with subhalo).
        
        Replicates prototype lines 329:
        config['lensing']['subhalo']['enabled'] = True
        
        Parameters
        ----------
        config : dict
            Original configuration.
            
        Returns
        -------
        test_config : dict
            Configuration with subhalo enabled.
        """
        test_config = deepcopy(config)
        if 'lensing' in test_config and 'subhalo' in test_config['lensing']:
            test_config['lensing']['subhalo']['enabled'] = True
        return test_config
    
    def _validate_config(self, config: Dict) -> None:
        """Deprecated: validation handled by validate_or_raise at entry."""
        return
    
    def _validate_detection_config(self, config: Dict) -> None:
        """Deprecated: validation handled by validate_or_raise at entry."""
        return
    


def run_pipeline(config_path: str, verbose: bool = True, modules: Optional[List[str]] = None) -> Tuple[Optional[LensingData], Optional[PSFData], Optional[ObservationData]]:
    """Run the complete HWO-SLAPS pipeline (legacy function for backward compatibility).
    
    This function maintains backward compatibility with existing code.
    For detection mode support, use `run_enhanced_pipeline()` instead.
    
    Parameters
    ----------
    config_path : str
        Path to the master configuration file.
    verbose : bool, optional
        Whether to print summaries at each step.
    modules : list of str, optional
        Specific modules to run. Options: 'lensing', 'psf', 'observation'.
        If None, runs all configured modules.
        
    Returns
    -------
    lensing_data : LensingData or None
        Generated lensing system (None if not run).
    psf_data : PSFData or None
        Generated PSF system (None if not run).
    observation_data : ObservationData or None
        Simulated observation (None if not run).
        
    Notes
    -----
    This function only supports standard observation mode. If the configuration
    contains `modeling.enabled: true`, it will be ignored. Use 
    `run_enhanced_pipeline()` for detection mode support.
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    validate_or_raise(config)
    
    # Force standard mode for backward compatibility
    if 'modeling' in config:
        config = deepcopy(config)
        config['modeling']['enabled'] = False
    
    # Use legacy logic for specific modules or full pipeline
    if modules is not None:
        return _run_legacy_modules(config, verbose, modules)
    
    # Otherwise use standard pipeline but return individual components
    pipeline = Pipeline(verbose=verbose)
    result = pipeline._run_standard_pipeline(config)
    
    # Return in legacy format - we can't easily separate components in the new architecture
    # This is a limitation of maintaining backward compatibility
    return None, None, result


def run_enhanced_pipeline(config_path: str, verbose: bool = True) -> Union[ObservationData, 'DetectionData']:
    """Run the enhanced HWO-SLAPS pipeline with detection mode support.
    
    This function automatically detects whether to run in standard mode or
    detection mode based on the configuration.
    
    Parameters
    ----------
    config_path : str
        Path to the master configuration file.
    verbose : bool, optional
        Whether to print progress information.
        
    Returns
    -------
    result : ObservationData or DetectionData
        - ObservationData in standard mode
        - DetectionData in detection mode (when modeling.enabled: true)
        
    Examples
    --------
    Standard observation simulation:
    
    >>> obs_data = run_enhanced_pipeline('standard_config.yaml')
    >>> print(f"Peak SNR: {obs_data.signal_to_noise_map.native.max():.2f}")
    
    Subhalo detection study:
    
    >>> detection_data = run_enhanced_pipeline('detection_config.yaml')  
    >>> print(f"5σ detection: {detection_data.is_detected_5sigma}")
    >>> print(f"Chi² value: {detection_data.chi2_value:.2f}")
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    validate_or_raise(config)
    
    # Create and run pipeline
    pipeline = Pipeline(verbose=verbose)
    return pipeline.run(config)


def run_pipeline_from_config(config: Dict, verbose: bool = True) -> Union[ObservationData, 'DetectionData']:
    """Run the HWO-SLAPS pipeline from a configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Complete pipeline configuration dictionary.
    verbose : bool, optional
        Whether to print progress information.
        
    Returns
    -------
    result : ObservationData or DetectionData
        - ObservationData in standard mode
        - DetectionData in detection mode (when modeling.enabled: true)
    """
    pipeline = Pipeline(verbose=verbose)
    return pipeline.run(config)


def _run_legacy_modules(config: Dict, verbose: bool, modules: List[str]) -> Tuple[Optional[LensingData], Optional[PSFData], Optional[ObservationData]]:
    """Legacy module-specific pipeline execution for backward compatibility."""
    # Initialize return values
    lensing_data = None
    psf_data = None
    observation_data = None
    
    # Step 1: Generate lensing system
    if 'lensing' in modules:
        if verbose:
            print("Generating lensing system...")
        lensing_data = generate_lensing_system(config['lensing'], full_config=config)
        if verbose:
            print_lensing_data_summary(lensing_data)
    
    # Step 2: Generate PSF system
    if 'psf' in modules:
        if verbose:
            print("\nGenerating PSF system...")
        psf_data = generate_psf_system(config['psf'], full_config=config)
        if verbose:
            print_psf_data_summary(psf_data)
    
    # Step 3: Generate observation
    if 'observation' in modules:
        # For observation, we need lensing and PSF data
        if lensing_data is None or psf_data is None:
            if verbose:
                print("\nObservation module requires lensing and PSF data. Running dependencies...")
            if lensing_data is None:
                if verbose:
                    print("Generating lensing system...")
                lensing_data = generate_lensing_system(config['lensing'], full_config=config)
                if verbose:
                    print_lensing_data_summary(lensing_data)
            if psf_data is None:
                if verbose:
                    print("Generating PSF system...")
                psf_data = generate_psf_system(config['psf'], full_config=config)
                if verbose:
                    print_psf_data_summary(psf_data)
        
        if verbose:
            print("\nSimulating observation...")
        observation_data = generate_observation(
            lensing_data=lensing_data,
            psf_data=psf_data,
            observation_config=config['observation'],
            full_config=config
        )
        if verbose:
            print_observation_summary(observation_data)
    
    # Step 4: Generate plots if enabled
    if config['plotting']['enabled']:
        if verbose:
            print("\nGenerating plots...")
        
        # Create context for automatic plot generation
        context = {
            'mode': 'standard',
            'has_subhalo': lensing_data.has_subhalo if lensing_data else False,
            'lensing_data': lensing_data,
            'psf_data': psf_data,
            'obs_data': observation_data,
            'run_name': config['run_name']
        }
        
        # Generate all applicable plots automatically
        generate_all_plots(context, config['plotting'], verbose=verbose)
    
    return lensing_data, psf_data, observation_data