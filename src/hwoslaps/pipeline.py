"""Main pipeline orchestration for HWO-SLAPS.

This module provides high-level functions to run the complete
strong lensing analysis pipeline, including both standard simulation
mode and subhalo detection mode.
"""

import yaml
from typing import Dict, Union
from copy import deepcopy

from .lensing import generate_lensing_system
from .psf import generate_psf_system  
from .observation import generate_observation
from .lensing.utils import print_lensing_data_summary
from .psf.utils import print_psf_data_summary
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
        elif detection_method == 'fisher':
            if self.verbose:
                print("\nPerforming Fisher v1 detectability (local/map Asimov metrics)...")
            from .modeling.generator_fisher import perform_fisher_detection
            from .modeling.utils_fisher import print_fisher_summary
            fisher_data = perform_fisher_detection(
                observation_baseline=obs_baseline,
                observation_test=obs_test,
                lensing_baseline=lensing_baseline,
                lensing_test=lensing_test,
                psf_data=psf_data,
                detection_config=config['modeling'],
                full_config=config,
            )
            if self.verbose:
                print("\n🎯 Fisher detectability analysis complete!")
                print_fisher_summary(fisher_data)
            detection_data = fisher_data
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
        """Standard single observation pipeline.
        
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
