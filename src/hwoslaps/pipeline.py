"""Main pipeline orchestration for HWO-SLAPS.

This module provides high-level functions to run the complete
strong lensing analysis pipeline, including both standard simulation
mode and subhalo detection mode.
"""

from copy import deepcopy
import os
from pathlib import Path
from typing import Dict, Union

import yaml

from .config.validation import validate_or_raise
from .lensing import generate_lensing_system
from .lensing.utils import print_lensing_data_summary
from .modeling.utils_fisher import FisherDetectionData, print_fisher_summary
from .observation import generate_observation
from .observation.utils import ObservationData, print_observation_summary
from .plotting import generate_all_plots
from .psf import generate_psf_system
from .psf.utils import print_psf_data_summary


def _resolve_relative_output_dir(config: Dict) -> None:
    plotting = config.get('plotting', {})
    if not isinstance(plotting, dict) or 'output_dir' not in plotting:
        return
    output_dir = Path(plotting['output_dir']).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[2] / output_dir
    plotting['output_dir'] = str(output_dir)


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

    def run(self, config: Dict) -> Union[ObservationData, FisherDetectionData]:
        """Run the pipeline, selecting the mode from the configuration.

        Parameters
        ----------
        config : `dict`
            Full pipeline configuration dictionary.

        Returns
        -------
        result : `ObservationData` or `FisherDetectionData`
            `ObservationData` in standard mode, or `FisherDetectionData` in
            detection mode (when ``modeling.enabled`` is true).
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

    def _run_detection_pipeline(self, config: Dict) -> FisherDetectionData:
        """Generate paired observations and perform Fisher detection analysis.

        Parameters
        ----------
        config : dict
            Full pipeline configuration.

        Returns
        -------
        detection_data : FisherDetectionData
            Complete Fisher detection results.
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

        # Legacy detector families were removed; only Fisher-based modeling
        # remains supported.
        detection_method = config['modeling'].get('detection', 'fisher').lower()
        if detection_method != 'fisher':
            raise ValueError(
                f"Unsupported modeling.detection={detection_method!r}. "
                "Only 'fisher' is supported."
            )

        if self.verbose:
            print("\nPerforming Fisher detectability (local/map Asimov metrics)...")
        from .modeling.generator_fisher import perform_fisher_detection
        detection_data = perform_fisher_detection(
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
            print_fisher_summary(detection_data)

        # Grid maps are stage-two analysis inputs; persist the arrays even
        # when plotting is disabled.
        if detection_data.has_grid_map:
            from .modeling.utils_fisher import save_fisher_grid_map_npz
            from .provenance import _git_hash, config_hash
            grid_map_dir = (
                Path(config['plotting']['output_dir']) / config['run_name'] / 'modeling'
            )
            snapshot_path = grid_map_dir.parent / 'config_used.yaml'
            detection_data.grid_map.config_hash = None
            if snapshot_path.is_file():
                with snapshot_path.open('r', encoding='utf-8') as stream:
                    snapshot_config = yaml.safe_load(stream)
                resolved_snapshot = deepcopy(snapshot_config)
                _resolve_relative_output_dir(resolved_snapshot)
                expected = config_hash(config)
                snapshot_hash = config_hash(snapshot_config)
                if (
                    snapshot_hash != expected
                    and config_hash(resolved_snapshot) != expected
                ):
                    raise ValueError(
                        f"Adjacent config snapshot {snapshot_path} does not "
                        "describe this run; refusing to bind the grid map "
                        "to it."
                    )
                detection_data.grid_map.config_hash = snapshot_hash
            detection_data.grid_map.git_hash = _git_hash(
                Path(__file__).resolve().parent
            )
            # S1-lite exports HWOSLAPS_CAMPAIGN_UUID to every campaign job;
            # standalone runs stay unbound.
            detection_data.grid_map.campaign_uuid = os.environ.get(
                'HWOSLAPS_CAMPAIGN_UUID'
            )
            grid_map_dir.mkdir(parents=True, exist_ok=True)
            grid_map_path = save_fisher_grid_map_npz(
                detection_data.grid_map,
                grid_map_dir / 'fisher_grid_map.npz',
            )
            if self.verbose:
                print(f"Fisher grid map arrays saved: {grid_map_path}")

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
        """Run the standard single-observation pipeline.

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


def run_enhanced_pipeline(config_path: str,
                          verbose: bool = True) -> Union[ObservationData, FisherDetectionData]:
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
    result : ObservationData or FisherDetectionData
        - ObservationData in standard mode
        - FisherDetectionData in detection mode (when modeling.enabled: true)

    Examples
    --------
    Standard observation simulation:

    >>> obs_data = run_enhanced_pipeline('standard_config.yaml')
    >>> print(f"Peak SNR: {obs_data.signal_to_noise_map.native.max():.2f}")

    Fisher detection study:

    >>> detection_data = run_enhanced_pipeline('detection_config.yaml')
    >>> print(f"Mode: {detection_data.mode}")
    >>> print(f"Pixels analyzed: {detection_data.pixels_unmasked}")
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    _resolve_relative_output_dir(config)
    validate_or_raise(config)

    # Create and run pipeline
    pipeline = Pipeline(verbose=verbose)
    return pipeline.run(config)
