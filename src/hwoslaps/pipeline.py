"""Main pipeline orchestration for HWO-SLAPS.

This module provides high-level functions to run the complete
strong lensing analysis pipeline, including both standard simulation
mode and subhalo detection mode.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from os import PathLike
from typing import Any, Dict, Union

from .artifacts import write_fisher_grid_map
from .config import load_config, merge_configs, resolve_config_paths, validate_or_raise
from .lensing import generate_lensing_system
from .lensing.utils import print_lensing_data_summary
from .modeling.utils_fisher import FisherDetectionData, print_fisher_summary
from .observation import generate_observation
from .observation.utils import ObservationData, print_observation_summary
from .plotting import generate_all_plots
from .psf import generate_psf_system
from .psf.utils import print_psf_data_summary


class Pipeline:
    """Enhanced HWO-SLAPS pipeline with detection mode support.

    This class provides automatic mode detection and handles both:
    - Standard mode: Single observation generation
    - Detection mode: Paired observation generation + subhalo detection
    """

    def __init__(self, verbose: bool = True, *, save_grid_maps: bool = True):
        """Initialize pipeline.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress information.
        save_grid_maps : bool, optional
            Persist Fisher grid-map arrays. Disable for in-memory forecasts;
            plotting and PSF high-resolution exports have their own switches.
        """
        self.verbose = verbose
        self.save_grid_maps = save_grid_maps

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

        if self.save_grid_maps:
            grid_map_path = write_fisher_grid_map(detection_data, config)
            if self.verbose and grid_map_path is not None:
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


def run_pipeline(
    config: Mapping[str, Any] | str | PathLike[str] | Sequence[str | PathLike[str]],
    *,
    verbose: bool = True,
    base_dir: str | PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    save_grid_maps: bool = True,
) -> Union[ObservationData, FisherDetectionData]:
    """Run a Python configuration or composed YAML files without CLI artifacts.

    File-declared paths resolve against each declaring file's directory.
    Python mappings use ``base_dir`` or the caller's working directory.
    ``overrides`` merges recursively and never modifies the caller's data.
    To retain historical repository-relative paths, pass the repository as
    ``base_dir`` explicitly. Use ``cli.run_with_artifacts`` for snapshot, log
    and provenance capture. Set ``save_grid_maps=False`` to retain grid-map
    results in memory without exporting their NPZ artifact.
    """
    if isinstance(config, Mapping):
        resolved = resolve_config_paths(config, base_dir=base_dir)
        if overrides is not None:
            resolved = merge_configs(resolved, resolve_config_paths(overrides, base_dir=base_dir))
    else:
        resolved = load_config(config, overrides=overrides, base_dir=base_dir, validate=False)
    return Pipeline(verbose=verbose, save_grid_maps=save_grid_maps).run(resolved)


def run_enhanced_pipeline(
    config_path: str,
    verbose: bool = True,
    *,
    base_dir: str | PathLike[str] | None = None,
) -> Union[ObservationData, FisherDetectionData]:
    """Compatibility entry point; new callers should use ``run_pipeline``."""
    return run_pipeline(config_path, verbose=verbose, base_dir=base_dir)
