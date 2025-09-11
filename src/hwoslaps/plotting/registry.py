"""
Plot registry system for automatic plot discovery and execution.

This module provides a system to automatically discover and call plotting
functions without manually adding them to the pipeline code.
"""

import inspect
import importlib
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from functools import wraps


@dataclass
class PlotMetadata:
    """Metadata for a plot function."""
    name: str
    function: Callable
    module_type: str  # 'lensing', 'psf', 'observation', 'detection'
    requires_subhalo: bool = False
    detection_mode_only: bool = False
    standard_mode_only: bool = False
    description: str = ""


def plot_function(module: str, requires_subhalo: bool = False, 
                 detection_mode_only: bool = False, standard_mode_only: bool = False,
                 description: str = ""):
    """Decorator to register a plot function with metadata.
    
    Parameters
    ----------
    module : str
        Module type: 'lensing', 'psf', 'observation', 'detection'
    requires_subhalo : bool, optional
        Whether this plot requires a subhalo to be present
    detection_mode_only : bool, optional
        Whether this plot should only run in detection mode
    standard_mode_only : bool, optional
        Whether this plot should only run in standard mode
    description : str, optional
        Description of what this plot shows
    """
    def decorator(func):
        func._plot_metadata = PlotMetadata(
            name=func.__name__,
            function=func,
            module_type=module,
            requires_subhalo=requires_subhalo,
            detection_mode_only=detection_mode_only,
            standard_mode_only=standard_mode_only,
            description=description
        )
        return func
    return decorator


class PlotRegistry:
    """Registry for automatic plot discovery and execution."""
    
    def __init__(self):
        self.plots: Dict[str, PlotMetadata] = {}
        self._discover_plots()
    
    def _discover_plots(self):
        """Automatically discover all plot functions in the plotting module."""
        # Import all plotting modules to trigger registration
        # Use relative imports since we're within the plotting package
        plotting_modules = [
            '.lensing_plots',
            '.psf_plots', 
            '.observation_plots',
            '.detection_plots'
        ]
        
        for module_name in plotting_modules:
            try:
                module = importlib.import_module(module_name, package='hwoslaps.plotting')
                self._discover_functions_in_module(module)
            except ImportError as e:
                print(f"Warning: Could not import {module_name}: {e}")
                continue
    
    def _discover_functions_in_module(self, module):
        """Discover plot functions in a specific module."""
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if hasattr(obj, '_plot_metadata'):
                # Function has explicit metadata from decorator
                metadata = obj._plot_metadata
                self.plots[name] = metadata
            elif name.startswith('plot_'):
                # Auto-discover based on naming convention
                metadata = self._infer_metadata_from_name_and_signature(name, obj)
                if metadata:
                    self.plots[name] = metadata
    
    def _infer_metadata_from_name_and_signature(self, name: str, func: Callable) -> Optional[PlotMetadata]:
        """Infer plot metadata from function name and signature."""
        # Determine module type from function name
        if 'lensing' in name:
            module_type = 'lensing'
        elif 'psf' in name:
            module_type = 'psf' 
        elif 'observation' in name:
            module_type = 'observation'
        elif 'detection' in name or 'chernoff' in name or 'mejiro' in name:
            module_type = 'detection'
        else:
            return None  # Unknown module type
        
        # Check if function requires subhalo based on name
        requires_subhalo = any(keyword in name.lower() for keyword in [
            'baseline', 'subhalo', 'comparison', 'difference', 'residual'
        ])
        
        # Check if detection mode only
        detection_mode_only = 'detection' in name or 'chernoff' in name or 'mejiro' in name
        
        return PlotMetadata(
            name=name,
            function=func,
            module_type=module_type,
            requires_subhalo=requires_subhalo,
            detection_mode_only=detection_mode_only,
            description=f"Auto-discovered {module_type} plot"
        )
    
    def get_applicable_plots(self, context: Dict[str, Any]) -> List[PlotMetadata]:
        """Get list of plots applicable to the current context.
        
        Parameters
        ----------
        context : dict
            Context dictionary containing:
            - 'mode': 'standard' or 'detection'  
            - 'has_subhalo': bool
            - 'lensing_data': LensingData or None
            - 'psf_data': PSFData or None
            - 'obs_data': ObservationData or None
            - 'detection_data': DetectionData or None
            
        Returns
        -------
        applicable_plots : list of PlotMetadata
            List of plots that should be executed in this context
        """
        applicable = []
        mode = context.get('mode', 'standard')
        has_subhalo = context.get('has_subhalo', False)
        
        for plot_meta in self.plots.values():
            # Check mode requirements
            if plot_meta.detection_mode_only and mode != 'detection':
                continue
            if plot_meta.standard_mode_only and mode != 'standard':
                continue
                
            # Check subhalo requirements
            if plot_meta.requires_subhalo and not has_subhalo:
                continue
                
            # Check data availability
            if plot_meta.module_type == 'lensing' and not context.get('lensing_data'):
                continue
            if plot_meta.module_type == 'psf' and not context.get('psf_data'):
                continue
            if plot_meta.module_type == 'observation' and not context.get('obs_data'):
                continue
            if plot_meta.module_type == 'detection' and not context.get('detection_data'):
                continue
                
            applicable.append(plot_meta)
        
        return applicable
    
    def execute_plots(self, context: Dict[str, Any], plot_config: Dict[str, Any], verbose: bool = True):
        """Execute all applicable plots for the given context.
        
        Parameters
        ----------
        context : dict
            Context dictionary with available data
        plot_config : dict
            Plotting configuration
        verbose : bool, optional
            Whether to print execution information
        """
        applicable_plots = self.get_applicable_plots(context)
        
        if verbose:
            print(f"\nExecuting {len(applicable_plots)} applicable plots...")
        
        for plot_meta in applicable_plots:
            try:
                self._execute_single_plot(plot_meta, context, plot_config, verbose)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to execute {plot_meta.name}: {e}")
    
    def _execute_single_plot(self, plot_meta: PlotMetadata, context: Dict[str, Any], 
                           plot_config: Dict[str, Any], verbose: bool):
        """Execute a single plot function with appropriate arguments."""
        func = plot_meta.function
        sig = inspect.signature(func)
        
        # Build arguments based on function signature
        kwargs = {}
        
        # Standard arguments
        if 'plot_config' in sig.parameters:
            kwargs['plot_config'] = plot_config
            
        # Data arguments based on module type and signature
        if plot_meta.module_type == 'lensing':
            if 'lensing_data' in sig.parameters:
                kwargs['lensing_data'] = context['lensing_data']
                
        elif plot_meta.module_type == 'psf':
            if 'psf_data' in sig.parameters:
                kwargs['psf_data'] = context['psf_data']
                
        elif plot_meta.module_type == 'observation':
            if 'lensing_data' in sig.parameters:
                kwargs['lensing_data'] = context['lensing_data']
            if 'psf_data' in sig.parameters:
                kwargs['psf_data'] = context['psf_data'] 
            if 'obs_data' in sig.parameters:
                kwargs['obs_data'] = context['obs_data']
            # Handle special save_path parameter
            if 'save_path' in sig.parameters:
                run_name = context.get('run_name', 'default')
                kwargs['save_path'] = f"{plot_config['output_dir']}/{run_name}/observation/observation_comparison.png"
                
        elif plot_meta.module_type == 'detection':
            if 'detection_data' in sig.parameters:
                kwargs['detection_data'] = context['detection_data']
            if 'obs_baseline' in sig.parameters:
                kwargs['obs_baseline'] = context.get('obs_baseline')
            if 'obs_test' in sig.parameters:
                kwargs['obs_test'] = context.get('obs_test')
            if 'run_name' in sig.parameters:
                kwargs['run_name'] = context.get('run_name')
        
        if verbose:
            print(f"  → {plot_meta.name}")
            
        # Execute the function
        func(**kwargs)


# Global registry instance
_plot_registry = None

def get_plot_registry() -> PlotRegistry:
    """Get the global plot registry instance."""
    global _plot_registry
    if _plot_registry is None:
        _plot_registry = PlotRegistry()
    return _plot_registry


def generate_all_plots(context: Dict[str, Any], plot_config: Dict[str, Any], verbose: bool = True):
    """Generate all applicable plots for the given context.
    
    This is the main entry point for the pipeline to generate plots.
    
    Parameters
    ----------
    context : dict
        Context dictionary containing available data
    plot_config : dict
        Plotting configuration
    verbose : bool, optional
        Whether to print execution information
    """
    registry = get_plot_registry()
    registry.execute_plots(context, plot_config, verbose)