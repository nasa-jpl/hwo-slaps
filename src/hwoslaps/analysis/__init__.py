"""Study-analysis layer for HWO-SLAPS sweeps.

This package is the horizontal layer that drives the four physics modules
(``lensing``, ``psf``, ``observation``, ``modeling``) over many
configurations: manifest expansion, cross-run aggregation, and ensemble
summaries. The physics modules must never import from this package.
"""

from .aggregate import RESULTS_CSV_COLUMNS, config_hash, study_provenance
from .manifest import RunSpec, expand_manifest, load_manifest
from .summarize import completeness_summary, wilson_interval

__all__ = (
    "RESULTS_CSV_COLUMNS",
    "RunSpec",
    "completeness_summary",
    "config_hash",
    "expand_manifest",
    "load_manifest",
    "study_provenance",
    "wilson_interval",
)
