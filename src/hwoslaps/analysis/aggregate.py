"""Cross-run aggregation schema and provenance capture.

This module fixes the aggregation contract for study results: the column
schema of the aggregate ``results.csv`` and the provenance record that
makes every row re-runnable. The run executor that populates the schema
is not yet promoted; see the package docstring.
"""

from __future__ import annotations

__all__ = ("RESULTS_CSV_COLUMNS", "config_hash", "study_provenance")

import hashlib
import importlib.metadata
import platform
import subprocess
from typing import Any, Dict, Optional, Sequence

import yaml

RESULTS_CSV_COLUMNS = (
    "study_name",
    "sweep",
    "run_name",
    "status",
    "error",
    "runtime_s",
    "run_dir",
    "config_path",
    "config_hash",
    "git_hash",
    "python",
    "mass_msun",
    "subhalo_model",
    "subhalo_position_y",
    "subhalo_position_x",
    "psf_case",
    "psf_family",
    "psf_mode",
    "psf_amplitude",
    "psf_units",
    "global_seed",
    "fisher_mode",
    "q_f",
    "z_f",
    "delta_log_l_f_equiv",
    "detected_scdd",
    "local_p_one_sided",
    "local_degradation",
    "local_absorbed_fraction",
    "sigma_amplitude_profiled",
    "pixels_unmasked",
    "n_nuisance",
    "gram_condition_number",
    "map_num_positions",
    "map_median_z_f",
    "map_max_z_f",
    "map_median_q_f",
    "map_max_q_f",
    "map_detectable_ring_fraction",
    "psf_strehl",
    "psf_raw_peak_ratio",
    "psf_total_rms_nm",
    "psf_segment_hexike_present",
    "psf_global_zernike_present",
    "psf_kernel_shape",
    "psf_kernel_sum",
    "psf_kernel_peak",
    "psf_kernel_diff_l2_norm",
    "psf_kernel_diff_l2_rel",
    "psf_fwhm_mas",
    "mode_scan_num_modes",
    "mode_scan_leading_mode",
    "mode_scan_leading_z_per_unit",
    "mode_scan_leading_one_sigma_z",
    "mode_scan_leading_tolerance",
    "mode_scan_top_modes",
)
"""Aggregate ``results.csv`` column schema (`tuple` of `str`).

One row per expanded run: identity and provenance fields, the local
Fisher/SCDD metrics, Einstein-ring map summaries, PSF diagnostics, and
the PSF mode-scan summary.
"""

_PROVENANCE_PACKAGES = (
    "numpy",
    "scipy",
    "matplotlib",
    "pyyaml",
    "autolens",
    "autofit",
    "hcipy",
    "hwoslaps",
)
"""Packages recorded by `study_provenance` (`tuple` of `str`)."""


def config_hash(config: Dict[str, Any]) -> str:
    """Return a stable short hash of a configuration dictionary.

    Parameters
    ----------
    config : `dict`
        Configuration to hash.

    Returns
    -------
    digest : `str`
        First 16 hex characters of the SHA-256 of the key-sorted YAML
        rendering. The rendering convention is part of the provenance
        contract; changing it invalidates recorded hashes.
    """
    rendered = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()[:16]


def _git_hash(cwd: Optional[str] = None) -> Optional[str]:
    """Return the short git hash of ``cwd``, or `None` outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def study_provenance(
    command: Sequence[str] | None = None, *, repo_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Build a provenance record for one study invocation.

    Parameters
    ----------
    command : `list` of `str`, optional
        Command-line argument vector to record, e.g. ``sys.argv``.
    repo_dir : `str`, optional
        Repository directory whose git hash is recorded. Defaults to the
        current working directory.

    Returns
    -------
    provenance : `dict`
        Record with the command line, git hash, python version, and the
        versions of the packages in ``_PROVENANCE_PACKAGES``.
    """
    versions: Dict[str, Optional[str]] = {}
    for name in _PROVENANCE_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "command": None if command is None else list(command),
        "git_hash": _git_hash(repo_dir),
        "python": platform.python_version(),
        "package_versions": versions,
    }
