"""Tests for Fisher detection plotting outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hwoslaps.modeling.utils_fisher import (
    FisherDetectionData,
    FisherLocalData,
    FisherMapData,
    FisherModeCouplingData,
    FisherModeScanData,
)
from hwoslaps.plotting.detection_plots import (
    plot_fisher_detection_map_summary,
    plot_fisher_local_summary,
    plot_fisher_map_degradation,
    plot_fisher_psf_mode_scan,
)


def _base_plot_config(tmp_path: Path) -> dict:
    return {"output_dir": str(tmp_path), "run_name": "plot-test"}


def _make_local_detection_data(with_psf_scan: bool = False) -> FisherDetectionData:
    local = FisherLocalData(
        snr_asimov=3.8,
        delta_chi2_raw=12.0,
        delta_chi2_profiled=8.0,
        degradation=0.67,
        pixels_unmasked=123,
        n_nuisance=4,
        gram_condition_number=15.0,
        true_subhalo_position=(0.1, -0.2),
        true_subhalo_mass=1.0e7,
        fisher_raw=12.0,
        fisher_profiled=8.0,
        sigma_amplitude_profiled=0.35,
        local_p_one_sided=1.2e-4,
        absorbed_fraction=0.33,
        residual_norm_whitened=2.5,
    )
    if with_psf_scan:
        local.psf_mode_scan = FisherModeScanData(
            couplings=[
                FisherModeCouplingData(
                    mode_name="psf.global_zernikes[4]",
                    amplitude_per_unit=0.2,
                    z_per_unit=0.08,
                    one_sigma_z=0.4,
                    tolerance_for_zmax=12.5,
                ),
                FisherModeCouplingData(
                    mode_name="psf.global_zernikes[5]",
                    amplitude_per_unit=-0.1,
                    z_per_unit=-0.03,
                    one_sigma_z=-0.15,
                    tolerance_for_zmax=33.0,
                ),
            ],
            sigma_amplitude_profiled=0.35,
            fisher_profiled=8.0,
            rms_spurious_z=0.43,
            z_tolerance=1.0,
        )

    return FisherDetectionData(
        mode="local",
        local=local,
        map=None,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=123,
        n_nuisance=4,
        gram_condition_number=15.0,
        pixel_scale=0.1,
    )


def _make_map_detection_data() -> FisherDetectionData:
    positions = np.asarray([[0.2, 0.0], [0.0, 0.2], [-0.2, 0.0], [0.0, -0.2]], dtype=float)
    snr = np.asarray([3.0, 4.0, 5.0, 4.5], dtype=float)
    degradation = np.asarray([0.4, 0.6, 0.8, 0.5], dtype=float)
    fmap = FisherMapData(
        positions_yx=positions,
        snr_asimov_by_position=snr,
        delta_chi2_profiled_by_position=snr**2,
        delta_chi2_raw_by_position=(snr + 1.0) ** 2,
        num_positions=4,
        median_snr_asimov=float(np.median(snr)),
        p25_snr_asimov=float(np.percentile(snr, 25)),
        p75_snr_asimov=float(np.percentile(snr, 75)),
        min_snr_asimov=float(np.min(snr)),
        max_snr_asimov=float(np.max(snr)),
        degradation_by_position=degradation,
        absorbed_fraction_by_position=1.0 - degradation,
    )
    return FisherDetectionData(
        mode="map",
        local=None,
        map=fmap,
        snr_threshold=3.0,
        include_background_offset=True,
        finite_diff={},
        map_config={},
        pixels_unmasked=321,
        n_nuisance=5,
        gram_condition_number=22.0,
        pixel_scale=0.1,
    )


def test_fisher_local_summary_plot_written(tmp_path):
    """Write the Fisher local summary figure to the modeling folder."""
    detection_data = _make_local_detection_data(with_psf_scan=False)

    plot_fisher_local_summary(detection_data=detection_data, plot_config=_base_plot_config(tmp_path))

    assert (tmp_path / "plot-test" / "modeling" / "fisher_local_summary.png").exists()


def test_fisher_psf_mode_scan_plot_written(tmp_path):
    """Write the PSF mode scan figure to the modeling folder."""
    detection_data = _make_local_detection_data(with_psf_scan=True)

    plot_fisher_psf_mode_scan(detection_data=detection_data, plot_config=_base_plot_config(tmp_path))

    assert (tmp_path / "plot-test" / "modeling" / "fisher_psf_mode_scan.png").exists()


def test_fisher_map_plots_written(tmp_path):
    """Write both Fisher map figures to the modeling folder."""
    detection_data = _make_map_detection_data()

    plot_fisher_detection_map_summary(detection_data=detection_data, plot_config=_base_plot_config(tmp_path))
    plot_fisher_map_degradation(detection_data=detection_data, plot_config=_base_plot_config(tmp_path))

    modeling_dir = tmp_path / "plot-test" / "modeling"
    assert (modeling_dir / "fisher_map_summary.png").exists()
    assert (modeling_dir / "fisher_map_degradation.png").exists()
