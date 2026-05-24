#!/usr/bin/env python
"""Diagnose Stage 0 nonlinear truth likelihoods without running a sampler."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/stage0_nonlinear_validation/truth_diagnostics.csv"
DEFAULT_RESULTS = REPO_ROOT / "outputs/stage0_internal_review/results.csv"
DEFAULT_CONFIG_DIR = REPO_ROOT / "outputs/stage0_internal_review/generated_configs"
DEFAULT_CASES = (
    "stage0_internal_review_mass_m1e7_perfect",
    "stage0_internal_review_mass_m10p7p25_perfect",
    "stage0_internal_review_mass_m10p7p75_perfect",
    "stage0_internal_review_hexike_s0_n2_a100p0nm_m1e7",
)


def _read_stage0_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["run_name"]: row for row in csv.DictReader(handle)}


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["plotting"]["enabled"] = False
    return config


def _fisher_mask_from_observation(observation: Any, config: Dict[str, Any]) -> np.ndarray:
    fisher_config = config["modeling"]["fisher"]
    mask_mode = str(fisher_config.get("mask_mode", "source_snr")).lower()
    if mask_mode == "all_pixels":
        return np.ones_like(observation.noiseless_source_eps, dtype=bool)
    if mask_mode != "source_snr":
        raise ValueError("Only source_snr and all_pixels Fisher masks are supported")

    source_adu = np.asarray(observation.source_electrons, dtype=float) / float(observation.gain)
    noise_adu = np.asarray(observation.noise_map.native, dtype=float)
    threshold = float(fisher_config["snr_threshold"])
    return source_adu / np.maximum(noise_adu, 1.0e-12) > threshold


def _exclude_psf_edge_pixels(use_mask: np.ndarray, psf_shape: tuple[int, int]) -> np.ndarray:
    use_mask = np.asarray(use_mask, dtype=bool).copy()
    y_half = int(psf_shape[0]) // 2
    x_half = int(psf_shape[1]) // 2
    if y_half > 0:
        use_mask[:y_half, :] = False
        use_mask[-y_half:, :] = False
    if x_half > 0:
        use_mask[:, :x_half] = False
        use_mask[:, -x_half:] = False
    return use_mask


def _chi2(data: np.ndarray, model: np.ndarray, noise: np.ndarray, mask: np.ndarray) -> float:
    residual = (np.asarray(data, dtype=float) - np.asarray(model, dtype=float))[mask]
    sigma = np.asarray(noise, dtype=float)[mask]
    return float(np.sum((residual / np.maximum(sigma, 1.0e-12)) ** 2))


def _scenario_row(
    *,
    base_row: Dict[str, Any],
    scenario: str,
    data: np.ndarray,
    smooth_model: np.ndarray,
    subhalo_model: np.ndarray,
    noise: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    chi2_smooth = _chi2(data, smooth_model, noise, mask)
    chi2_subhalo = _chi2(data, subhalo_model, noise, mask)
    signed_delta_log_l = -0.5 * (chi2_subhalo - chi2_smooth)
    q_truth = 2.0 * max(0.0, signed_delta_log_l)
    return {
        **base_row,
        "scenario": scenario,
        "chi2_smooth_truth": chi2_smooth,
        "chi2_subhalo_truth": chi2_subhalo,
        "signed_delta_log_l_truth": signed_delta_log_l,
        "q_truth": q_truth,
        "z_truth": q_truth**0.5,
        "q_truth_over_q_fisher": q_truth / float(base_row["fisher_q"]),
    }


def diagnose_case(run_name: str, stage0_row: Dict[str, str], config_dir: Path) -> list[Dict[str, Any]]:
    from hwoslaps.config.validation import validate_or_raise
    from hwoslaps.lensing import generate_lensing_system
    from hwoslaps.modeling.nonlinear.dataset_builder import imaging_from_observation
    from hwoslaps.observation import generate_observation
    from hwoslaps.psf import generate_psf_system
    from hwoslaps.psf.utils import pyauto_kernel_native

    config = _load_config(config_dir / f"{run_name}.yaml")
    validate_or_raise(config)

    psf_data = generate_psf_system(config["psf"], full_config=config)

    baseline_config = _load_config(config_dir / f"{run_name}.yaml")
    baseline_config["lensing"]["subhalo"]["enabled"] = False
    validate_or_raise(baseline_config)
    lensing_baseline = generate_lensing_system(
        baseline_config["lensing"],
        full_config=baseline_config,
    )
    obs_baseline = generate_observation(
        lensing_data=lensing_baseline,
        psf_data=psf_data,
        observation_config=baseline_config["observation"],
        full_config=baseline_config,
    )

    test_config = _load_config(config_dir / f"{run_name}.yaml")
    test_config["lensing"]["subhalo"]["enabled"] = True
    validate_or_raise(test_config)
    lensing_test = generate_lensing_system(test_config["lensing"], full_config=test_config)
    obs_test = generate_observation(
        lensing_data=lensing_test,
        psf_data=psf_data,
        observation_config=test_config["observation"],
        full_config=test_config,
    )

    psf_shape = tuple(pyauto_kernel_native(psf_data.kernel).shape)
    mask = _exclude_psf_edge_pixels(
        _fisher_mask_from_observation(obs_baseline, config),
        psf_shape=psf_shape,
    )
    exposure = float(obs_test.exposure_time)
    gain = float(obs_test.gain)

    base_row = {
        "run_name": run_name,
        "psf_case": stage0_row["psf_case"],
        "mass_msun": float(stage0_row["mass_msun"]),
        "fisher_q": float(stage0_row["q_f"]),
        "fisher_z": float(stage0_row["z_f"]),
        "n_unmasked_pixels": int(np.count_nonzero(mask)),
        "exposure_time": exposure,
        "gain": gain,
    }

    smooth_eps = np.asarray(obs_baseline.noiseless_source_eps, dtype=float)
    subhalo_eps = np.asarray(obs_test.noiseless_source_eps, dtype=float)
    noise_adu = np.asarray(obs_test.noise_map.native, dtype=float)

    rows = [
        _scenario_row(
            base_row=base_row,
            scenario="current_dataset_data_adu_model_eps",
            data=np.asarray(obs_test.source_electrons, dtype=float) / gain,
            smooth_model=smooth_eps,
            subhalo_model=subhalo_eps,
            noise=noise_adu,
            mask=mask,
        ),
        _scenario_row(
            base_row=base_row,
            scenario="consistent_adu_exposure_scaled_model",
            data=np.asarray(obs_test.source_electrons, dtype=float) / gain,
            smooth_model=smooth_eps * exposure / gain,
            subhalo_model=subhalo_eps * exposure / gain,
            noise=noise_adu,
            mask=mask,
        ),
        _scenario_row(
            base_row=base_row,
            scenario="consistent_eps_rate_units",
            data=subhalo_eps,
            smooth_model=smooth_eps,
            subhalo_model=subhalo_eps,
            noise=noise_adu * gain / exposure,
            mask=mask,
        ),
    ]
    dataset, _ = imaging_from_observation(
        obs_test,
        psf_for_fit=None,
        dataset_kind="asimov",
        background_treatment="subtract_known",
        mask_bool_use=mask,
        psf_truth_label=str(stage0_row["psf_case"]),
        psf_fit_label=str(stage0_row["psf_case"]),
    )
    from autolens.imaging.fit_imaging import FitImaging

    fit_smooth = FitImaging(dataset=dataset, tracer=lensing_baseline.tracer)
    fit_subhalo = FitImaging(dataset=dataset, tracer=lensing_test.tracer)
    signed_delta_log_l = float(
        fit_subhalo.figure_of_merit - fit_smooth.figure_of_merit
    )
    q_truth = 2.0 * max(0.0, signed_delta_log_l)
    rows.append(
        {
            **base_row,
            "scenario": "pyautolens_fitimaging_truth_tracers",
            "chi2_smooth_truth": float(fit_smooth.chi_squared),
            "chi2_subhalo_truth": float(fit_subhalo.chi_squared),
            "signed_delta_log_l_truth": signed_delta_log_l,
            "q_truth": q_truth,
            "z_truth": q_truth**0.5,
            "q_truth_over_q_fisher": q_truth / float(base_row["fisher_q"]),
        }
    )
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--cases", nargs="*", default=list(DEFAULT_CASES))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage0_rows = _read_stage0_rows(Path(args.stage0_results).resolve())
    all_rows = []
    for run_name in args.cases:
        all_rows.extend(
            diagnose_case(
                run_name=run_name,
                stage0_row=stage0_rows[run_name],
                config_dir=Path(args.config_dir).resolve(),
            )
        )
    _write_csv(Path(args.output).resolve(), all_rows)
    print(f"Wrote truth diagnostics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
