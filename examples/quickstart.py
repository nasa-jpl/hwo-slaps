"""Run a small, synthetic subhalo forecast in the supported science environment."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import erf
import yaml

from hwoslaps import forecast, mass_reach, prepare_forecast, summarize_forecast
from hwoslaps.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", choices=("reference", "jax"), default="reference")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/master_config.yaml", validate=False)
    config.pop("plotting", None)
    config["run_name"] = "synthetic_example"
    config["lensing"]["grid"].update(shape=[41, 41], pixel_scale=0.1)
    config["lensing"]["source_galaxy"]["light"]["intensity"] = 500.0
    config["lensing"]["subhalo"]["enabled"] = False
    config["observation"]["exposure_time"] = 2000.0
    # Integrate a Gaussian over each detector pixel. This is an illustrative
    # response, not a telescope performance or source-population prediction.
    edges = np.arange(-3.5, 4.0)
    bins = np.diff(0.5 * erf(edges / (np.sqrt(2) * 0.75)))
    kernel = np.outer(bins, bins)
    np.save(output / "kernel.npy", kernel)
    config["psf"] = {"provider": "kernel", "kernel": {
        "path": str(output / "kernel.npy"), "pixel_scale_arcsec": 0.1,
    }}
    fisher = config["modeling"]["fisher"]
    fisher.update(compute_psf_mode_scan=False, include_psf_nuisance=False)
    fisher["map"] = {
        "type": "grid", "engine": args.backend, "num_workers": 1, "batch_size": 32,
        "grid": {"spacing_arcsec": 0.2, "half_width_arcsec": 1.2, "annulus": None},
        "detection_q_threshold": 10.0,
    }
    prepared = prepare_forecast(config)
    result = forecast(prepared, masses=np.logspace(7, 9, 7))
    selection = np.linalg.norm(result.positions_yx, axis=1) <= 1.2
    summary = summarize_forecast(
        result, q_threshold=10, selection=selection, cell_areas_arcsec2=0.2**2,
    )
    reach = mass_reach(result.masses_msun, summary.detectable_fraction, target=0.1)
    result.save_npz(output / "forecast.npz")
    with (output / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(prepared.config, stream, sort_keys=False)
    print("mass_msun  max_q  sensitive_fraction")
    for mass, peak, fraction in zip(result.masses_msun, summary.q_max, summary.detectable_fraction):
        print(f"{mass:.6g}  {peak:.6g}  {fraction:.6g}")
    print(f"10% area reach: {reach.status}; mass={reach.mass_msun}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
