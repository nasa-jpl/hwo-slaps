#!/usr/bin/env python
"""Create aggregate Stage 0 study figures from results.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

SCDD_Q_THRESHOLD = 10.0
SCDD_Z_THRESHOLD = np.sqrt(SCDD_Q_THRESHOLD)
THRESHOLD_CAPTION = r"SCDD threshold: $q_F > 10$ ($Z_F > \sqrt{10}$)."


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return np.nan if value in ("", None) else float(value)


def _ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finish(fig, path: Path, caption: str) -> None:
    fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_mass_sweep(rows: List[Dict[str, str]], output_dir: Path) -> None:
    mass_rows = [row for row in rows if row["sweep"] == "perfect_mass"]
    mass_rows = sorted(mass_rows, key=lambda row: _float(row, "mass_msun"))
    masses = np.asarray([_float(row, "mass_msun") for row in mass_rows])
    q_values = np.asarray([_float(row, "q_f") for row in mass_rows])
    z_values = np.asarray([_float(row, "z_f") for row in mass_rows])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(masses, q_values, marker="o", color="#285c8f", linewidth=2)
    ax.axhline(SCDD_Q_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.5, label="SCDD q=10")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Subhalo mass (M_sun)")
    ax.set_ylabel("q_F")
    ax.set_title("Perfect-PSF Fisher Detectability")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    _finish(fig, output_dir / "stage0_mass_sweep_qf.png", THRESHOLD_CAPTION)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(masses, z_values, marker="o", color="#2d6f55", linewidth=2)
    ax.axhline(SCDD_Z_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.5, label="SCDD sqrt(10)")
    ax.set_xscale("log")
    ax.set_xlabel("Subhalo mass (M_sun)")
    ax.set_ylabel("Z_F")
    ax.set_title("Perfect-PSF Local Significance")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    _finish(fig, output_dir / "stage0_mass_sweep_zf.png", THRESHOLD_CAPTION)


def _plot_one_psf_sweep(
    rows: List[Dict[str, str]],
    output_dir: Path,
    *,
    sweep_name: str,
    title_name: str,
    file_prefix: str,
) -> None:
    sweep_rows = [row for row in rows if row["sweep"] == sweep_name]
    if not sweep_rows:
        return
    sweep_rows = sorted(sweep_rows, key=lambda row: _float(row, "psf_amplitude"))
    amp = np.asarray([_float(row, "psf_amplitude") for row in sweep_rows])
    q_values = np.asarray([_float(row, "q_f") for row in sweep_rows])
    z_values = np.asarray([_float(row, "z_f") for row in sweep_rows])
    degradation = np.asarray([_float(row, "local_degradation") for row in sweep_rows])
    strehl = np.asarray([_float(row, "psf_strehl") for row in sweep_rows])
    units = sweep_rows[0].get("psf_units") or "amplitude units"
    caption = f"{THRESHOLD_CAPTION} PSF amplitudes are in {units}."

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    axes[0].plot(amp, q_values, marker="o", color="#285c8f", linewidth=2, label="q_F")
    axes[0].axhline(SCDD_Q_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.5, label="SCDD q=10")
    axes[0].set_ylabel("q_F")
    axes[0].set_title(f"{title_name} Sweep at 1e7 M_sun")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(amp, degradation, marker="o", color="#6b5b95", linewidth=2, label="Profiled / raw information")
    axes[1].set_xlabel(f"{title_name} amplitude ({units})")
    axes[1].set_ylabel("Profiling degradation")
    axes[1].grid(alpha=0.25)

    ax2 = axes[1].twinx()
    ax2.plot(amp, strehl, marker="s", color="#b35c1e", linewidth=1.6, label="Strehl")
    ax2.set_ylabel("Strehl")
    _finish(fig, output_dir / f"{file_prefix}_degradation.png", caption)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(amp, z_values, marker="o", color="#2d6f55", linewidth=2)
    ax.axhline(SCDD_Z_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.5, label="SCDD sqrt(10)")
    ax.set_xlabel(f"{title_name} amplitude ({units})")
    ax.set_ylabel("Z_F")
    ax.set_title(f"{title_name} Significance at 1e7 M_sun")
    ax.grid(alpha=0.25)
    ax.legend()
    _finish(fig, output_dir / f"{file_prefix}_zf.png", caption)


def plot_psf_sweeps(rows: List[Dict[str, str]], output_dir: Path) -> None:
    _plot_one_psf_sweep(
        rows,
        output_dir,
        sweep_name="segment_hexike_amplitude",
        title_name="Segment Hexike Noll 2",
        file_prefix="stage0_hexike",
    )
    _plot_one_psf_sweep(
        rows,
        output_dir,
        sweep_name="global_zernike_amplitude",
        title_name="Global Zernike Noll 4",
        file_prefix="stage0_global_zernike",
    )


def plot_ring_fraction(rows: List[Dict[str, str]], output_dir: Path) -> None:
    map_rows = [
        row
        for row in rows
        if row.get("map_detectable_ring_fraction") not in ("", None)
    ]
    labels = []
    values = []
    for row in map_rows:
        if row["sweep"] == "perfect_mass":
            label = "perfect\n1e7"
        else:
            label = f"{row['psf_amplitude']} nm\n1e7"
        labels.append(label)
        values.append(_float(row, "map_detectable_ring_fraction"))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.bar(labels, values, color="#3f6c8c")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Detectable-ring fraction")
    ax.set_title("Ring-Map Demonstration")
    ax.grid(alpha=0.25, axis="y")
    _finish(
        fig,
        output_dir / "stage0_detectable_ring_fraction.png",
        "Detectable-ring fraction is the fraction of sampled ring positions with q_F > 10.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="outputs/stage0_internal_review/results.csv",
        help="Aggregate Stage 0 results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/stage0_internal_review/figures",
        help="Directory for aggregate figures.",
    )
    args = parser.parse_args()

    rows = _read_rows(Path(args.results))
    output_dir = _ensure_output(Path(args.output_dir))
    plot_mass_sweep(rows, output_dir)
    plot_psf_sweeps(rows, output_dir)
    plot_ring_fraction(rows, output_dir)
    print(f"Wrote Stage 0 aggregate figures to {output_dir}")


if __name__ == "__main__":
    main()
