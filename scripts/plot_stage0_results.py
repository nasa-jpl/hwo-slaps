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


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return np.nan if value in ("", None) else float(value)


def _ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    fig.tight_layout()
    fig.savefig(output_dir / "stage0_mass_sweep_qf.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(masses, z_values, marker="o", color="#2d6f55", linewidth=2)
    ax.axhline(SCDD_Z_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.5, label="SCDD sqrt(10)")
    ax.set_xscale("log")
    ax.set_xlabel("Subhalo mass (M_sun)")
    ax.set_ylabel("Z_F")
    ax.set_title("Perfect-PSF Local Significance")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stage0_mass_sweep_zf.png", dpi=180)
    plt.close(fig)


def plot_hexike_sweep(rows: List[Dict[str, str]], output_dir: Path) -> None:
    sweep_rows = [row for row in rows if row["sweep"] == "segment_hexike_amplitude"]
    sweep_rows = sorted(sweep_rows, key=lambda row: _float(row, "psf_amplitude"))
    amp = np.asarray([_float(row, "psf_amplitude") for row in sweep_rows])
    q_values = np.asarray([_float(row, "q_f") for row in sweep_rows])
    z_values = np.asarray([_float(row, "z_f") for row in sweep_rows])
    degradation = np.asarray([_float(row, "local_degradation") for row in sweep_rows])
    strehl = np.asarray([_float(row, "psf_strehl") for row in sweep_rows])

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    axes[0].plot(amp, q_values, marker="o", color="#285c8f", linewidth=2, label="q_F")
    axes[0].axhline(SCDD_Q_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.5, label="SCDD q=10")
    axes[0].set_ylabel("q_F")
    axes[0].set_title("Segment Hexike Noll 2 Sweep at 1e7 M_sun")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(amp, degradation, marker="o", color="#6b5b95", linewidth=2, label="Profiled / raw information")
    axes[1].set_xlabel("Segment hexike amplitude (nm RMS)")
    axes[1].set_ylabel("Profiling degradation")
    axes[1].grid(alpha=0.25)

    ax2 = axes[1].twinx()
    ax2.plot(amp, strehl, marker="s", color="#b35c1e", linewidth=1.6, label="Strehl")
    ax2.set_ylabel("Strehl")
    fig.tight_layout()
    fig.savefig(output_dir / "stage0_hexike_degradation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(amp, z_values, marker="o", color="#2d6f55", linewidth=2)
    ax.axhline(SCDD_Z_THRESHOLD, color="#9d2f2f", linestyle="--", linewidth=1.5, label="SCDD sqrt(10)")
    ax.set_xlabel("Segment hexike amplitude (nm RMS)")
    ax.set_ylabel("Z_F")
    ax.set_title("Segment Hexike Significance at 1e7 M_sun")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stage0_hexike_zf.png", dpi=180)
    plt.close(fig)


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
    fig.tight_layout()
    fig.savefig(output_dir / "stage0_detectable_ring_fraction.png", dpi=180)
    plt.close(fig)


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
    plot_hexike_sweep(rows, output_dir)
    plot_ring_fraction(rows, output_dir)
    print(f"Wrote Stage 0 aggregate figures to {output_dir}")


if __name__ == "__main__":
    main()
