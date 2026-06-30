#!/usr/bin/env python
"""Package PSF-marginalized mass-completeness validation results."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path("outputs/stage0_psf_marginalized_mass_completeness_20260601")
SPIE_DIR = Path("outputs/spie_draft_results")
ONE_E7_CASES = SPIE_DIR / "csv" / "psf_marginalized_1e7_weekend_all_cases.csv"
ONE_E7_MASS = 1.0e7


def first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def parse_amplitude(run_name: object) -> float:
    text = str(run_name)
    if "perfect" in text:
        return 0.0
    match = re.search(r"_a(\d+)p(\d+)nm", text)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    match = re.search(r"_a(\d+)nm", text)
    if match:
        return float(match.group(1))
    return np.nan


def normalize_family(df: pd.DataFrame) -> pd.Series:
    if "psf_family" in df.columns:
        series = df["psf_family"].copy()
    elif "truth_psf_case" in df.columns:
        series = df["truth_psf_case"].copy()
    elif "family" in df.columns:
        series = df["family"].copy()
    else:
        series = pd.Series("unknown", index=df.index)

    series = series.replace({"perfect_reference": "perfect"}).fillna("unknown")
    run_col = "stage0_run_name" if "stage0_run_name" in df.columns else "run_name"
    if run_col in df.columns:
        series.loc[df[run_col].astype(str).str.contains("perfect", na=False)] = "perfect"
    return series


def normalize_detection_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q_col = first_existing(out, ["q_fit_psf_profile", "q_fit", "q_fit_profile"])
    dz_col = first_existing(
        out,
        [
            "delta_log_evidence_psf_marg",
            "delta_log_evidence",
            "delta_logz",
            "dlogz",
        ],
    )
    if q_col is None:
        raise ValueError("Could not find q-fit column in input table")
    if dz_col is None:
        raise ValueError("Could not find evidence column in input table")
    if q_col != "q_fit_psf_profile":
        out["q_fit_psf_profile"] = out[q_col]
    if dz_col != "delta_log_evidence_psf_marg":
        out["delta_log_evidence_psf_marg"] = out[dz_col]
    return out


def load_new_mass_sweep() -> pd.DataFrame:
    results = normalize_detection_columns(pd.read_csv(RUN_DIR / "marginalized_results.csv"))
    selected = pd.read_csv(RUN_DIR / "selected_mass_completeness_cases.csv")
    merged = results.merge(
        selected[
            [
                "run_name",
                "mass_msun",
                "psf_family",
                "psf_amplitude",
                "psf_total_rms_nm",
            ]
        ],
        left_on="stage0_run_name",
        right_on="run_name",
        how="left",
    )
    merged["source"] = "mass_completeness_20260601"
    merged["psf_amplitude_nm"] = merged["psf_amplitude"]
    return merged


def load_one_e7_sweep() -> pd.DataFrame:
    one = normalize_detection_columns(pd.read_csv(ONE_E7_CASES))
    one["source"] = "psf_marginalized_1e7_weekend_20260530"
    one["mass_msun"] = ONE_E7_MASS

    if "stage0_run_name" not in one.columns:
        if "run_name" in one.columns:
            one["stage0_run_name"] = one["run_name"]
        elif "case" in one.columns:
            one["stage0_run_name"] = one["case"]
        else:
            one["stage0_run_name"] = one.index.astype(str)

    one["psf_family"] = normalize_family(one)

    amp_col = first_existing(
        one, ["psf_amplitude_nm", "psf_amplitude", "truth_psf_amp_nm", "amplitude_nm"]
    )
    if amp_col is not None:
        one["psf_amplitude_nm"] = one[amp_col]
    else:
        one["psf_amplitude_nm"] = one["stage0_run_name"].map(parse_amplitude)

    rms_col = first_existing(one, ["psf_total_rms_nm", "truth_psf_total_rms_nm", "total_rms_nm"])
    if rms_col is None:
        one["psf_total_rms_nm"] = np.nan
    elif rms_col != "psf_total_rms_nm":
        one["psf_total_rms_nm"] = one[rms_col]

    return one


def wilson_interval(k: int, n: int, z: float = 1.0) -> tuple[float, float, float]:
    if n == 0:
        return np.nan, np.nan, np.nan
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half_width = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return phat, max(0.0, center - half_width), min(1.0, center + half_width)


def summarize(cases: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in cases.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(group)
        q_count = int(group["q10_detected"].sum())
        evidence_count = int(group["evidence_detected"].sum())
        q_frac, q_lo, q_hi = wilson_interval(q_count, n)
        evidence_frac, evidence_lo, evidence_hi = wilson_interval(evidence_count, n)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n": n,
                "q10_detected": q_count,
                "q10_fraction": q_frac,
                "q10_wilson_lo_1sigma": q_lo,
                "q10_wilson_hi_1sigma": q_hi,
                "evidence_detected": evidence_count,
                "evidence_fraction": evidence_frac,
                "evidence_wilson_lo_1sigma": evidence_lo,
                "evidence_wilson_hi_1sigma": evidence_hi,
                "median_q_fit": float(group["q_fit_psf_profile"].median()),
                "p16_q_fit": float(group["q_fit_psf_profile"].quantile(0.16)),
                "p84_q_fit": float(group["q_fit_psf_profile"].quantile(0.84)),
                "median_delta_log_evidence": float(
                    group["delta_log_evidence_psf_marg"].median()
                ),
                "p16_delta_log_evidence": float(
                    group["delta_log_evidence_psf_marg"].quantile(0.16)
                ),
                "p84_delta_log_evidence": float(
                    group["delta_log_evidence_psf_marg"].quantile(0.84)
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def clipped_yerr(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[np.ndarray]:
    return [np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]


def write_plots(by_mass: pd.DataFrame, by_family: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plot_dir = SPIE_DIR / "plots"
    x = by_mass["mass_msun"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for metric, color, label in [
        ("q10", "#2a6fbb", "q_fit >= 10"),
        ("evidence", "#b1463c", "Delta log Z > 5"),
    ]:
        y = by_mass[f"{metric}_fraction"].to_numpy()
        lo = by_mass[f"{metric}_wilson_lo_1sigma"].to_numpy()
        hi = by_mass[f"{metric}_wilson_hi_1sigma"].to_numpy()
        ax.errorbar(
            x,
            y,
            yerr=clipped_yerr(y, lo, hi),
            marker="o",
            lw=2.2,
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("Injected subhalo mass [Msun]")
    ax.set_ylabel("Detection fraction")
    ax.set_title("PSF-bank-marginalized PyAutoLens mass completeness")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(plot_dir / "psf_marginalized_mass_completeness_detection_curve.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharex=True)
    for ax, med, p16, p84, ylabel, thresh, color in [
        (axes[0], "median_q_fit", "p16_q_fit", "p84_q_fit", "q_fit", 10.0, "#2a6fbb"),
        (
            axes[1],
            "median_delta_log_evidence",
            "p16_delta_log_evidence",
            "p84_delta_log_evidence",
            "Delta log Z",
            5.0,
            "#b1463c",
        ),
    ]:
        y = by_mass[med].to_numpy()
        lo = by_mass[p16].to_numpy()
        hi = by_mass[p84].to_numpy()
        ax.errorbar(x, y, yerr=clipped_yerr(y, lo, hi), marker="o", lw=2, capsize=3, color=color)
        ax.axhline(thresh, ls="--", lw=1.2, color="0.25")
        ax.set_xscale("log")
        ax.set_xlabel("Injected subhalo mass [Msun]")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Fit detection statistic")
    axes[1].set_title("Bayesian evidence statistic")
    fig.tight_layout()
    fig.savefig(plot_dir / "psf_marginalized_mass_completeness_statistics.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = {
        "perfect": "#333333",
        "segment_only": "#2a6fbb",
        "combined": "#6a994e",
        "global_only": "#b1463c",
        "unknown": "#777777",
    }
    families = [
        family
        for family in ["perfect", "segment_only", "combined", "global_only", "unknown"]
        if family in set(by_family["psf_family"])
    ]
    for family in families:
        subset = by_family[by_family["psf_family"] == family].sort_values("mass_msun")
        ax.plot(
            subset["mass_msun"],
            subset["evidence_fraction"],
            marker="o",
            lw=2,
            color=colors.get(family, "#777777"),
            label=family,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.04)
    ax.set_xlabel("Injected subhalo mass [Msun]")
    ax.set_ylabel("Evidence detection fraction (Delta log Z > 5)")
    ax.set_title("Mass completeness by PSF-error family")
    ax.legend(frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "psf_marginalized_mass_completeness_by_family.png", dpi=220)
    plt.close(fig)


def main() -> None:
    for subdir in ["csv", "plots", "metadata"]:
        (SPIE_DIR / subdir).mkdir(parents=True, exist_ok=True)

    cases = pd.concat([load_new_mass_sweep(), load_one_e7_sweep()], ignore_index=True)
    keep = [
        "source",
        "stage0_run_name",
        "mass_msun",
        "psf_family",
        "psf_amplitude_nm",
        "psf_total_rms_nm",
        "status",
        "q_fit_psf_profile",
        "delta_log_evidence_psf_marg",
    ]
    for column in keep:
        if column not in cases.columns:
            cases[column] = np.nan
    cases = cases[keep].copy()
    cases["q10_detected"] = cases["q_fit_psf_profile"] >= 10.0
    cases["evidence_detected"] = cases["delta_log_evidence_psf_marg"] > 5.0
    cases["log10_mass"] = np.log10(cases["mass_msun"])
    cases["psf_family"] = normalize_family(cases)

    by_mass = summarize(cases, ["mass_msun"])
    by_family = summarize(cases, ["mass_msun", "psf_family"])

    cases.to_csv(SPIE_DIR / "csv" / "psf_marginalized_mass_completeness_all_cases.csv", index=False)
    by_mass.to_csv(SPIE_DIR / "csv" / "psf_marginalized_mass_completeness_by_mass.csv", index=False)
    by_family.to_csv(
        SPIE_DIR / "csv" / "psf_marginalized_mass_completeness_by_mass_family.csv",
        index=False,
    )
    write_plots(by_mass, by_family)

    summary = {
        "source_dirs": [
            str(RUN_DIR),
            "outputs/stage0_psf_marginalized_1e7_weekend_20260530",
        ],
        "total_cases": int(len(cases)),
        "success_cases": int((cases["status"] == "success").sum()),
        "masses_msun": [float(value) for value in sorted(cases["mass_msun"].dropna().unique())],
        "by_mass": by_mass.to_dict(orient="records"),
    }
    (SPIE_DIR / "metadata" / "psf_marginalized_mass_completeness_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(by_mass.to_string(index=False))
    print(f"Wrote mass-completeness results to {SPIE_DIR}")


if __name__ == "__main__":
    main()
