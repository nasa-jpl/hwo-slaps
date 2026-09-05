"""Publication figures for the PSF knowledge-error block of the RASTI manuscript.

Reads the harvest review files of the psf_knowledge_fisher_v1 and
psf_knowledge_nonlinear_v1 campaigns and renders two figures as vector PDF plus
a 300-dpi PNG. The Fisher retention and spurious ratios are re-derived from the
per-direction cell counts and compared with the harvest quantiles; the
nonlinear counts and their exact Clopper-Pearson intervals are copied from the
harvest summaries. Every plotted headline number is checked against the frozen
reference values below and the checks are recorded in the stats file.
The numbers behind both figures are written to figures_stats_psf_knowledge.json
next to the figures, on the production figures_stats.json pattern.

Usage:
    python scripts/make_psf_knowledge_figures.py \
        --fisher-review <campaigns>/psf_knowledge_fisher_v1/harvest/review.json \
        --nonlinear-review <campaigns>/psf_knowledge_nonlinear_v1/harvest/review.json \
        --out <directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COL = 3.35  # single column, inches

PAPER = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#dcdbd4"
AXIS = "#b8b7ae"
SEL = "#2a78d6"
PAR = "#eb6834"
ORD = ("#5598e7", "#256abf", "#0d366b")

plt.rcParams.update(
    {
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.top": True,
        "ytick.right": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.borderpad": 0.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 6,
        "savefig.bbox": None,
    }
)

RUNGS = (1.0, 2.0, 5.0, 10.0, 20.0, 35.0)
ENDPOINT_NM = 35.0
N_SYSTEMS = 12
N_DIRECTIONS = 8
NONLINEAR_DELTAS = (2.0, 5.0, 10.0, 20.0)
N_DRAWS = 36
CLASSES = (
    ("m10", r"$M_{10}$", ORD[1], "s"),
    ("m50", r"$M_{50}$", ORD[2], "^"),
)
DELTA_LABEL = r"PSF knowledge error  $\delta$  (nm RMS)"

# Frozen reference values from the 2026-09-05 harvests (review.json, both CLEAN).
REFERENCE = {
    "fisher": {
        "verdict": "CLEAN",
        "expected_maps": 1764,
        "total_map_wall_hours": 15.633,
        "tier_delta_star": {"m10": (10.0, (5.0, 20.0)), "m50": (5.0, (2.0, 10.0))},
        "tier_delta_star_sensitivity": {"m10": 20.0, "m50": 10.0},
        "q10_R_median": {("m10", 10.0): 0.917, ("m50", 5.0): 0.935, ("m50", 10.0): 0.860},
        "directions_with_spurious_at_20": {"m10": 5, "m50": 1},
        "m_best_below_floor": 12,
    },
    "nonlinear": {
        "integrity": "CLEAN",
        "rows": 288,
        "total_fit_wall_hours": 18.265,
        "control_q_ge_10_counts": {2.0: 0, 5.0: 0, 10.0: 0, 20.0: 1},
        "control_dlogz_counts": {2.0: 0, 5.0: 0, 10.0: 0, 20.0: 0},
        "injected_q_ge_10_counts": {2.0: 27, 5.0: 27, 10.0: 28, 20.0: 26},
        "injected_dlogz_counts": {2.0: 8, 5.0: 9, 10.0: 12, 20.0: 12},
        "reference_q_ge_10": (10, 12),
        "reference_dlogz": (2, 12),
        "pooled_null_q_ge_10": (3, 590),
        "pooled_null_dlogz": (0, 590),
        "pooled_null_composition": (531, 59),
        "selected_null_q_ge_10": (1, 120),
        "selected_null_dlogz": (0, 120),
    },
}

checks: list[dict] = []


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check(label, got, want, tol=5e-4):
    if isinstance(want, str):
        ok = got == want
    elif want is None:
        ok = got is None
    elif got is None:
        ok = False
    else:
        ok = abs(got - want) <= tol
    checks.append({"label": label, "computed": got, "reference": want, "pass": ok})
    if not ok:
        raise AssertionError(f"{label}: computed {got!r} != reference {want!r}")


def quantile(values, p):
    """Linear-interpolation quantile, the numpy default used by the harvest."""
    ordered = sorted(values)
    n = len(ordered)
    position = p * (n - 1)
    low = int(position)
    high = min(low + 1, n - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def save(fig, out, name):
    pdf = os.path.join(out, f"{name}.pdf")
    png = os.path.join(out, f"{name}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(
        f"  wrote {os.path.basename(pdf)}  {os.path.getsize(pdf) / 1024:.0f} kB"
        f"   {os.path.basename(png)}  {os.path.getsize(png) / 1024:.0f} kB"
    )


# ------------------------------------------------------------------ fisher block
def fisher_statistics(review):
    ref = REFERENCE["fisher"]
    check("fisher verdict", review["verdict"], ref["verdict"])
    check("fisher expected maps", review["expected_maps"], ref["expected_maps"], 0)
    check("fisher map wall hours", review["total_map_wall_hours"], ref["total_map_wall_hours"], 1e-2)
    require(review["integrity_findings"] == [] and review["missing"] == [], 'review["integrity_findings"] == [] and review["missing"] == []')
    for receipt in review["science"]["receipt"]:
        require(receipt["findings"] == [], receipt)
        require(receipt["matched_cells"] == receipt["production_cells"], receipt)
    science = review["science"]
    per_class = science["per_system_rung_class"]
    gates = science["gates"]
    require(gates["default"] == [0.9, 0.1] and gates["sensitivity"] == [[0.8, 0.2]], 'gates["default"] == [0.9, 0.1] and gates["sensitivity"] == [[0.8, 0.2]]')

    m_best = per_class["m_best"]
    n_floor = sum(1 for entry in m_best if entry["below_ratio_floor"])
    check("m_best rungs below ratio floor", n_floor, ref["m_best_below_floor"], 0)

    out = {
        "gates": gates,
        "ratio_floor": science["ratio_floor"],
        "endpoint_anchor_nm": science["endpoint_anchor_nm"],
        "m_best": {
            "n_below_ratio_floor": n_floor,
            "matched_cells": sorted(entry["matched_cells"] for entry in m_best),
            "logm": sorted(entry["logm"] for entry in m_best),
        },
        "classes": {},
    }
    for cls, _, _, _ in CLASSES:
        entries = per_class[cls]
        require(len(entries) == N_SYSTEMS, 'len(entries) == N_SYSTEMS')
        require(not any(entry["below_ratio_floor"] for entry in entries), 'not any(entry["below_ratio_floor"] for entry in entries)')
        rows = {}
        for delta in RUNGS:
            key = f"{delta:g}"
            q10_r, q50_r, q90_f, max_f = [], [], [], []
            n_spurious_dirs = 0
            for entry in entries:
                per_delta = entry["per_delta"][key]
                dirs = per_delta["directions"]
                require(len(dirs) == N_DIRECTIONS and per_delta["zero_area_exclusion_count"] == 0, 'len(dirs) == N_DIRECTIONS and per_delta["zero_area_exclusion_count"] == 0')
                matched = entry["matched_cells"]
                r_vals = [d["mismatch_cells"] / matched for d in dirs]
                f_vals = [d["spurious_cells"] / matched for d in dirs]
                if delta == ENDPOINT_NM:
                    require(per_delta["endpoint_anchor"], 'per_delta["endpoint_anchor"]')
                    require(all(d["R"] is None and d["F"] is None for d in dirs), 'all(d["R"] is None and d["F"] is None for d in dirs)')
                else:
                    require(not per_delta["endpoint_anchor"], 'not per_delta["endpoint_anchor"]')
                    for d, r, f in zip(dirs, r_vals, f_vals):
                        require(abs(d["R"] - r) < 1e-12 and abs(d["F"] - f) < 1e-12, 'abs(d["R"] - r) < 1e-12 and abs(d["F"] - f) < 1e-12')
                    require(abs(quantile(r_vals, 0.1) - per_delta["quantiles"]["R"]["q10"]) < 1e-12, 'abs(quantile(r_vals, 0.1) - per_delta["quantiles"]["R"]["q10"]) < 1e-12')
                    require(abs(quantile(f_vals, 0.9) - per_delta["quantiles"]["F"]["q90"]) < 1e-12, 'abs(quantile(f_vals, 0.9) - per_delta["quantiles"]["F"]["q90"]) < 1e-12')
                q10_r.append(quantile(r_vals, 0.1))
                q50_r.append(quantile(r_vals, 0.5))
                q90_f.append(quantile(f_vals, 0.9))
                max_f.append(max(f_vals))
                n_spurious_dirs += sum(1 for d in dirs if d["spurious_cells"] > 0)
            rows[key] = {
                "delta_nm": delta,
                "endpoint_anchor": delta == ENDPOINT_NM,
                "n_systems": len(entries),
                "n_directions_total": len(entries) * N_DIRECTIONS,
                "n_directions_with_spurious_cells": n_spurious_dirs,
                "q10_R_median": statistics.median(q10_r),
                "q10_R_min": min(q10_r),
                "q10_R_max": max(q10_r),
                "q50_R_median": statistics.median(q50_r),
                "q90_F_median": statistics.median(q90_f),
                "q90_F_max": max(q90_f),
                "F_max": max(max_f),
            }
        for (ref_cls, ref_delta), want in ref["q10_R_median"].items():
            if ref_cls == cls:
                check(f"{cls} Q10[R] median at {ref_delta:g} nm", rows[f"{ref_delta:g}"]["q10_R_median"], want, 1e-3)
        check(f"{cls} directions with spurious cells at 20 nm", rows["20"]["n_directions_with_spurious_cells"], ref["directions_with_spurious_at_20"][cls], 0)
        for delta in (1.0, 2.0, 5.0, 10.0):
            check(f"{cls} directions with spurious cells at {delta:g} nm", rows[f"{delta:g}"]["n_directions_with_spurious_cells"], 0, 0)

        default = [entry["delta_star"]["default"]["delta_star"] for entry in entries]
        sensitivity = [entry["delta_star"]["sensitivity_0.8_0.2"]["delta_star"] for entry in entries]
        require(all(value is not None for value in default + sensitivity), 'all(value is not None for value in default + sensitivity)')
        tier = science["tier_summary"][cls]
        want_median, want_range = ref["tier_delta_star"][cls]
        check(f"{cls} tier median delta*", statistics.median(default), want_median, 0)
        check(f"{cls} tier median delta* (harvest)", tier["default"]["median_delta_star"], want_median, 0)
        check(f"{cls} tier delta* min", min(default), want_range[0], 0)
        check(f"{cls} tier delta* max", max(default), want_range[1], 0)
        check(f"{cls} tier median delta* at gates 0.8/0.2", statistics.median(sensitivity), ref["tier_delta_star_sensitivity"][cls], 0)
        counts = {f"{delta:g}": default.count(delta) for delta in RUNGS}
        out["classes"][cls] = {
            "per_delta": rows,
            "delta_star_default_per_system": {entry["system_id"]: entry["delta_star"]["default"]["delta_star"] for entry in entries},
            "delta_star_sensitivity_per_system": {entry["system_id"]: entry["delta_star"]["sensitivity_0.8_0.2"]["delta_star"] for entry in entries},
            "delta_star_default_counts": counts,
            "tier_median_delta_star": statistics.median(default),
            "tier_range_delta_star": [min(default), max(default)],
            "tier_median_delta_star_sensitivity": statistics.median(sensitivity),
            "tier_range_delta_star_sensitivity": [min(sensitivity), max(sensitivity)],
            "logm": {entry["system_id"]: entry["logm"] for entry in entries},
            "matched_cells": {entry["system_id"]: entry["matched_cells"] for entry in entries},
        }
    return out


def fig_psf_knowledge_requirement(stats, out):
    fig = plt.figure(figsize=(COL, 4.75))
    ax_strip = fig.add_axes((0.175, 0.905, 0.79, 0.075))
    ax_r = fig.add_axes((0.175, 0.475, 0.79, 0.415))
    ax_f = fig.add_axes((0.175, 0.105, 0.79, 0.325))
    xlim = (0.78, 47.0)
    gate_r, gate_f = stats["gates"]["default"]

    for ax in (ax_strip, ax_r, ax_f):
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(RUNGS)
        ax.set_axisbelow(True)
    ax_strip.set_xticklabels([])
    ax_r.set_xticklabels([])
    ax_f.set_xticklabels([f"{r:g}" for r in RUNGS])
    ax_strip.tick_params(axis="x", top=False, bottom=False)
    ax_r.tick_params(axis="x", top=False)
    ax_f.tick_params(axis="x", top=False)

    # per-system delta* strip
    ax_strip.set_ylim(0, 9.2)
    ax_strip.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_strip.spines[spine].set_visible(False)
    ax_strip.spines["bottom"].set_color(AXIS)
    offsets = {"m10": 0.94, "m50": 1.065}
    for cls, _, color, marker in CLASSES:
        counts = stats["classes"][cls]["delta_star_default_counts"]
        for delta in RUNGS:
            n = counts[f"{delta:g}"]
            for level in range(n):
                ax_strip.plot([delta * offsets[cls]], [0.9 + level * 1.02], linestyle="none",
                              marker=marker, markersize=3.1, markerfacecolor=color,
                              markeredgecolor=PAPER, markeredgewidth=0.45, zorder=4)
    ax_strip.text(-0.005, 0.5, r"$\delta^{*}$" + "\nper system", transform=ax_strip.transAxes,
                  color=INK2, fontsize=6.4, ha="right", va="center", linespacing=1.05)

    # retention panel
    ax_r.set_ylim(0.25, 1.09)
    ax_r.grid(axis="y", zorder=0.5)
    ax_r.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax_r.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax_r.axhline(gate_r, color=INK2, linewidth=0.75, linestyle=(0, (2.6, 1.8)), zorder=2)
    ax_r.text(0.83, gate_r + 0.018, f"retention gate {gate_r:g}", color=INK2, fontsize=6.6, ha="left", va="bottom")
    ax_r.axvline(ENDPOINT_NM, color=MUTED, linewidth=0.7, linestyle=(0, (1.2, 1.6)), zorder=1.5)
    ax_f.axvline(ENDPOINT_NM, color=MUTED, linewidth=0.7, linestyle=(0, (1.2, 1.6)), zorder=1.5)
    ax_r.text(ENDPOINT_NM * 0.955, 0.275, "endpoint anchor", color=INK2, fontsize=6.2, ha="right", va="bottom", rotation=90)

    for cls, label, color, marker in CLASSES:
        rows = [stats["classes"][cls]["per_delta"][f"{d:g}"] for d in RUNGS]
        x = [row["delta_nm"] for row in rows]
        med = [row["q10_R_median"] for row in rows]
        lo = [row["q10_R_min"] for row in rows]
        hi = [row["q10_R_max"] for row in rows]
        ax_r.fill_between(x, lo, hi, color=color, alpha=0.16, linewidth=0, zorder=2)
        ax_r.plot(x[:-1], med[:-1], color=color, linewidth=1.3, zorder=3)
        ax_r.plot(x[-2:], med[-2:], color=color, linewidth=1.0, linestyle=(0, (2.0, 1.6)), zorder=3)
        ax_r.plot(x[:-1], med[:-1], linestyle="none", marker=marker, markersize=4.0,
                  markerfacecolor=color, markeredgecolor=PAPER, markeredgewidth=0.7, zorder=4)
        ax_r.plot([x[-1]], [med[-1]], linestyle="none", marker=marker, markersize=4.0,
                  markerfacecolor=PAPER, markeredgecolor=color, markeredgewidth=0.9, zorder=4)
        star = stats["classes"][cls]["tier_median_delta_star"]
        ax_r.plot([star], [gate_r], linestyle="none", marker="*", markersize=9.5,
                  markerfacecolor=color, markeredgecolor=PAPER, markeredgewidth=0.7, zorder=6)

        f_rows = rows
        f_med = [row["q90_F_median"] for row in f_rows]
        f_max = [row["q90_F_max"] for row in f_rows]
        ax_f.plot(x, f_max, color=color, linewidth=0.8, linestyle=(0, (1.0, 1.4)), zorder=3)
        ax_f.plot(x[:-1], f_med[:-1], color=color, linewidth=1.3, zorder=3)
        ax_f.plot(x[-2:], f_med[-2:], color=color, linewidth=1.0, linestyle=(0, (2.0, 1.6)), zorder=3)
        ax_f.plot(x[:-1], f_med[:-1], linestyle="none", marker=marker, markersize=4.0,
                  markerfacecolor=color, markeredgecolor=PAPER, markeredgewidth=0.7, zorder=4)
        ax_f.plot([x[-1]], [min(f_med[-1], 0.27)], linestyle="none", marker=marker, markersize=4.0,
                  markerfacecolor=PAPER, markeredgecolor=color, markeredgewidth=0.9, zorder=4)

    m10 = stats["classes"]["m10"]
    m50 = stats["classes"]["m50"]
    ax_r.text(0.02, 0.04,
              r"$\delta^{*}$" + f" median  {m10['tier_median_delta_star']:g} nm ({label_of('m10')}),  "
              f"{m50['tier_median_delta_star']:g} nm ({label_of('m50')})",
              transform=ax_r.transAxes, color=INK, fontsize=6.6, ha="left", va="bottom")
    ax_r.set_ylabel(r"retention  $Q_{10}[R]$  over 8 directions")
    ax_r.legend(
        handles=[
            Line2D([], [], color=c, linewidth=1.3, marker=m, markersize=3.8,
                   markeredgecolor=PAPER, markeredgewidth=0.6, label=f"{l}  median of 12 systems")
            for _, l, c, m in CLASSES
        ] + [Patch(facecolor=INK2, alpha=0.16, linewidth=0, label="range over systems")],
        loc="upper left", bbox_to_anchor=(-0.01, 0.62), ncol=1,
    )

    # spurious panel
    ax_f.set_ylim(-0.012, 0.285)
    ax_f.grid(axis="y", zorder=0.5)
    ax_f.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_f.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax_f.axhline(gate_f, color=INK2, linewidth=0.75, linestyle=(0, (2.6, 1.8)), zorder=2)
    ax_f.text(0.83, gate_f + 0.008, f"spurious gate {gate_f:g}", color=INK2, fontsize=6.6, ha="left", va="bottom")
    clipped = m10["per_delta"]["35"]["q90_F_max"]
    ax_f.text(19.0, 0.262, f"max {clipped:.2f}, off scale", color=INK2, fontsize=6.0, ha="right", va="top")
    ax_f.set_ylabel(r"spurious fraction  $Q_{90}[F]$")
    ax_f.set_xlabel(DELTA_LABEL)
    ax_f.legend(
        handles=[
            Line2D([], [], color=INK2, linewidth=1.3, label="median over systems"),
            Line2D([], [], color=INK2, linewidth=0.8, linestyle=(0, (1.0, 1.4)), label="maximum over systems"),
        ],
        loc="upper left", bbox_to_anchor=(-0.01, 1.02), ncol=1,
    )
    save(fig, out, "fig_psf_knowledge_requirement")


def label_of(cls):
    return next(label for key, label, _, _ in CLASSES if key == cls)


# --------------------------------------------------------------- nonlinear block
def nonlinear_statistics(review):
    ref = REFERENCE["nonlinear"]
    check("nonlinear integrity", review["integrity"], ref["integrity"])
    check("nonlinear rows", review["rows"], ref["rows"], 0)
    check("nonlinear fit wall hours", review["total_fit_wall_hours"], ref["total_fit_wall_hours"], 1e-2)
    require(review["integrity_findings"] == [] and review["missing"] == [] and review["amendments"] == [], 'review["integrity_findings"] == [] and review["missing"] == [] and review["amendments"] == []')
    require(review["quality_flag_tally"] == {}, 'review["quality_flag_tally"] == {}')
    science = review["science"]
    require(science["findings"] == [] and science["expected_delta_draws"] == N_DRAWS, 'science["findings"] == [] and science["expected_delta_draws"] == N_DRAWS')

    def tally(block):
        return {
            "count": block["count"],
            "n": block["n_tested"],
            "fraction": block["fraction"],
            "interval": block["interval"],
            "confidence": block["confidence"],
        }

    out = {"control": {}, "injected": {}, "reference": {}, "matched_null": {}}
    for delta in NONLINEAR_DELTAS:
        key = f"{delta:g}"
        ctrl = science["control"]["per_delta"][key]
        require(ctrl["n_draws"] == N_DRAWS and ctrl["n_draws_expected"] == N_DRAWS, 'ctrl["n_draws"] == N_DRAWS and ctrl["n_draws_expected"] == N_DRAWS')
        require(ctrl["q_fit_ge_10"]["n_none"] == 0, 'ctrl["q_fit_ge_10"]["n_none"] == 0')
        check(f"control q_fit>=10 count at {key} nm", ctrl["q_fit_ge_10"]["count"], ref["control_q_ge_10_counts"][delta], 0)
        check(f"control dlogZ>5 count at {key} nm", ctrl["dlogZ_gt_5"]["count"], ref["control_dlogz_counts"][delta], 0)
        out["control"][key] = {
            "delta_nm": delta,
            "q_fit_ge_10": tally(ctrl["q_fit_ge_10"]),
            "dlogZ_gt_5": tally(ctrl["dlogZ_gt_5"]),
            "median_q_fit": ctrl["median_q_fit"],
            "median_dlogZ": ctrl["median_dlogZ"],
            "q_fit_max": ctrl["q_fit_quantiles"]["max"],
            "q_fit_q99": ctrl["q_fit_quantiles"]["quantiles"]["99"],
            "boundary_tally": ctrl["diagnostics"]["boundary_tally"],
        }
        inj = science["recovery"]["per_delta"][key]
        require(inj["n_draws"] == N_DRAWS and inj["n"] == N_DRAWS, 'inj["n_draws"] == N_DRAWS and inj["n"] == N_DRAWS')
        check(f"injected q_fit>=10 count at {key} nm", inj["q_fit_ge_10"]["count"], ref["injected_q_ge_10_counts"][delta], 0)
        check(f"injected dlogZ>5 count at {key} nm", inj["dlogZ_gt_5"]["count"], ref["injected_dlogz_counts"][delta], 0)
        out["injected"][key] = {
            "delta_nm": delta,
            "q_fit_ge_10": tally(inj["q_fit_ge_10"]),
            "dlogZ_gt_5": tally(inj["dlogZ_gt_5"]),
            "median_q_fit": inj["median_q_fit"],
            "median_q_fit_shift": inj["median_q_fit_shift"],
            "median_dlogZ": inj["median_dlogZ"],
            "mass_bias_median_dex": inj["bias"]["mass_bias_median_dex"],
            "position_offset_median_arcsec": inj["bias"]["position_offset_median_arcsec"],
            "boundary_tally": inj["diagnostics"]["boundary_tally"],
        }
    require(science["control"]["first_separating_delta"] is None, 'science["control"]["first_separating_delta"] is None')
    reference = science["recovery"]["delta_0_reference"]
    check("reference q_fit>=10 count", reference["q_fit_ge_10"]["count"], ref["reference_q_ge_10"][0], 0)
    check("reference n", reference["q_fit_ge_10"]["n_tested"], ref["reference_q_ge_10"][1], 0)
    check("reference dlogZ>5 count", reference["dlogZ_gt_5"]["count"], ref["reference_dlogz"][0], 0)
    require(reference["label"] == "v1 noisy_injected delta 0", reference["label"])
    out["reference"] = {
        "label": reference["label"],
        "q_fit_ge_10": tally(reference["q_fit_ge_10"]),
        "dlogZ_gt_5": tally(reference["dlogZ_gt_5"]),
        "median_q_fit": reference["median_q_fit"],
    }
    null = science["matched_null"]
    check("pooled null q_fit>=10 count", null["pooled_590"]["q_fit_ge_10"]["count"], ref["pooled_null_q_ge_10"][0], 0)
    check("pooled null n", null["pooled_590"]["q_fit_ge_10"]["n_tested"], ref["pooled_null_q_ge_10"][1], 0)
    check("selected null q_fit>=10 count", null["selected12_120"]["q_fit_ge_10"]["count"], ref["selected_null_q_ge_10"][0], 0)
    check("selected null n", null["selected12_120"]["q_fit_ge_10"]["n_tested"], ref["selected_null_q_ge_10"][1], 0)
    check("pooled null dlogZ>5 count", null["pooled_590"]["dlogZ_gt_5"]["count"], ref["pooled_null_dlogz"][0], 0)
    check("selected null dlogZ>5 count", null["selected12_120"]["dlogZ_gt_5"]["count"], ref["selected_null_dlogz"][0], 0)
    n_null_rows = science["null_source"]["n_rows"]
    n_v1_controls = science["reference_source"]["n_control_rows"]
    check("null campaign rows", n_null_rows, ref["pooled_null_composition"][0], 0)
    check("v1 control rows", n_v1_controls, ref["pooled_null_composition"][1], 0)
    check("pooled null composition sums", n_null_rows + n_v1_controls, ref["pooled_null_q_ge_10"][1], 0)
    check("reference injected rows", science["reference_source"]["n_injected_rows"], ref["reference_q_ge_10"][1], 0)
    out["matched_null"] = {
        "composition": {
            "null_campaign_rows": n_null_rows,
            "null_campaign_uuid": science["null_source"]["campaign_uuid"],
            "v1_control_rows": n_v1_controls,
            "v1_campaign_uuid": science["reference_source"]["campaign_uuid"],
            "total": n_null_rows + n_v1_controls,
            "note": "rows share systems, scenes and positions; the pooled tally is not a set of independent trials",
        },
        "pooled_590": {"q_fit_ge_10": tally(null["pooled_590"]["q_fit_ge_10"]), "dlogZ_gt_5": tally(null["pooled_590"]["dlogZ_gt_5"])},
        "selected12_120": {"q_fit_ge_10": tally(null["selected12_120"]["q_fit_ge_10"]), "dlogZ_gt_5": tally(null["selected12_120"]["dlogZ_gt_5"])},
    }
    out["first_separating_delta"] = science["control"]["first_separating_delta"]
    return out


def errorbar(ax, x, block, color, marker, filled, size=4.2, zorder=4):
    frac = block["fraction"]
    lo, hi = block["interval"]
    ax.errorbar([x], [frac], yerr=[[frac - lo], [hi - frac]], fmt="none", ecolor=color,
                elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=zorder - 0.5)
    ax.plot([x], [frac], linestyle="none", marker=marker, markersize=size,
            markerfacecolor=color if filled else PAPER, markeredgecolor=PAPER if filled else color,
            markeredgewidth=0.7 if filled else 0.9, zorder=zorder)


def fig_psf_knowledge_nonlinear(stats, out):
    fig = plt.figure(figsize=(COL, 4.35))
    ax_c = fig.add_axes((0.175, 0.565, 0.79, 0.395))
    ax_i = fig.add_axes((0.175, 0.105, 0.79, 0.395))
    positions = {0.0: 0.0, 2.0: 1.0, 5.0: 2.0, 10.0: 3.0, 20.0: 4.0}
    labels = ["0", "2", "5", "10", "20"]
    for ax in (ax_c, ax_i):
        ax.set_xlim(-0.55, 4.55)
        ax.set_xticks(list(positions.values()))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.grid(axis="y", zorder=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", top=False)
    ax_c.set_xticklabels([])
    ax_i.set_xticklabels(labels)

    null = stats["matched_null"]["pooled_590"]["q_fit_ge_10"]
    lo, hi = null["interval"]
    ax_c.axhspan(lo, hi, color=PAR, alpha=0.16, linewidth=0, zorder=1)
    ax_c.axhline(null["fraction"], color=PAR, linewidth=0.9, zorder=2)
    errorbar(ax_c, positions[0.0], null, PAR, "o", True)
    errorbar(ax_c, positions[0.0] + 0.13, stats["matched_null"]["pooled_590"]["dlogZ_gt_5"], PAR, "D", False, size=3.4)
    for delta in NONLINEAR_DELTAS:
        row = stats["control"][f"{delta:g}"]
        errorbar(ax_c, positions[delta], row["q_fit_ge_10"], SEL, "o", True)
        errorbar(ax_c, positions[delta] + 0.13, row["dlogZ_gt_5"], SEL, "D", False, size=3.4)
    ax_c.set_ylim(-0.006, 0.20)
    ax_c.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax_c.set_ylabel("control exceedance fraction\n(no-subhalo arm)")
    ax_c.text(positions[0.0] + 0.07, 0.026, f"matched null\n{null['count']}/{null['n']} at " + r"$q_{\mathrm{fit}}\geq10$", color=PAR, fontsize=6.4, ha="center", va="bottom", linespacing=1.05)
    ax_c.text(0.015, 0.96, f"n = {N_DRAWS} draws\nat each non-zero " + r"$\delta$", transform=ax_c.transAxes, color=MUTED, fontsize=6.4, ha="left", va="top")
    ax_c.legend(
        handles=[
            Line2D([], [], color=SEL, linestyle="none", marker="o", markersize=3.8, markeredgecolor=PAPER, markeredgewidth=0.6, label=r"$q_{\mathrm{fit}}\geq 10$"),
            Line2D([], [], color=SEL, linestyle="none", marker="D", markersize=3.0, markerfacecolor=PAPER, markeredgewidth=0.9, label=r"$\Delta\log Z>5$"),
            Patch(facecolor=PAR, alpha=0.16, linewidth=0, label=r"matched null $q_{\mathrm{fit}}\geq10$, 95% interval"),
        ],
        loc="upper right", bbox_to_anchor=(1.01, 1.01), ncol=1,
    )

    reference = stats["reference"]
    errorbar(ax_i, positions[0.0], reference["q_fit_ge_10"], PAR, "s", True)
    errorbar(ax_i, positions[0.0] + 0.13, reference["dlogZ_gt_5"], PAR, "D", False, size=3.4)
    for delta in NONLINEAR_DELTAS:
        row = stats["injected"][f"{delta:g}"]
        errorbar(ax_i, positions[delta], row["q_fit_ge_10"], ORD[1], "s", True)
        errorbar(ax_i, positions[delta] + 0.13, row["dlogZ_gt_5"], ORD[1], "D", False, size=3.4)
    ax_i.set_ylim(-0.02, 1.02)
    ax_i.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax_i.set_ylabel("recovery fraction\n(injected bracket-top rung)")
    ax_i.set_xlabel(DELTA_LABEL)
    ax_i.text(positions[0.0] + 0.28, 0.50, f"matched PSF\nv1 noisy_injected\n{reference['q_fit_ge_10']['n']} selected systems", color=PAR, fontsize=6.2, ha="left", va="center", linespacing=1.05)
    ax_i.legend(
        handles=[
            Line2D([], [], color=ORD[1], linestyle="none", marker="s", markersize=3.8, markeredgecolor=PAPER, markeredgewidth=0.6, label=r"$q_{\mathrm{fit}}\geq 10$"),
            Line2D([], [], color=ORD[1], linestyle="none", marker="D", markersize=3.0, markerfacecolor=PAPER, markeredgewidth=0.9, label=r"$\Delta\log Z>5$"),
        ],
        loc="lower right", bbox_to_anchor=(1.01, -0.01), ncol=1,
    )
    save(fig, out, "fig_psf_knowledge_nonlinear")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fisher-review", required=True)
    parser.add_argument("--nonlinear-review", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.abspath(__file__), "rb") as handle:
        script_sha = hashlib.sha256(handle.read()).hexdigest()
    fisher, fisher_sha = load(args.fisher_review)
    nonlinear, nonlinear_sha = load(args.nonlinear_review)
    stats = {
        "provenance": {
            "generated_unix": int(time.time()),
            "figure_script": os.path.basename(__file__),
            "figure_script_sha256": script_sha,
            "fisher_review_sha256": fisher_sha,
            "fisher_campaign_uuid": fisher["campaign_uuid"],
            "nonlinear_review_sha256": nonlinear_sha,
            "nonlinear_campaign_uuid": nonlinear["campaign_uuid"],
            "code_revision": fisher["code_revision"],
            "note": (
                "Retention R and spurious fraction F are re-derived from the per-direction "
                "cell counts; the 35 nm endpoint anchor has no harvest ratios and is "
                "computed here from the same counts for display only and excluded from "
                "delta*. Nonlinear counts and Clopper-Pearson intervals are copied from "
                "the harvest summaries, not recomputed."
            ),
        },
        "fig_psf_knowledge_requirement": fisher_statistics(fisher),
        "fig_psf_knowledge_nonlinear": nonlinear_statistics(nonlinear),
    }
    require(nonlinear["code_revision"] == fisher["code_revision"], "campaign code revisions differ")
    print("figure 1")
    fig_psf_knowledge_requirement(stats["fig_psf_knowledge_requirement"], args.out)
    print("figure 2")
    fig_psf_knowledge_nonlinear(stats["fig_psf_knowledge_nonlinear"], args.out)
    stats["sanity_checks"] = checks
    stats["sanity_checks_all_pass"] = all(item["pass"] for item in checks)
    with open(os.path.join(args.out, "figures_stats_psf_knowledge.json"), "w") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    print(f"{len(checks)} sanity checks, all pass: {stats['sanity_checks_all_pass']}")


if __name__ == "__main__":
    main()
