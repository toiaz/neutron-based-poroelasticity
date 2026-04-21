#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = ROOT / "batch_summary.csv"
R_MIN_A = 10.0
R_MAX_A = 15000.0
WINDOW_LABEL = "1-1500 nm (10-15000 A)"
PSD_CHANGE_BAND_A = (1.0e3, 3.0e3)
SF_Q_MAX = 5.0e-3
COMMON_BASELINE_SAMPLE = "TH6-0.1"
AXIAL_PRESSURE_MAP = {
    0.1: 0.7,
    1.0: 6.8,
    2.0: 13.5,
    3.0: 20.3,
    4.0: 27.1,
    5.0: 33.9,
    6.0: 40.7,
    8.0: 54.2,
    10.0: 67.7,
}


def convert_axial_pressure(nominal_value: float) -> float:
    for nominal, actual in AXIAL_PRESSURE_MAP.items():
        if np.isclose(float(nominal_value), nominal, atol=1.0e-9):
            return actual
    raise KeyError(f"No converted axial pressure defined for {nominal_value}")


def format_state_label(row, confined_suffix=True):
    pressure = row["axial_pressure_mpa"]
    if row["side_pressure_mpa"] > 0.0:
        return f"{pressure:.1f} MPa, confined" if confined_suffix else f"{pressure:.1f} MPa"
    return f"{pressure:.1f} MPa, no side pressure" if confined_suffix else f"{pressure:.1f} MPa"


def load_summary(path: Path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed = dict(row)
            for key in (
                "side_pressure_mpa",
                "axial_pressure_mpa",
                "thickness_um",
                "thickness_mm",
                "q_min",
                "q_max",
                "C",
                "D",
                "R_hs",
                "eta",
                "porosity_pct",
                "partial_porosity_pct",
                "pore_vol_mm3",
                "partial_pore_vol_mm3",
            ):
                parsed[key] = float(parsed[key])
            parsed["axial_pressure_nominal_mpa"] = parsed["axial_pressure_mpa"]
            parsed["axial_pressure_mpa"] = convert_axial_pressure(parsed["axial_pressure_mpa"])
            parsed["output_dir"] = Path(parsed["output_dir"])
            rows.append(parsed)
    rows.sort(key=lambda item: (item["side_pressure_mpa"], item["axial_pressure_mpa"]))
    return rows


def split_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["side_pressure_mpa"], []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda item: item["axial_pressure_mpa"])
    return groups


def build_focus_rows(rows):
    focus_rows = []
    groups = split_groups(rows)
    baseline = get_row_by_name(rows, COMMON_BASELINE_SAMPLE)
    base_vol = baseline["partial_pore_vol_mm3"]
    base_phi = baseline["partial_porosity_pct"]
    for side_pressure, group_rows in groups.items():
        for row in group_rows:
            focus = dict(row)
            focus["pore_volume_1_1500_nm_mm3"] = row["partial_pore_vol_mm3"]
            focus["porosity_1_1500_nm_pct"] = row["partial_porosity_pct"]
            focus["delta_pore_volume_mm3_vs_group_baseline"] = row["partial_pore_vol_mm3"] - base_vol
            focus["delta_porosity_pct_points_vs_group_baseline"] = row["partial_porosity_pct"] - base_phi
            focus["ratio_pore_volume_vs_group_baseline"] = row["partial_pore_vol_mm3"] / base_vol
            focus["ratio_porosity_vs_group_baseline"] = row["partial_porosity_pct"] / base_phi
            # Explicit common-baseline aliases (0.7 MPa TH6-0.1) for manuscript consistency.
            focus["delta_pore_volume_mm3_vs_common_baseline"] = focus["delta_pore_volume_mm3_vs_group_baseline"]
            focus["delta_porosity_pct_points_vs_common_baseline"] = focus["delta_porosity_pct_points_vs_group_baseline"]
            focus["ratio_pore_volume_vs_common_baseline"] = focus["ratio_pore_volume_vs_group_baseline"]
            focus["ratio_porosity_vs_common_baseline"] = focus["ratio_porosity_vs_group_baseline"]
            focus_rows.append(focus)
    focus_rows.sort(key=lambda item: (item["side_pressure_mpa"], item["axial_pressure_mpa"]))
    return focus_rows


def build_shared_axial_comparison(focus_rows):
    groups = split_groups(focus_rows)
    if 0.0 not in groups or 1.1 not in groups:
        return []

    no_side = {row["axial_pressure_mpa"]: row for row in groups[0.0]}
    side = {row["axial_pressure_mpa"]: row for row in groups[1.1]}
    shared_axial = sorted(set(no_side).intersection(side))

    comparisons = []
    for axial in shared_axial:
        base = no_side[axial]
        conf = side[axial]
        comparisons.append(
            {
                "axial_pressure_mpa": axial,
                "pore_volume_no_side_mm3": base["pore_volume_1_1500_nm_mm3"],
                "pore_volume_side_mm3": conf["pore_volume_1_1500_nm_mm3"],
                "delta_pore_volume_mm3": conf["pore_volume_1_1500_nm_mm3"] - base["pore_volume_1_1500_nm_mm3"],
                "ratio_pore_volume_side_vs_no_side": conf["pore_volume_1_1500_nm_mm3"] / base["pore_volume_1_1500_nm_mm3"],
                "porosity_no_side_pct": base["porosity_1_1500_nm_pct"],
                "porosity_side_pct": conf["porosity_1_1500_nm_pct"],
                "delta_porosity_pct_points": conf["porosity_1_1500_nm_pct"] - base["porosity_1_1500_nm_pct"],
                "ratio_porosity_side_vs_no_side": conf["porosity_1_1500_nm_pct"] / base["porosity_1_1500_nm_pct"],
            }
        )
    return comparisons


def write_focus_summary(focus_rows, shared_rows):
    summary_path = ROOT / "th6_1nm_focus_summary.csv"
    fieldnames = [
        "sample_key",
        "sample_name",
        "side_pressure_mpa",
        "axial_pressure_nominal_mpa",
        "axial_pressure_mpa",
        "pore_volume_1_1500_nm_mm3",
        "porosity_1_1500_nm_pct",
        "delta_pore_volume_mm3_vs_group_baseline",
        "delta_porosity_pct_points_vs_group_baseline",
        "ratio_pore_volume_vs_group_baseline",
        "ratio_porosity_vs_group_baseline",
        "delta_pore_volume_mm3_vs_common_baseline",
        "delta_porosity_pct_points_vs_common_baseline",
        "ratio_pore_volume_vs_common_baseline",
        "ratio_porosity_vs_common_baseline",
        "output_dir",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in focus_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    shared_path = ROOT / "th6_1nm_shared_axial_comparison.csv"
    if shared_rows:
        fieldnames = list(shared_rows[0].keys())
        with shared_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shared_rows)


def write_structure_factor_summary(rows):
    out_path = ROOT / "th6_structure_factor_summary.csv"
    q_probes = (3.0e-4, 1.0e-3)
    fieldnames = [
        "sample_key",
        "sample_name",
        "side_pressure_mpa",
        "axial_pressure_nominal_mpa",
        "axial_pressure_mpa",
        "R_hs_A",
        "eta",
        "S_at_3e-4_A^-1",
        "S_at_1e-3_A^-1",
        "output_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            s_probe = structure_factor_hardsphere_py(np.array(q_probes), row["R_hs"], row["eta"])
            writer.writerow(
                {
                    "sample_key": row["sample_key"],
                    "sample_name": row["sample_name"],
                    "side_pressure_mpa": row["side_pressure_mpa"],
                    "axial_pressure_nominal_mpa": row["axial_pressure_nominal_mpa"],
                    "axial_pressure_mpa": row["axial_pressure_mpa"],
                    "R_hs_A": row["R_hs"],
                    "eta": row["eta"],
                    "S_at_3e-4_A^-1": s_probe[0],
                    "S_at_1e-3_A^-1": s_probe[1],
                    "output_dir": row["output_dir"],
                }
            )


def structure_factor_hardsphere_py(q, radius_hs, eta):
    x = np.asarray(q, dtype=float) * 2.0 * float(radius_hs)
    eta = float(eta)
    alpha = (1.0 + 2.0 * eta) ** 2 / (1.0 - eta) ** 4
    beta = -6.0 * eta * (1.0 + eta / 2.0) ** 2 / (1.0 - eta) ** 4
    gamma = 0.5 * eta * alpha

    term1 = np.empty_like(x)
    term2 = np.empty_like(x)
    term3 = np.empty_like(x)

    small = np.abs(x) < 1.0e-3
    xs = x[~small]
    if xs.size:
        sinx = np.sin(xs)
        cosx = np.cos(xs)
        term1[~small] = (sinx - xs * cosx) / xs ** 2
        term2[~small] = (2.0 * xs * sinx + (2.0 - xs ** 2) * cosx - 2.0) / xs ** 3
        term3_num = (-xs ** 4) * cosx + 4.0 * (
            (3.0 * xs ** 2 - 6.0) * cosx + (xs ** 3 - 6.0 * xs) * sinx + 6.0
        )
        term3[~small] = term3_num / xs ** 5

    if np.any(small):
        x0 = x[small]
        term1[small] = x0 / 3.0 - x0 ** 3 / 30.0 + x0 ** 5 / 840.0
        term2[small] = x0 / 4.0 - x0 ** 3 / 36.0 + x0 ** 5 / 960.0
        term3[small] = x0 / 6.0 - x0 ** 3 / 48.0 + x0 ** 5 / 1200.0

    g_term = alpha * term1 + beta * term2 + gamma * term3
    structure = np.empty_like(x)
    zero = np.abs(x) < 1.0e-12
    structure[zero] = 1.0 / alpha
    nonzero = ~zero
    structure[nonzero] = 1.0 / (1.0 + 24.0 * eta * g_term[nonzero] / x[nonzero])
    return structure


def get_row_by_name(rows, sample_name):
    for row in rows:
        if row["sample_name"] == sample_name:
            return row
    raise KeyError(f"Sample not found: {sample_name}")


def load_psd_curve(row):
    psd_path = row["output_dir"] / "psd_result.csv"
    data = np.genfromtxt(psd_path, delimiter=",", names=True)
    radius = data["r_center"]
    density = data["psd_density_per_log_r"]
    mask = (radius >= R_MIN_A) & (radius <= R_MAX_A)
    return radius[mask], density[mask]


def load_iq_fit_curve(row):
    iq_path = row["output_dir"] / "iq_fit.csv"
    data = np.genfromtxt(iq_path, delimiter=",", names=True)
    q = data["q_A1"]
    intensity = data["intensity"]
    sigma = data["sigma"]
    fit = data["intensity_fit"]
    return q, intensity, sigma, fit


def plot_metric_ratio_overview(focus_rows):
    groups = split_groups(focus_rows)
    colors = {0.0: "#0B6E4F", 1.1: "#B23A48"}
    labels = {0.0: "No side pressure", 1.1: "Confining + pore-fluid pressure"}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)

    specs = [
        (
            "pore_volume_1_1500_nm_mm3",
            "Absolute pore volume",
            "Pore volume (mm^3)",
            False,
        ),
        (
            "porosity_1_1500_nm_pct",
            "Apparent porosity in the same window",
            "Porosity (%)",
            False,
        ),
        (
            "ratio_pore_volume_vs_group_baseline",
            "Pore-volume ratio to common 0.7 MPa baseline",
            "Volume / common baseline",
            True,
        ),
        (
            "ratio_porosity_vs_group_baseline",
            "Porosity ratio to common 0.7 MPa baseline",
            "Porosity / common baseline",
            True,
        ),
    ]

    for ax, (metric, title, ylabel, is_ratio) in zip(axes.flat, specs):
        for side_pressure, group_rows in groups.items():
            x = [item["axial_pressure_mpa"] for item in group_rows]
            y = [item[metric] for item in group_rows]
            ax.plot(
                x,
                y,
                marker="o",
                ms=6,
                lw=2.2,
                color=colors[side_pressure],
                label=labels[side_pressure],
            )
        if is_ratio:
            ax.axhline(1.0, color="#555555", lw=1.0, ls=":")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Axial pressure (MPa)")
        ax.set_ylabel(ylabel)
        ax.grid(True, ls="--", lw=0.6, alpha=0.45)

    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle(
        "TH6 metrics separate absolute pore-volume response from geometry-driven porosity",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(ROOT / "th6_1nm_metric_ratio_overview.png", dpi=240, bbox_inches="tight")
    fig.savefig(ROOT / "th6_1nm_metric_ratio_overview.svg", bbox_inches="tight")
    plt.close(fig)


def plot_shared_comparison(shared_rows):
    if not shared_rows:
        return

    axial = [row["axial_pressure_mpa"] for row in shared_rows]
    vol_ratio = [row["ratio_pore_volume_side_vs_no_side"] for row in shared_rows]
    phi_ratio = [row["ratio_porosity_side_vs_no_side"] for row in shared_rows]

    x = np.arange(len(axial))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    ax.bar(x - width / 2, vol_ratio, width=width, color="#B23A48", label="Pore-volume ratio")
    ax.bar(x + width / 2, phi_ratio, width=width, color="#D98E04", label="Porosity ratio")
    ax.axhline(1.0, color="#555555", lw=1.0, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g} MPa" for value in axial])
    ax.set_ylabel("Confined / unconfined")
    ax.set_xlabel("Matched axial pressure")
    ax.set_title("Confinement shifts TH6 to a lower-volume trajectory", fontsize=12, weight="bold")
    ax.legend(frameon=False, loc="best")
    ax.grid(True, axis="y", ls="--", lw=0.6, alpha=0.45)
    fig.savefig(ROOT / "th6_1nm_shared_axial_comparison.png", dpi=240, bbox_inches="tight")
    fig.savefig(ROOT / "th6_1nm_shared_axial_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def plot_psd_change_overview(rows):
    state_pairs = [
        ("TH6-3", "TH6-3-1.1"),
        ("TH6-4", "TH6-4-1.1"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    overlay_ax, diff_ax = axes

    styles = {
        "TH6-3": ("#1B7F5A", "-"),
        "TH6-3-1.1": ("#B23A48", "-"),
        "TH6-4": ("#1B7F5A", "--"),
        "TH6-4-1.1": ("#B23A48", "--"),
    }

    for sample_name in ("TH6-3", "TH6-3-1.1", "TH6-4", "TH6-4-1.1"):
        row = get_row_by_name(rows, sample_name)
        radius, density = load_psd_curve(row)
        color, ls = styles[sample_name]
        label = format_state_label(row)
        overlay_ax.plot(radius, density, color=color, ls=ls, lw=2.0, label=label)

    overlay_ax.set_xscale("log")
    overlay_ax.set_xlim(R_MIN_A, R_MAX_A)
    overlay_ax.set_xlabel("Radius (A)")
    overlay_ax.set_ylabel("dV / d(ln r)")
    overlay_ax.set_title("Matched-state PSD overlays", fontsize=12, weight="bold")
    overlay_ax.grid(True, which="both", ls="--", lw=0.55, alpha=0.4)
    overlay_ax.legend(frameon=False, fontsize=8.5, loc="best")

    diff_colors = {"TH6-3": "#6A1B4D", "TH6-4": "#D98E04"}
    for base_name, conf_name in state_pairs:
        base_row = get_row_by_name(rows, base_name)
        conf_row = get_row_by_name(rows, conf_name)
        radius_base, density_base = load_psd_curve(base_row)
        radius_conf, density_conf = load_psd_curve(conf_row)
        if not np.allclose(radius_base, radius_conf):
            raise ValueError("PSD grids are inconsistent across matched states.")
        diff_ax.plot(
            radius_base,
            density_conf - density_base,
            lw=2.2,
            color=diff_colors[base_name],
            label=f"confined - unconfined, {base_row['axial_pressure_mpa']:.1f} MPa",
        )

    diff_ax.axhline(0.0, color="#555555", lw=1.0, ls=":")
    diff_ax.axvspan(PSD_CHANGE_BAND_A[0], PSD_CHANGE_BAND_A[1], color="#808080", alpha=0.08)
    diff_ax.set_xscale("log")
    diff_ax.set_xlim(R_MIN_A, R_MAX_A)
    diff_ax.set_xlabel("Radius (A)")
    diff_ax.set_ylabel("Delta dV / d(ln r)")
    diff_ax.set_title("Largest PSD differences are concentrated near 100-300 nm", fontsize=12, weight="bold")
    diff_ax.grid(True, which="both", ls="--", lw=0.55, alpha=0.4)
    diff_ax.legend(frameon=False, fontsize=8.5, loc="best")

    fig.suptitle("Confinement changes the PSD mainly in the intermediate-size pore band", fontsize=14, weight="bold")
    fig.savefig(ROOT / "th6_1nm_psd_change_overview.png", dpi=240, bbox_inches="tight")
    fig.savefig(ROOT / "th6_1nm_psd_change_overview.svg", bbox_inches="tight")
    plt.close(fig)


def plot_structure_factor_overview(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    sf_ax, eta_ax = axes

    selected = [
        ("TH6-0.1", "#1B7F5A", "-"),
        ("TH6-4", "#0B6E4F", "--"),
        ("TH6-3-1.1", "#B23A48", "-"),
        ("TH6-10-1.1", "#7A0019", "--"),
    ]
    q_min = min(row["q_min"] for row in rows)
    q_vals = np.logspace(np.log10(q_min), np.log10(SF_Q_MAX), 400)
    for sample_name, color, ls in selected:
        row = get_row_by_name(rows, sample_name)
        sf_curve = structure_factor_hardsphere_py(q_vals, row["R_hs"], row["eta"])
        label = format_state_label(row)
        sf_ax.plot(q_vals, sf_curve, color=color, ls=ls, lw=2.1, label=label)

    sf_ax.set_xscale("log")
    sf_ax.set_xlabel("Q (A^-1)")
    sf_ax.set_ylabel("S(q)")
    sf_ax.set_title("Inferred structure-factor curves", fontsize=12, weight="bold")
    sf_ax.grid(True, which="both", ls="--", lw=0.55, alpha=0.4)
    sf_ax.legend(frameon=False, fontsize=8.5, loc="best")

    groups = split_groups(rows)
    colors = {0.0: "#0B6E4F", 1.1: "#B23A48"}
    labels = {0.0: "No side pressure", 1.1: "Confining + pore-fluid pressure"}
    for side_pressure, group_rows in groups.items():
        eta_ax.plot(
            [item["axial_pressure_mpa"] for item in group_rows],
            [item["eta"] for item in group_rows],
            marker="o",
            ms=6,
            lw=2.0,
            color=colors[side_pressure],
            label=labels[side_pressure],
        )
    eta_ax.set_xlabel("Axial pressure (MPa)")
    eta_ax.set_ylabel("Packing fraction eta")
    eta_ax.set_title("Inferred eta varies only modestly", fontsize=12, weight="bold")
    eta_ax.grid(True, ls="--", lw=0.6, alpha=0.45)
    eta_ax.legend(frameon=False, loc="best")
    eta_ax.text(
        0.98,
        0.04,
        r"$R_{hs} \approx 4153\ \AA$ for all states",
        transform=eta_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    fig.suptitle("Structure-factor evolution is concentrated at the lowest q", fontsize=14, weight="bold")
    fig.savefig(ROOT / "th6_structure_factor_overview.png", dpi=240, bbox_inches="tight")
    fig.savefig(ROOT / "th6_structure_factor_overview.svg", bbox_inches="tight")
    plt.close(fig)


def plot_raw_iq_summary(rows):
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12.8,
            "axes.titlesize": 13.6,
            "axes.labelsize": 13.2,
            "legend.fontsize": 10.3,
            "xtick.labelsize": 11.8,
            "ytick.labelsize": 11.8,
            "axes.linewidth": 0.95,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
        }
    )

    selected = [
        ("TH6-0.1", "a"),
        ("TH6-3", "b"),
        ("TH6-3-1.1", "c"),
        ("TH6-4-1.1", "d"),
    ]
    fig = plt.figure(figsize=(13.2, 8.9))
    outer = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.34)

    for idx, (sample_name, panel_id) in enumerate(selected):
        row = get_row_by_name(rows, sample_name)
        q, intensity, sigma, fit = load_iq_fit_curve(row)
        valid = np.isfinite(q) & np.isfinite(intensity) & np.isfinite(fit) & (q > 0.0) & (intensity > 0.0) & (fit > 0.0)
        qv = q[valid]
        iv = intensity[valid]
        fv = fit[valid]
        sigv = sigma[valid] if sigma is not None else None

        r = idx // 2
        c = idx % 2
        inner = outer[r, c].subgridspec(2, 1, height_ratios=[3.1, 1.0], hspace=0.04)
        ax_top = fig.add_subplot(inner[0, 0])
        ax_bottom = fig.add_subplot(inner[1, 0], sharex=ax_top)

        ax_top.loglog(
            qv,
            iv,
            linestyle="none",
            marker="o",
            ms=3.4,
            alpha=0.72,
            color="#2F6DA8",
            mec="white",
            mew=0.35,
            label="Measured",
        )
        ax_top.loglog(
            qv,
            fv,
            "-",
            lw=2.0,
            color="#C44E18",
            label="Bayesian-MaxEnt fit",
        )
        ax_top.set_ylabel("Scattering intensity, I(q)")
        ax_top.set_title(format_state_label(row), pad=5.5)
        ax_top.grid(True, which="both", ls="-", lw=0.55, color="#E6E6E6")
        for spine in ax_top.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.95)

        ax_top.text(
            0.02,
            0.98,
            panel_id,
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=15.8,
            fontweight="bold",
        )
        ax_top.text(
            0.98,
            0.94,
            f"thickness = {row['thickness_um']:.1f} um",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=9.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
        )
        ax_top.legend(
            loc="upper right",
            bbox_to_anchor=(0.985, 0.82),
            frameon=True,
            facecolor="white",
            edgecolor="#D3D3D3",
            framealpha=0.92,
            borderpad=0.35,
            handlelength=2.1,
        )

        if sigv is not None and sigv.shape == fv.shape:
            sigma_safe = np.where(np.isfinite(sigv) & (sigv > 0.0), sigv, np.nan)
            residual = (iv - fv) / sigma_safe
            ylabel = r"(I-I$_{fit}$)/$\sigma$"
        else:
            residual = (iv - fv) / fv
            ylabel = r"(I-I$_{fit}$)/I$_{fit}$"

        finite_res = np.isfinite(residual)
        ax_bottom.plot(qv[finite_res], residual[finite_res], "-", lw=1.1, color="#575757")
        ax_bottom.axhline(0.0, color="#3A3A3A", lw=0.95, ls=":")
        ax_bottom.set_xscale("log")
        ax_bottom.set_xlabel(r"Scattering vector, q ($\mathrm{\AA}^{-1}$)")
        ax_bottom.set_ylabel(ylabel, fontsize=10.2)
        ax_bottom.grid(True, which="both", ls="-", lw=0.5, color="#EAEAEA")
        ax_bottom.tick_params(axis="y", labelsize=10.2)
        for spine in ax_bottom.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)

    fig.savefig(ROOT / "th6_raw_iq_summary.png", dpi=320, bbox_inches="tight")
    fig.savefig(ROOT / "th6_raw_iq_summary.svg", bbox_inches="tight")
    plt.close(fig)


def plot_psd_overlay(rows):
    focus_rows = build_focus_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)
    groups = split_groups(focus_rows)
    cmap_map = {0.0: plt.cm.Greens, 1.1: plt.cm.Reds}

    for ax, side_pressure in zip(axes, sorted(groups.keys())):
        group_rows = groups[side_pressure]
        cmap = cmap_map.get(side_pressure, plt.cm.Blues)
        color_positions = np.linspace(0.35, 0.9, len(group_rows))

        for color_pos, row in zip(color_positions, group_rows):
            radius, density = load_psd_curve(row)
            ax.plot(
                radius,
                density,
                lw=2.0,
                color=cmap(color_pos),
                label=f"axial = {row['axial_pressure_mpa']:g} MPa",
            )

        ax.set_xscale("log")
        ax.set_xlim(R_MIN_A, R_MAX_A)
        ax.set_xlabel("Radius (A)")
        ax.set_ylabel("dV / d(ln r)")
        ax.set_title(f"Side pressure = {side_pressure:g} MPa", fontsize=12, weight="bold")
        ax.grid(True, which="both", ls="--", lw=0.55, alpha=0.4)
        ax.legend(frameon=False, fontsize=8.5, loc="best")

    fig.suptitle(f"TH6 PSD overlay restricted to {WINDOW_LABEL}", fontsize=14, weight="bold")
    fig.savefig(ROOT / "th6_1nm_psd_overlay.png", dpi=240, bbox_inches="tight")
    fig.savefig(ROOT / "th6_1nm_psd_overlay.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.unicode_minus"] = False

    rows = load_summary(SUMMARY_CSV)
    focus_rows = build_focus_rows(rows)
    shared_rows = build_shared_axial_comparison(focus_rows)

    write_focus_summary(focus_rows, shared_rows)
    write_structure_factor_summary(rows)
    plot_metric_ratio_overview(focus_rows)
    plot_shared_comparison(shared_rows)
    plot_psd_change_overview(rows)
    plot_structure_factor_overview(rows)
    plot_psd_overlay(rows)
    plot_raw_iq_summary(rows)

    print(f"Plots written to: {ROOT}")
    print("Generated:")
    for name in (
        "th6_1nm_focus_summary.csv",
        "th6_1nm_shared_axial_comparison.csv",
        "th6_structure_factor_summary.csv",
        "th6_1nm_metric_ratio_overview.png",
        "th6_1nm_metric_ratio_overview.svg",
        "th6_1nm_shared_axial_comparison.png",
        "th6_1nm_shared_axial_comparison.svg",
        "th6_1nm_psd_change_overview.png",
        "th6_1nm_psd_change_overview.svg",
        "th6_structure_factor_overview.png",
        "th6_structure_factor_overview.svg",
        "th6_1nm_psd_overlay.png",
        "th6_1nm_psd_overlay.svg",
        "th6_raw_iq_summary.png",
        "th6_raw_iq_summary.svg",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
