"""
05_figures_agreement.py
=======================
Agreement figures for each comparison.

For the NTR and DAC-restored comparisons (all eight stations):
    scatter_all_stations.png      SWOT vs. gauge, one panel per station
    scatter_<station>.png         the same, one station per figure
    timeseries_<station>.png      SWOT and gauge values through time
    difference_<station>.png      SWOT - gauge through time, with mean bias
    station_rmse_r.png            RMSE and Pearson r by station
    slope_summary.png             regression slope, deviation from 1, r, slope vs r

For the TWL comparison (Anchorage, Nikiski, Kodiak):
    the scatter, time-series and difference figures above, plus
    twl_vs_ntr_comparison.png     NTR and TWL scatter side by side per station

Each scatter shows the OLS regression line and the 1:1 line; the stats box
gives r, RMSE, bias and n from OUTPUT_DIR/stats/stats_by_station_<key>.csv.

Inputs:  matched pairs (script 03) and statistics (script 04)
Outputs: OUTPUT_DIR/figures/<ntr|dac|twl>/

Run:
    python 05_figures_agreement.py
"""

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from swot_utils import (PIPELINES, REGION_COLORS, fig_dir, keep_box_clear, load_pairs,
                        station_color, stats_path, title_tag)

PERIOD = "Ice-Free Periods 2023–2026"


def load_stats(key):
    return pd.read_csv(stats_path(f"stats_by_station_{key}.csv"))


def square_limits(x, y, key, scale):
    """Axis limits for a 1:1 scatter: symmetric about 0 for anomalies (NTR/DAC),
    data range for total water level (TWL)."""
    if key == "twl":
        return min(x.min(), y.min()) * 1.02, max(x.max(), y.max()) * 1.02
    lim = max(abs(np.concatenate([x, y])).max() * scale, 0.1)
    return -lim, lim


def scatter_panel(ax, sub, key, row, big=False):
    """One SWOT-vs-gauge scatter with regression and 1:1 lines and a stats box."""
    p = PIPELINES[key]
    x, y = sub[p["x"]].values, sub[p["y"]].values
    color = station_color(sub["station"].iloc[0])
    ax.scatter(x, y, c=color, alpha=0.75 if big else 0.7, edgecolors="white",
               linewidths=0.6 if big else 0.5, s=80 if big else 60, zorder=3)
    if len(x) >= 3:
        slope, intercept, *_ = stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, slope * xl + intercept, color=color, linewidth=2.5 if big else 2,
                alpha=1 if big else 0.9,
                label=f"Regression (slope={slope:.2f})" if big else None)
    lo, hi = square_limits(x, y, key, 1.15 if big else 1.1)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2 if big else 1, alpha=0.5 if big else 0.4,
            label="1:1 line" if big else None)
    if key != "twl":
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if big:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.5)

    if big:
        text = (f"r = {row['r']:.3f}  (p = {row['p_value']:.3f})\nRMSE = {row['rmse_m']:.3f} m\n"
                f"Bias = {row['bias_m']:.3f} m\nn = {int(row['n'])}")
        if key == "twl":
            text += f"\nMSS - MSL offset = {row['mean_datum_offset_m']:.3f} m"
        box = dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85,
                   edgecolor=color, linewidth=1.5)
    else:
        text = (f"r = {row['r']:.3f}\nRMSE = {row['rmse_m']:.3f} m\n"
                f"Bias = {row['bias_m']:.3f} m\nn = {row['n']}")
        box = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color)
    t = ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10 if big else 8.5,
                verticalalignment="top", bbox=box)
    if key != "twl":                    # anomaly panels: move the box off the points
        keep_box_clear(ax, t, x, y)
    if not big:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linewidth=0.5)


def scatter_figures(df, st, key, out):
    p = PIPELINES[key]
    stations = [s for s in p["stations"] if s in df["station"].values]

    # All stations in one figure: 3 x 3 grid (NTR/DAC) or 1 x 3 (TWL).
    if key == "twl":
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[len(stations):]:
        ax.set_visible(False)
    for ax, station in zip(axes, stations):
        scatter_panel(ax, df[df["station"] == station], key,
                      st[st["station"] == station].iloc[0])
        ax.set_title(station, fontweight="bold", fontsize=11, color=station_color(station))
        ax.set_xlabel(p["x_label"], fontsize=9)
        ax.set_ylabel(p["y_label"], fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle(f"{p['swot_name']} vs. {p['gauge_name']} — All Matched Pairs{title_tag(key)}\n"
                 f"Southcentral Alaska, {PERIOD}", fontsize=13, fontweight="bold",
                 y=1.01 if key != "twl" else None)
    if key != "twl":
        handles = [mpatches.Patch(facecolor=c, label=r) for r, c in REGION_COLORS.items()]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04),
                   fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out / "scatter_all_stations.png", dpi=300, bbox_inches="tight")
    plt.close()

    # One figure per station.
    for station in stations:
        sub = df[df["station"] == station]
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter_panel(ax, sub, key, st[st["station"] == station].iloc[0], big=True)
        ax.set_xlabel(p["x_label"], fontsize=11)
        ax.set_ylabel(p["y_label"], fontsize=11)
        ax.set_title(f"{station}  |  {sub['region'].iloc[0]}{title_tag(key, 'bracket')}",
                     fontsize=12, fontweight="bold", color=station_color(station))
        ax.legend(fontsize=9, loc="lower right")
        plt.tight_layout()
        plt.savefig(out / f"scatter_{station.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()


def time_figures(df, st, key, out):
    """Time series of SWOT and gauge values, and of their difference."""
    p = PIPELINES[key]
    for station in [s for s in p["stations"] if s in df["station"].values]:
        sub = df[df["station"] == station].sort_values("overpass_time")
        color, region = station_color(station), sub["region"].iloc[0]
        row = st[st["station"] == station].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(sub["overpass_time"], sub[p["x"]], "o--", color="gray", linewidth=1,
                markersize=5, alpha=0.7, label=p["gauge_name"])
        ax.plot(sub["overpass_time"], sub[p["y"]], "o-", color=color, linewidth=1.5,
                markersize=6, alpha=0.9, label=p["swot_name"] + title_tag(key))
        if key != "twl":
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Date (UTC)", fontsize=10)
        ax.set_ylabel("Total water level (m)" if key == "twl" else "Water level anomaly (m)",
                      fontsize=10)
        ax.set_title(f"{station}  |  {region} — {p['swot_name']} vs. {p['gauge_name']}{title_tag(key)}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out / f"timeseries_{station.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(sub["overpass_time"], sub[p["y"]].values - sub[p["x"]].values,
               color=color, alpha=0.7, width=5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(row["bias_m"], color=color, linewidth=1.5, linestyle="--",
                   label=f"Mean bias = {row['bias_m']:.3f} m")
        ax.set_xlabel("Date (UTC)", fontsize=10)
        ax.set_ylabel(f"{p['diff_label']} (m)", fontsize=10)
        ax.set_title(f"{station}  |  {region} — Difference ({p['diff_label']})",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(out / f"difference_{station.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()


def station_bar_figure(st, key, out):
    """RMSE and Pearson r by station (NTR and DAC-restored)."""
    p = PIPELINES[key]
    s = st.set_index("station").loc[[x for x in p["stations"] if x in st["station"].values]]
    colors = [REGION_COLORS[r] for r in s["region"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(s.index, s["rmse_m"], color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax1.set_ylabel("RMSE (m)", fontsize=11)
    ax1.set_title(f"RMSE by Station{title_tag(key)}", fontsize=12, fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, s["rmse_m"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    bars = ax2.bar(s.index, s["r"], color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax2.set_ylabel("Pearson r", fontsize=11)
    ax2.set_title(f"Correlation (r) by Station{title_tag(key)}", fontsize=12, fontweight="bold")
    ax2.tick_params(axis="x", rotation=30)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylim(-1, 1)
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, s["r"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + (0.03 if val >= 0 else -0.07),
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    handles = [mpatches.Patch(facecolor=c, label=r) for r, c in REGION_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle(f"SWOT Agreement with NOAA Tide Gauges by Station — All Matched Pairs{title_tag(key)}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out / "station_rmse_r.png", dpi=300, bbox_inches="tight")
    plt.close()


def slope_summary_figure(key, out):
    """Four panels: slope with 95% CI, slope - 1, Pearson r, and slope vs. r."""
    sl = pd.read_csv(stats_path(f"regression_slopes_{key}.csv"))
    colors = [REGION_COLORS[r] for r in sl["region"]]
    x = np.arange(len(sl))
    tag = title_tag(key, "bracket")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"SWOT-NOAA Regression Slope Analysis — All Matched Pairs{title_tag(key)}\n"
                 f"Southcentral Alaska, {PERIOD}", fontsize=13, fontweight="bold")

    def station_ticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(sl["station"], rotation=25, ha="right", fontsize=9)

    # Panel 1: slope with 95% CI; labels sit just above each whisker.
    ax = axes[0, 0]
    bars = ax.bar(x, sl["slope"], color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, sl["slope"], yerr=sl["slope_ci95"], fmt="none", color="#333333",
                capsize=5, lw=1.5, zorder=5)
    ax.axhline(1.0, color="black", lw=1.8, linestyle="--", label="Ideal slope = 1.0", zorder=4)
    ax.axhline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax.set_ylabel("Regression Slope", fontsize=11)
    ax.set_title(f"Regression Slope (SWOT ~ gauge){tag}\nError bars = 95% CI",
                 fontsize=10, fontweight="bold")
    station_ticks(ax)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val, ci in zip(bars, sl["slope"], sl["slope_ci95"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + ci + 0.06, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(ax.get_ylim()[0], (sl["slope"] + sl["slope_ci95"]).max() * 1.18)

    # Panel 2: deviation from the ideal slope of 1.
    ax = axes[0, 1]
    dev = sl["slope_dev_from_1"].values
    bars = ax.bar(x, dev, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, dev, yerr=sl["slope_ci95"], fmt="none", color="#333333", capsize=5,
                lw=1.5, zorder=5)
    ax.axhline(0.0, color="black", lw=1.8, linestyle="--", label="Ideal (slope = 1)", zorder=4)
    ax.set_ylabel("Slope - 1.0  (deviation from ideal)", fontsize=11)
    ax.set_title(f"Slope Deviation from Ideal (1.0){tag}\n"
                 "Positive = over-response, Negative = under-response",
                 fontsize=10, fontweight="bold")
    station_ticks(ax)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val, ci in zip(bars, dev, sl["slope_ci95"]):
        above = val >= 0
        ax.text(bar.get_x() + bar.get_width() / 2, val + ci + 0.08 if above else val - ci - 0.08,
                f"{val:+.2f}", ha="center", va="bottom" if above else "top", fontsize=8)
    ax.set_ylim((dev - sl["slope_ci95"]).min() - 0.40, (dev + sl["slope_ci95"]).max() + 0.40)

    # Panel 3: Pearson r.
    ax = axes[1, 0]
    bars = ax.bar(x, sl["r"], color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title(f"Pearson r by Station{tag}", fontsize=10, fontweight="bold")
    station_ticks(ax)
    ax.set_ylim(-0.1, sl["r"].max() * 1.22)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, sl["r"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012 if val >= 0 else val - 0.035,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Panel 4: slope vs. r. Labels for most stations are stacked in a column on
    # the left with leader lines; stations with slope > 2 are labelled in place.
    ax = axes[1, 1]
    for _, row in sl.iterrows():
        ax.scatter(row["r"], row["slope"], c=REGION_COLORS[row["region"]], s=120,
                   edgecolors="white", linewidths=0.8, zorder=3)
    ax.axhline(1.0, color="black", lw=1.2, linestyle="--", alpha=0.6, label="Ideal slope")
    ax.axvline(0.0, color="gray", lw=0.6, alpha=0.5)
    ax.set_xlabel("Pearson r", fontsize=11)
    ax.set_ylabel("Regression Slope", fontsize=11)
    ax.set_title(f"Slope vs. r{tag}\n(each point = one station)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.0, sl["r"].max() + 0.12)
    ax.set_ylim(sl["slope"].min() - 0.25, sl["slope"].max() + 0.80)
    label_box = dict(boxstyle="square,pad=0.12", fc="white", alpha=0.92, ec="none")
    for _, row in sl[sl["slope"] > 2.0].iterrows():
        c = REGION_COLORS[row["region"]]
        ax.annotate(row["station"], xy=(row["r"], row["slope"]), xytext=(-14, 8),
                    textcoords="offset points", fontsize=8.5, color=c, ha="right", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.4, alpha=0.6), bbox=label_box)
    cluster = sl[sl["slope"] <= 2.0].sort_values("slope")
    y0 = sl["slope"].min() - 0.10
    for i, (_, row) in enumerate(cluster.iterrows()):
        c = REGION_COLORS[row["region"]]
        ax.annotate(row["station"], xy=(row["r"], row["slope"]), xytext=(0.03, y0 + i * 0.20),
                    fontsize=8.5, color=c, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, alpha=0.6), bbox=label_box)

    handles = [mpatches.Patch(facecolor=c, label=r) for r, c in REGION_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "slope_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


def ntr_vs_twl_figure(out):
    """NTR scatter (left) and TWL scatter (right) for each TWL station."""
    df_ntr, st_ntr = load_pairs("ntr"), load_stats("ntr").set_index("station")
    df_twl, st_twl = load_pairs("twl"), load_stats("twl").set_index("station")
    stations = [s for s in PIPELINES["twl"]["stations"] if s in df_twl["station"].values]
    fig = plt.figure(figsize=(14, 5 * len(stations)))
    gs = gridspec.GridSpec(len(stations), 2, figure=fig, hspace=0.45, wspace=0.35)

    for i, station in enumerate(stations):
        color, region = station_color(station), config.STATIONS[station]["region"]
        for col, key, df, st in [(0, "ntr", df_ntr, st_ntr), (1, "twl", df_twl, st_twl)]:
            p = PIPELINES[key]
            ax = fig.add_subplot(gs[i, col])
            sub = df[df["station"] == station]
            x, y = sub[p["x"]].values, sub[p["y"]].values
            ax.scatter(x, y, c=color, alpha=0.6, s=55, edgecolors="white", linewidths=0.5, zorder=3)
            if len(x) >= 3:
                sl, ic, *_ = stats.linregress(x, y)
                xl = np.linspace(x.min(), x.max(), 100)
                ax.plot(xl, sl * xl + ic, color=color, lw=2, alpha=0.9)
            lo, hi = square_limits(x, y, key, 1.1)
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.4)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            s = st.loc[station]
            ax.text(0.05, 0.95, f"r = {s['r']:.3f}\nRMSE = {s['rmse_m']:.3f} m\n"
                                f"Bias = {s['bias_m']:.3f} m\nn = {int(s['n'])}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85,
                              ec="#777777" if key == "ntr" else color))
            ax.set_xlabel(p["x_label"], fontsize=9)
            ax.set_ylabel(p["y_label"], fontsize=9)
            ax.set_title(f"{station} ({region}) — {'NTR' if key == 'ntr' else 'TWL'}",
                         fontsize=10, fontweight="bold",
                         color="#555555" if key == "ntr" else color)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.25)

    fig.suptitle(f"NTR vs. TWL Comparison — Side by Side\nAnchorage, Nikiski, Kodiak  |  {PERIOD}",
                 fontsize=13, fontweight="bold")
    plt.savefig(out / "twl_vs_ntr_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    for key in PIPELINES:
        try:
            df, st = load_pairs(key), load_stats(key)
        except FileNotFoundError as e:
            print(f"[skip] {key}: {e}")
            continue
        out = fig_dir(key)
        print(f"{PIPELINES[key]['title']}: writing figures to {out}")
        scatter_figures(df, st, key, out)
        time_figures(df, st, key, out)
        if key in ("ntr", "dac"):
            station_bar_figure(st, key, out)
            slope_summary_figure(key, out)
        else:
            ntr_vs_twl_figure(out)
    print("Done.")


if __name__ == "__main__":
    main()
