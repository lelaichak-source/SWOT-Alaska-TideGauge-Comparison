"""
09_pipeline_comparisons.py
==========================
Figures that compare the comparisons with each other.

1. NTR pathway: standard NTR -> DAC-restored -> DAC-restored and
   bias-corrected, side by side for each station.
       ntr_pathway_twl_stations.png   Anchorage, Nikiski, Kodiak (pairs with the
                                      TWL bias-corrected figure from script 07)
       ntr_pathway_all_stations.png   all eight stations
2. Regression slope with 95% CI for two comparisons at each station:
       slope_ntr_vs_dac.png   standard vs. DAC-restored NTR (all stations);
                              a grey line joins each station's pair because DAC
                              restoration is a step applied to the same data
       slope_dac_vs_twl.png   DAC-restored NTR vs. TWL (TWL stations); two
                              independent frameworks, so no connecting line
   If any slope is 2 or more, the y-axis is split so the remaining stations
   are still readable.

Outputs:
    OUTPUT_DIR/figures/comparison/*.png
    OUTPUT_DIR/stats/slope_ntr_vs_dac.csv, slope_dac_vs_twl.csv

Run:
    python 09_pipeline_comparisons.py
"""

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from swot_utils import (PIPELINES, TWL_STATION_ORDER, agreement_stats,
                        fig_dir, keep_box_clear, load_pairs, region_colors_for,
                        regression_with_ci, station_color, stats_path)

BREAK_AT = 2.0      # split the slope axis if any slope is at or above this


# =============================================================================
# 1. NTR pathway
# =============================================================================
def pathway_panel(ax, x, y, color, s, title, corrected=False, small=False):
    ax.scatter(x, y, c=color, alpha=0.65, s=55, edgecolors="white", linewidths=0.4, zorder=3)
    if len(x) >= 3:
        slope, intercept, *_ = stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, slope * xl + intercept, color=color, lw=2,
                linestyle="--" if corrected else "-", alpha=0.9)
    lim = max(abs(np.concatenate([x, y])).max() * 1.15, 0.1)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, lw=0.4)
    box = ax.text(0.05, 0.95, f"r = {s['r']:.3f}\nRMSE = {s['rmse']:.3f} m\n"
                              f"Bias = {s['bias']:+.3f} m\nn = {int(s['n'])}",
                  transform=ax.transAxes, fontsize=8 if small else 8.5, va="top",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88,
                            ec=color if corrected else "#666666", lw=1.2))
    keep_box_clear(ax, box, x, y)
    ax.set_xlabel("NOAA NTR (m)", fontsize=8.5 if small else 9)
    ax.set_ylabel("SWOT SSHA (m)", fontsize=8.5 if small else 9)
    ax.set_title(title, fontsize=9 if small else 9.5, fontweight="bold",
                 color=color if corrected else "#333333")
    if small:
        ax.tick_params(labelsize=7.5)


def rounded(x, y):
    s = agreement_stats(x, y)
    return {k: (round(v, 4) if k != "n" else v) for k, v in s.items()}


def pathway_station(fig, gs, row, col0, station, ntr, dac, small):
    """Draw the three pathway panels for one station starting at grid column col0."""
    color = station_color(station)
    sub = ntr[ntr["station"] == station]
    pathway_panel(fig.add_subplot(gs[row, col0]), sub["ntr_m"].values,
                  sub["ssha_median_m"].values, color,
                  rounded(sub["ntr_m"], sub["ssha_median_m"]),
                  f"{station} — NTR (standard)", small=small)
    sub = dac[dac["station"] == station]
    x, y = sub["ntr_m"].values, sub["ssha_median_m"].values
    pathway_panel(fig.add_subplot(gs[row, col0 + 1]), x, y, color, rounded(x, y),
                  f"{station} — DAC-restored", small=small)
    bias = float(np.mean(y - x))
    pathway_panel(fig.add_subplot(gs[row, col0 + 2]), x, y - bias, color, rounded(x, y - bias),
                  f"{station} — DAC-restored, Bias-Corrected\n(bias removed: {bias:+.3f} m)",
                  corrected=True, small=small)


def ntr_pathway(ntr, dac, out):
    # Three TWL stations, one per row.
    n = len(TWL_STATION_ORDER)
    fig = plt.figure(figsize=(15, 5 * n))
    gs = gridspec.GridSpec(n, 3, figure=fig, hspace=0.42, wspace=0.32, top=0.92, bottom=0.05)
    for row, station in enumerate(TWL_STATION_ORDER):
        pathway_station(fig, gs, row, 0, station, ntr, dac, small=False)
    fig.suptitle("NTR Pathway: Standard -> DAC-Restored -> Bias-Corrected\n"
                 "Anchorage, Nikiski, Kodiak  |  Ice-Free Periods 2023-2026",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(out / "ntr_pathway_twl_stations.png", dpi=300, bbox_inches="tight")
    plt.close()

    # All stations: Cook Inlet on the left, Prince William Sound on the right
    # (row-aligned), and the two Kodiak stations on the bottom row.
    pairs = [("Anchorage", "Cordova"), ("Nikiski", "Seward"),
             ("Seldovia", "Valdez"), ("Alitak", "Kodiak")]
    fig = plt.figure(figsize=(28, 4.8 * len(pairs)))
    gs = gridspec.GridSpec(len(pairs), 6, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.94, bottom=0.02)
    for row, pair in enumerate(pairs):
        for block, station in enumerate(pair):
            if station in ntr["station"].values:
                pathway_station(fig, gs, row, 3 * block, station, ntr, dac, small=True)
    fig.suptitle("NTR Pathway: Standard -> DAC-Restored -> Bias-Corrected — All Stations\n"
                 "Cook Inlet (left)  |  Prince William Sound (right)  |  Kodiak (bottom row)\n"
                 "Southcentral Alaska, Ice-Free Periods 2023-2026",
                 fontsize=15, fontweight="bold", y=0.985)
    plt.savefig(out / "ntr_pathway_all_stations.png", dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 2. Slope comparisons
# =============================================================================
def slope_table(key_a, key_b, stations):
    """Slope, 95% CI, r and p at each station for two comparisons."""
    dfs = {k: load_pairs(k) for k in (key_a, key_b)}
    rows = []
    for station in stations:
        row = {"station": station, "region": dfs[key_a].loc[
            dfs[key_a]["station"] == station, "region"].iloc[0]}
        ok = True
        for k, df in dfs.items():
            p = PIPELINES[k]
            sub = df[df["station"] == station]
            reg = regression_with_ci(sub[p["x"]], sub[p["y"]])
            if reg is None:
                ok = False
                break
            row.update({f"n_{k}": reg["n"], f"slope_{k}": round(reg["slope"], 4),
                        f"ci95_{k}": round(reg["ci95"], 4), f"r_{k}": round(reg["r"], 4),
                        f"p_{k}": round(reg["p"], 4)})
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def slope_figure(tbl, key_a, key_b, style_a, style_b, connect, title, ylabel, out_png,
                 figsize, label_rotation):
    """Dodged slope +/- 95% CI for two comparisons per station, split axis if needed."""
    x = np.arange(len(tbl))
    dodge = 0.15
    use_break = bool(((tbl[f"slope_{key_a}"] >= BREAK_AT) | (tbl[f"slope_{key_b}"] >= BREAK_AT)).any())
    if use_break:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                             gridspec_kw={"height_ratios": [1, 3], "hspace": 0.08})
        axes = (ax_top, ax_bot)
    else:
        fig, ax_bot = plt.subplots(figsize=(figsize[0], figsize[1] - 1))
        ax_top, axes = ax_bot, (ax_bot,)

    for ax in axes:
        if connect:
            for i, row in tbl.iterrows():
                ax.plot([x[i] - dodge, x[i] + dodge], [row[f"slope_{key_a}"], row[f"slope_{key_b}"]],
                        color="#cccccc", lw=1.2, zorder=1)
        for i, row in tbl.iterrows():
            c = station_color(row["station"])
            for k, style, offset in [(key_a, style_a, -dodge), (key_b, style_b, dodge)]:
                kwargs = dict(fmt=style["marker"], ecolor=c, markersize=style["size"], capsize=4,
                              lw=1.5, zorder=3, markeredgecolor=c,
                              markerfacecolor="none" if style["hollow"] else c,
                              markeredgewidth=2 if style["hollow"] else 1)
                _, _, (bars,) = ax.errorbar(x[i] + offset, row[f"slope_{k}"],
                                            yerr=row[f"ci95_{k}"], **kwargs)
                bars.set_linestyle(style["line"])
        ax.axhline(1.0, color="black", lw=1.5, linestyle="--", alpha=0.7, zorder=2)
        ax.grid(axis="y", alpha=0.3)

    # Bottom panel: everything below the break (whiskers capped at the break).
    lows, highs = [], []
    for k in (key_a, key_b):
        keep = tbl[f"slope_{k}"] < BREAK_AT if use_break else slice(None)
        lows.append((tbl.loc[keep, f"slope_{k}"] - tbl.loc[keep, f"ci95_{k}"]).min())
        highs.append((tbl.loc[keep, f"slope_{k}"] + tbl.loc[keep, f"ci95_{k}"]).max())
    lo, hi = min(lows), max(highs)
    if use_break:
        hi = min(hi, BREAK_AT)
    margin = 0.06 * (hi - lo)
    ax_bot.set_ylim(lo - margin, hi + margin)

    if use_break:
        top = max((tbl[f"slope_{k}"] + tbl[f"ci95_{k}"]).max() for k in (key_a, key_b)) + 0.2
        ax_top.set_ylim(BREAK_AT, top)
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        d = 0.015                                         # diagonal break marks
        kw = dict(transform=ax_top.transAxes, color="k", clip_on=False, lw=1)
        ax_top.plot((-d, +d), (-d, +d), **kw)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
        kw.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kw)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(tbl["station"], rotation=label_rotation,
                           ha="right" if label_rotation else "center", fontsize=10)
    ax_bot.set_ylabel(ylabel, fontsize=11)

    handles = [mlines.Line2D([], [], marker=s["marker"], linestyle=s["line"], color="#555555",
                             markersize=8, markerfacecolor="none" if s["hollow"] else "#555555",
                             markeredgecolor="#555555", markeredgewidth=2 if s["hollow"] else 1,
                             label=PIPELINES[k]["title"])
               for k, s in [(key_a, style_a), (key_b, style_b)]]
    handles.append(mlines.Line2D([], [], color="black", lw=1.5, ls="--", alpha=0.7,
                                 label="Ideal slope = 1.0"))
    leg = ax_top.legend(handles=handles, fontsize=9, loc="upper right", title="Comparison",
                        title_fontsize=9)
    ax_top.add_artist(leg)
    ax_top.legend(handles=[mpatches.Patch(facecolor=c, label=r)
                           for r, c in region_colors_for(tbl["station"]).items()],
                  fontsize=9, loc="lower right" if use_break else "upper center",
                  title="Region", title_fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    out = fig_dir("comparison")
    ntr, dac = load_pairs("ntr"), load_pairs("dac")
    ntr_pathway(ntr, dac, out)

    hollow_dashed = dict(marker="o", size=8, hollow=True, line="--")
    filled_solid = dict(marker="o", size=8, hollow=False, line="-")
    filled_square = dict(marker="s", size=9, hollow=False, line=":")

    stations = [s for s in PIPELINES["ntr"]["stations"] if s in ntr["station"].values]
    tbl = slope_table("ntr", "dac", stations)
    tbl["slope_shift"] = (tbl["slope_dac"] - tbl["slope_ntr"]).round(4)
    tbl.to_csv(stats_path("slope_ntr_vs_dac.csv"), index=False)
    slope_figure(tbl, "ntr", "dac", hollow_dashed, filled_solid, connect=True,
                 title="Regression Slope: Standard NTR vs. DAC-Restored NTR\n"
                       "Southcentral Alaska, Ice-Free Periods 2023-2026",
                 ylabel="Regression slope (SWOT SSHA ~ NOAA NTR)",
                 out_png=out / "slope_ntr_vs_dac.png", figsize=(13, 8), label_rotation=25)

    tbl = slope_table("dac", "twl", TWL_STATION_ORDER)
    tbl.to_csv(stats_path("slope_dac_vs_twl.csv"), index=False)
    slope_figure(tbl, "dac", "twl", dict(filled_solid, size=9), filled_square, connect=False,
                 title="Regression Slope: DAC-Restored NTR vs. TWL\n"
                       "Anchorage, Nikiski, Kodiak  |  Ice-Free Periods 2023-2026",
                 ylabel="Regression slope", out_png=out / "slope_dac_vs_twl.png",
                 figsize=(8, 8), label_rotation=0)

    print(pd.read_csv(stats_path("slope_ntr_vs_dac.csv"))
          [["station", "slope_ntr", "slope_dac", "slope_shift"]].to_string(index=False))
    print(pd.read_csv(stats_path("slope_dac_vs_twl.csv"))
          [["station", "slope_dac", "slope_twl"]].to_string(index=False))


if __name__ == "__main__":
    main()
