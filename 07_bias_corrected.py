"""
07_bias_corrected.py
====================
Bias-corrected comparisons for each comparison (NTR, DAC-restored, TWL).

Each station's mean bias, mean(SWOT - gauge), is removed from that station's
SWOT values. Pearson r is unchanged by this; RMSE drops to the scatter about
the mean, showing how much of the RMSE is a constant offset.

Per station:  original and bias-corrected scatter side by side, plus a panel
              with all stations, and a before/after statistics table.
By region:    all stations' bias-corrected pairs pooled by region on one
              scatter (colour = region, marker = station), with a regression
              line and statistics (n, r, slope +/- 95% CI, RMSE) per region
              and for all stations together.

Outputs:
    OUTPUT_DIR/figures/<key>/bias_corrected_<station>.png
    OUTPUT_DIR/figures/<key>/bias_corrected_all_stations.png
    OUTPUT_DIR/figures/<key>/regional_bias_corrected.png
    OUTPUT_DIR/stats/bias_correction_<key>.csv
    OUTPUT_DIR/stats/regional_bias_corrected_<key>.csv

Run:
    python 07_bias_corrected.py
"""

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from swot_utils import (PIPELINES, REGION_COLORS, REGION_ORDER, agreement_stats, fig_dir,
                        keep_box_clear, load_pairs, regression_with_ci,
                        station_bias_corrected, station_color, stats_path, title_tag)

# Marker per station on the regional plot (shape = station, colour = region).
STATION_MARKERS = {"Anchorage": "o", "Nikiski": "s", "Seldovia": "^",
                   "Seward": "o", "Valdez": "s", "Cordova": "^",
                   "Kodiak": "o", "Alitak": "s"}


def rounded_stats(x, y):
    s = agreement_stats(x, y)
    return {k: (round(v, 4) if k != "n" else v) for k, v in s.items()}


def draw_scatter(ax, x, y, color, title, key, corrected=False):
    """Scatter of SWOT vs. gauge with regression and 1:1 lines and a stats box."""
    p = PIPELINES[key]
    s = rounded_stats(x, y)
    ax.scatter(x, y, c=color, alpha=0.68, s=65, edgecolors="white", linewidths=0.5, zorder=3)
    if s["n"] >= 3:
        slope, intercept, *_ = stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 200)
        ax.plot(xl, slope * xl + intercept, color=color, lw=2.2,
                linestyle="--" if corrected else "-", alpha=0.9,
                label=f"Regression (slope={slope:.2f})")

    if key == "twl":            # absolute heights: use the data range
        lo, hi = min(x.min(), y.min()) * 1.02, max(x.max(), y.max()) * 1.02
    else:                       # anomalies: symmetric about zero
        hi = max(abs(np.concatenate([x, y])).max() * 1.18, 0.15)
        lo = -hi
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, alpha=0.4, label="1:1 line")
    if key != "twl":
        ax.axhline(0, color="gray", lw=0.5, alpha=0.35)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.35)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, lw=0.45)

    box = ax.text(0.04, 0.96, f"r = {s['r']:.3f}  (p = {s['p']:.3f})\nRMSE = {s['rmse']:.3f} m\n"
                              f"Bias = {s['bias']:+.3f} m\nn = {s['n']}",
                  transform=ax.transAxes, fontsize=9, va="top",
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88,
                            ec=color if corrected else "#444444", lw=1.4))
    keep_box_clear(ax, box, x, y)
    ax.set_title(title, fontsize=10, fontweight="bold", color=color if corrected else "#222222")
    ax.set_xlabel(p["x_label"], fontsize=10)
    ax.set_ylabel(p["y_label"], fontsize=10)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    return s


def per_station(key):
    p = PIPELINES[key]
    df = load_pairs(key)
    out = fig_dir(key)
    stations = [s for s in p["stations"] if s in df["station"].values]
    name = p["swot_name"] + title_tag(key)
    rows = []

    for station in stations:
        sub = df[df["station"] == station]
        color, region = station_color(station), sub["region"].iloc[0]
        x, y = sub[p["x"]].values, sub[p["y"]].values
        bias = float(np.mean(y - x))
        y_corr = y - bias
        s0, s1 = rounded_stats(x, y), rounded_stats(x, y_corr)
        rows.append({
            "station": station, "region": region, "n": s0["n"], "r": s0["r"],
            "bias_orig_m": s0["bias"], "rmse_orig_m": s0["rmse"], "std_diff_orig": s0["std_diff"],
            "bias_corr_m": round(s1["bias"], 4), "rmse_corr_m": s1["rmse"],
            "std_diff_corr": s1["std_diff"],
            "rmse_reduction_m": round(s0["rmse"] - s1["rmse"], 4),
            "rmse_reduction_pct": round(100 * (s0["rmse"] - s1["rmse"]) / s0["rmse"], 1),
        })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        draw_scatter(ax1, x, y, color, f"{station} — Original {name}", key)
        draw_scatter(ax2, x, y_corr, color, f"{station} — Bias-Corrected {name}", key,
                     corrected=True)
        d_rmse = s1["rmse"] - s0["rmse"]
        fig.text(0.5, 0.01, f"Bias removed: {bias:+.3f} m   |   DRMSE = {d_rmse:+.3f} m "
                            f"({100 * abs(d_rmse) / s0['rmse']:.1f}% reduction)   |   "
                            f"r unchanged = {s0['r']:.3f}",
                 ha="center", fontsize=9.5, color="#333333",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc", lw=1))
        fig.suptitle(f"{station}  |  {region}{title_tag(key, 'bracket')}\n"
                     f"{p['swot_name']} vs. {p['gauge_name']} — Original and Bias-Corrected",
                     fontsize=12, fontweight="bold", color=color)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(out / f"bias_corrected_{station.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # All stations: original (left) and bias-corrected (right), one row each.
    n = len(stations)
    fig = plt.figure(figsize=(11 if key == "twl" else 14, 5.5 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.45, wspace=0.32,
                           top=0.94 if key == "twl" else 0.97,
                           bottom=0.02 if key == "twl" else 0.01)
    for i, station in enumerate(stations):
        sub = df[df["station"] == station]
        color, region = station_color(station), sub["region"].iloc[0]
        x, y = sub[p["x"]].values, sub[p["y"]].values
        bias = float(np.mean(y - x))
        ax_l, ax_r = fig.add_subplot(gs[i, 0]), fig.add_subplot(gs[i, 1])
        draw_scatter(ax_l, x, y, color, f"{station} ({region}) — Original", key)
        draw_scatter(ax_r, x, y - bias, color,
                     f"{station} — Bias-Corrected  [{bias:+.3f} m removed]", key, corrected=True)
        for ax in (ax_l, ax_r):
            ax.tick_params(labelsize=8)
    where = ("Anchorage, Nikiski, Kodiak  |  Ice-Free Periods 2023-2026" if key == "twl" else
             "All Matched Pairs  |  Southcentral Alaska, Ice-Free Periods 2023-2026")
    fig.suptitle(f"All Stations — Original vs. Bias-Corrected {name}\n{where}",
                 fontsize=13, fontweight="bold", y=0.985 if key == "twl" else 0.99)
    plt.savefig(out / "bias_corrected_all_stations.png", dpi=300, bbox_inches="tight")
    plt.close()

    res = pd.DataFrame(rows)
    res.to_csv(stats_path(f"bias_correction_{key}.csv"), index=False)
    print(f"\n{p['title']}: bias correction (r unchanged; RMSE after = SD of differences)")
    for _, r in res.iterrows():
        print(f"  {r['station']:<10} bias {r['bias_orig_m']:+.3f} m   RMSE {r['rmse_orig_m']:.3f} "
              f"-> {r['rmse_corr_m']:.3f} m ({r['rmse_reduction_pct']:.1f}% lower)")


def region_stats(x, y):
    reg = regression_with_ci(x, y)
    if reg is None:
        return dict(n=len(x), r=np.nan, p=np.nan, rmse=np.nan, slope=np.nan,
                    slope_ci95=np.nan, intercept=np.nan)
    return dict(n=reg["n"], r=reg["r"], p=reg["p"],
                rmse=float(np.sqrt(np.mean((np.asarray(y) - np.asarray(x)) ** 2))),
                slope=reg["slope"], slope_ci95=reg["ci95"], intercept=reg["intercept"])


def regional(key):
    """All stations' bias-corrected pairs, pooled and coloured by region."""
    p = PIPELINES[key]
    df = load_pairs(key)
    df["y_corr"], df["station_bias_m"] = station_bias_corrected(df, p["x"], p["y"])
    fig, ax = plt.subplots(figsize=(9, 9))
    rows = []
    for region in REGION_ORDER:
        sub_r = df[df["region"] == region]
        if sub_r.empty:
            continue
        color = REGION_COLORS[region]
        for station, sub_s in sub_r.groupby("station"):
            ax.scatter(sub_s[p["x"]], sub_s["y_corr"], c=color,
                       marker=STATION_MARKERS.get(station, "o"), s=42, alpha=0.6,
                       edgecolors="white", linewidths=0.4, zorder=3)
        s = region_stats(sub_r[p["x"]], sub_r["y_corr"])
        s.update(region=region, stations=", ".join(sorted(sub_r["station"].unique())))
        rows.append(s)
        if s["n"] >= 3:
            xl = np.linspace(sub_r[p["x"]].min(), sub_r[p["x"]].max(), 100)
            ax.plot(xl, s["slope"] * xl + s["intercept"], color=color, lw=2.4, zorder=4)

    allv = np.concatenate([df[p["x"]].to_numpy(), df["y_corr"].to_numpy()])
    if key == "twl":
        pad = 0.05 * (allv.max() - allv.min())
        lo, hi = allv.min() - pad, allv.max() + pad
    else:
        hi = max(np.abs(allv).max() * 1.08, 0.1)
        lo = -hi
        ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, alpha=0.5, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, lw=0.4)

    s_all = region_stats(df[p["x"]], df["y_corr"])
    width = 13 if key == "twl" else 21
    dp = 3 if key == "twl" else 2      # TWL r and slopes sit close to 1: show 3 decimals
    lines = [f"{s['region']:<{width}} n={s['n']:>3}  r={s['r']:.{dp}f}  "
             f"slope={s['slope']:.{dp}f}±{s['slope_ci95']:.{dp}f}  RMSE={s['rmse']:.3f} m"
             for s in rows + [dict(s_all, region="All stations")]]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8,
            family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9,
                                          ec="#999999"))

    present = [r for r in REGION_ORDER if r in df["region"].values]
    handles = [mlines.Line2D([], [], color=REGION_COLORS[r], lw=2.4, marker="o", markersize=7,
                             label=r) for r in present]
    handles.append(mlines.Line2D([], [], color="k", ls="--", lw=1.1, label="1:1 line"))
    leg = ax.legend(handles=handles, loc="lower right", fontsize=9,
                    title="Region (line = regression)", title_fontsize=9)
    ax.add_artist(leg)
    ax.legend(handles=[mlines.Line2D([], [], ls="", marker=STATION_MARKERS.get(st, "o"),
                                     color=REGION_COLORS[r], markersize=7, label=st)
                       for r in present
                       for st in sorted(df.loc[df["region"] == r, "station"].unique())],
              loc="center right", bbox_to_anchor=(1.0, 0.36), fontsize=8, title="Station",
              title_fontsize=8)
    ax.set_xlabel(p["x_label"], fontsize=11)
    ax.set_ylabel(p["y_label"][:-4] + ", bias-corrected (m)", fontsize=11)
    who = "Anchorage, Nikiski, Kodiak" if key == "twl" else "All 8 Stations"
    ax.set_title(f"SWOT vs. NOAA by Region — {p['title']}, Bias-Corrected\n"
                 f"{who}  |  Ice-Free Periods 2023–2026", fontsize=12, fontweight="bold")
    plt.savefig(fig_dir(key) / "regional_bias_corrected.png", dpi=300, bbox_inches="tight")
    plt.close()

    res = pd.DataFrame(rows + [dict(s_all, region="All stations",
                                    stations=", ".join(sorted(df["station"].unique())))])
    res = res[["region", "stations", "n", "r", "p", "slope", "slope_ci95", "intercept",
               "rmse"]].round(4)
    res.to_csv(stats_path(f"regional_bias_corrected_{key}.csv"), index=False)
    print(f"\n{p['title']}: regional statistics after removing each station's bias")
    print(res.to_string(index=False))


def main():
    for key in PIPELINES:
        per_station(key)
        regional(key)


if __name__ == "__main__":
    main()
