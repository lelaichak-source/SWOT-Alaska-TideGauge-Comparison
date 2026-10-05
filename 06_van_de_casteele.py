"""
06_van_de_casteele.py
=====================
Van de Casteele diagrams: does SWOT error depend on tidal state?

NTR and DAC-restored comparisons (all stations)
    error = SWOT value - NOAA NTR, plotted against the NOAA predicted tide at
    the matched gauge record. A running mean (moving average over the pairs
    sorted by tidal level, window = 20% of the station's pairs) is overlaid,
    and the Pearson correlation between error and predicted tide is reported
    (* = p < 0.05). A flat running mean means a constant offset; a slope or
    curve means tidal-state-dependent error (for example, a mismatch between
    the model tide removed from SWOT and the NOAA harmonic prediction).

TWL comparison (Anchorage, Nikiski, Kodiak)
    error = SWOT TWL - gauge TWL, plotted against gauge water level relative
    to the station mean, with a least-squares trend line. A non-zero trend
    means the error scales with water level.

Points are coloured by season (spring Apr-May, summer Jun-Aug, autumn Sep-Oct).

Outputs:
    OUTPUT_DIR/figures/<ntr|dac|twl>/vdc_<station>.png, vdc_all_stations.png
    OUTPUT_DIR/stats/vdc_tide_correlation_{ntr,dac}.csv
    OUTPUT_DIR/stats/vdc_trend_twl.csv

Run:
    python 06_van_de_casteele.py
"""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from swot_utils import (PIPELINES, REGION_COLORS, SEASON_COLORS, fig_dir, load_pairs,
                        month_to_season, station_color, stats_path, title_tag)

PERIOD = "Southcentral Alaska, Ice-Free Periods 2023-2026"


# =============================================================================
# NTR and DAC-restored: error vs. predicted tide
# =============================================================================
def vdc_panel_ntr(ax, sub, key):
    color = station_color(sub["station"].iloc[0])
    x, y = sub["noaa_predicted_m"].values, sub["error_m"].values

    for season, sc in SEASON_COLORS.items():
        m = sub["season"] == season
        if m.any():
            ax.scatter(sub.loc[m, "noaa_predicted_m"], sub.loc[m, "error_m"], c=sc, s=55,
                       alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3, label=season)

    ax.axhline(0, color="black", lw=0.9, alpha=0.6, zorder=2)
    mean_err = float(np.mean(y))
    ax.axhline(mean_err, color=color, lw=1.5, linestyle="--", alpha=0.85,
               label=f"Mean error = {mean_err:+.3f} m")
    ax.axvline(np.percentile(x, 10), color="gray", lw=0.7, linestyle=":", alpha=0.6)
    ax.axvline(np.percentile(x, 90), color="gray", lw=0.7, linestyle=":", alpha=0.6,
               label="P10/P90 predicted tide")

    # Running mean over pairs sorted by predicted tide (window = 20% of pairs).
    if len(x) >= 8:
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        window = max(3, len(xs) // 5)
        running = np.convolve(ys, np.ones(window) / window, mode="valid")
        ax.plot(xs[window // 2: window // 2 + len(running)], running, color=color, lw=2.5,
                alpha=0.6, zorder=4, label="Running mean")

    pad = (x.max() - x.min()) * 0.08 + 0.2
    yr = max(abs(y).max() * 1.2, 0.1)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(-yr, yr)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_xlabel("NOAA Predicted Tide (m)", fontsize=9)
    ax.set_ylabel(f"{PIPELINES[key]['diff_label']} (m)", fontsize=9)

    rmse = float(np.sqrt(np.mean(y ** 2)))
    r, p = stats.pearsonr(x, y) if len(x) >= 3 else (np.nan, np.nan)
    ax.text(0.03, 0.97, f"n = {len(x)}\nMean error = {mean_err:+.3f} m\nRMSE = {rmse:.3f} m\n"
                        f"r(error~tide) = {r:.3f}{' *' if p < 0.05 else ''}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec=color, lw=1.2))
    tidal_range = np.percentile(x, 95) - np.percentile(x, 5)
    ax.text(0.97, 0.03, f"Approx. tidal range: {tidal_range:.2f} m", transform=ax.transAxes,
            fontsize=7.5, va="bottom", ha="right", color="#555555")
    return dict(n=len(x), mean_error_m=mean_err, rmse_m=rmse, r_error_tide=r, p_value=p,
                tidal_range_m=tidal_range)


def vdc_ntr(key):
    p = PIPELINES[key]
    df = load_pairs(key)
    df["error_m"] = df[p["y"]] - df[p["x"]]
    df["season"] = df["overpass_time"].dt.month.apply(month_to_season)
    out = fig_dir(key)
    stations = [s for s in p["stations"] if s in df["station"].values]
    tag = title_tag(key, "bracket")

    rows = []
    for station in stations:
        sub = df[df["station"] == station]
        fig, ax = plt.subplots(figsize=(8, 6))
        res = vdc_panel_ntr(ax, sub, key)
        rows.append({"station": station, "region": sub["region"].iloc[0], **res})
        ax.set_title(f"{station}  |  {sub['region'].iloc[0]}{tag}", fontsize=12,
                     fontweight="bold", color=station_color(station))
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        fig.suptitle(f"Van de Casteele Diagram — All Matched Pairs{title_tag(key)}\n"
                     "SWOT error vs. Predicted Tidal Level", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out / f"vdc_{station.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()

    nrows = int(np.ceil(len(stations) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 5 * nrows))
    axes = axes.flatten()
    for ax, station in zip(axes, stations):
        sub = df[df["station"] == station]
        vdc_panel_ntr(ax, sub, key)
        ax.set_title(f"{station}  ({sub['region'].iloc[0]}){tag}", fontsize=10,
                     fontweight="bold", color=station_color(station))
        ax.tick_params(labelsize=8)
    for ax in axes[len(stations):]:
        ax.set_visible(False)
    fig.legend(handles=[mpatches.Patch(facecolor=c, label=s) for s, c in SEASON_COLORS.items()],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.065), fontsize=9,
               title="Season", title_fontsize=9)
    fig.legend(handles=[mpatches.Patch(facecolor=c, label=r) for r, c in REGION_COLORS.items()],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015), fontsize=9,
               title="Region", title_fontsize=9)
    fig.suptitle(f"Van de Casteele Diagrams — All Matched Pairs{title_tag(key)}, All Stations\n"
                 f"SWOT Error vs. Predicted Tidal Level\n{PERIOD}", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])   # room for two legend rows and a 3-line title
    plt.savefig(out / "vdc_all_stations.png", dpi=300, bbox_inches="tight")
    plt.close()

    res = pd.DataFrame(rows)
    res.round(4).to_csv(stats_path(f"vdc_tide_correlation_{key}.csv"), index=False)
    print(f"\n{p['title']}: r(error ~ predicted tide)")
    for _, r in res.iterrows():
        print(f"  {r['station']:<10} r = {r['r_error_tide']:+.3f}  p = {r['p_value']:.4f}"
              f"{'  (significant)' if r['p_value'] < 0.05 else ''}")


# =============================================================================
# TWL: error vs. gauge water level
# =============================================================================
def twl_trend(sub):
    """Least-squares fit of error on level (NaNs if fewer than 3 pairs)."""
    x, y = sub["level_m"].to_numpy(float), sub["error_m"].to_numpy(float)
    n = len(x)
    if n < 3:
        return dict(n=n, slope=np.nan, slope_ci95=np.nan, intercept=np.nan, r=np.nan,
                    p=np.nan, level_range_m=np.nan, change_m=np.nan)
    lr = stats.linregress(x, y)
    rng = float(x.max() - x.min())
    return dict(n=n, slope=lr.slope, slope_ci95=stats.t.ppf(0.975, n - 2) * lr.stderr,
                intercept=lr.intercept, r=lr.rvalue, p=lr.pvalue, level_range_m=rng,
                change_m=lr.slope * rng)


def vdc_panel_twl(ax, sub):
    color = station_color(sub["station"].iloc[0])
    x, y = sub["level_m"].to_numpy(float), sub["error_m"].to_numpy(float)
    for season, sc in SEASON_COLORS.items():
        m = (sub["season"] == season).to_numpy()
        if m.any():
            ax.scatter(x[m], y[m], c=sc, s=55, alpha=0.8, edgecolors="white",
                       linewidths=0.4, zorder=3)
    ax.axhline(0, color="black", lw=0.9, alpha=0.6, zorder=2)
    mean_err = float(np.mean(y))
    ax.axhline(mean_err, color=color, lw=1.5, ls="--", alpha=0.85, zorder=2)
    for q in (10, 90):
        ax.axvline(np.percentile(x, q), color="gray", lw=0.7, ls=":", alpha=0.6)

    t = twl_trend(sub)
    if np.isfinite(t["slope"]):
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, t["slope"] * xl + t["intercept"], color="black", lw=1.8, alpha=0.8, zorder=4)

    # Centre the y-range on the mean error, with headroom so the stats box
    # does not cover data.
    half = max(np.abs(y - mean_err).max() * 1.15, 0.1)
    ax.set_ylim(mean_err - half, max(y.max(), mean_err + half) + 0.7 * half)
    pad = (x.max() - x.min()) * 0.06 + 0.1
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_xlabel("Gauge water level, relative to station mean (m)", fontsize=9)
    ax.set_ylabel("SWOT TWL - gauge TWL (m)", fontsize=9)

    rmse = float(np.sqrt(np.mean(y ** 2)))
    ax.text(0.03, 0.97,
            f"n = {len(x)}\nMean error = {mean_err:+.3f} m\nRMSE = {rmse:.3f} m\n"
            f"Trend = {t['slope']:+.3f} ± {t['slope_ci95']:.3f} m/m{' *' if t['p'] < 0.05 else ''}\n"
            f"Change over {t['level_range_m']:.1f} m range = {t['change_m']:+.2f} m",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88, ec=color, lw=1.2))
    return t, mean_err, rmse


def twl_legend(mean_color, mean_label="Mean error"):
    h = [mpatches.Patch(facecolor=c, label=s) for s, c in SEASON_COLORS.items()]
    h += [mlines.Line2D([], [], color="black", lw=1.8, label="Trend (error ~ level)"),
          mlines.Line2D([], [], color=mean_color, lw=1.5, ls="--", label=mean_label),
          mlines.Line2D([], [], color="gray", lw=0.7, ls=":", label="P10 / P90 water level")]
    return h


def vdc_twl():
    p = PIPELINES["twl"]
    df = load_pairs("twl")
    df["error_m"] = df[p["y"]] - df[p["x"]]
    # Water level relative to each station's mean, so stations share an axis.
    df["level_m"] = df[p["x"]] - df.groupby("station")[p["x"]].transform("mean")
    df["season"] = df["overpass_time"].dt.month.apply(month_to_season)
    out = fig_dir("twl")
    stations = [s for s in p["stations"] if s in df["station"].values]

    rows = []
    for station in stations:
        sub = df[df["station"] == station]
        color, region = station_color(station), sub["region"].iloc[0]
        fig, ax = plt.subplots(figsize=(8, 6))
        t, mean_err, rmse = vdc_panel_twl(ax, sub)
        rows.append({"station": station, "region": region, "mean_error_m": mean_err,
                     "rmse_m": rmse, **t})
        ax.set_title(f"{station}  |  {region}  [TWL]", fontsize=12, fontweight="bold", color=color)
        ax.legend(handles=twl_legend(color), fontsize=8, loc="upper right", framealpha=0.9)
        fig.suptitle("Van de Casteele Diagram — TWL, All Matched Pairs\n"
                     "SWOT total water level error vs. gauge water level",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out / f"vdc_{station.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()

    fig, axes = plt.subplots(1, len(stations), figsize=(6 * len(stations), 6.4))
    for ax, station in zip(np.atleast_1d(axes), stations):
        sub = df[df["station"] == station]
        vdc_panel_twl(ax, sub)
        ax.set_title(f"{station}  ({sub['region'].iloc[0]})", fontsize=11, fontweight="bold",
                     color=station_color(station))
        ax.tick_params(labelsize=8)
    fig.legend(handles=twl_legend("gray", "Mean error (region colour)"), loc="lower center",
               ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Van de Casteele Diagrams — TWL, All Matched Pairs\n"
                 "SWOT Total Water Level Error vs. Gauge Water Level  |  Ice-Free Periods 2023–2026",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(out / "vdc_all_stations.png", dpi=300, bbox_inches="tight")
    plt.close()

    res = pd.DataFrame(rows)[["station", "region", "n", "mean_error_m", "rmse_m", "slope",
                              "slope_ci95", "r", "p", "level_range_m", "change_m"]]
    res.round(4).to_csv(stats_path("vdc_trend_twl.csv"), index=False)
    print("\nTWL: trend of error with water level")
    for _, r in res.iterrows():
        print(f"  {r['station']:<10} trend = {r['slope']:+.3f} m/m (p = {r['p']:.2g}), "
              f"change over {r['level_range_m']:.1f} m = {r['change_m']:+.2f} m")


def main():
    for key in ("ntr", "dac"):
        vdc_ntr(key)
    vdc_twl()


if __name__ == "__main__":
    main()
