"""
11_thesis_figures.py
====================
Main-text Results figures, sized for a portrait page (6.5 in wide) with
readable fonts, shared axis labels, and no in-panel statistics boxes (the
numbers are in the thesis tables).

    fig_ntr_vs_dac_scatter.png   SWOT vs. gauge NTR at each station: standard
                                 NTR (grey) and DAC-restored (colour) overlaid
    fig_slopes.png               regression slope +/- 95% CI, standard vs.
                                 DAC-restored, each station
    fig_regional_dac.png         DAC-restored, station bias removed, pooled by
                                 region
    fig_station_differences.png  SWOT - gauge difference for every pair, all
                                 stations on one axis by longitude (DAC-restored)
    fig_tidal_range.png          RMSE and bias vs. tidal range (DAC-restored)
    fig_vdc_dac.png              Van de Casteele diagrams, DAC-restored NTR
                                 (points coloured by season)
    fig_vdc_twl.png              Van de Casteele diagrams, TWL (by season)
    fig_vdc_twl_flood_ebb.png    same, points marked flood or ebb tide
    fig_ntr_vs_dac_scatter_flood_ebb.png
                                 NTR vs. DAC scatter, DAC points marked flood/ebb
    fig_twl_scatter.png          SWOT TWL vs. gauge TWL (optional figure)

Van de Casteele diagrams follow the conventional layout: water level on the
y-axis and SWOT - gauge difference on the x-axis, with a least-squares line of
difference on water level. A vertical line means a constant offset; a tilted
line means the difference changes with water level.

Flood or ebb is taken from the NOAA tidal prediction: an overpass is "flood"
if the predicted tide is higher 12 minutes after the overpass than 12 minutes
before it, and "ebb" otherwise.

Inputs:  matched pairs (script 03); NOAA tidal predictions (script 01)
Outputs: OUTPUT_DIR/figures/thesis/

Run:
    python 11_thesis_figures.py
"""

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from swot_utils import fig_dir, load_pairs, month_to_season  # also adds the figure footer

# ── Style ────────────────────────────────────────────────────────────────────
# Region colours chosen to stay distinguishable for colour-blind readers.
REGION_COLORS = {"Cook Inlet": "#E63946", "Prince William Sound": "#2F6FB3",
                 "Kodiak": "#2E8B57"}
NTR_GREY = "#9A9A9A"
SEASONS = {  # colour + marker, so seasons are distinguishable without colour
    "Spring (Apr-May)":  ("#2E8B57", "^"),
    "Summer (Jun-Aug)":  ("#E08A00", "o"),
    "Autumn (Sep-Oct)":  ("#7B3FB5", "s"),
}
TIDE_PHASE = {  # colour + marker (rising / falling triangles)
    "Flood (rising tide)": ("#C0392B", "^"),
    "Ebb (falling tide)":  ("#2F6FB3", "v"),
}
# Stations grouped by region, for 4 x 2 panel grids (read across, then down).
ORDER = ["Anchorage", "Nikiski", "Seldovia", "Cordova",
         "Seward", "Valdez", "Alitak", "Kodiak"]
TWL_ORDER = [s for s in ORDER if s in config.TWL_STATIONS]
PAGE_W = 6.5

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ECECEC", "grid.linewidth": 0.6,
    "axes.axisbelow": True,          # grid lines sit behind the data
    "savefig.dpi": 300,
})


def color(station):
    return REGION_COLORS[config.STATIONS[station]["region"]]


def panel_label(ax, text, c="#222222"):
    ax.set_title(text, loc="left", fontsize=10, fontweight="bold", color=c, pad=4)


def save(fig, name):
    path = fig_dir("thesis") / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def fit_line(ax, x, y, c, ls="-", lw=1.8):
    if len(x) >= 3:
        lr = stats.linregress(x, y)
        xl = np.linspace(np.min(x), np.max(x), 50)
        ax.plot(xl, lr.slope * xl + lr.intercept, color=c, ls=ls, lw=lw, zorder=4)


def tide_phase(df, minutes=12):
    """Flood or ebb at each overpass, from the NOAA tidal prediction."""
    out = pd.Series(index=df.index, dtype=object)
    dt = pd.Timedelta(minutes=minutes)
    for st, g in df.groupby("station"):
        sid = config.STATIONS[st]["id"]
        p = pd.read_csv(config.NOAA_DIR / f"NOAA_{sid}_{st}_predicted.csv")
        p["datetime"] = pd.to_datetime(p["datetime"], utc=True)
        p = p.dropna(subset=["water_level_m"]).drop_duplicates("datetime").sort_values("datetime")
        t0 = pd.Timestamp("2000-01-01", tz="UTC")       # times as seconds, for np.interp
        x = (p["datetime"] - t0).dt.total_seconds().values
        t = (pd.to_datetime(g["overpass_time"], utc=True) - t0).dt.total_seconds().values
        before = np.interp(t - dt.total_seconds(), x, p["water_level_m"])
        after = np.interp(t + dt.total_seconds(), x, p["water_level_m"])
        out[g.index] = np.where(after > before, *TIDE_PHASE.keys())
    return out


# ── 1. NTR vs. DAC-restored scatter ──────────────────────────────────────────
def fig_scatter(ntr, dac, by_phase=False):
    """by_phase=True draws the DAC-restored points as flood / ebb triangles."""
    if by_phase:
        dac = dac.assign(phase=tide_phase(dac))
    fig, axes = plt.subplots(4, 2, figsize=(PAGE_W, 10.2))
    for ax, st in zip(axes.flat, ORDER):
        a, b = ntr[ntr.station == st], dac[dac.station == st]
        ax.scatter(a.ntr_m, a.ssha_median_m, s=12, facecolor="none", edgecolor=NTR_GREY,
                   linewidth=0.7, zorder=2)
        if by_phase:
            for name, (c, mk) in TIDE_PHASE.items():
                m = b.phase == name
                ax.scatter(b.ntr_m[m], b.ssha_median_m[m], s=14, color=c, marker=mk,
                           alpha=0.9, edgecolor="white", linewidth=0.3, zorder=3)
        else:
            ax.scatter(b.ntr_m, b.ssha_median_m, s=12, color=color(st), alpha=0.9,
                       edgecolor="white", linewidth=0.3, zorder=3)
        fit_line(ax, a.ntr_m.values, a.ssha_median_m.values, NTR_GREY, ls="--", lw=1.4)
        fit_line(ax, b.ntr_m.values, b.ssha_median_m.values, color(st))
        vals = np.concatenate([a.ntr_m, a.ssha_median_m, b.ssha_median_m])
        lo, hi = vals.min(), vals.max()
        pad = 0.08 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        panel_label(ax, st, color(st))
    fig.supxlabel("NOAA non-tidal residual (m)", y=0.045)
    fig.supylabel("SWOT SSHA (m)", x=0.01)
    handles = [
        mlines.Line2D([], [], marker="o", ls="--", color=NTR_GREY, markerfacecolor="none",
                      label="Standard NTR"),
        mlines.Line2D([], [], marker="o", ls="-", color="#555555", label="DAC-restored"),
        mlines.Line2D([], [], ls=":", color="black", label="1:1"),
    ]
    if by_phase:   # legend fills by column: row 1 = points, row 2 = lines
        flood, ebb = [mlines.Line2D([], [], marker=mk, ls="", color=c,
                                    label=f"DAC-restored, {n.split()[0].lower()}")
                      for n, (c, mk) in TIDE_PHASE.items()]
        handles = [mlines.Line2D([], [], marker="o", ls="", color=NTR_GREY,
                                 markerfacecolor="none", label="Standard NTR"),
                   mlines.Line2D([], [], ls="-", color="#555555", label="Fit (dashed = standard)"),
                   flood, handles[2], ebb]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0 if not by_phase else 1.015))
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.975 if not by_phase else 0.955])
    save(fig, "fig_ntr_vs_dac_scatter_flood_ebb.png" if by_phase else "fig_ntr_vs_dac_scatter.png")


# ── 2. Regression slopes ─────────────────────────────────────────────────────
def slope_ci(df, st, x, y):
    s = df[df.station == st]
    lr = stats.linregress(s[x], s[y])
    return lr.slope, stats.t.ppf(0.975, len(s) - 2) * lr.stderr


def fig_slopes(ntr, dac):
    fig, ax = plt.subplots(figsize=(PAGE_W, 3.6))
    xs = np.arange(len(ORDER))
    for i, st in enumerate(ORDER):
        for df, off, hollow in [(ntr, -0.15, True), (dac, 0.15, False)]:
            m, ci = slope_ci(df, st, "ntr_m", "ssha_median_m")
            ax.errorbar(i + off, m, yerr=ci, fmt="o", ms=6, capsize=3, lw=1.3, zorder=3,
                        color=color(st), markerfacecolor="white" if hollow else color(st),
                        markeredgewidth=1.5)
    ax.axhline(1, color="black", lw=1, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(ORDER, rotation=30, ha="right")
    ax.set_ylabel("Regression slope (SWOT vs. gauge)")
    ax.grid(axis="x", visible=False)
    handles = [mlines.Line2D([], [], marker="o", ls="", color="#555555",
                             markerfacecolor="white", markeredgewidth=1.5, label="Standard NTR"),
               mlines.Line2D([], [], marker="o", ls="", color="#555555", label="DAC-restored"),
               mlines.Line2D([], [], ls="--", color="black", label="Slope = 1")]
    ax.legend(handles=handles, loc="upper right", frameon=False)
    fig.tight_layout()
    save(fig, "fig_slopes.png")


# ── 3. Regional DAC-restored, bias-corrected ─────────────────────────────────
MARKERS = {"Anchorage": "o", "Nikiski": "s", "Seldovia": "^", "Cordova": "^",
           "Seward": "o", "Valdez": "s", "Alitak": "s", "Kodiak": "o"}


def fig_regional(dac):
    d = dac.copy()
    d["bias"] = d.groupby("station").ssha_median_m.transform("mean") - \
        d.groupby("station").ntr_m.transform("mean")
    d["y"] = d.ssha_median_m - d.bias
    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W))
    for st in ORDER:
        s = d[d.station == st]
        ax.scatter(s.ntr_m, s.y, s=16, marker=MARKERS[st], color=color(st), alpha=0.9,
                   edgecolor="white", linewidth=0.3, label=st)
    for region, c in REGION_COLORS.items():
        s = d[d.station.map(lambda x: config.STATIONS[x]["region"]) == region]
        fit_line(ax, s.ntr_m.values, s.y.values, c, lw=2.2)
    lim = np.abs(np.concatenate([d.ntr_m, d.y])).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls=":", zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("NOAA non-tidal residual (m)")
    ax.set_ylabel("SWOT SSHA, DAC-restored, station bias removed (m)")
    ax.legend(loc="lower right", ncol=2, frameon=False, markerscale=1.4,
              title="Station (line = regional fit)", title_fontsize=9)
    fig.tight_layout()
    save(fig, "fig_regional_dac.png")


# ── 4. All stations on one axis ──────────────────────────────────────────────
def fig_station_diffs(dac):
    """Every DAC-restored SWOT - gauge difference, each station placed at its
    longitude (west to east), after the layout of Lichtman et al. (2025)."""
    d = dac.assign(err=dac.ssha_median_m - dac.ntr_m)
    lon = {s: config.STATIONS[s]["lon"] for s in ORDER}
    order = sorted(ORDER, key=lon.get)
    rng = np.random.default_rng(0)         # fixed jitter so the figure is repeatable
    fig, ax = plt.subplots(figsize=(PAGE_W, 3.8))
    ax.grid(False)
    for yv in (-0.5, 0.5):
        ax.axhline(yv, color="#777777", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="#C0392B", lw=0.8, ls=":", zorder=1)
    for i, st in enumerate(order):
        c = color(st)
        ax.axvline(lon[st], color=c, lw=1.2, alpha=0.5, zorder=2)
        e = d[d.station == st].err.values
        ax.scatter(lon[st] + rng.uniform(-0.07, 0.07, len(e)), e, s=6, color=c,
                   alpha=0.85, edgecolor="none", zorder=3)
    lo, hi = d.err.min(), d.err.max()
    ax.set_ylim(min(lo - 0.08, -0.6), max(hi + 0.15, 0.6))
    ax.set_xlim(min(lon.values()) - 0.55, max(lon.values()) + 0.55)
    top = ax.get_ylim()[1]
    span = top - ax.get_ylim()[0]
    for i, st in enumerate(order):         # two rows so neighbouring names don't overlap
        ax.text(lon[st], top + span * (0.075 if i % 2 else 0.02), st, ha="center",
                va="bottom", fontsize=8.5, color=color(st), fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{-v:.0f}\u00b0W"))
    ax.set_xlabel("Longitude of tide gauge")
    ax.set_ylabel("SWOT SSHA (DAC-restored)\n\u2212 NOAA NTR (m)")
    fig.tight_layout()
    save(fig, "fig_station_differences.png")


# ── 5. Tidal range ───────────────────────────────────────────────────────────
# Label positions (data units: tidal range, value) for each station, placed by
# hand so every label has a short leader line and no two leaders cross.
# If the data change, these may need adjusting.
LABEL_AT = {
    "rmse": {"Kodiak": (2.2, 0.36), "Seward": (2.9, 0.42), "Valdez": (3.75, 0.37),
             "Cordova": (2.95, 0.07), "Alitak": (4.4, 0.10), "Seldovia": (6.4, 0.22),
             "Nikiski": (5.0, 0.62), "Anchorage": (6.6, 0.80)},
    "bias": {"Kodiak": (2.2, -0.08), "Seward": (2.75, 0.32), "Valdez": (3.95, 0.31),
             "Alitak": (4.45, 0.08), "Cordova": (4.3, -0.19), "Seldovia": (6.4, 0.19),
             "Nikiski": (5.0, -0.36), "Anchorage": (6.85, -0.62)},
}


def fig_tidal(dac):
    rows = []
    for st in ORDER:
        s = dac[dac.station == st]
        rng = np.percentile(s.noaa_predicted_m, 95) - np.percentile(s.noaa_predicted_m, 5)
        d = s.ssha_median_m - s.ntr_m
        rows.append(dict(station=st, range=rng, rmse=np.sqrt(np.mean(d ** 2)), bias=d.mean()))
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 3.2), sharex=True)
    for ax, col, lab in [(axes[0], "rmse", "RMSE (m)"), (axes[1], "bias", "Mean bias, SWOT - gauge (m)")]:
        ax.scatter(t["range"], t[col], s=40, c=[color(s) for s in t.station], zorder=4,
                   edgecolor="white", linewidth=0.6)
        lr = stats.linregress(t["range"], t[col])
        xl = np.linspace(t["range"].min(), t["range"].max(), 20)
        ax.plot(xl, lr.slope * xl + lr.intercept, color="#555555", lw=1, ls="--", zorder=2)
        ax.set_title(f"r = {lr.rvalue:.2f}, p = {lr.pvalue:.3f}", loc="right", fontsize=8.5,
                     color="#333333", pad=4)
        for _, r in t.iterrows():
            ax.annotate(r.station, (r["range"], r[col]), xytext=LABEL_AT[col][r.station],
                        fontsize=7.5, color="#333333", ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9),
                        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6,
                                        shrinkA=2, shrinkB=4))
        ys = [v[1] for v in LABEL_AT[col].values()] + list(t[col])
        ax.set_ylim(min(ys) - 0.06, max(ys) + 0.06)
        if col == "bias":
            ax.axhline(0, color="black", lw=0.6, zorder=1)
        ax.set_ylabel(lab)
    for ax in axes:
        ax.set_xlim(t["range"].min() - 0.6, t["range"].max() + 0.4)
    fig.supxlabel("Tidal range sampled, P5-P95 of predicted tide (m)", y=0.03)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "fig_tidal_range.png")


# ── 6-7. Van de Casteele diagrams ────────────────────────────────────────────
def vdc_panel(ax, level, err, group, styles):
    for name, (c, mk) in styles.items():
        m = group == name
        ax.scatter(err[m], level[m], s=14, color=c, marker=mk, alpha=0.9,
                   edgecolor="white", linewidth=0.3, zorder=3)
    ax.axvline(0, color="black", lw=0.7, zorder=1)
    ax.axvline(err.mean(), color="#555555", lw=1, ls="--", zorder=2)
    lr = stats.linregress(level, err)              # difference as a function of level
    yl = np.linspace(level.min(), level.max(), 50)
    ax.plot(lr.slope * yl + lr.intercept, yl, color="black", lw=1.6, zorder=4)
    star = "*" if lr.pvalue < 0.05 else ""
    ax.set_title(f"r = {lr.rvalue:+.2f}{star}", loc="right", fontsize=8.5, color="#333333", pad=4)


def vdc_legend(fig, y, styles, ncol=3):
    handles = [mlines.Line2D([], [], marker=mk, ls="", color=c, label=n)
               for n, (c, mk) in styles.items()]
    handles += [mlines.Line2D([], [], color="black", lw=1.6, label="Fit (difference vs. level)"),
                mlines.Line2D([], [], color="#555555", lw=1, ls="--", label="Mean difference")]
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y))


def fig_vdc_dac(dac):
    d = dac.assign(err=dac.ssha_median_m - dac.ntr_m,
                   season=dac.overpass_time.dt.month.map(month_to_season))
    fig, axes = plt.subplots(4, 2, figsize=(PAGE_W, 10.2))
    for ax, st in zip(axes.flat, ORDER):
        s = d[d.station == st]
        vdc_panel(ax, s.noaa_predicted_m.values, s.err.values, s.season.values, SEASONS)
        panel_label(ax, st, color(st))
    fig.supxlabel("SWOT SSHA (DAC-restored) - NOAA NTR (m)", y=0.045)
    fig.supylabel("NOAA predicted tide (m, MLLW)", x=0.01)
    vdc_legend(fig, 1.01, SEASONS)
    fig.tight_layout(rect=[0.02, 0.05, 1, 0.965])
    save(fig, "fig_vdc_dac.png")


def fig_vdc_twl(twl, x, by_phase=False):
    """Season colours (original), or flood / ebb triangles with by_phase=True."""
    d = twl.assign(err=twl.twl_median_m - twl[x],
                   level=twl[x] - twl.groupby("station")[x].transform("mean"))
    if by_phase:
        d["group"], styles = tide_phase(twl), TIDE_PHASE
    else:
        d["group"], styles = d.overpass_time.dt.month.map(month_to_season), SEASONS
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 3.4), sharey=True)
    for ax, st in zip(axes, TWL_ORDER):
        s = d[d.station == st]
        vdc_panel(ax, s.level.values, s.err.values, s.group.values, styles)
        panel_label(ax, st, color(st))
    fig.supxlabel("Elevation difference, SWOT \u2212 gauge (m)", y=0.04)
    axes[0].set_ylabel("Gauge water level,\nrelative to station mean (m)")
    vdc_legend(fig, 1.12, styles, ncol=2 if by_phase else 3)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "fig_vdc_twl_flood_ebb.png" if by_phase else "fig_vdc_twl.png")


# ── 8. TWL scatter ───────────────────────────────────────────────────────────
def fig_twl_scatter(twl, x):
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, 2.6))
    for ax, st in zip(axes, TWL_ORDER):
        s = twl[twl.station == st]
        ax.scatter(s[x], s.twl_median_m, s=10, color=color(st), alpha=0.9, edgecolor="white",
                   linewidth=0.3)
        lo = min(s[x].min(), s.twl_median_m.min()) - 0.3
        hi = max(s[x].max(), s.twl_median_m.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls=":", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        panel_label(ax, st, color(st))
    fig.supxlabel("Gauge TWL, MSS-referenced (m)", y=0.02)
    axes[0].set_ylabel("SWOT TWL (m)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_twl_scatter.png")


def main():
    ntr, dac, twl = load_pairs("ntr"), load_pairs("dac"), load_pairs("twl")
    x_twl = "gauge_twl_m"
    print(f"Writing thesis figures to {fig_dir('thesis')}")
    fig_scatter(ntr, dac)
    fig_scatter(ntr, dac, by_phase=True)
    fig_slopes(ntr, dac)
    fig_regional(dac)
    fig_station_diffs(dac)
    fig_tidal(dac)
    fig_vdc_dac(dac)
    fig_vdc_twl(twl, x_twl)
    fig_vdc_twl(twl, x_twl, by_phase=True)
    fig_twl_scatter(twl, x_twl)


if __name__ == "__main__":
    main()
