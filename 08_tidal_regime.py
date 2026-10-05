"""
08_tidal_regime.py
==================
Does SWOT-gauge agreement depend on tidal range?

For the NTR and DAC-restored comparisons, each station's approximate tidal
range is taken as the 5th-95th percentile spread of the NOAA predicted tide
at the matched overpasses (the range actually sampled). RMSE, bias and
Pearson r (from script 04) are plotted against it, each with an OLS trend
line across stations, alongside a bar chart of tidal range by station.

With only eight stations, a trend across stations should be read cautiously.

Outputs:
    OUTPUT_DIR/figures/<ntr|dac>/tidal_regime.png
    OUTPUT_DIR/stats/tidal_range_<ntr|dac>.csv

Run:
    python 08_tidal_regime.py
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from swot_utils import (PIPELINES, REGION_COLORS, fig_dir, load_pairs, stats_path,
                        title_tag)

# Hand-placed label offsets for crowded panels, as fractions of the axis range:
# {pipeline: {panel: {station: (dx, dy, horizontal alignment)}}}.
LABEL_OVERRIDES = {
    "ntr": {
        "rmse_m": {"Valdez": (0.0, 0.10, "center"), "Seward": (-0.015, -0.09, "right")},
        "r": {"Kodiak": (0.0, 0.05, "center"), "Alitak": (0.01, -0.09, "center"),
              "Valdez": (0.04, -0.07, "left"), "Seward": (-0.015, -0.09, "right"),
              "Cordova": (-0.03, 0.04, "right")},
    },
    "dac": {
        "rmse_m": {"Valdez": (0.0, 0.10, "center")},
        "r": {"Kodiak": (0.0, 0.05, "center"), "Alitak": (0.01, 0.105, "center"),
              "Valdez": (0.02, 0.105, "left"), "Seward": (-0.015, -0.09, "right"),
              "Cordova": (0.02, -0.09, "left")},
    },
}


def annotate_points(ax, xs, ys, labels, colors, fontsize=8.5, overrides=None):
    """Label scatter points inside the axes without covering other points or labels.

    For each point (left to right) up to four positions are tried: beside the
    point on its open side, above or below, then the opposite side. The first
    position whose text box avoids every other point and earlier label is
    used. `overrides` fixes the position of named labels instead.
    """
    overrides = overrides or {}
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    xr, yr = x1 - x0, y1 - y0
    dx, dy = 0.07 * xr, 0.10 * yr
    tw, th, margin = 0.18 * xr, 0.05 * yr, 0.005 * xr     # approximate label box size

    def box(tx, ha, ty):
        left = {"left": tx, "right": tx - tw}.get(ha, tx - tw / 2)
        return left, left + tw, ty - th, ty + th

    def clamp(tx, ha, ty):
        if ha == "left":
            tx = min(x1 - tw - margin, max(x0 + margin, tx))
        elif ha == "right":
            tx = max(x0 + tw + margin, min(x1 - margin, tx))
        else:
            tx = max(x0 + tw / 2 + margin, min(x1 - tw / 2 - margin, tx))
        return tx, max(y0 + th + margin, min(y1 - th - margin, ty))

    def collides(skip, tx, ha, ty, placed):
        b = box(tx, ha, ty)
        if any(b[0] <= xs[j] <= b[1] and b[2] <= ys[j] <= b[3]
               for j in range(len(xs)) if j != skip):
            return True
        return any(b[0] <= p1 and b[1] >= p0 and b[2] <= q1 and b[3] >= q0
                   for p0, p1, q0, q1 in placed)

    placed, flip, last_xf = [], 1, None
    style = dict(xycoords="data", textcoords="data", fontsize=fontsize, va="center",
                 bbox=dict(boxstyle="square,pad=0.15", fc="white", alpha=0.92, ec="none"))
    for i in sorted(range(len(xs)), key=lambda k: xs[k]):
        x, y, label, c = xs[i], ys[i], labels[i], colors[i]
        xf = (x - x0) / xr
        arrow = dict(arrowstyle="-", color=c, lw=0.4, alpha=0.5)
        if label in overrides:
            odx, ody, ha = overrides[label]
            tx, ty = clamp(x + odx * xr, ha, y + ody * yr)
        else:
            # Alternate above/below for points close together in x.
            flip = -flip if last_xf is not None and abs(xf - last_xf) < 0.12 else 1
            side = ("right", x - dx) if xf > 0.55 else ("left", x + dx)
            other = ("left", x + dx) if side[0] == "right" else ("right", x - dx)
            candidates = [(side[0], side[1], y + dy * flip), (side[0], side[1], y - dy * flip),
                          (other[0], other[1], y + dy * flip), (other[0], other[1], y - dy * flip)]
            ha = candidates[0][0]
            tx, ty = clamp(candidates[0][1], ha, candidates[0][2])
            for cha, ctx, cty in candidates:
                ctx, cty = clamp(ctx, cha, cty)
                if not collides(i, ctx, cha, cty, placed):
                    ha, tx, ty = cha, ctx, cty
                    break
        placed.append(box(tx, ha, ty))
        ax.annotate(label, xy=(x, y), xytext=(tx, ty), color=c, ha=ha, arrowprops=arrow, **style)
        last_xf = xf


def tidal_regime(key):
    p = PIPELINES[key]
    df = load_pairs(key)
    st = pd.read_csv(stats_path(f"stats_by_station_{key}.csv"))

    rng = {s: round(np.percentile(g["noaa_predicted_m"], 95)
                    - np.percentile(g["noaa_predicted_m"], 5), 3)
           for s, g in df.groupby("station")}
    st["tidal_range_m"] = st["station"].map(rng)
    st["color"] = st["region"].map(REGION_COLORS)
    st = st.set_index("station").loc[[s for s in p["stations"] if s in rng]].reset_index()
    st[["station", "region", "tidal_range_m", "rmse_m", "bias_m", "r"]].to_csv(
        stats_path(f"tidal_range_{key}.csv"), index=False)

    tag = title_tag(key, "bracket")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Tidal Regime vs. SWOT-NOAA Agreement — All Matched Pairs{title_tag(key)}\n"
                 "Southcentral Alaska, Ice-Free Periods 2023-2026", fontsize=13, fontweight="bold")

    x = st["tidal_range_m"].values
    x_line = np.linspace(x.min(), x.max(), 100)
    panels = [(axes[0, 0], "rmse_m", "RMSE (m)", "RMSE vs. Tidal Range", False),
              (axes[0, 1], "bias_m", "Bias: mean(SWOT - NTR)  (m)", "Bias vs. Tidal Range", True),
              (axes[1, 0], "r", "Pearson r", "Pearson r vs. Tidal Range", True)]
    for ax, col, ylabel, title, zero_line in panels:
        ax.scatter(x, st[col], c=st["color"], s=120, edgecolors="white", linewidths=0.8, zorder=3)
        lr = stats.linregress(x, st[col])
        ax.plot(x_line, lr.slope * x_line + lr.intercept, "k--", lw=1.2, alpha=0.5,
                label=f"OLS  r={lr.rvalue:.2f}  p={lr.pvalue:.3f}")
        ax.legend(fontsize=8)
        if zero_line:
            ax.axhline(0, color="black" if col == "bias_m" else "gray",
                       lw=0.8 if col == "bias_m" else 0.6, alpha=0.5)
        ax.set_xlabel("Approx. tidal range (m, P5-P95)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title + tag, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        # Widen the axes to leave room for labels, then place them.
        (a, b), (c, d) = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(a - 0.10 * (b - a), b + 0.28 * (b - a))
        ax.set_ylim(c - 0.12 * (d - c), d + 0.12 * (d - c))
        annotate_points(ax, list(x), list(st[col]), list(st["station"]), list(st["color"]),
                        overrides=LABEL_OVERRIDES.get(key, {}).get(col))

    ax = axes[1, 1]
    bars = ax.bar(range(len(st)), x, color=st["color"], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xticks(range(len(st)))
    ax.set_xticklabels(st["station"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Approx. tidal range (m)", fontsize=10)
    ax.set_title("Tidal Range by Station (P5-P95 of predicted tide)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, x):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f} m",
                ha="center", va="bottom", fontsize=8.5)

    fig.legend(handles=[mpatches.Patch(facecolor=c, label=r) for r, c in REGION_COLORS.items()],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir(key) / "tidal_regime.png", dpi=300, bbox_inches="tight")
    plt.close()

    r, pval = stats.pearsonr(x, st["rmse_m"])
    print(f"{p['title']}: r(tidal range, RMSE) = {r:.3f}, p = {pval:.4f}")


def main():
    for key in ("ntr", "dac"):
        tidal_regime(key)


if __name__ == "__main__":
    main()
