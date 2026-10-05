"""
swot_utils.py
=============
Shared helpers used by the numbered pipeline scripts:

* geometry and time conversion for SWOT files
* the pixel-level MAD outlier filter
* definitions of the three comparison pipelines (NTR, DAC-restored, TWL)
* agreement statistics and regression with confidence intervals
* plotting conventions (colours, station order, seasons) and two small
  matplotlib helpers (a data-version footer and a stats-box placer)

Nothing here reads or writes data on its own; the numbered scripts do that.
"""

import re

import numpy as np
import pandas as pd
from scipy import stats

import config

# =============================================================================
# Geometry and time
# =============================================================================
EARTH_RADIUS_KM = 6371.0
SWOT_EPOCH = pd.Timestamp("2000-01-01", tz="UTC")   # SWOT time = seconds since this


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) from one point to an array of points."""
    r = np.pi / 180.0
    dlat = (lat2 - lat1) * r
    dlon = (lon2 - lon1) * r
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1 * r) * np.cos(lat2 * r) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def swot_time_to_utc(seconds_since_2000):
    """Convert a SWOT time value (seconds since 2000-01-01) to a UTC Timestamp."""
    return SWOT_EPOCH + pd.to_timedelta(float(seconds_since_2000), unit="s")


def parse_cycle_pass(filename):
    """Return (cycle, pass) from a SWOT L3 file name, e.g. ..._031_459_... -> (31, 459)."""
    m = re.search(r"_(\d{3})_(\d{3})_", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# =============================================================================
# Pixel-level outlier filter
# =============================================================================
def mad_inliers(values, threshold=config.MAD_THRESHOLD):
    """Boolean mask of pixels kept by the median-absolute-deviation filter.

    A pixel is dropped when it lies more than `threshold` x MAD from the median.
    The filter is applied only when there are at least three pixels and the
    MAD is greater than zero; otherwise every pixel is kept.
    """
    values = np.asarray(values, dtype=float)
    keep = np.ones(values.shape, dtype=bool)
    if len(values) >= 3:
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad > 0:
            keep = np.abs(values - med) <= threshold * mad
    return keep


# =============================================================================
# The three comparison pipelines
# =============================================================================
STATION_ORDER = ["Anchorage", "Nikiski", "Seldovia",
                 "Cordova", "Seward", "Valdez",
                 "Alitak", "Kodiak"]
TWL_STATION_ORDER = [s for s in STATION_ORDER if s in config.TWL_STATIONS]

# x = the NOAA (reference) value, y = the SWOT value, for each pipeline.
PIPELINES = {
    "ntr": dict(
        title="Standard NTR",
        tag="",                              # appended to figure titles
        pairs_file="matched_pairs_ntr.csv",
        x="ntr_m", y="ssha_median_m",
        gauge_name="NOAA NTR", swot_name="SWOT SSHA",
        x_label="NOAA NTR (m)",
        y_label="SWOT SSHA (m)",
        diff_label="SWOT SSHA - NOAA NTR",
        stations=STATION_ORDER,
    ),
    "dac": dict(
        title="DAC-Restored NTR",
        tag="DAC-Restored",
        pairs_file="matched_pairs_dac.csv",
        # In the DAC files, ssha_median_m holds ssha_unfiltered + dac.
        x="ntr_m", y="ssha_median_m",
        gauge_name="NOAA NTR", swot_name="SWOT SSHA",
        x_label="NOAA NTR (m)",
        y_label="SWOT SSHA, DAC-restored (m)",
        diff_label="SWOT SSHA (DAC-restored) - NOAA NTR",
        stations=STATION_ORDER,
    ),
    "twl": dict(
        title="Total Water Level",
        tag="TWL",
        pairs_file="matched_pairs_twl.csv",
        # Both values share the SWOT MSS reference (see script 03).
        x="gauge_twl_m", y="twl_median_m",
        gauge_name="Gauge TWL", swot_name="SWOT TWL",
        x_label="Gauge TWL, MSS-referenced (m)",
        y_label="SWOT TWL (m)",
        diff_label="SWOT TWL - gauge TWL",
        stations=TWL_STATION_ORDER,
    ),
}


def pairs_path(key):
    return config.OUTPUT_DIR / PIPELINES[key]["pairs_file"]


def load_pairs(key):
    """Load the matched SWOT-gauge pairs for one pipeline ('ntr', 'dac' or 'twl')."""
    path = pairs_path(key)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 03_match_noaa_swot.py first.")
    return pd.read_csv(path, parse_dates=["overpass_time"])


def stats_path(filename):
    config.STATS_DIR.mkdir(parents=True, exist_ok=True)
    return config.STATS_DIR / filename


def fig_dir(*parts):
    d = config.FIG_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Statistics
# =============================================================================
def agreement_stats(x, y):
    """Pearson r, bias, RMSE and SD of differences for SWOT (y) vs. gauge (x).

    Bias and differences are SWOT minus gauge. Returns NaNs if n < 3.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return dict(n=n, r=np.nan, p=np.nan, bias=np.nan, rmse=np.nan, std_diff=np.nan)
    r, p = stats.pearsonr(x, y)
    d = y - x
    return dict(n=n, r=float(r), p=float(p), bias=float(np.mean(d)),
                rmse=float(np.sqrt(np.mean(d ** 2))), std_diff=float(np.std(d)))


def regression_with_ci(x, y, level=0.95):
    """OLS of y on x with a two-sided confidence interval on the slope.

    The half-width of the interval is t(1 - alpha/2, n - 2) x SE(slope).
    Returns None if n < 3.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return None
    lr = stats.linregress(x, y)
    ci = stats.t.ppf(1 - (1 - level) / 2, df=n - 2) * lr.stderr
    return dict(n=n, slope=lr.slope, intercept=lr.intercept, r=lr.rvalue,
                p=lr.pvalue, se=lr.stderr, ci95=ci)


def station_bias_corrected(df, x, y):
    """Return (corrected_y, bias) after removing each station's mean bias.

    bias = mean(y) - mean(x) for that station; corrected_y = y - bias.
    Pearson r is unchanged by this; RMSE drops to the scatter about the mean.
    """
    bias = df.groupby("station")[y].transform("mean") - df.groupby("station")[x].transform("mean")
    return df[y] - bias, bias


# =============================================================================
# Plotting conventions
# =============================================================================
REGION_COLORS = {
    "Cook Inlet":           "#E63946",
    "Prince William Sound": "#457B9D",
    "Kodiak":               "#2D6A4F",
}
REGION_ORDER = list(REGION_COLORS)

SEASON_COLORS = {
    "Spring (Apr-May)": "#4CAF50",
    "Summer (Jun-Aug)": "#FF9800",
    "Autumn (Sep-Oct)": "#9C27B0",
}


def month_to_season(month):
    if month in (4, 5):
        return "Spring (Apr-May)"
    if month in (6, 7, 8):
        return "Summer (Jun-Aug)"
    return "Autumn (Sep-Oct)"


def station_region(station):
    return config.STATIONS[station]["region"]


def station_color(station):
    return REGION_COLORS[station_region(station)]


def region_colors_for(stations):
    """Region-colour mapping restricted to the regions present in `stations`."""
    present = {station_region(s) for s in stations}
    return {r: c for r, c in REGION_COLORS.items() if r in present}


def title_tag(key, style="dash"):
    """Pipeline label for figure titles: '' for NTR, otherwise e.g. ' (DAC-Restored)'."""
    tag = PIPELINES[key]["tag"]
    if not tag:
        return ""
    return f"  [{tag}]" if style == "bracket" else f" ({tag})"


# =============================================================================
# Matplotlib helpers
# =============================================================================
def _install_version_stamp():
    """Add a small grey data-version footer to every figure saved with savefig.

    Also retries a save that fails because the image is open elsewhere (a
    common Windows problem with image viewers and preview panes).
    """
    import matplotlib
    matplotlib.use("Agg")                       # scripts only save figures
    from matplotlib.figure import Figure
    if getattr(Figure.savefig, "_version_stamp", False):
        return
    original_savefig = Figure.savefig

    def savefig(self, *args, **kwargs):
        if config.STAMP_FIGURES and not getattr(self, "_version_stamped", False):
            if kwargs.get("bbox_inches") == "tight":
                # Place the label just below the lowest drawn element.
                bb = self.get_tightbbox(self.canvas.get_renderer())
                w, h = self.get_size_inches()
                self.text(min(bb.x1 / w, 0.995), bb.y0 / h - 0.06 / h,
                          config.DATA_VERSION_LABEL, ha="right", va="top",
                          fontsize=7, color="0.4", style="italic")
            else:
                self.text(0.995, 0.004, config.DATA_VERSION_LABEL, ha="right",
                          va="bottom", fontsize=7, color="0.4", style="italic")
            self._version_stamped = True
        import time
        for attempt in range(4):
            try:
                return original_savefig(self, *args, **kwargs)
            except OSError as e:
                if e.errno not in (13, 22) or attempt == 3:
                    if e.errno in (13, 22):
                        raise OSError(e.errno, "Could not save the figure because the "
                                      "file is open in another program. Close it and "
                                      "re-run.", e.filename) from None
                    raise
                time.sleep(2)

    savefig._version_stamp = True
    Figure.savefig = savefig


_install_version_stamp()


def keep_box_clear(ax, txt, x, y, max_zoom=1.6, pad=0.015):
    """Keep a stats text box (placed in axes coordinates) off the data points.

    Tries, in order: the box's current corner with a slight zoom-out; the other
    three corners (skipping any corner a legend occupies); and finally the
    corner covering the fewest points, which are then drawn on top of the box.
    Zooming is about the panel centre, and any 1:1 line is stretched to match.
    """
    fig = ax.figure
    r = fig.canvas.get_renderer()
    ax.apply_aspect()
    x, y = np.asarray(x, float), np.asarray(y, float)
    (xl, xh), (yl, yh) = ax.get_xlim(), ax.get_ylim()
    cx, cy, hx, hy = (xl + xh) / 2, (yl + yh) / 2, (xh - xl) / 2, (yh - yl) / 2
    inv = ax.transAxes.inverted()

    leg_boxes = []
    for leg in [ax.get_legend()] + [a for a in ax.artists if hasattr(a, "get_frame")]:
        if leg is not None:
            lb = leg.get_window_extent(r).transformed(inv)
            leg_boxes.append((lb.x0, lb.x1, lb.y0, lb.y1))

    def box_frac():
        patch = txt.get_bbox_patch()
        if patch is not None:
            txt.update_bbox_position_size(r)
            bb = patch.get_window_extent(r).transformed(inv)
        else:
            bb = txt.get_window_extent(r).transformed(inv)
        return bb.x0 - pad, bb.x1 + pad, bb.y0 - pad, bb.y1 + pad

    def clear_at(s, b):
        x0, x1, y0, y1 = b
        fx = (x - (cx - s * hx)) / (2 * s * hx)
        fy = (y - (cy - s * hy)) / (2 * s * hy)
        return not ((fx > x0) & (fx < x1) & (fy > y0) & (fy < y1)).any()

    def hits_legend(b):
        x0, x1, y0, y1 = b
        return any(x0 < L1 and L0 < x1 and y0 < T1 and T0 < y1 for L0, L1, T0, T1 in leg_boxes)

    def apply_zoom(s):
        if s <= 1.0:
            return
        nxl, nxh, nyl, nyh = cx - s * hx, cx + s * hx, cy - s * hy, cy + s * hy
        for ln in ax.lines:                                    # stretch the 1:1 line
            xd, yd = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if len(xd) == 2 and np.allclose(xd, yd):
                lo, hi = min(nxl, nyl), max(nxh, nyh)
                ln.set_data([lo, hi], [lo, hi])
        ax.set_xlim(nxl, nxh)
        ax.set_ylim(nyl, nyh)

    # 1. Current spot, modest zoom.
    b = box_frac()
    s = 1.0
    while s <= 1.25 + 1e-9:
        if clear_at(s, b):
            apply_zoom(s)
            return True
        s += 0.03
    # 2. Other corners: no zoom first, then the full zoom allowance.
    px, py = txt.get_position()
    corners = [(1 - px, 1 - py, "right", "bottom"), (1 - px, py, "right", "top"),
               (px, 1 - py, "left", "bottom")]
    orig = (px, py, txt.get_ha(), txt.get_va())
    for zoom_limit in (1.0, max_zoom):
        for cxp, cyp, ha, va in corners:
            txt.set_position((cxp, cyp))
            txt.set_ha(ha)
            txt.set_va(va)
            b = box_frac()
            if hits_legend(b):
                continue
            s = 1.0
            while s <= zoom_limit + 1e-9:
                if clear_at(s, b):
                    apply_zoom(s)
                    return True
                s += 0.03
    # 1b. Original spot with the full zoom allowance.
    txt.set_position(orig[:2])
    txt.set_ha(orig[2])
    txt.set_va(orig[3])
    b = box_frac()
    s = 1.0
    while s <= max_zoom + 1e-9:
        if clear_at(s, b):
            apply_zoom(s)
            return True
        s += 0.03

    # 3. Last resort: the corner covering the fewest points; draw them on top.
    def n_inside(bx):
        x0, x1, y0, y1 = bx
        fx = (x - xl) / (xh - xl)
        fy = (y - yl) / (yh - yl)
        return int(((fx > x0) & (fx < x1) & (fy > y0) & (fy < y1)).sum())

    best = None
    for pos in [orig] + [(cxp, cyp, ha, va) for cxp, cyp, ha, va in corners]:
        txt.set_position(pos[:2])
        txt.set_ha(pos[2])
        txt.set_va(pos[3])
        bx = box_frac()
        if hits_legend(bx):
            continue
        n = n_inside(bx)
        if best is None or n < best[0]:
            best = (n, pos)
    pos = best[1] if best else orig
    txt.set_position(pos[:2])
    txt.set_ha(pos[2])
    txt.set_va(pos[3])
    txt.set_zorder(2)
    return False
