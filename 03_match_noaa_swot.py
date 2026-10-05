"""
03_match_noaa_swot.py
=====================
Pair each SWOT overpass with the nearest NOAA gauge record.

NTR and DAC-restored comparisons
--------------------------------
The NOAA non-tidal residual is computed for every 6-minute record:
    NTR = observed water level - harmonic tidal prediction   (both MLLW)
Each observation is paired with the nearest prediction within 6 minutes.
Each SWOT overpass is then paired with the nearest NTR record within
+/- 3 minutes of the overpass time (no interpolation).

TWL comparison
--------------
Gauge water levels (relative to station datum, STND) are placed on the same
mean-sea-surface reference as the reconstructed SWOT heights:

    gauge_twl = gauge_STND + (MSS_SWOT - MSL_STND)

MSS_SWOT is the median SWOT mean sea surface over the retained pixels for that
overpass, and MSL_STND is the station's mean sea level above station datum
(config.TWL_STATIONS). The same MSS enters the SWOT reconstruction, so it
largely cancels in the SWOT - gauge difference: the comparison evaluates total
water level variability, not the absolute ellipsoidal height of either record.

Inputs:  OUTPUT_DIR/swot_overpasses_{ntr,dac,twl}.csv   (script 02)
         NOAA_DIR/NOAA_<id>_<name>_{verified,predicted,verified_stnd}.csv (script 01)
Outputs: OUTPUT_DIR/matched_pairs_{ntr,dac,twl}.csv

Run:
    python 03_match_noaa_swot.py
"""

import pandas as pd

import config


def read_noaa(path, value_name):
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.rename(columns={"water_level_m": value_name})


def load_ntr(name, station_id):
    """Observed, predicted and NTR for one station, indexed by UTC time."""
    v_path = config.NOAA_DIR / f"NOAA_{station_id}_{name}_verified.csv"
    p_path = config.NOAA_DIR / f"NOAA_{station_id}_{name}_predicted.csv"
    for p in (v_path, p_path):
        if not p.exists():
            print(f"  [warn] missing {p.name}")
            return None
    merged = pd.merge_asof(
        read_noaa(v_path, "verified_m").sort_values("datetime"),
        read_noaa(p_path, "predicted_m").sort_values("datetime"),
        on="datetime", direction="nearest",
        tolerance=pd.Timedelta(minutes=config.PREDICTION_TOL_MIN),
    )
    merged["ntr_m"] = merged["verified_m"] - merged["predicted_m"]
    return merged.dropna(subset=["ntr_m"]).set_index("datetime")


def load_stnd(name, station_id):
    """Observed water level relative to station datum, indexed by UTC time."""
    path = config.NOAA_DIR / f"NOAA_{station_id}_{name}_verified_stnd.csv"
    if not path.exists():
        print(f"  [warn] missing {path.name}")
        return None
    return read_noaa(path, "gauge_stnd_m").dropna(subset=["gauge_stnd_m"]).set_index("datetime")


def nearest_record(noaa, overpass_time):
    """Nearest gauge record within TEMPORAL_TOL_MIN of the overpass, or None.

    Returns (row, minutes between overpass and record).
    """
    t = pd.Timestamp(overpass_time)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    tol = pd.Timedelta(minutes=config.TEMPORAL_TOL_MIN)
    window = noaa[(noaa.index >= t - tol) & (noaa.index <= t + tol)]
    if window.empty:
        return None
    offsets = pd.Series((window.index - t).total_seconds(), index=window.index)
    nearest = offsets.abs().idxmin()
    return window.loc[nearest], abs((nearest - t).total_seconds()) / 60.0


def match_ntr(swot, stations):
    """Pair SWOT overpasses (NTR or DAC-restored) with the NOAA NTR."""
    pairs = []
    for name in stations:
        sid = config.STATIONS[name]["id"]
        noaa = load_ntr(name, sid)
        if noaa is None:
            continue
        sub = swot[swot["station"] == name]
        n = 0
        for _, row in sub.iterrows():
            hit = nearest_record(noaa, row["overpass_time"])
            if hit is None:
                continue
            rec, dt = hit
            pair = row.to_dict()
            pair.update(noaa_verified_m=round(float(rec["verified_m"]), 4),
                        noaa_predicted_m=round(float(rec["predicted_m"]), 4),
                        ntr_m=round(float(rec["ntr_m"]), 4),
                        time_diff_min=round(dt, 2))
            pairs.append(pair)
            n += 1
        print(f"  {name:<10} {len(sub):>4} overpasses -> {n:>4} pairs")
    cols = ["station", "region", "pass_num", "cycle_num", "overpass_time",
            "ssha_median_m", "ssha_std_m", "n_pixels", "min_dist_km",
            "noaa_verified_m", "noaa_predicted_m", "ntr_m", "time_diff_min", "nc_file"]
    return pd.DataFrame(pairs)[cols]


def match_twl(swot):
    """Pair SWOT TWL overpasses with MSS-referenced gauge water levels."""
    pairs = []
    for name, twl_info in config.TWL_STATIONS.items():
        sid = config.STATIONS[name]["id"]
        noaa = load_stnd(name, sid)
        if noaa is None:
            continue
        sub = swot[swot["station"] == name]
        n = 0
        for _, row in sub.iterrows():
            hit = nearest_record(noaa, row["overpass_time"])
            if hit is None:
                continue
            rec, dt = hit
            gauge_stnd = float(rec["gauge_stnd_m"])
            offset = float(row["mss_median_m"]) - twl_info["msl_stnd_m"]   # MSS - MSL_STND
            pair = row.to_dict()
            pair.update(gauge_stnd_m=round(gauge_stnd, 4),
                        gauge_twl_m=round(gauge_stnd + offset, 4),
                        datum_offset_m=round(offset, 4),
                        time_diff_min=round(dt, 2))
            pairs.append(pair)
            n += 1
        print(f"  {name:<10} {len(sub):>4} overpasses -> {n:>4} pairs "
              f"(MSL_STND = {twl_info['msl_stnd_m']} m)")
    cols = ["station", "region", "pass_num", "cycle_num", "overpass_time",
            "twl_median_m", "twl_std_m", "mss_median_m", "n_pixels", "min_dist_km",
            "gauge_stnd_m", "gauge_twl_m", "datum_offset_m", "time_diff_min", "nc_file"]
    return pd.DataFrame(pairs)[cols]


def main():
    if not config.NOAA_DIR.exists():
        raise SystemExit(f"NOAA folder not found: {config.NOAA_DIR}. Check NOAA_DIR in config.py.")
    for key in ("ntr", "dac", "twl"):
        src = config.OUTPUT_DIR / f"swot_overpasses_{key}.csv"
        if not src.exists():
            raise SystemExit(f"{src} not found - run 02_process_swot.py first.")
        swot = pd.read_csv(src, parse_dates=["overpass_time"])
        print(f"\n{key.upper()}: {len(swot)} SWOT overpasses")
        df = match_twl(swot) if key == "twl" else match_ntr(swot, list(config.STATIONS))
        df = df.sort_values(["station", "overpass_time"]).reset_index(drop=True)
        out = config.OUTPUT_DIR / f"matched_pairs_{key}.csv"
        df.to_csv(out, index=False)
        print(f"  {len(df)} matched pairs -> {out.name} "
              f"(max time offset {df['time_diff_min'].max():.1f} min)")


if __name__ == "__main__":
    main()
