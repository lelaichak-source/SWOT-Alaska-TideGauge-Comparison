"""
01_download_noaa.py
===================
Download NOAA CO-OPS 6-minute water levels from the CO-OPS data API.

Two sets of files are written to NOAA_DIR:

1. For all eight stations, relative to Mean Lower Low Water (MLLW), used by
   the NTR and DAC-restored comparisons:
       NOAA_<id>_<name>_verified.csv     observed (verified) water level
       NOAA_<id>_<name>_predicted.csv    harmonic tidal prediction
   These cover the full study period (STUDY_START to STUDY_END).

2. For the three TWL stations, relative to Station Datum (STND), used by the
   total-water-level comparison:
       NOAA_<id>_<name>_verified_stnd.csv
   These cover the ice-free months of each year in the study period.

Each CSV has two columns: datetime (UTC) and water_level_m (metres).
Times are requested in GMT, which is UTC. The API limits each request to 31
days, so data are fetched one calendar month at a time.

Existing files are skipped; delete a file to download it again.

Run:
    python 01_download_noaa.py
"""

import calendar
import sys
import time

import pandas as pd
import requests

import config

API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
PAUSE_S = 1.0                     # polite pause between requests
APPLICATION = "swot_noaa_calval"  # identifies the requests to NOAA


def month_windows(start, end):
    """Yield (YYYYMMDD, YYYYMMDD) for each calendar month between two dates."""
    for month_start in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        last_day = calendar.monthrange(month_start.year, month_start.month)[1]
        month_end = min(month_start.replace(day=last_day), pd.Timestamp(end))
        yield max(month_start, pd.Timestamp(start)).strftime("%Y%m%d"), month_end.strftime("%Y%m%d")


def fetch(station_id, product, datum, begin, end):
    """Fetch one window of 6-minute data. Returns a DataFrame or None."""
    params = {
        "station": station_id, "product": product, "datum": datum,
        "time_zone": "GMT", "units": "metric", "format": "json",
        "begin_date": begin, "end_date": end, "application": APPLICATION,
    }
    try:
        r = requests.get(API_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    request error: {e}")
        return None
    if "error" in data:
        print(f"    API error: {data['error'].get('message', data['error'])}")
        return None

    # Observations come back under "data", predictions under "predictions".
    records = data.get("data") or data.get("predictions")
    if not records:
        return None
    df = pd.DataFrame(records).rename(columns={"t": "datetime", "v": "water_level_m"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["water_level_m"] = pd.to_numeric(df["water_level_m"], errors="coerce")
    return df[["datetime", "water_level_m"]].dropna()


def download(station_id, product, datum, windows, out_path):
    """Download all windows for one station/product and save one CSV."""
    if out_path.exists():
        print(f"  [skip] {out_path.name} already exists")
        return
    frames = []
    for begin, end in windows:
        df = fetch(station_id, product, datum, begin, end)
        print(f"    {begin}-{end}: {0 if df is None else len(df)} records")
        if df is not None:
            frames.append(df)
        time.sleep(PAUSE_S)
    if not frames:
        print(f"  [warn] no data retrieved for {out_path.name}")
        return
    out = (pd.concat(frames).drop_duplicates("datetime")
           .sort_values("datetime").reset_index(drop=True))
    out.to_csv(out_path, index=False)
    print(f"  saved {len(out)} records -> {out_path.name}")


def main():
    config.NOAA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {config.NOAA_DIR}\n")

    # 1. MLLW observations and predictions, all stations, full study period.
    full_period = list(month_windows(config.STUDY_START, config.STUDY_END))
    for name, info in config.STATIONS.items():
        print(f"{name} ({info['id']}) - MLLW")
        for product, suffix in [("water_level", "verified"), ("predictions", "predicted")]:
            out = config.NOAA_DIR / f"NOAA_{info['id']}_{name}_{suffix}.csv"
            download(info["id"], product, "MLLW", full_period, out)

    # 2. STND observations, TWL stations, ice-free months only.
    years = range(pd.Timestamp(config.STUDY_START).year, pd.Timestamp(config.STUDY_END).year + 1)
    ice_free = [(f"{y}{m:02d}01", f"{y}{m:02d}{calendar.monthrange(y, m)[1]:02d}")
                for y in years for m in config.ICE_FREE_MONTHS]
    for name in config.TWL_STATIONS:
        sid = config.STATIONS[name]["id"]
        print(f"{name} ({sid}) - STND")
        download(sid, "water_level", "STND", ice_free,
                 config.NOAA_DIR / f"NOAA_{sid}_{name}_verified_stnd.csv")

    print("\nDone.")


if __name__ == "__main__":
    sys.exit(main())
