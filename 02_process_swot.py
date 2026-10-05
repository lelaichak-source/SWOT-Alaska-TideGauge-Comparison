"""
02_process_swot.py
==================
Extract one SWOT value per station per overpass for the three comparisons.

For every SWOT file of every pass associated with a station (config.STATION_PASSES):

1. Representative overpass time = median of the file's valid (non-zero) time
   records. Files are spatial subsets of the study region, so these span only
   about two minutes. Overpasses outside the ice-free months are skipped.
2. Pixels are kept if they lie within the station's search radius (Haversine
   distance; 6 km, or 50 km at Seward and Valdez), have quality_flag == 0, and
   have valid (unmasked) values for the variables that comparison needs.
3. The comparison value is formed at each pixel:
       NTR : ssha_unfiltered
       DAC : ssha_unfiltered + dac
       TWL : ssha_unfiltered + ocean_tide + dac + mss + internal_tide
             (internal_tide added where present; TWL stations only, 6 km)
4. Pixel outliers are removed with the MAD filter (swot_utils.mad_inliers).
5. The median of the retained pixels is the SWOT value for that overpass. The
   pixel count, standard deviation and nearest-pixel distance are also saved.

Longitudes are converted from 0-360 to -180..180. netCDF4 applies each
variable's scale factor and offset automatically when it is read.

Outputs (OUTPUT_DIR):
    swot_overpasses_ntr.csv
    swot_overpasses_dac.csv    (ssha_median_m holds the DAC-restored value)
    swot_overpasses_twl.csv    (also stores the footprint MSS median)

Run:
    python 02_process_swot.py
"""

import netCDF4 as nc
import numpy as np
import pandas as pd

import config
from swot_utils import haversine_km, mad_inliers, parse_cycle_pass, swot_time_to_utc

TWL_VARS = ["ssha_unfiltered", "ocean_tide", "dac", "mss"]


def valid(arr):
    """True where a (possibly masked) array has a value."""
    return ~np.ma.getmaskarray(arr)


def summarise(values, dist, keep, extra=None):
    """Median, SD, count and nearest distance of the MAD-retained pixels."""
    values, dist = values[keep], dist[keep]
    if len(values) == 0:
        return None
    row = {"median": round(float(np.median(values)), 6),
           "std": round(float(np.std(values)), 6),
           "n_pixels": int(len(values)),
           "min_dist_km": round(float(np.min(dist)), 4)}
    if extra is not None:
        row["mss_median_m"] = round(float(np.median(extra[keep])), 6)
    return row


def process_file(nc_path, station, info):
    """Return {'ntr': row, 'dac': row, 'twl': row} for one file and station
    (a value is None when that comparison has no usable pixels)."""
    out = {"ntr": None, "dac": None, "twl": None}
    try:
        ds = nc.Dataset(nc_path)
    except Exception as e:
        print(f"    [skip] could not open {nc_path.name}: {e}")
        return out

    with ds:
        v = ds.variables

        # Overpass time; skip anything outside the ice-free season.
        t = v["time"][:]
        t = t[~np.ma.getmaskarray(np.ma.masked_equal(t, 0))]
        if len(t) == 0:
            return out
        overpass_time = swot_time_to_utc(np.nanmedian(t))
        if overpass_time.month not in config.ICE_FREE_MONTHS:
            return out

        lat = v["latitude"][:]
        lon = v["longitude"][:]
        lon = np.where(lon > 180, lon - 360, lon)
        dist = haversine_km(info["lat"], info["lon"], lat, lon)
        good = np.isin(v["quality_flag"][:], [config.GOOD_FLAG])
        ssha = v["ssha_unfiltered"][:]

        cycle, pass_num = parse_cycle_pass(nc_path.name)
        base = {"station": station, "region": info["region"], "pass_num": pass_num,
                "cycle_num": cycle, "overpass_time": overpass_time}

        def finish(s):
            if s is None:
                return None
            row = dict(base)
            row.update(s)
            row["nc_file"] = nc_path.name
            return row

        radius = (config.RADIUS_KM_WIDE if station in config.WIDE_RADIUS_STATIONS
                  else config.RADIUS_KM)
        in_area = dist <= radius

        # ── NTR: ssha_unfiltered ─────────────────────────────────────────────
        m = good & valid(ssha) & in_area
        if m.any():
            vals = np.array(ssha[m], dtype=np.float64)
            out["ntr"] = finish(summarise(vals, dist[m], mad_inliers(vals)))

        # ── DAC-restored: ssha_unfiltered + dac ─────────────────────────────
        if "dac" in v:
            dac = v["dac"][:]
            m = good & valid(ssha) & valid(dac) & in_area
            if m.any():
                vals = (np.array(ssha[m], dtype=np.float64)
                        + np.array(dac[m], dtype=np.float64))
                out["dac"] = finish(summarise(vals, dist[m], mad_inliers(vals)))

        # ── TWL: ssha_unfiltered + ocean_tide + dac + mss (+ internal_tide) ──
        if station in config.TWL_STATIONS and all(name in v for name in TWL_VARS):
            arrays = {name: v[name][:] for name in TWL_VARS}
            m = good & (dist <= config.RADIUS_KM)
            for a in arrays.values():
                m &= valid(a)
            if m.any():
                parts = {k: np.array(a[m], dtype=np.float64) for k, a in arrays.items()}
                vals = (parts["ssha_unfiltered"] + parts["ocean_tide"]
                        + parts["dac"] + parts["mss"])
                if "internal_tide" in v:
                    it = np.ma.filled(v["internal_tide"][:][m].astype(np.float64), 0.0)
                    vals += np.where(np.isfinite(it), it, 0.0)
                out["twl"] = finish(summarise(vals, dist[m], mad_inliers(vals),
                                              extra=parts["mss"]))
    return out


# Column names used in each output file.
COLUMN_NAMES = {
    "ntr": {"median": "ssha_median_m", "std": "ssha_std_m"},
    "dac": {"median": "ssha_median_m", "std": "ssha_std_m"},
    "twl": {"median": "twl_median_m", "std": "twl_std_m"},
}
COLUMN_ORDER = {
    "ntr": ["station", "region", "pass_num", "cycle_num", "overpass_time",
            "ssha_median_m", "ssha_std_m", "n_pixels", "min_dist_km", "nc_file"],
    "twl": ["station", "region", "pass_num", "cycle_num", "overpass_time",
            "twl_median_m", "twl_std_m", "mss_median_m", "n_pixels", "min_dist_km", "nc_file"],
}
COLUMN_ORDER["dac"] = COLUMN_ORDER["ntr"]


def main():
    print(f"SWOT files: {config.SWOT_DIR}")
    rows = {"ntr": [], "dac": [], "twl": []}
    n_files = 0

    for station, info in config.STATIONS.items():
        print(f"\n{station} (passes {config.STATION_PASSES[station]})")
        for pass_num in config.STATION_PASSES[station]:
            files = sorted((config.SWOT_DIR / f"Pass{pass_num}").glob("*.nc"))
            if not files:
                print(f"  [warn] no files for pass {pass_num}")
                continue
            found = {"ntr": 0, "dac": 0, "twl": 0}
            for f in files:
                n_files += 1
                for key, row in process_file(f, station, info).items():
                    if row is not None:
                        rows[key].append(row)
                        found[key] += 1
            print(f"  pass {pass_num}: {len(files)} files -> "
                  f"{found['ntr']} NTR, {found['dac']} DAC, {found['twl']} TWL overpasses")

    if n_files == 0:
        raise SystemExit(f"\nNo SWOT files found under {config.SWOT_DIR}.\n"
                         "Check DATA_ROOT / SWOT_DIR in config.py (expected Pass<NNN> "
                         "sub-folders containing .nc files).")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nScanned {n_files} station-file combinations.")
    for key, key_rows in rows.items():
        if not key_rows:
            print(f"  {key.upper()}: no overpasses found")
            continue
        df = (pd.DataFrame(key_rows).rename(columns=COLUMN_NAMES[key])
              [COLUMN_ORDER[key]]
              .sort_values(["station", "overpass_time"]).reset_index(drop=True))
        out = config.OUTPUT_DIR / f"swot_overpasses_{key}.csv"
        df.to_csv(out, index=False)
        print(f"  {key.upper()}: {len(df)} overpasses -> {out.name}")


if __name__ == "__main__":
    main()
