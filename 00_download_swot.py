"""
00_download_swot.py
===================
Download SWOT Level-3 KaRIn Low-Rate SSH "Unsmoothed" v3.0 files from AVISO+
as spatial subsets covering the study region.

Only a Gulf of Alaska box around the eight stations and only the variables
the pipeline reads are requested, so each file is a few MB instead of ~200 MB.
Files are saved to SWOT_DIR/Pass<NNN>/ with their original AVISO names.

Requirements
------------
* A free AVISO+ account with access to the SWOT L3 products.
* The official AVISO download package:  pip install altimetry_downloader_aviso
  On the first run the package asks for your AVISO username and password and
  stores them in your user folder (~/.altimetry/.netrc). No credentials are
  stored in this repository.

Notes
-----
* 2023 is requested by science-orbit cycle number (cycles 1-5, late July to
  October 2023). A date window starting in July 2023 overlaps the end of the
  1-day calibration orbit and makes the request fail.
* Existing files are skipped, so re-running the script resumes or fills gaps.

Dataset DOI: 10.24400/527896/A01-2024.003

Run:
    python 00_download_swot.py
"""

# The AVISO package must be imported before netCDF4/xarray.
import altimetry_downloader_aviso as dl_aviso

import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import config

# ── Request settings ─────────────────────────────────────────────────────────
PRODUCT = "SWOT_L3_LR_SSH_Unsmoothed"
VERSION = "3.0"

# (lon_min, lat_min, lon_max, lat_max): the stations plus a margin larger than
# the 50 km wide search radius.
BOX = (-155.6, 56.3, -144.5, 61.8)

VARIABLES = ["time", "latitude", "longitude", "quality_flag",
             "ssha_unfiltered", "dac", "ocean_tide", "mss", "internal_tide"]

# Ice-free seasons. 2023 uses cycle numbers (see note above); others use dates.
SEASON_CYCLES = {2023: [1, 2, 3, 4, 5]}
SEASONS = {
    2024: ("2024-04-01", "2024-10-31"),
    2025: ("2025-04-01", "2025-10-31"),
    2026: ("2026-04-01", "2026-10-31"),
}
RUN_SEASONS = [2023, 2024, 2025, 2026]

PASSES = sorted({p for plist in config.STATION_PASSES.values() for p in plist})
N_WORKERS = 4      # parallel download processes


def run_job(pass_num, year):
    """Download one pass for one season. Returns (pass, year, n_files, seconds, warnings)."""
    t0 = time.time()
    out_dir = config.SWOT_DIR / f"Pass{pass_num}"
    out_dir.mkdir(parents=True, exist_ok=True)
    when = ({"cycle_number": SEASON_CYCLES[year]} if year in SEASON_CYCLES
            else {"time": SEASONS[year]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        files = dl_aviso.subset(PRODUCT, output_dir=out_dir, pass_number=pass_num,
                                version=VERSION, box=BOX,
                                selected_variables=VARIABLES, overwrite=False, **when)
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    return pass_num, year, len(files or []), time.time() - t0, msgs


def check_one_file():
    """Print the variables and time units of one downloaded file as a sanity check."""
    import netCDF4 as nc
    files = sorted(config.SWOT_DIR.glob("Pass*/*.nc"))
    if not files:
        print("  (no .nc files yet to check)")
        return
    with nc.Dataset(files[0]) as ds:
        print(f"\nCheck file: {files[0].name}")
        missing = [v for v in VARIABLES if v not in ds.variables]
        if missing:
            print(f"  [WARN] missing variables: {missing}")
        units = getattr(ds.variables["time"], "units", "")
        print(f"  time units: {units}")
        if "seconds since 2000-01-01" not in units:
            print("  [WARN] time units differ from 'seconds since 2000-01-01'; "
                  "update swot_utils.SWOT_EPOCH before running script 02.")


def main():
    jobs = [(p, y) for y in RUN_SEASONS for p in PASSES]
    print(f"{PRODUCT} v{VERSION} subsets -> {config.SWOT_DIR}")
    print(f"{len(PASSES)} passes x {len(RUN_SEASONS)} seasons = {len(jobs)} jobs\n")

    # Run the first job in this process so the AVISO package can ask for (and
    # save) your login before the parallel workers start.
    first = jobs.pop(0)
    print(f"First job (Pass {first[0]}, {first[1]}) - enter your AVISO login if asked...")
    p, y, n, dt, msgs = run_job(*first)
    print(f"  Pass {p} {y}: {n} files in {dt:.0f}s")
    for m in msgs:
        print(f"    [warn] {m}")
    check_one_file()

    total, failed = n, []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(run_job, p, y): (p, y) for p, y in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            p, y = futures[fut]
            try:
                _, _, n, dt, msgs = fut.result()
                total += n
                print(f"[{i}/{len(jobs)}] Pass {p} {y}: {n} files in {dt:.0f}s", flush=True)
                for m in msgs:
                    print(f"    [warn] {m}")
            except Exception as e:                      # keep going; report at the end
                failed.append((p, y))
                print(f"[{i}/{len(jobs)}] Pass {p} {y}: FAILED ({type(e).__name__}: {e})",
                      flush=True)

    print(f"\nDone. {total} files in {config.SWOT_DIR}")
    if failed:
        print(f"Failed jobs (re-run the script to retry): {failed}")


if __name__ == "__main__":
    main()
