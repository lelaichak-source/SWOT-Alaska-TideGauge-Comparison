"""
config.py
=========
Central settings for the SWOT vs. NOAA tide-gauge analysis. Every script in
this repository imports from here, so this is the only file you should need
to edit to run the pipeline on your own machine.

Where the data live
-------------------
By default, data are read from and written to a ``data/`` folder next to this
file. To use a different location, either edit ``DATA_ROOT`` below or set the
environment variable ``SWOT_THESIS_DATA`` before running, e.g.

    Windows (PowerShell):  $env:SWOT_THESIS_DATA = "D:\\path\\to\\data"
    macOS / Linux:         export SWOT_THESIS_DATA=/path/to/data

Expected layout inside DATA_ROOT::

    SWOT/Pass125/*.nc, SWOT/Pass136/*.nc, ...   SWOT L3 files, one folder per pass
    NOAA/                                       tide-gauge CSVs (script 01)
    output/                                     everything the pipeline produces
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(os.environ.get("SWOT_THESIS_DATA",
                                Path(__file__).resolve().parent / "data"))

SWOT_DIR   = DATA_ROOT / "SWOT"      # one sub-folder per pass: Pass459, Pass470, ...
NOAA_DIR   = DATA_ROOT / "NOAA"      # NOAA CO-OPS CSVs
OUTPUT_DIR = DATA_ROOT / "output"    # matched pairs, statistics, figures, tables

STATS_DIR  = OUTPUT_DIR / "stats"
FIG_DIR    = OUTPUT_DIR / "figures"
TABLE_DIR  = OUTPUT_DIR / "tables"

# ── Study period ─────────────────────────────────────────────────────────────
# NOAA records are downloaded for the full span below. Only the ice-free
# months are used in the comparison (applied to SWOT overpasses in script 02
# and to the station-datum download in script 01).
STUDY_START = "2023-04-01"
STUDY_END   = "2026-10-31"
ICE_FREE_MONTHS = [4, 5, 6, 7, 8, 9, 10]          # April-October

# ── Tide-gauge stations ──────────────────────────────────────────────────────
# NOAA CO-OPS station ID, gauge coordinates (decimal degrees, longitude
# negative = west), and the regional group used throughout the thesis.
STATIONS = {
    "Anchorage": {"id": "9455920", "lat": 61.2381, "lon": -149.8918, "region": "Cook Inlet"},
    "Nikiski":   {"id": "9455760", "lat": 60.6831, "lon": -151.3964, "region": "Cook Inlet"},
    "Seldovia":  {"id": "9455500", "lat": 59.4397, "lon": -151.7197, "region": "Cook Inlet"},
    "Seward":    {"id": "9455090", "lat": 60.1197, "lon": -149.4267, "region": "Prince William Sound"},
    "Valdez":    {"id": "9454240", "lat": 61.1253, "lon": -146.3567, "region": "Prince William Sound"},
    "Cordova":   {"id": "9454050", "lat": 60.5581, "lon": -145.7547, "region": "Prince William Sound"},
    "Kodiak":    {"id": "9457292", "lat": 57.7317, "lon": -152.5119, "region": "Kodiak"},
    "Alitak":    {"id": "9457804", "lat": 56.8983, "lon": -154.2483, "region": "Kodiak"},
}

# SWOT science-orbit passes that cross each station's search area (Table 1).
STATION_PASSES = {
    "Anchorage": [164, 181, 459, 470],
    "Nikiski":   [181, 192, 459, 498],
    "Seldovia":  [181, 220, 459, 498],
    "Kodiak":    [153, 248, 459, 526],
    "Alitak":    [181, 276, 459, 554],
    "Seward":    [153, 192, 431, 470],
    "Valdez":    [125, 136, 431, 442],
    "Cordova":   [125, 136, 403, 442],
}

# ── Total water level (TWL) stations ─────────────────────────────────────────
# The TWL comparison uses three stations. They were selected because they are
# the study stations with published station-datum-to-NAVD88 relationships
# (the first step toward an ellipsoidal tie). A full tie to the ellipsoid was
# not possible in Alaska, so the comparison is mean-referenced instead
# (see script 03 and thesis Section 3.6). NAVD88 itself is not used.
#
# msl_stnd_m = mean sea level above station datum (STND), metres, from the
# NOAA CO-OPS datums API, retrieved 2026-07-03.
TWL_STATIONS = {
    "Anchorage": {"msl_stnd_m": 6.931},
    "Nikiski":   {"msl_stnd_m": 5.453},
    "Kodiak":    {"msl_stnd_m": 9.074},
}

# ── Matching parameters ──────────────────────────────────────────────────────
RADIUS_KM            = 6.0     # search radius around each gauge (km)
RADIUS_KM_WIDE       = 50.0    # used only for the fjord-head gauges below
WIDE_RADIUS_STATIONS = {"Seward", "Valdez"}
MAD_THRESHOLD        = 3.0     # drop pixels more than N x MAD from the median
TEMPORAL_TOL_MIN     = 3       # max minutes between SWOT overpass and gauge record
PREDICTION_TOL_MIN   = 6       # max minutes between gauge observation and prediction
GOOD_FLAG            = 0       # SWOT quality_flag value kept ("good")

# ── Figures ──────────────────────────────────────────────────────────────────
# Each saved figure gets a small grey footer naming the data version.
DATA_VERSION_LABEL = ("Data: SWOT L3_LR_SSH Unsmoothed v3.0 "
                      "(ice-free seasons 2023-2026, through Aug 2026)")
STAMP_FIGURES = True
