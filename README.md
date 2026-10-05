# SWOT vs. NOAA Tide Gauges in Southcentral Alaska

Analysis code for an MSES thesis (Lauren Oschak, Alaska Pacific University) comparing
sea surface height from the **SWOT** satellite (Level-3 KaRIn Low-Rate SSH, Unsmoothed,
version 3.0) with water levels at eight **NOAA CO-OPS** tide gauges in Cook Inlet,
Prince William Sound and the Kodiak Archipelago, for the ice-free seasons
(April–October) of 2023–2026.

## What the pipeline does

Three comparisons are run side by side:

| Comparison | SWOT value (per pixel) | Gauge value | Stations |
|---|---|---|---|
| **NTR** (standard) | `ssha_unfiltered` | non-tidal residual = observed − predicted tide (MLLW) | all 8 |
| **DAC-restored** | `ssha_unfiltered + dac` | same NTR | all 8 |
| **TWL** (total water level) | `ssha_unfiltered + ocean_tide + dac + mss + internal_tide` | observed level (station datum) + (MSS − MSL) | Anchorage, Nikiski, Kodiak |

For each station and overpass, SWOT pixels within 6 km of the gauge (50 km at Seward and
Valdez) with `quality_flag == 0` are kept, outliers beyond 3 × MAD of the median are
removed, and the median of the remaining pixels is paired with the nearest 6-minute gauge
record within ±3 minutes. All matched pairs are used (no overpass-level screening).

**About the TWL comparison.** A tie between station datum and the ellipsoid was not
available in Alaska, so the gauge record is placed on the SWOT mean-sea-surface (MSS)
reference using the station's published mean sea level. Because the same MSS enters both
sides, it largely cancels: the TWL comparison tests whether SWOT, with the modelled tide
and atmospheric corrections restored, reproduces total water level variability at the
gauge. It is not an absolute test of SWOT heights.

## Scripts

Run them in order (or use `run_all.py`). Each script's header explains its method,
inputs and outputs in more detail.

| Script | Purpose |
|---|---|
| `config.py` | All settings: data location, stations, SWOT passes, radii, thresholds |
| `swot_utils.py` | Shared helpers: geometry, MAD filter, statistics, plotting conventions |
| `00_download_swot.py` | Download SWOT L3 Unsmoothed v3.0 subsets from AVISO+ |
| `01_download_noaa.py` | Download NOAA observations and predictions (MLLW and station datum) |
| `02_process_swot.py` | One SWOT value per station per overpass, for all three comparisons |
| `03_match_noaa_swot.py` | Compute the NTR and pair each overpass with the nearest gauge record |
| `04_statistics.py` | r, bias, RMSE, SD of differences (by station and region), regression slopes with 95% CI, mixed-effects models |
| `05_figures_agreement.py` | Scatter, time-series, difference, station-summary and slope figures |
| `06_van_de_casteele.py` | Van de Casteele diagrams (error vs. tidal state / water level) |
| `07_bias_corrected.py` | Per-station bias-corrected comparisons and regional pooled plots |
| `08_tidal_regime.py` | Agreement vs. tidal range across stations |
| `09_pipeline_comparisons.py` | NTR → DAC-restored → bias-corrected pathway; slope comparisons |
| `10_thesis_tables.py` | Thesis Tables 1–4 as CSV and a formatted Excel workbook |
| `run_all.py` | Runs 02–10 in order (add `--download` for 00–01) |

## Setup

Python 3.10 or later.

```
pip install -r requirements.txt
```

Point the code at a data folder, either by editing `DATA_ROOT` in `config.py` or by
setting an environment variable:

```
# Windows PowerShell
$env:SWOT_THESIS_DATA = "D:\path\to\data"
# macOS / Linux
export SWOT_THESIS_DATA=/path/to/data
```

The data folder is organised as:

```
data/
├── SWOT/Pass125/*.nc, SWOT/Pass136/*.nc, ...   (script 00)
├── NOAA/*.csv                                  (script 01)
└── output/                                     (everything else)
    ├── swot_overpasses_{ntr,dac,twl}.csv
    ├── matched_pairs_{ntr,dac,twl}.csv
    ├── stats/      statistics tables
    ├── figures/    ntr/, dac/, twl/, comparison/
    └── tables/     thesis tables
```

## Running

```
python run_all.py --download     # first time: download, then run everything
python run_all.py                # re-run processing, statistics, figures and tables
python run_all.py --from 05      # re-run from a later step
```

Downloading SWOT data needs a free [AVISO+](https://www.aviso.altimetry.fr) account; the
download package asks for your login once and stores it in your user folder, not in this
repository. NOAA data need no account.

## Data sources

* SWOT Level-3 KaRIn Low Rate SSH Unsmoothed, v3.0. AVISO/DUACS, CNES.
  doi:[10.24400/527896/A01-2024.003](https://doi.org/10.24400/527896/A01-2024.003).
  Product handbook: SALP-MU-P-EA-23629-CLS.
* NOAA CO-OPS water levels, tidal predictions and datums:
  <https://tidesandcurrents.noaa.gov>.

The SWOT L3 product was produced by AVISO and DUACS teams on behalf of CNES.

## Notes

* Times are UTC throughout. SWOT longitudes (0–360°) are converted to −180–180°.
* In `matched_pairs_dac.csv`, the column `ssha_median_m` holds the DAC-restored value.
* Figures are stamped with a small data-version footer (`STAMP_FIGURES` in `config.py`).
