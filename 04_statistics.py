"""
04_statistics.py
================
Agreement statistics for the three comparisons (NTR, DAC-restored, TWL).

For each comparison, with x = NOAA value and y = SWOT value:

* Per station and per region (stations pooled):
      n, Pearson r and its p-value, bias = mean(SWOT - gauge),
      RMSE, and the standard deviation of the differences.
* Per station OLS regression of SWOT on gauge: slope, its standard error,
  95% confidence half-width (t distribution, n - 2 df), deviation from 1,
  intercept, r, p, bias and RMSE. A slope whose interval excludes 1 means
  SWOT systematically over- or under-represents gauge variability.
* A linear mixed-effects model across stations,
      SWOT ~ gauge + (1 | station),
  fitted by restricted maximum likelihood (statsmodels MixedLM).

Outputs (OUTPUT_DIR/stats/):
    stats_by_station_{ntr,dac,twl}.csv
    stats_by_region_{ntr,dac,twl}.csv
    regression_slopes_{ntr,dac,twl}.csv
    mixed_effects_{ntr,dac,twl}.txt

Run:
    python 04_statistics.py
"""

import pandas as pd
import statsmodels.formula.api as smf

from swot_utils import PIPELINES, agreement_stats, load_pairs, regression_with_ci, stats_path


def stats_row(group, x, y):
    s = agreement_stats(group[x], group[y])
    return {"n": s["n"], "r": round(s["r"], 4), "p_value": round(s["p"], 4),
            "bias_m": round(s["bias"], 4), "rmse_m": round(s["rmse"], 4),
            "std_diff_m": round(s["std_diff"], 4)}


def per_station(df, key):
    p = PIPELINES[key]
    rows = []
    for station, g in df.groupby("station"):
        row = {"station": station, "region": g["region"].iloc[0], **stats_row(g, p["x"], p["y"])}
        if key == "twl":
            # Mean MSS - MSL_STND offset applied to the gauge (sanity check).
            row["mean_datum_offset_m"] = round(float(g["datum_offset_m"].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def per_region(df, key):
    p = PIPELINES[key]
    return pd.DataFrame([{"region": region, **stats_row(g, p["x"], p["y"])}
                         for region, g in df.groupby("region")])


def slopes(df, key):
    p = PIPELINES[key]
    rows = []
    for station in p["stations"]:
        g = df[df["station"] == station]
        reg = regression_with_ci(g[p["x"]], g[p["y"]])
        if reg is None:
            continue
        s = agreement_stats(g[p["x"]], g[p["y"]])
        rows.append({
            "station": station, "region": g["region"].iloc[0], "n": reg["n"],
            "slope": round(reg["slope"], 4), "slope_se": round(reg["se"], 4),
            "slope_ci95": round(reg["ci95"], 4),
            "slope_dev_from_1": round(reg["slope"] - 1.0, 4),
            "intercept": round(reg["intercept"], 4), "r": round(reg["r"], 4),
            "p_value": round(reg["p"], 4), "bias_m": round(s["bias"], 4),
            "rmse_m": round(s["rmse"], 4),
        })
    return pd.DataFrame(rows)


def mixed_model(df, key):
    """Fit SWOT ~ gauge + (1 | station) and return the summary text."""
    p = PIPELINES[key]
    data = df.rename(columns={p["y"]: "swot", p["x"]: "gauge"})
    result = smf.mixedlm("swot ~ gauge", data=data, groups=data["station"]).fit(reml=True)
    header = (f"Linear mixed-effects model - {p['title']} (all matched pairs)\n"
              f"SWOT ({p['y']}) ~ gauge ({p['x']}) + (1 | station), fitted by REML\n"
              f"{df['station'].nunique()} stations, {len(df)} pairs\n\n")
    return header + str(result.summary()), result


def main():
    for key, p in PIPELINES.items():
        try:
            df = load_pairs(key)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        print(f"\n{'=' * 70}\n{p['title']}: {len(df)} pairs\n{'=' * 70}")

        st = per_station(df, key)
        st.to_csv(stats_path(f"stats_by_station_{key}.csv"), index=False)
        print(st.to_string(index=False))

        per_region(df, key).to_csv(stats_path(f"stats_by_region_{key}.csv"), index=False)
        slopes(df, key).to_csv(stats_path(f"regression_slopes_{key}.csv"), index=False)

        try:
            text, result = mixed_model(df, key)
            stats_path(f"mixed_effects_{key}.txt").write_text(text, encoding="utf-8")
            print(f"\nMixed model: gauge coefficient = {result.params['gauge']:.4f} "
                  f"(p = {result.pvalues['gauge']:.4g})")
        except Exception as e:          # e.g. too few groups to converge
            print(f"Mixed model failed: {e}")

    print("\nSaved statistics to OUTPUT_DIR/stats/")


if __name__ == "__main__":
    main()
