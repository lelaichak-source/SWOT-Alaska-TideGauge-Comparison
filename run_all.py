"""
run_all.py
==========
Run the pipeline scripts in order and stop at the first error.

    python run_all.py              # processing, statistics, figures, tables (02-10)
    python run_all.py --from 04    # start at a later step, e.g. after a fix
    python run_all.py --download   # also run the download steps (00 and 01) first

Everything the scripts print is also saved to run_all_log.txt.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

DOWNLOADS = ["00_download_swot.py", "01_download_noaa.py"]
STEPS = [
    "02_process_swot.py",
    "03_match_noaa_swot.py",
    "04_statistics.py",
    "05_figures_agreement.py",
    "06_van_de_casteele.py",
    "07_bias_corrected.py",
    "08_tidal_regime.py",
    "09_pipeline_comparisons.py",
    "10_thesis_tables.py",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--from", dest="start", default="02",
                    help="two-digit number of the first step to run (default 02)")
    ap.add_argument("--download", action="store_true", help="run 00 and 01 first")
    args = ap.parse_args()

    todo = [s for s in STEPS if s[:2] >= args.start]
    if args.download:
        todo = DOWNLOADS + todo
    if not todo:
        sys.exit(f"No steps at or after {args.start}.")

    # Force UTF-8 so characters such as "–" print correctly on Windows.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    log_path = HERE / "run_all_log.txt"
    t_all = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for i, script in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {script} ...", end=" ", flush=True)
            t0 = time.time()
            proc = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE, env=env,
                                  capture_output=True, text=True, encoding="utf-8",
                                  errors="replace")
            log.write(f"\n{'=' * 78}\n{script}\n{'=' * 78}\n{proc.stdout}{proc.stderr}")
            log.flush()
            if proc.returncode != 0:
                print("FAILED")
                print("\n".join((proc.stderr or proc.stdout).strip().splitlines()[-15:]))
                sys.exit(f"\nStopped. Fix {script}, then resume with: "
                         f"python run_all.py --from {script[:2]}\nFull log: {log_path}")
            print(f"ok ({time.time() - t0:.0f} s)")
    print(f"\nFinished {len(todo)} scripts in {(time.time() - t_all) / 60:.1f} min. Log: {log_path}")


if __name__ == "__main__":
    main()
