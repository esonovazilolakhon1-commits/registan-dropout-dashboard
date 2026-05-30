# run_live_pipeline.py  (LIVE / MONTHLY UPDATE PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────
# Fetches fresh data from the Edutizim API, cleans it, builds features,
# and generates live dropout predictions for currently active students.
#
# Runs automatically every last day of the month via GitHub Actions.
# Output: data/processed/live_predictions.parquet
#
# How to run manually:
#   python3 api_on_process/run_live_pipeline.py
#
# Companion script for static/historical data:
#   python3 run_historical_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"live_pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"

STEPS = [
    ("Fetch recent data from API",    "api_on_process/extract_recent.py"),
    ("Clean branches",                "src/clean/clean_branches.py"),
    ("Merge users (historical + new)","api_on_process/merge_users.py"),
    ("Clean courses",                 "src/clean/clean_courses.py"),
    ("Clean levels",                  "src/clean/clean_levels.py"),
    ("Clean groups",                  "src/clean/clean_groups.py"),
    ("Clean lessons",                 "src/clean/clean_lessons.py"),
    ("Clean students",                "src/clean/clean_students.py"),
    ("Clean student-groups",          "src/clean/clean_studentgroups.py"),
    ("Clean student-teachers",        "src/clean/clean_studentteachers.py"),
    ("Clean group histories",         "src/clean/clean_grouphistories.py"),
    ("Clean student histories",       "src/clean/clean_studenthistories.py"),
    ("Clean attendance",              "src/clean/clean_attendance.py"),
    ("Clean transactions",            "src/clean/clean_transactions.py"),
    ("Clean orders",                  "src/clean/clean_orders.py"),
    ("Build master feature table",    "src/features/build_master.py"),
    ("Build snapshot panel",          "src/features/build_snapshots.py"),
    ("Build snapshot features",       "src/features/build_snapshot_features.py"),
    ("Encode snapshots",              "src/features/encode_snapshots.py"),
    ("Generate live predictions",     "api_on_process/predict_live.py"),
]

TOTAL = len(STEPS)
WIDTH = 40  # progress bar width



def progress_bar(done: int, total: int) -> str:
    pct  = done / total
    fill = int(WIDTH * pct)
    bar  = "█" * fill + "░" * (WIDTH - fill)
    return f"[{bar}] {pct*100:5.1f}%  ({done}/{total})"


def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       REGISTAN LIVE PIPELINE  (API → clean → predict)       ║")
    print(f"║  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                              ║")
    print(f"║  Fetch   : 2026-03-19 → now                                ║")
    print(f"║  Output  : data/processed/live_predictions.parquet         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def run_step(step_num: int, label: str, script_path: str, log_f) -> bool:
    script = ROOT / script_path
    pct    = (step_num - 1) / TOTAL

    print(f"\n{'─' * 64}")
    print(f"  Step {step_num:>2}/{TOTAL}  {progress_bar(step_num - 1, TOTAL)}")
    print(f"  ▶  {label}")
    print(f"{'─' * 64}")

    if not script.exists():
        msg = f"  ✗  Script not found: {script}"
        print(msg)
        log_f.write(msg + "\n")
        return False

    # Pass LIVE_PIPELINE=1 so config.py redirects all paths to
    # data/live_raw, data/live_interim, data/live_processed —
    # the original data/ directories are NEVER touched.
    env = os.environ.copy()
    env["LIVE_PIPELINE"] = "1"

    # Stream output line-by-line so user sees progress in real time
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True,
        cwd=str(ROOT),
        bufsize=1,
        env=env,
    )

    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(f"     {line}")
        log_f.write(f"     {line}\n")
        output_lines.append(line)

    proc.wait()

    if proc.returncode != 0:
        print(f"\n  ✗  FAILED  (exit code {proc.returncode})")
        log_f.write(f"  FAILED exit code {proc.returncode}\n")
        return False

    print(f"  ✓  Done")
    log_f.write(f"  Done\n")
    return True


def run():
    started_at = datetime.now()
    print_header()

    with open(log_file, "w", encoding="utf-8") as log_f:
        log_f.write(f"Live pipeline started: {started_at}\n\n")

        for i, (label, script) in enumerate(STEPS, start=1):
            ok = run_step(i, label, script, log_f)

            if not ok:
                print()
                print("╔══════════════════════════════════════════════════════════════╗")
                print("║                   ✗  PIPELINE CRASHED                       ║")
                print(f"║  Stopped at step {i:>2}: {label:<40}║")
                print(f"║  Log saved to: {str(log_file)[-46:]:<46}  ║")
                print("╚══════════════════════════════════════════════════════════════╝")
                print()
                log_f.write(f"\nPIPELINE CRASHED at step {i}: {label}\n")
                sys.exit(1)

        elapsed = (datetime.now() - started_at).seconds
        minutes, seconds = divmod(elapsed, 60)

        print()
        print(f"\n  {progress_bar(TOTAL, TOTAL)}")
        print()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║                  ✓  PIPELINE COMPLETE                       ║")
        print(f"║  Time   : {minutes}m {seconds}s                                          ║")
        print(f"║  Steps  : {TOTAL}/{TOTAL} completed                                    ║")
        print(f"║  Output : live_predictions.parquet                         ║")
        print(f"║  Note   : predictions.parquet was NOT modified              ║")
        print("╠══════════════════════════════════════════════════════════════╣")
        print("║  → Refresh the dashboard → open 'Live Predictions' page     ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()

        log_f.write(f"\nPIPELINE COMPLETE in {minutes}m {seconds}s\n")


if __name__ == "__main__":
    run()
