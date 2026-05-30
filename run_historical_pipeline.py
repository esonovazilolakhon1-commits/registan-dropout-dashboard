# run_historical_pipeline.py  (STATIC / HISTORICAL DATA PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────
# Processes the original historical dataset (data/raw/) and generates static
# dropout predictions for all students in the dissertation study.
#
# What it does:
#   clean → build features → encode → generate predictions
# Output: data/processed/predictions.parquet
#
# Prerequisites: data/raw/ files must already be present (from MongoDB export).
# Does NOT call the API — use api_on_process/run_live_pipeline.py for live data.
#
# How to run:
#   python3 run_historical_pipeline.py
#
# Companion script for monthly live updates:
#   python3 api_on_process/run_live_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Pipeline steps (in order) ─────────────────────────────────────────────────
STEPS = [
    # --- CLEAN (one script per raw data file) ---
    ("Clean branches",          "src/clean/clean_branches.py"),
    ("Clean users",             "src/clean/clean_users.py"),
    ("Clean courses",           "src/clean/clean_courses.py"),
    ("Clean levels",            "src/clean/clean_levels.py"),
    ("Clean groups",            "src/clean/clean_groups.py"),
    ("Clean lessons",           "src/clean/clean_lessons.py"),
    ("Clean students",          "src/clean/clean_students.py"),
    ("Clean student-groups",    "src/clean/clean_studentgroups.py"),
    ("Clean student-teachers",  "src/clean/clean_studentteachers.py"),
    ("Clean group histories",   "src/clean/clean_grouphistories.py"),
    ("Clean student histories", "src/clean/clean_studenthistories.py"),
    ("Clean attendance",        "src/clean/clean_attendance.py"),
    ("Clean transactions",      "src/clean/clean_transactions.py"),
    ("Clean orders",            "src/clean/clean_orders.py"),

    # --- FEATURES ---
    ("Build master table",      "src/features/build_master.py"),
    ("Build snapshots",         "src/features/build_snapshots.py"),
    ("Build snapshot features", "src/features/build_snapshot_features.py"),
    ("Encode snapshots",        "src/features/encode_snapshots.py"),

    # --- MODEL ---
    ("Generate predictions",    "src/models/predict.py"),
]


def install_requirements():
    """Install all libraries from requirements.txt before running the pipeline."""
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        log.warning("requirements.txt not found — skipping library installation.")
        return

    log.info("Installing required libraries from requirements.txt ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        log.info("✓ All libraries installed successfully.")
    else:
        log.warning("Some libraries may not have installed cleanly:")
        for line in result.stderr.splitlines():
            log.warning(f"  {line}")
        log.warning("Continuing anyway — if a step fails, install manually with:")
        log.warning(f"  pip install -r requirements.txt")


def run_step(label: str, script_path: str) -> bool:
    """Run one pipeline step, stream its output, return True if successful."""
    log.info(f"{'─' * 60}")
    log.info(f"  STEP : {label}")
    log.info(f"  FILE : {script_path}")
    log.info(f"{'─' * 60}")

    script = ROOT / script_path
    if not script.exists():
        log.error(f"  ✗ Script not found: {script}")
        return False

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    for line in result.stdout.splitlines():
        log.info(f"    {line}")
    for line in result.stderr.splitlines():
        log.warning(f"    {line}")

    if result.returncode != 0:
        log.error(f"  ✗ FAILED with exit code {result.returncode}")
        return False

    log.info(f"  ✓ Done")
    return True


def run():
    started_at = datetime.now()

    log.info("=" * 60)
    log.info("  REGISTAN HISTORICAL PIPELINE  (static data, no API)")
    log.info(f"  Started : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Log     : {log_file}")
    log.info("=" * 60)

    # Step 0: install libraries
    install_requirements()
    log.info("")

    # Steps 1–N: run each script
    failed = []
    for label, script in STEPS:
        ok = run_step(label, script)
        if not ok:
            failed.append(label)
            log.error("=" * 60)
            log.error("  PIPELINE STOPPED — fix the error above and re-run.")
            log.error("=" * 60)
            sys.exit(1)

    # Summary
    elapsed = (datetime.now() - started_at).seconds
    minutes, seconds = divmod(elapsed, 60)

    log.info("")
    log.info("=" * 60)
    log.info(f"  PIPELINE COMPLETE  ✓  ({minutes}m {seconds}s)")
    log.info(f"  {len(STEPS)} steps ran successfully.")
    log.info("")
    log.info("  Predictions saved to:  data/processed/predictions.parquet")
    log.info("")
    log.info("  To launch the dashboard, run:")
    log.info("    streamlit run src/dashboard/eda_dashboard.py")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
