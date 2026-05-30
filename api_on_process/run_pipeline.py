# run_pipeline.py — runs the full dropout prediction pipeline end to end
# Steps: extract → clean → feature engineering → encode → predict
# The pipeline stops immediately on any failure to avoid silently running
# scoring on stale data.
#
# How to run:
#   python3 src/pipeline/run_pipeline.py

import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

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

STEPS = [
    ("Extract from MongoDB",          "src/pipeline/extract.py"),

    ("Clean branches",                "src/clean/clean_branches.py"),
    ("Clean users",                   "src/clean/clean_users.py"),
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

    ("Generate predictions",          "src/models/predict.py"),
]


def run_step(label: str, script_path: str) -> bool:
    """Run one pipeline step as a subprocess, log stdout/stderr, return success flag."""
    log.info(f"{'─' * 55}")
    log.info(f"  STEP: {label}")
    log.info(f"        {script_path}")
    log.info(f"{'─' * 55}")

    result = subprocess.run(
        [sys.executable, str(ROOT / script_path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    if result.stdout:
        for line in result.stdout.splitlines():
            log.info(f"    {line}")
    if result.stderr:
        for line in result.stderr.splitlines():
            log.warning(f"    STDERR: {line}")

    if result.returncode != 0:
        log.error(f"  ✗ FAILED with exit code {result.returncode}")
        return False

    log.info(f"  ✓ Done")
    return True


def run():
    started_at = datetime.now()
    log.info("=" * 60)
    log.info("REGISTAN DROPOUT PIPELINE — STARTED")
    log.info(f"  {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Log file: {log_file}")
    log.info("=" * 60)

    for label, script in STEPS:
        success = run_step(label, script)
        if not success:
            log.error("=" * 60)
            log.error("PIPELINE ABORTED — fix the error above and re-run.")
            log.error("=" * 60)
            sys.exit(1)

    elapsed = (datetime.now() - started_at).seconds
    minutes, seconds = divmod(elapsed, 60)

    log.info("=" * 60)
    log.info(f"PIPELINE COMPLETE  ({minutes}m {seconds}s)")
    log.info("  predictions.parquet has been updated.")
    log.info("  Refresh the Streamlit dashboard to see new scores.")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
