# scheduler.py — runs the dropout pipeline on a weekly schedule
# Uses the 'schedule' library so no cron setup is needed for the demo environment.
# Equivalent cron entry: 0 8 * * 1  cd /path/to/project && python3 src/pipeline/run_pipeline.py
#
# How to start:
#   python3 src/pipeline/scheduler.py
#   nohup python3 src/pipeline/scheduler.py &   (background)
#
# Edit SCHEDULE_DAY and SCHEDULE_TIME in .env to change the schedule.

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import schedule
import time

from src.pipeline.run_pipeline import run as run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SCHEDULE_DAY  = os.getenv("SCHEDULE_DAY",  "monday")
SCHEDULE_TIME = os.getenv("SCHEDULE_TIME", "08:00")

VALID_DAYS = {
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday"
}
if SCHEDULE_DAY.lower() not in VALID_DAYS:
    raise ValueError(
        f"SCHEDULE_DAY='{SCHEDULE_DAY}' is not valid. "
        f"Use one of: {sorted(VALID_DAYS)}"
    )


def job():
    log.info(f"Scheduled run triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        run_pipeline()
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        log.error("Scheduler will try again on the next scheduled run.")


def main():
    log.info("=" * 60)
    log.info("REGISTAN DROPOUT SCHEDULER — STARTED")
    log.info(f"  Schedule : every {SCHEDULE_DAY.capitalize()} at {SCHEDULE_TIME}")
    log.info(f"  To stop  : press Ctrl+C")
    log.info("=" * 60)

    getattr(schedule.every(), SCHEDULE_DAY).at(SCHEDULE_TIME).do(job)

    log.info(f"Next run: {schedule.next_run()}")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
