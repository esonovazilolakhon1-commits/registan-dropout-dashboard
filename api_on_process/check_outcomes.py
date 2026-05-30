# check_outcomes.py
# ─────────────────────────────────────────────────────────────────────────────
# After each monthly pipeline run, this script checks what ACTUALLY happened
# to the students we predicted last month.
#
# Logic:
#   1. Load last month's predictions  (live_predictions.parquet)
#   2. Load last month's features     (live_features.parquet — saved by predict_live.py)
#   3. Load FRESH student states      (live_interim/students.parquet from today's API pull)
#   4. For each predicted student:
#        - state == 'active'   → stayed       → actual_dropout = 0
#        - state == 'archive'  → dropped out  → actual_dropout = 1
#        - state == 'graduate' → graduated    → excluded (not a dropout)
#   5. Append labeled rows to data/processed/outcomes_history.parquet
#      (this file grows every month and feeds retrain_incremental.py)
#
# Run order in pipeline:
#   extract → clean → features → check_outcomes → retrain → predict
#
# How to run manually:
#   python3 api_on_process/check_outcomes.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths
ORIGINAL_PROCESSED = ROOT / "data" / "processed"
LIVE_INTERIM       = ROOT / "data" / "live_interim"
OUTCOMES_FILE      = ORIGINAL_PROCESSED / "outcomes_history.parquet"


def run():
    log.info("=" * 60)
    log.info("CHECK OUTCOMES — labelling last month's predictions")
    log.info("=" * 60)

    # ── Step 1: Load last month's predictions ────────────────────────────────
    pred_path = ORIGINAL_PROCESSED / "live_predictions.parquet"
    feat_path = ORIGINAL_PROCESSED / "live_features.parquet"

    if not pred_path.exists():
        log.warning("No live_predictions.parquet found — nothing to label yet.")
        log.warning("This is normal on the first run. Skipping outcome check.")
        return

    if not feat_path.exists():
        log.warning("No live_features.parquet found — cannot label without features.")
        log.warning("Features are saved by predict_live.py on first run. Skipping.")
        return

    pred = pd.read_parquet(pred_path)
    feat = pd.read_parquet(feat_path)
    log.info(f"Loaded {len(pred):,} previous predictions  (snapshotMonths: "
             f"{sorted(pred['snapshotMonth'].unique())})")
    log.info(f"Loaded {len(feat):,} feature rows")

    # ── Step 2: Load current student states from fresh API data ─────────────
    stu_path = LIVE_INTERIM / "students.parquet"
    if not stu_path.exists():
        log.warning("live_interim/students.parquet not found.")
        log.warning("Make sure extract + clean ran before check_outcomes.")
        return

    stu = pd.read_parquet(stu_path)[['studentId', 'state', 'archiveDate', 'graduatedAt']]
    log.info(f"Loaded {len(stu):,} current student records")
    log.info(f"State distribution:\n{stu['state'].value_counts().to_string()}")

    # ── Step 3: Join predictions with current states ─────────────────────────
    merged = pred[['studentId', 'courseName', 'snapshotMonth', 'dropout_probability', 'risk_level']].merge(
        stu, on='studentId', how='left'
    )

    # Label outcomes:
    #   archive  → actual_dropout = 1  (left the school)
    #   active   → actual_dropout = 0  (still here)
    #   graduate → exclude             (successful completion, not a dropout)
    #   unknown  → exclude             (no state info, can't label reliably)

    merged = merged[merged['state'].isin(['active', 'archive'])]  # exclude graduates & unknowns
    merged['actual_dropout'] = (merged['state'] == 'archive').astype(int)

    n_dropped = merged['actual_dropout'].sum()
    n_stayed  = (merged['actual_dropout'] == 0).sum()
    log.info(f"\nOutcome summary:")
    log.info(f"  Dropped out : {n_dropped:,}  ({n_dropped/len(merged)*100:.1f}%)")
    log.info(f"  Still active: {n_stayed:,}   ({n_stayed/len(merged)*100:.1f}%)")

    # ── Step 4: Join with features so retraining has actual training rows ────
    labeled = merged[['studentId', 'courseName', 'snapshotMonth', 'actual_dropout']].merge(
        feat, on=['studentId', 'courseName'], how='inner'
    )
    labeled['labeled_at'] = datetime.now().strftime('%Y-%m')
    log.info(f"Matched {len(labeled):,} labeled rows with features (ready for retraining)")

    # ── Step 5: Append to outcomes_history.parquet ───────────────────────────
    if OUTCOMES_FILE.exists():
        existing = pd.read_parquet(OUTCOMES_FILE)
        # Avoid duplicating the same snapshot month if pipeline is re-run
        existing = existing[~existing['snapshotMonth'].isin(labeled['snapshotMonth'].unique())]
        combined = pd.concat([existing, labeled], ignore_index=True)
        log.info(f"Appended to existing history "
                 f"({len(existing):,} old + {len(labeled):,} new = {len(combined):,} total)")
    else:
        combined = labeled
        log.info(f"Created new outcomes_history.parquet with {len(combined):,} rows")

    combined.to_parquet(OUTCOMES_FILE, index=False)
    log.info(f"Saved → {OUTCOMES_FILE}")
    log.info("=" * 60)
    log.info("OUTCOME CHECK COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
