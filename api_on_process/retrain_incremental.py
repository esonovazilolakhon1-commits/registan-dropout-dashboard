# retrain_incremental.py
# ─────────────────────────────────────────────────────────────────────────────
# Retrains the dropout prediction model by combining:
#   - Original training data       (data/processed/snap_train.parquet)
#   - New labeled outcomes         (data/processed/outcomes_history.parquet)
#
# Each month the model gains experience from real outcomes at Registan —
# students who were predicted to drop out and either did or did not leave.
# This is continuous / incremental learning.
#
# The new model overwrites models/best_model.pkl. A dated backup of the
# previous model is saved to models/archive/ so training history is preserved.
#
# Run order in pipeline:
#   extract → clean → features → check_outcomes → retrain → predict
#
# How to run manually:
#   python3 api_on_process/retrain_incremental.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import pickle
import shutil
import logging
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PROCESSED, MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

ORIGINAL_PROCESSED = ROOT / "data" / "processed"
MODEL_PATH         = MODELS / "best_model.pkl"
MODEL_ARCHIVE_DIR  = MODELS / "archive"
OUTCOMES_FILE      = ORIGINAL_PROCESSED / "outcomes_history.parquet"

META = ['studentId', 'courseName', 'snapshotMonth', 'label',
        'rowSet', 'labeled_at', 'actual_dropout']


def run():
    log.info("=" * 60)
    log.info("INCREMENTAL RETRAIN — combining history + new outcomes")
    log.info("=" * 60)

    # ── Step 1: Load original training data ──────────────────────────────────
    snap_train_path = ORIGINAL_PROCESSED / "snap_train.parquet"
    if not snap_train_path.exists():
        raise FileNotFoundError(f"snap_train.parquet not found at {snap_train_path}")

    original = pd.read_parquet(snap_train_path)
    log.info(f"Original training data: {len(original):,} rows  "
             f"(dropout rate: {original['label'].mean():.1%})")

    # ── Step 2: Load new labeled outcomes (if any) ───────────────────────────
    if not OUTCOMES_FILE.exists():
        log.warning("No outcomes_history.parquet found.")
        log.warning("Running check_outcomes.py first produces labeled data.")
        log.warning("Skipping retrain — model unchanged.")
        return

    outcomes = pd.read_parquet(OUTCOMES_FILE)
    log.info(f"New labeled outcomes: {len(outcomes):,} rows  "
             f"(actual dropout rate: {outcomes['actual_dropout'].mean():.1%})")

    # Rename actual_dropout → label so it matches original training schema
    outcomes = outcomes.rename(columns={'actual_dropout': 'label'})
    outcomes['rowSet'] = 'live_outcome'

    # ── Step 3: Combine datasets ──────────────────────────────────────────────
    # Keep only columns that exist in both datasets (features + label)
    original_features = [c for c in original.columns if c not in META]
    outcome_features  = [c for c in outcomes.columns  if c not in META]
    shared_features   = [f for f in original_features if f in outcome_features]

    if not shared_features:
        log.error("No shared features between original training data and new outcomes.")
        log.error("Cannot retrain. Check that predict_live.py saves live_features.parquet correctly.")
        return

    log.info(f"Shared features: {len(shared_features)}  "
             f"(original had {len(original_features)}, outcomes had {len(outcome_features)})")

    orig_subset  = original[shared_features + ['label', 'studentId', 'snapshotMonth']].copy()
    new_subset   = outcomes[shared_features + ['label', 'studentId', 'snapshotMonth']].copy()
    combined     = pd.concat([orig_subset, new_subset], ignore_index=True)

    log.info(f"Combined training set: {len(combined):,} rows  "
             f"(dropout rate: {combined['label'].mean():.1%})")

    # ── Step 4: Backup current model ─────────────────────────────────────────
    MODEL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        ts = datetime.now().strftime('%Y-%m-%d')
        backup = MODEL_ARCHIVE_DIR / f"best_model_{ts}.pkl"
        shutil.copy2(MODEL_PATH, backup)
        log.info(f"Backed up previous model → {backup}")

    # ── Step 5: Retrain LightGBM on combined data ────────────────────────────
    # Use the same hyperparameters as the original training (train_models.py)
    # so the model stays consistent. Only the training data grows.
    X = combined[shared_features]
    y = combined['label'].astype(int)

    # Balance classes — same approach as original training
    spw = (y == 0).sum() / max((y == 1).sum(), 1)

    log.info(f"\nRetraining LightGBM on {len(X):,} rows...")
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    # ── Step 6: Quick performance check ──────────────────────────────────────
    proba = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, proba)
    roc_auc = roc_auc_score(y, proba)
    log.info(f"\nIn-sample performance (training data — informational only):")
    log.info(f"  PR-AUC  : {pr_auc:.4f}")
    log.info(f"  ROC-AUC : {roc_auc:.4f}")
    log.info(f"  (In-sample figures are optimistic — out-of-sample tested monthly)")

    # ── Step 7: Save new model ────────────────────────────────────────────────
    bundle = {
        "model":       model,
        "model_name":  "LightGBM_incremental",
        "features":    shared_features,
        "trained_at":  datetime.now().strftime('%Y-%m-%d %H:%M'),
        "n_train_rows": len(combined),
        "n_original":   len(orig_subset),
        "n_new_labeled": len(new_subset),
        "pr_auc_insample": round(pr_auc, 4),
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    log.info(f"\nNew model saved → {MODEL_PATH}")
    log.info(f"  Original rows : {len(orig_subset):,}")
    log.info(f"  New labeled   : {len(new_subset):,}")
    log.info(f"  Total trained : {len(combined):,}")
    log.info("=" * 60)
    log.info("INCREMENTAL RETRAIN COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
