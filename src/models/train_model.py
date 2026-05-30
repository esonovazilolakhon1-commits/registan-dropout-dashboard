# =============================================================
# train_model.py
# =============================================================
# LightGBM Prototype — Early Baseline Model
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why this script still exists alongside train_models.py:
#   This was the first complete model I trained on the Registan data.
#   At the time I was using train_encoded.parquet — a single-row-per-
#   student design that I later discovered contained data leakage (features
#   like final attendance rate were computed over the student's full
#   history, including the period after dropout). I kept this file as
#   a historical reference so I can show my supervisor the progression:
#   prototype → leakage discovery → snapshot redesign → multi-model
#   comparison. It also documents the StratifiedKFold approach I used
#   before I switched to StratifiedGroupKFold (grouped by studentId) to
#   prevent within-student leakage in the cross-validation.
#
# Why LightGBM as the prototype:
#   I chose LightGBM as the first model to try because it is fast,
#   handles class imbalance natively with class_weight='balanced', and
#   supports early stopping — which means I do not need to hand-tune
#   the number of trees. I planned to compare it against other models
#   later (which I did in train_models.py), but I wanted a working
#   end-to-end pipeline first.
#
# Why I did NOT use StratifiedGroupKFold here (important caveat):
#   This script uses plain StratifiedKFold, which splits rows without
#   knowing which rows belong to the same student. At the time I wrote
#   this I had not yet identified the within-student leakage problem.
#   In train_models.py I fixed this by using StratifiedGroupKFold with
#   groups=studentId. The CV scores from this script are therefore
#   overoptimistic and should NOT be cited in the dissertation —
#   I report results from train_models.py only.
#
# Input:
#   data/processed/train_encoded.parquet
#   data/processed/test_encoded.parquet
#
# Output:
#   models/lgbm_baseline.pkl
#   reports/figures/feature_importance_lgbm.png
#   reports/figures/confusion_matrix_lgbm.png
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from src.config import *

print("=" * 60)
print("LIGHTGBM BASELINE MODEL")
print("=" * 60)

# =============================================================
# LOAD DATA
# =============================================================
print("\nLoading encoded data...")
train = pd.read_parquet(PROCESSED / "train_encoded.parquet")
test  = pd.read_parquet(PROCESSED / "test_encoded.parquet")

print(f"  → Train : {len(train):,} rows")
print(f"  → Test  : {len(test):,} rows")

# =============================================================
# DEFINE FEATURES AND TARGET
# =============================================================
# studentId is an identifier, not a behavioural signal — including it
# would let the model memorise individual students rather than learn
# patterns. dropout is the label I want to predict, not a feature.
ID_COLS   = ['studentId']
TARGET    = 'dropout'
DROP_COLS = ID_COLS + [TARGET]

feature_cols = [c for c in train.columns if c not in DROP_COLS]

X_train = train[feature_cols].copy()
y_train = train[TARGET].astype(int)

X_test  = test[feature_cols].copy()
y_test  = test[TARGET].astype(int)

print(f"  → Features used : {len(feature_cols)}")

# =============================================================
# 5-FOLD CROSS VALIDATION ON TRAINING SET
# =============================================================
# Note: I use plain StratifiedKFold here (not grouped). This was
# written before I identified the within-student leakage issue. The
# corrected version (StratifiedGroupKFold by studentId) is in
# train_models.py. CV scores below are kept for reference only.
print("\nRunning 5-fold stratified cross validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_roc  = []
cv_pr   = []
cv_acc  = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    m = lgb.LGBMClassifier(
        n_estimators      = 500,
        learning_rate     = 0.05,
        num_leaves        = 31,
        min_child_samples = 20,
        class_weight      = 'balanced',
        random_state      = 42,
        verbose           = -1
    )
    # early_stopping(50) tells LightGBM to stop training if the
    # validation PR-AUC does not improve for 50 consecutive rounds.
    # This prevents overfitting without requiring me to manually
    # determine the right n_estimators via a grid search.
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(-1)
        ]
    )

    y_prob = m.predict_proba(X_val)[:, 1]
    y_pred = m.predict(X_val)

    cv_roc.append(roc_auc_score(y_val, y_prob))
    cv_pr.append(average_precision_score(y_val, y_prob))
    cv_acc.append((y_pred == y_val).mean())

    print(f"  Fold {fold}: ROC-AUC={cv_roc[-1]:.4f} | PR-AUC={cv_pr[-1]:.4f} | Acc={cv_acc[-1]:.4f}")

print(f"\n  ── CV Summary ──────────────────────────")
print(f"  ROC-AUC  : {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
print(f"  PR-AUC   : {np.mean(cv_pr):.4f} ± {np.std(cv_pr):.4f}")
print(f"  Accuracy : {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")

# 0.70 was my initial PR-AUC target before I had a proper baseline.
# The final reported threshold from the dissertation comes from the
# comparison in train_models.py using the snapshot pipeline.
if np.mean(cv_pr) >= 0.70:
    print(f"\n  ✅ PR-AUC target met (>0.70)!")
else:
    print(f"\n  ⚠️  PR-AUC below 0.70 — needs improvement")

# =============================================================
# TRAIN FINAL MODEL ON FULL TRAINING SET
# =============================================================
# I refit on the full training set (no validation fold) because the
# CV step was only for performance estimation. The model that goes
# into production should see as much data as possible.
print("\nTraining final model on full training set...")

final_model = lgb.LGBMClassifier(
    n_estimators      = 500,
    learning_rate     = 0.05,
    num_leaves        = 31,
    min_child_samples = 20,
    class_weight      = 'balanced',
    random_state      = 42,
    verbose           = -1
)
final_model.fit(X_train, y_train)

# =============================================================
# EVALUATE ON HOLDOUT TEST SET
# =============================================================
print("\n── Holdout Test Set Evaluation ─────────────────────────")
y_test_prob = final_model.predict_proba(X_test)[:, 1]
y_test_pred = final_model.predict(X_test)

test_roc = roc_auc_score(y_test, y_test_prob)
test_pr  = average_precision_score(y_test, y_test_prob)
test_acc = (y_test_pred == y_test).mean()

print(f"  ROC-AUC  : {test_roc:.4f}")
print(f"  PR-AUC   : {test_pr:.4f}")
print(f"  Accuracy : {test_acc:.4f}")
print(f"\n{classification_report(y_test, y_test_pred, target_names=['Completer','Dropout'])}")

# =============================================================
# FEATURE IMPORTANCE
# =============================================================
# I redefine FIGURES here rather than relying on the config import
# because I want this script to be self-contained — running it from
# any directory should still write to the right place.
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

importance = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(importance.head(20).to_string(index=False))

plt.figure(figsize=(10, 8))
top20 = importance.head(20)
# I colour the top 5 features red to draw the reader's eye to the
# features that matter most. The remaining 15 are blue. This is purely
# a visual aid for the dissertation appendix figure.
colors = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(top20))]
plt.barh(top20['feature'][::-1], top20['importance'][::-1], color=colors[::-1])
plt.xlabel('Feature Importance (number of splits)')
plt.title('LightGBM — Top 20 Feature Importance')
plt.tight_layout()
plt.savefig(FIGURES / "feature_importance_lgbm.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Feature importance saved → reports/figures/feature_importance_lgbm.png")

# =============================================================
# CONFUSION MATRIX
# =============================================================
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Completer', 'Dropout'],
    yticklabels=['Completer', 'Dropout']
)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix — LightGBM (Test Set)')
plt.tight_layout()
plt.savefig(FIGURES / "confusion_matrix_lgbm.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Confusion matrix saved → reports/figures/confusion_matrix_lgbm.png")

# =============================================================
# SAVE MODEL
# =============================================================
# I also redefine MODELS locally for the same self-containment reason
# as FIGURES above.
MODELS = Path("models")
MODELS.mkdir(parents=True, exist_ok=True)

# I bundle the feature list alongside the model so that any script that
# loads lgbm_baseline.pkl can reconstruct the exact feature matrix
# without importing or re-running this script.
model_data = {
    'model'        : final_model,
    'feature_cols' : feature_cols,
    'cv_roc_auc'   : np.mean(cv_roc),
    'cv_pr_auc'    : np.mean(cv_pr),
    'cv_accuracy'  : np.mean(cv_acc),
    'test_roc_auc' : test_roc,
    'test_pr_auc'  : test_pr,
    'test_accuracy': test_acc,
}

with open(MODELS / "lgbm_baseline.pkl", 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved → models/lgbm_baseline.pkl")
print(f"\n{'='*60}")
print(f"FINAL SUMMARY")
print(f"{'='*60}")
print(f"  CV  ROC-AUC  : {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
print(f"  CV  PR-AUC   : {np.mean(cv_pr):.4f} ± {np.std(cv_pr):.4f}")
print(f"  CV  Accuracy : {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"  Test ROC-AUC : {test_roc:.4f}")
print(f"  Test PR-AUC  : {test_pr:.4f}")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Features     : {len(feature_cols)}")
