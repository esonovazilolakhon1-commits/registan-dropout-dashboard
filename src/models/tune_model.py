# =============================================================
# tune_model.py
# =============================================================
# Hyperparameter tuning for XGBoost using Optuna (Bayesian
# optimisation with TPE sampler and median pruning).
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why Optuna over GridSearch or RandomSearch:
#   GridSearch is exhaustive but exponentially slow — with 9 XGBoost
#   hyperparameters and 3 values each that would be 3^9 = 19,683 fits.
#   RandomSearch is faster but wastes trials on uninformative regions.
#   Optuna uses Tree-structured Parzen Estimators (TPE): a Bayesian
#   surrogate model that learns which regions of hyperparameter space
#   are promising and focuses sampling there. This typically finds
#   better solutions in 50–100 trials than RandomSearch finds in 500.
#
# Why PR-AUC as the optimisation metric:
#   My dataset has 32% dropout (positive class). In imbalanced settings
#   ROC-AUC is optimistic because it treats all thresholds equally,
#   including very low-recall regions that are useless in practice.
#   PR-AUC directly measures how well the model ranks true dropouts
#   against false alarms — which is exactly what Registan's moderators
#   care about when they work through a ranked intervention list.
#
# Why StratifiedGroupKFold (not plain StratifiedKFold):
#   Each student appears in multiple snapshot rows (one per month).
#   Without grouping by studentId, the same student's rows could
#   appear in both the training fold and the validation fold, making
#   CV scores artificially high (within-student leakage). Grouping
#   ensures every validation fold contains completely unseen students.
#
# What hyperparameters are tuned and why:
#   max_depth        — controls tree complexity; too deep = overfit
#   learning_rate    — step size per tree; smaller = more robust but slower
#   n_estimators     — number of trees; tuned jointly with learning_rate
#   min_child_weight — minimum sample weight in a leaf; regularises splits
#   subsample        — row sampling per tree; reduces variance
#   colsample_bytree — column sampling per tree; reduces correlation
#   gamma            — min gain to split; explicit regularisation
#   reg_alpha        — L1 penalty; drives small weights to zero
#   reg_lambda       — L2 penalty; shrinks all weights
#
# Input  : data/processed/snap_train.parquet
#          data/processed/snap_test.parquet
#          models/best_model.pkl  (baseline to beat)
# Output : models/best_model.pkl  (overwritten ONLY if tuned model wins)
#          models/tuned_xgb.pkl   (always saved regardless)
#          reports/figures/optuna_importance.png
#          reports/figures/optuna_history.png
# =============================================================

import sys
import pickle
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

from src.config import PROCESSED, MODELS, FIGURES

optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial noise

N_TRIALS    = 100   # increase for better results; 100 is a good dissertation baseline
N_CV_FOLDS  = 5
RANDOM_SEED = 42

print("=" * 65)
print("XGBOOST HYPERPARAMETER TUNING — OPTUNA (TPE + MEDIAN PRUNING)")
print("=" * 65)

# ── load data ────────────────────────────────────────────────────────────────
print("\nLoading train / test splits ...")
train = pd.read_parquet(PROCESSED / "snap_train.parquet")
test  = pd.read_parquet(PROCESSED / "snap_test.parquet")

META     = ['studentId', 'courseName', 'snapshotMonth', 'label', 'rowSet']
FEATURES = [c for c in train.columns if c not in META]

X_train = train[FEATURES].values
y_train = train['label'].astype(int).values
groups  = train['studentId'].values

X_test  = test[FEATURES].values
y_test  = test['label'].astype(int).values

spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

print(f"  → Train : {len(train):,} rows | {len(FEATURES)} features")
print(f"  → Test  : {len(test):,} rows")
print(f"  → Dropout rate train/test: {y_train.mean():.1%} / {y_test.mean():.1%}")
print(f"  → scale_pos_weight = {spw:.2f}")

# ── load current best model to compare against ────────────────────────────────
baseline_pkl = MODELS / "best_model.pkl"
baseline_pr  = None
if baseline_pkl.exists():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(baseline_pkl, "rb") as f:
            baseline_bundle = pickle.load(f)
    bl_model = baseline_bundle["model"]
    bl_name  = baseline_bundle["model_name"]
    bl_feats = baseline_bundle["features"]
    # evaluate baseline on test set using its own feature list
    X_test_bl = test[bl_feats].values
    p_bl = bl_model.predict_proba(X_test_bl)[:, 1]
    baseline_pr  = average_precision_score(y_test, p_bl)
    baseline_roc = roc_auc_score(y_test, p_bl)
    print(f"\nBaseline ({bl_name}):")
    print(f"  Test PR-AUC  = {baseline_pr:.4f}")
    print(f"  Test ROC-AUC = {baseline_roc:.4f}")
else:
    print("\nNo existing best_model.pkl found — will save tuned model unconditionally.")

# ── Optuna objective ──────────────────────────────────────────────────────────
sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True,
                             random_state=RANDOM_SEED)

def objective(trial):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators",      200, 800),
        "max_depth"        : trial.suggest_int("max_depth",          3, 10),
        "learning_rate"    : trial.suggest_float("learning_rate",    0.01, 0.3,  log=True),
        "min_child_weight" : trial.suggest_int("min_child_weight",   1, 20),
        "subsample"        : trial.suggest_float("subsample",        0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma"            : trial.suggest_float("gamma",            0.0, 5.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha",        1e-4, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda",       1e-4, 10.0, log=True),
        "scale_pos_weight" : spw,
        "eval_metric"      : "aucpr",
        "random_state"     : RANDOM_SEED,
        "n_jobs"           : -1,
    }

    fold_scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
            sgkf.split(X_train, y_train, groups)):

        Xtr, Xva = X_train[tr_idx], X_train[va_idx]
        ytr, yva = y_train[tr_idx], y_train[va_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        fold_scores.append(average_precision_score(yva, p))

        # report intermediate value so Optuna can prune bad trials
        trial.report(np.mean(fold_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(fold_scores)


# ── run study ─────────────────────────────────────────────────────────────────
print(f"\nRunning {N_TRIALS} Optuna trials "
      f"(TPE sampler + MedianPruner, {N_CV_FOLDS}-fold CV) ...")
print("  Each dot = 1 completed trial\n  ", end="", flush=True)

completed = [0]
def progress_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        completed[0] += 1
        print(".", end="", flush=True)
        if completed[0] % 50 == 0:
            best = study.best_value
            print(f"  [{completed[0]}/{N_TRIALS}] best CV PR-AUC so far: {best:.4f}")
            print("  ", end="", flush=True)

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=RANDOM_SEED),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress_callback])

print()  # newline after dots

# ── results ───────────────────────────────────────────────────────────────────
best_params = study.best_params
best_cv_pr  = study.best_value

print(f"\n{'─'*65}")
print(f"  Best CV PR-AUC : {best_cv_pr:.4f}")
print(f"  Best params    :")
for k, v in best_params.items():
    print(f"    {k:25s} = {v}")

# ── refit on full training set with best params ───────────────────────────────
print(f"\nRefitting on full training set with best params ...")
final_params = {**best_params,
                "scale_pos_weight": spw,
                "eval_metric"     : "aucpr",
                "random_state"    : RANDOM_SEED,
                "n_jobs"          : -1}

tuned_model = xgb.XGBClassifier(**final_params)
tuned_model.fit(X_train, y_train)

p_tuned      = tuned_model.predict_proba(X_test)[:, 1]
tuned_pr_auc = average_precision_score(y_test, p_tuned)
tuned_roc    = roc_auc_score(y_test, p_tuned)

# ── comparison table ─────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  BEFORE vs AFTER TUNING")
print(f"{'─'*65}")
print(f"  {'Metric':<20} {'Baseline':>12} {'Tuned':>12} {'Δ':>10}")
print(f"{'─'*65}")
if baseline_pr is not None:
    delta_pr  = tuned_pr_auc - baseline_pr
    delta_roc = tuned_roc    - baseline_roc
    print(f"  {'Test PR-AUC':<20} {baseline_pr:>12.4f} {tuned_pr_auc:>12.4f} {delta_pr:>+10.4f}")
    print(f"  {'Test ROC-AUC':<20} {baseline_roc:>12.4f} {tuned_roc:>12.4f} {delta_roc:>+10.4f}")
else:
    print(f"  {'Test PR-AUC':<20} {'N/A':>12} {tuned_pr_auc:>12.4f}")
    print(f"  {'Test ROC-AUC':<20} {'N/A':>12} {tuned_roc:>12.4f}")
print(f"{'─'*65}")
print(f"  CV PR-AUC (tuned, 5-fold): {best_cv_pr:.4f}")
print(f"{'='*65}")

# ── precision / recall at threshold 0.5 and optimal threshold ────────────────
def best_threshold(y_true, y_prob):
    """Find threshold that maximises F1 for the dropout (positive) class."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.where((prec + rec) == 0, 1, prec + rec)
    idx = np.argmax(f1[:-1])   # last element has no threshold
    return thresholds[idx], prec[idx], rec[idx], f1[idx]

def eval_at_threshold(y_true, y_prob, threshold, label):
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\n  ── {label} (threshold = {threshold:.2f}) ──")
    print(f"  {'':20s} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─'*55}")
    for cls, name in [(0, 'Retained'), (1, 'Dropout')]:
        mask = y_true == cls
        p = precision_score(y_true, y_pred, pos_label=cls, zero_division=0)
        r = recall_score(y_true, y_pred, pos_label=cls, zero_division=0)
        f = f1_score(y_true, y_pred, pos_label=cls, zero_division=0)
        print(f"  {name:<20} {p:>10.3f} {r:>10.3f} {f:>10.3f} {mask.sum():>10,}")
    acc = (y_pred == y_true).mean()
    print(f"\n  Accuracy : {acc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    print(f"  Interpretation: of {(y_true==1).sum():,} actual dropouts,")
    print(f"    model catches {cm[1,1]:,} ({cm[1,1]/(y_true==1).sum():.0%}) — misses {cm[1,0]:,} ({cm[1,0]/(y_true==1).sum():.0%})")
    print(f"    raises {cm[0,1]:,} false alarms out of {(y_true==0).sum():,} retained students ({cm[0,1]/(y_true==0).sum():.0%})")

print(f"\n{'='*65}")
print(f"  PRECISION / RECALL ANALYSIS")
print(f"{'='*65}")

# baseline at 0.5
if baseline_pr is not None:
    eval_at_threshold(y_test, p_bl, 0.5, f"BASELINE ({bl_name}) at 0.50")
    bl_opt_thresh, bl_opt_p, bl_opt_r, bl_opt_f1 = best_threshold(y_test, p_bl)
    eval_at_threshold(y_test, p_bl, bl_opt_thresh,
                      f"BASELINE ({bl_name}) at optimal threshold")

# tuned at 0.5
eval_at_threshold(y_test, p_tuned, 0.5, "TUNED (Optuna XGBoost) at 0.50")

# tuned at optimal threshold
opt_thresh, opt_p, opt_r, opt_f1 = best_threshold(y_test, p_tuned)
eval_at_threshold(y_test, p_tuned, opt_thresh,
                  f"TUNED (Optuna XGBoost) at optimal threshold")

print(f"\n{'─'*65}")
print(f"  Optimal threshold for tuned model: {opt_thresh:.3f}")
print(f"  At optimal: Precision={opt_p:.3f}  Recall={opt_r:.3f}  F1={opt_f1:.3f}")
if baseline_pr is not None:
    print(f"  vs baseline optimal: P={bl_opt_p:.3f}  R={bl_opt_r:.3f}  F1={bl_opt_f1:.3f}")
print(f"{'='*65}")

# ── PR curve: baseline vs tuned ───────────────────────────────────────────────
plt.figure(figsize=(8, 6))
if baseline_pr is not None:
    prec_bl, rec_bl, _ = precision_recall_curve(y_test, p_bl)
    plt.plot(rec_bl, prec_bl, color="#3498db", linewidth=2,
             label=f"Baseline {bl_name} (AP={baseline_pr:.4f})")
prec_t, rec_t, _ = precision_recall_curve(y_test, p_tuned)
plt.plot(rec_t, prec_t, color="#e74c3c", linewidth=2,
         label=f"Tuned XGBoost (AP={tuned_pr_auc:.4f})")
plt.axhline(y_test.mean(), color="grey", linestyle="--", alpha=0.6,
            label=f"No-skill baseline = {y_test.mean():.3f}")
plt.scatter([opt_r], [opt_p], color="#e74c3c", s=100, zorder=5,
            label=f"Optimal threshold ({opt_thresh:.2f}): P={opt_p:.3f} R={opt_r:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve — Baseline vs Optuna-Tuned XGBoost")
plt.legend(loc="upper right"); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(FIGURES / "pr_curve_tuned_vs_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ Figure saved → reports/figures/pr_curve_tuned_vs_baseline.png")

# ── always save tuned model ───────────────────────────────────────────────────
MODELS.mkdir(parents=True, exist_ok=True)
tuned_bundle = {
    "model"      : tuned_model,
    "model_name" : "XGBoost (Optuna-tuned)",
    "features"   : FEATURES,
    "best_params": best_params,
    "cv_pr_auc"  : best_cv_pr,
    "test_pr_auc": tuned_pr_auc,
    "test_roc_auc": tuned_roc,
}
with open(MODELS / "tuned_xgb.pkl", "wb") as f:
    pickle.dump(tuned_bundle, f)
print(f"\n✅ Tuned model saved → models/tuned_xgb.pkl")

# ── overwrite best_model.pkl only if tuned model wins ────────────────────────
if baseline_pr is None or tuned_pr_auc > baseline_pr:
    with open(MODELS / "best_model.pkl", "wb") as f:
        pickle.dump({
            "model"     : tuned_model,
            "model_name": "XGBoost (Optuna-tuned)",
            "features"  : FEATURES,
            "metrics"   : {"cv_pr_auc": best_cv_pr,
                           "test_pr_auc": tuned_pr_auc,
                           "test_roc_auc": tuned_roc},
        }, f)
    print(f"✅ best_model.pkl UPDATED — tuned model is better "
          f"(+{tuned_pr_auc - (baseline_pr or 0):.4f} PR-AUC)")
else:
    print(f"⚠️  best_model.pkl NOT updated — baseline ({bl_name}) "
          f"is still better ({baseline_pr:.4f} vs {tuned_pr_auc:.4f})")

# ── figures ───────────────────────────────────────────────────────────────────
FIGURES.mkdir(parents=True, exist_ok=True)

# optimisation history
values = [t.value for t in study.trials
          if t.state == optuna.trial.TrialState.COMPLETE]
best_so_far = np.maximum.accumulate(values)

plt.figure(figsize=(9, 4))
plt.plot(values, alpha=0.4, color="#3498db", label="Trial PR-AUC")
plt.plot(best_so_far, color="#e74c3c", linewidth=2, label="Best so far")
if baseline_pr:
    plt.axhline(baseline_pr, color="grey", linestyle="--",
                label=f"Baseline ({bl_name}) = {baseline_pr:.4f}")
plt.xlabel("Trial"); plt.ylabel("CV PR-AUC")
plt.title("Optuna Optimisation History — XGBoost PR-AUC")
plt.legend(); plt.tight_layout()
plt.savefig(FIGURES / "optuna_history.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Figure saved → reports/figures/optuna_history.png")

# hyperparameter importance
try:
    imp = optuna.importance.get_param_importances(study)
    names  = list(imp.keys())
    scores = list(imp.values())
    plt.figure(figsize=(8, 5))
    bars = plt.barh(names[::-1], scores[::-1], color="#2ecc71")
    plt.xlabel("Importance (fraction of variance explained)")
    plt.title("Optuna Hyperparameter Importance — XGBoost")
    plt.tight_layout()
    plt.savefig(FIGURES / "optuna_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Figure saved → reports/figures/optuna_importance.png")
    print(f"\n  Top hyperparameters by importance:")
    for n, s in list(imp.items())[:5]:
        print(f"    {n:25s} {s:.3f}")
except Exception as e:
    print(f"  (Importance plot skipped: {e})")

print(f"\n{'='*65}")
print(f"  DONE — run 'streamlit run src/dashboard/eda_dashboard.py'")
print(f"  to see the updated predictions.")
print(f"{'='*65}")
