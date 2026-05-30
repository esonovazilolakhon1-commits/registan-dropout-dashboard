# =============================================================
# evaluate_tuned.py
# =============================================================
# Full precision / recall evaluation comparing baseline XGBoost
# vs Optuna-tuned XGBoost at two thresholds:
#   1. Default threshold 0.50
#   2. Optimal threshold (maximises F1 for the Dropout class)
#
# Also produces a PR curve figure for the dissertation.
#
# How to run:
#   python3 src/models/evaluate_tuned.py
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

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from src.config import PROCESSED, MODELS, FIGURES

# ── load test set ─────────────────────────────────────────────────────────────
test   = pd.read_parquet(PROCESSED / "snap_test.parquet")
y_test = test['label'].astype(int).values

# ── load baseline (backup saved by resave_model or tune_model) ────────────────
backup_path = MODELS / "best_model_backup.pkl"
if not backup_path.exists():
    print("⚠️  best_model_backup.pkl not found — comparing tuned vs itself at two thresholds only.")
    bl_available = False
else:
    bl_available = True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(backup_path, "rb") as f:
            bl_bundle = pickle.load(f)
    bl_name  = bl_bundle["model_name"]
    bl_feats = bl_bundle["features"]
    p_bl     = bl_bundle["model"].predict_proba(test[bl_feats].values)[:, 1]

# ── load tuned model ──────────────────────────────────────────────────────────
with open(MODELS / "tuned_xgb.pkl", "rb") as f:
    tun_bundle = pickle.load(f)
p_tuned = tun_bundle["model"].predict_proba(test[tun_bundle["features"]].values)[:, 1]

# ── helpers ───────────────────────────────────────────────────────────────────
def optimal_threshold(y_true, y_prob):
    """Threshold that maximises F1 for the positive (Dropout) class."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / np.where((prec + rec) == 0, 1, prec + rec)
    idx = np.argmax(f1[:-1])
    return thresholds[idx], prec[idx], rec[idx], f1[idx]

def print_eval(y_true, y_prob, threshold, title):
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    n_drop = (y_true == 1).sum()
    n_ret  = (y_true == 0).sum()

    print(f"\n  ┌─ {title} (threshold = {threshold:.2f}) ────────────────────")
    print(f"  │  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  │  {'─'*52}")
    for cls, name in [(0, 'Retained'), (1, 'Dropout')]:
        p = precision_score(y_true, y_pred, pos_label=cls, zero_division=0)
        r = recall_score   (y_true, y_pred, pos_label=cls, zero_division=0)
        f = f1_score       (y_true, y_pred, pos_label=cls, zero_division=0)
        n = (y_true == cls).sum()
        print(f"  │  {name:<12} {p:>10.3f} {r:>10.3f} {f:>10.3f} {n:>10,}")
    print(f"  │")
    acc = (y_pred == y_true).mean()
    print(f"  │  Accuracy : {acc:.4f}")
    print(f"  │  TN={cm[0,0]:>5,}  FP={cm[0,1]:>5,}")
    print(f"  │  FN={cm[1,0]:>5,}  TP={cm[1,1]:>5,}")
    print(f"  │")
    print(f"  │  ✓ Catches {cm[1,1]:,} of {n_drop:,} real dropouts  ({cm[1,1]/n_drop:.0%} recall)")
    print(f"  │  ✗ Misses  {cm[1,0]:,} of {n_drop:,} real dropouts  ({cm[1,0]/n_drop:.0%})")
    print(f"  │  ⚠ False alarms: {cm[0,1]:,} of {n_ret:,} retained students ({cm[0,1]/n_ret:.0%})")
    print(f"  └{'─'*60}")
    return {
        "precision_dropout": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_dropout"   : recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_dropout"       : f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy"         : acc,
        "tp": cm[1,1], "fn": cm[1,0], "fp": cm[0,1], "tn": cm[0,0],
    }

# ── run evaluation ────────────────────────────────────────────────────────────
print("=" * 65)
print("  PRECISION / RECALL EVALUATION — BASELINE vs TUNED XGBoost")
print("=" * 65)

if bl_available:
    baseline_pr  = average_precision_score(y_test, p_bl)
    baseline_roc = roc_auc_score(y_test, p_bl)
    print(f"\n  Baseline ({bl_name}):  PR-AUC={baseline_pr:.4f}  ROC-AUC={baseline_roc:.4f}")

tuned_pr  = average_precision_score(y_test, p_tuned)
tuned_roc = roc_auc_score(y_test, p_tuned)
print(f"  Tuned (Optuna XGBoost): PR-AUC={tuned_pr:.4f}  ROC-AUC={tuned_roc:.4f}")

if bl_available:
    r_bl05  = print_eval(y_test, p_bl, 0.50, f"BASELINE ({bl_name}) — default")
    bl_ot, bl_op, bl_or, bl_of = optimal_threshold(y_test, p_bl)
    r_blopt = print_eval(y_test, p_bl, bl_ot, f"BASELINE ({bl_name}) — optimal")

r_t05  = print_eval(y_test, p_tuned, 0.50, "TUNED XGBoost — default")
ot, op, or_, of = optimal_threshold(y_test, p_tuned)
r_topt = print_eval(y_test, p_tuned, ot, "TUNED XGBoost — optimal")

# ── summary comparison table ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SUMMARY — Dropout class (what matters for intervention)")
print(f"{'─'*65}")
if bl_available:
    print(f"  {'Metric':<28} {'Baseline@0.5':>13} {'Tuned@0.5':>12} {'Tuned@opt':>12}")
    print(f"  {'─'*63}")
    metrics = [
        ("Precision (Dropout)",  r_bl05["precision_dropout"],  r_t05["precision_dropout"],  r_topt["precision_dropout"]),
        ("Recall    (Dropout)",  r_bl05["recall_dropout"],     r_t05["recall_dropout"],     r_topt["recall_dropout"]),
        ("F1        (Dropout)",  r_bl05["f1_dropout"],         r_t05["f1_dropout"],         r_topt["f1_dropout"]),
        ("Accuracy",             r_bl05["accuracy"],           r_t05["accuracy"],           r_topt["accuracy"]),
        ("PR-AUC",               baseline_pr,                  tuned_pr,                    tuned_pr),
        ("ROC-AUC",              baseline_roc,                 tuned_roc,                   tuned_roc),
    ]
    for name, v_bl, v_t05, v_topt in metrics:
        delta = v_topt - v_bl
        print(f"  {name:<28} {v_bl:>13.3f} {v_t05:>12.3f} {v_topt:>12.3f}  (Δ{delta:+.3f})")
else:
    print(f"  {'Metric':<28} {'Tuned@0.50':>13} {'Tuned@optimal':>15}")
    print(f"  {'─'*58}")
    for name, v05, vopt in [
        ("Precision (Dropout)", r_t05["precision_dropout"], r_topt["precision_dropout"]),
        ("Recall    (Dropout)", r_t05["recall_dropout"],    r_topt["recall_dropout"]),
        ("F1        (Dropout)", r_t05["f1_dropout"],        r_topt["f1_dropout"]),
        ("Accuracy",            r_t05["accuracy"],          r_topt["accuracy"]),
        ("PR-AUC",              tuned_pr,                   tuned_pr),
        ("ROC-AUC",             tuned_roc,                  tuned_roc),
    ]:
        print(f"  {name:<28} {v05:>13.3f} {vopt:>15.3f}")

print(f"\n  Recommended deployment threshold: {ot:.2f}")
print(f"  At this threshold: {r_topt['recall_dropout']:.0%} of dropouts caught,")
print(f"  {r_topt['fp']:,} false alarms per {(y_test==0).sum():,} retained students.")
print(f"{'='*65}")

# ── PR curve figure ───────────────────────────────────────────────────────────
FIGURES.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(8, 6))

if bl_available:
    prec_bl, rec_bl, _ = precision_recall_curve(y_test, p_bl)
    plt.plot(rec_bl, prec_bl, color="#3498db", lw=2,
             label=f"Baseline {bl_name} (AP={baseline_pr:.4f})")
    plt.scatter([bl_or], [bl_op], color="#3498db", s=100, zorder=5, marker="^",
                label=f"Baseline optimal ({bl_ot:.2f}): P={bl_op:.3f} R={bl_or:.3f}")

prec_t, rec_t, _ = precision_recall_curve(y_test, p_tuned)
plt.plot(rec_t, prec_t, color="#e74c3c", lw=2,
         label=f"Tuned XGBoost (AP={tuned_pr:.4f})")
plt.scatter([or_], [op], color="#e74c3c", s=100, zorder=5,
            label=f"Tuned optimal ({ot:.2f}): P={op:.3f} R={or_:.3f}")
plt.axhline(y_test.mean(), color="grey", ls="--", alpha=0.6,
            label=f"No-skill baseline = {y_test.mean():.3f}")

plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curve — Baseline vs Optuna-Tuned XGBoost", fontsize=13)
plt.legend(loc="upper right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "pr_curve_tuned_vs_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✅ PR curve saved → reports/figures/pr_curve_tuned_vs_baseline.png")
