# =============================================================
# explain_model.py
# =============================================================
# SHAP Explainability for the Winning Dropout Prediction Model
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why I added an explainability step at all:
#   My supervisor emphasised that in an educational setting, a model
#   that predicts dropout without explaining why is not useful in
#   practice. A moderator cannot act on a probability score alone —
#   she needs to know whether the alert is driven by missed lessons,
#   overdue payments, or something else, so she can choose the right
#   intervention. SHAP values give her that.
#
# Why SHAP rather than feature importances from the model itself:
#   XGBoost's built-in 'gain' or 'split count' importances are
#   computed from the training set only and can be misleading when
#   features are correlated (e.g. attendanceLast30Days and
#   attendanceLast60Days are related). SHAP (SHapley Additive
#   exPlanations) attributes each prediction to its features in a
#   way that is theoretically grounded in cooperative game theory —
#   every feature gets its fair contribution, and the values sum
#   exactly to the model's output. My supervisor accepted SHAP as
#   the standard for interpretability in the literature, which is why
#   I use it here rather than a simpler alternative.
#
# Why TreeExplainer specifically:
#   shap.TreeExplainer is the fast, exact algorithm for tree-based
#   models. It runs in O(TLD^2) time (T=trees, L=leaves, D=depth)
#   rather than the exponential time of the naive Shapley calculation.
#   For my XGBoost model with 400 estimators it finishes in under a
#   minute on 4,000 rows, whereas the model-agnostic KernelExplainer
#   would take hours.
#
# Why two figures (beeswarm + bar chart):
#   The beeswarm shows BOTH magnitude AND direction for each feature —
#   I can see not just that attendanceLast30Days is the most important
#   feature but also that low values push the model toward dropout (red
#   dots on the left). The bar chart shows only magnitude (mean |SHAP|)
#   which is easier to read in a dissertation table or appendix figure
#   when the audience just wants a ranked list. I include both so the
#   dissertation has a publication-quality figure for the main text
#   and a clean ranking for the appendix.
#
# Inputs : models/best_model.pkl, data/processed/snap_test.parquet
# Outputs: reports/figures/shap_summary.png
#          reports/figures/shap_importance_bar.png
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from src.config import PROCESSED, MODELS, FIGURES

print("=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)

with open(MODELS / "best_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model, features, name = bundle["model"], bundle["features"], bundle["model_name"]
print(f"\nLoaded winner: {name} ({len(features)} features)")

test = pd.read_parquet(PROCESSED / "snap_test.parquet")
# I sample 4,000 rows rather than running SHAP on the full test set.
# 4,000 is enough for the mean |SHAP| estimates to be stable (I verified
# by comparing 1,000 vs 4,000 vs the full set — the ranking and magnitudes
# did not change materially). Using the full set would take ~10× longer
# for no meaningful gain in the dissertation figures.
X = test[features].sample(n=min(4000, len(test)), random_state=42)

print("Computing SHAP values (TreeExplainer)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
# For binary classification, older versions of the shap library return
# a list of two arrays [class-0 values, class-1 values]. Newer versions
# return a single array for the positive class directly. I handle both
# cases here so the script does not break if the shap version changes.
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Beeswarm summary — shows each feature's SHAP value for every sampled
# row, coloured by the feature's actual value (red = high, blue = low).
# I cap at 20 features because beyond that the plot becomes too crowded
# to read in a dissertation figure at normal font size.
plt.figure()
shap.summary_plot(shap_values, X, show=False, max_display=20)
plt.tight_layout(); plt.savefig(FIGURES / "shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"✅ {FIGURES / 'shap_summary.png'}")

# Bar chart of mean |SHAP| — a clean global importance ranking that is
# easier to reference in text than the beeswarm. Same 20-feature cap.
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=20)
plt.tight_layout(); plt.savefig(FIGURES / "shap_importance_bar.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"✅ {FIGURES / 'shap_importance_bar.png'}")

# I also print the ranked table to stdout so I can copy the top-15
# directly into the dissertation without running a separate analysis.
imp = pd.DataFrame({"feature": features,
                    "mean_abs_shap": np.abs(shap_values).mean(0)}).sort_values("mean_abs_shap", ascending=False)
print("\nTop 15 features by mean |SHAP|:")
print(imp.head(15).to_string(index=False))
