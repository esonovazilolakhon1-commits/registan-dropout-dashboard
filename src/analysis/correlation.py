# =============================================================
# correlation.py
# =============================================================
# Feature Correlation Analysis
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan
#               Chilonzor Branch using Machine Learning
#
# Why I wrote this script:
#   Before finalising the feature set I needed to check two things:
#   (1) whether any features are so highly correlated that they give
#   the model redundant information, and (2) which features have the
#   strongest linear relationship with the dropout label. High
#   pairwise correlation can distort SHAP importance by splitting
#   the same signal across two near-identical features — a problem
#   I wanted to identify before training rather than explain away
#   afterwards. The target correlation ranking also gives me a quick
#   sanity check: if attendanceRate is not near the top, something
#   has gone wrong in the feature engineering.
#
#   This script produces three outputs:
#   1. Highly correlated feature pairs (|r| > 0.85) flagged for
#      review — these are candidates for removal in build_ml_ready.py.
#   2. Feature-target correlation ranking — a preliminary linear
#      ordering of predictors, useful for the dissertation literature
#      review section when I compare my findings with prior work.
#   3. Full correlation heatmap — a visual overview of the feature
#      correlation structure saved as a high-resolution PNG for the
#      dissertation appendix.
#
# Why Pearson and not Spearman or mutual information:
#   Pearson is the most common choice in the educational data mining
#   literature, which makes my results directly comparable to prior
#   work. Spearman would be more appropriate if I suspected heavy
#   monotonic but non-linear relationships; mutual information would
#   detect arbitrary dependencies. I use Pearson here as a first pass
#   and rely on the tree models themselves (LightGBM, XGBoost) to
#   capture non-linear structure during training.
#
# Output:
#   reports/figures/correlation_matrix.png
#   reports/figures/target_correlation.png
#   Console output: high-correlation pairs + top-15 features
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from src.config import *

print("=" * 60)
print("FEATURE CORRELATION ANALYSIS")
print("=" * 60)

# =============================================================
# LOAD TRAINING DATA
# =============================================================
# I use only the labelled training set (label ∈ {0, 1}).
# The scoring set contains students whose forward attendance window
# extends past the reference date — their label is genuinely unknown.
# Including them would introduce noise into the correlation estimates
# and would not represent the population I am trying to model.
train = pd.read_parquet(PROCESSED / "snap_train.parquet")

print(f"\nLoaded snapshot training set : {len(train):,} rows × {len(train.columns)} columns")
print(f"  label = 1 (dropout)        : {(train['label']==1).sum():,}")
print(f"  label = 0 (retained)       : {(train['label']==0).sum():,}")

# =============================================================
# DEFINE NUMERIC FEATURE COLUMNS
# =============================================================
# I exclude identifiers, the snapshot time index, the target label,
# and the row-set tag. Everything else in snap_train is a numeric
# feature (categoricals are one-hot encoded, so they are already
# numeric and can be included in the Pearson calculation).
EXCLUDE_COLS = [
    'studentId',      # identifier — memorising it would be leakage
    'snapshotMonth',  # time index, not a behavioural signal
    'label',          # target — excluded from the feature matrix
    'rowSet',         # split metadata, not a predictor
]

num_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
print(f"\nNumeric features analysed : {len(num_cols)}")

# compute full Pearson correlation matrix on the training set
corr = train[num_cols].corr()

# =============================================================
# 1. HIGHLY CORRELATED FEATURE PAIRS
# =============================================================
# I set the threshold at 0.85 following Kuhn & Johnson (2013),
# who recommend removing one feature from any pair with |r| > 0.85
# to reduce multicollinearity. In practice, for my tree-based models
# the main risk is not multicollinearity (trees are immune to it)
# but diluted SHAP importance — two features that encode almost the
# same information will each receive roughly half the importance
# signal, making the bar chart misleading. I use 0.85 rather than
# a stricter cutoff (e.g. 0.95) because I want to catch near-
# duplicates early without discarding features that are merely
# related but still carry distinct signals.
CORRELATION_THRESHOLD = 0.85

print(f"\n{'='*60}")
print(f"HIGHLY CORRELATED PAIRS (|r| > {CORRELATION_THRESHOLD})")
print(f"{'='*60}")

high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > CORRELATION_THRESHOLD:
            high_corr_pairs.append({
                'feature_1': corr.columns[i],
                'feature_2': corr.columns[j],
                'correlation': round(r, 4)
            })
            print(f"  r = {r:+.3f}  |  {corr.columns[i]}  ←→  {corr.columns[j]}")

if not high_corr_pairs:
    print("  ✅ No highly correlated pairs found.")
    print("     All features are sufficiently independent for model training.")
else:
    print(f"\n  → {len(high_corr_pairs)} pair(s) exceed the threshold.")
    print("     Review build_ml_ready.py DROP_COLS to address these.")

# =============================================================
# 2. FEATURE-TARGET CORRELATION RANKING
# =============================================================
# I compute the Pearson correlation between each numeric feature and
# the binary dropout label. A positive value means higher feature
# values are associated with dropout = 1; a negative value means they
# are associated with retention (label = 0).
#
# This ranking is a linear approximation only — my tree models capture
# non-linear interactions that Pearson cannot see, so a low |r| does
# not mean a feature is uninformative. What this ranking does tell me
# is whether the features are pointing in the expected direction.
# For instance, I expect attendanceLast30Days to be negatively
# correlated (students who attend more are less likely to drop out)
# and debtRate to be positively correlated (students in debt more
# often are more likely to drop out). Any reversal would indicate a
# bug in the feature engineering.
print(f"\n{'='*60}")
print(f"FEATURE-TARGET CORRELATION RANKING (top 20)")
print(f"{'='*60}")

target_corr = train[num_cols + ['label']].corr()['label'].drop('label')
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

print(f"\n  {'Rank':<5} {'Feature':<40} {'|r|':<8} {'Direction'}")
print(f"  {'-'*4} {'-'*39} {'-'*7} {'-'*10}")
for rank, (feat, abs_r) in enumerate(target_corr_sorted.head(20).items(), 1):
    raw_r = target_corr[feat]
    direction = '↑ dropout' if raw_r > 0 else '↓ dropout'
    print(f"  {rank:<5} {feat:<40} {abs_r:.3f}    {direction}")

# I save the bar chart to include in the dissertation. Red bars are
# features positively associated with dropout (higher value → more
# likely to drop out), blue bars are negatively associated (higher
# value → more likely to be retained). This colour convention matches
# the dashboard's Dropout=red / Retained=green scheme.
FIGURES = Path("reports/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 8))
top20 = target_corr.reindex(target_corr_sorted.head(20).index)
colors = ['#d73027' if v > 0 else '#4575b4' for v in top20.values]
bars = ax.barh(range(len(top20)), top20.values, color=colors, height=0.6)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('Pearson Correlation with Dropout Label', fontsize=10)
ax.set_title(
    'Feature Correlation with Dropout — Snapshot Model (top 20)\n'
    'Red = positively associated with dropout | Blue = negatively associated',
    fontsize=11, pad=12
)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
plt.tight_layout()
plt.savefig(FIGURES / "target_correlation.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✅ Target correlation chart saved → reports/figures/target_correlation.png")

# =============================================================
# 3. FULL CORRELATION HEATMAP
# =============================================================
# I generate a lower-triangular heatmap of all pairwise Pearson
# correlations. I mask the upper triangle because each pair appears
# twice in the full matrix (corr[A,B] = corr[B,A]) — showing only
# the lower triangle halves the visual clutter without losing
# information. I use the RdBu_r diverging colormap centred at zero
# so that the reader can immediately distinguish positive (red) from
# negative (blue) correlations. The heatmap is 26×22 inches at
# 150 dpi because the feature set is large enough (~50 features)
# that a smaller figure would make the axis labels illegible.
print(f"\nGenerating full correlation heatmap...")

fig, ax = plt.subplots(figsize=(26, 22))

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,
    annot=False,
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.3,
    linecolor='#f0f0f0',
    cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
    ax=ax
)

ax.set_title(
    'Feature Correlation Matrix — Registan Dropout Prediction (Snapshot Model)\n'
    'Numeric features only | Snapshot training set (n = {:,})'.format(len(train)),
    fontsize=13,
    pad=16
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7.5)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7.5)

plt.tight_layout()
plt.savefig(
    FIGURES / "correlation_matrix.png",
    dpi=150,
    bbox_inches='tight'
)
plt.close()

print(f"✅ Correlation heatmap saved    → reports/figures/correlation_matrix.png")

# =============================================================
# SUMMARY
# =============================================================
print(f"\n{'='*60}")
print(f"CORRELATION ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"  Features analysed          : {len(num_cols)}")
print(f"  High-correlation pairs     : {len(high_corr_pairs)} (threshold |r| > {CORRELATION_THRESHOLD})")
print(f"  Strongest predictor        : {target_corr_sorted.index[0]} (|r| = {target_corr_sorted.iloc[0]:.3f})")
print(f"  Weakest predictor (top 20) : {target_corr_sorted.index[19]} (|r| = {target_corr_sorted.iloc[19]:.3f})")
print(f"\n  Figures saved to: reports/figures/")
print(f"    → correlation_matrix.png")
print(f"    → target_correlation.png")
