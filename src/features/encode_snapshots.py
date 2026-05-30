# =============================================================
# encode_snapshots.py
# =============================================================
# Encode the Snapshot Feature Panel and Split into Train / Test / Score
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why this script is a separate step from build_snapshot_features.py:
#   build_snapshot_features.py produces raw (un-encoded) features so
#   I can inspect distributions and spot engineering bugs before any
#   encoding is applied. Once I am satisfied with the features, this
#   script handles the methodological decisions that must happen after
#   feature engineering but before training: removing borderline-leaky
#   features, one-hot encoding categoricals, and splitting students
#   into train / test / score sets.
#
# Why I split by studentId and not by row:
#   One student contributes many monthly snapshot rows. If I split by
#   row, some of a student's months could land in train and others in
#   test. The model would then have seen that student's history during
#   training and would effectively be recognising a known individual
#   rather than generalising to new students. Every metric would be
#   overoptimistic. Splitting by studentId guarantees that every
#   student in the test set is completely unseen during training.
#
# Input : data/processed/snapshots_features.parquet
# Output: data/processed/snap_train.parquet
#         data/processed/snap_test.parquet
#         data/processed/snap_score.parquet  (active students to predict)
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import PROCESSED

print("=" * 60)
print("ENCODING SNAPSHOT PANEL + SPLIT")
print("=" * 60)

df = pd.read_parquet(PROCESSED / "snapshots_features.parquet")
print(f"\nLoaded: {len(df):,} rows × {len(df.columns)} cols")

# -------------------------------------------------------------
# Exclude borderline-leaky features (methodological decision)
# -------------------------------------------------------------
# n_removedFromGroup_course and wasEverRemoved record whether a
# student was administratively removed from a group. The problem is
# that this event is often logged in the same calendar month the
# student stops attending — so by the time the feature is non-zero,
# the decision has already been made and there is nothing a moderator
# can do. Including it would also inflate the model's PR-AUC by
# essentially letting it see a delayed echo of the outcome.
#
# I report a sensitivity analysis in the dissertation that includes
# these two features — it scores ROC 0.90 / PR 0.80 versus the clean
# model's 0.88 / 0.75 — to show my supervisor exactly how much the
# exclusion costs and why I am willing to pay that cost for a model
# that is genuinely actionable.
LEAKY_EXCLUDE = ['n_removedFromGroup_course', 'wasEverRemoved']
df = df.drop(columns=[c for c in LEAKY_EXCLUDE if c in df.columns])
print(f"  → Excluded borderline-leaky features: {LEAKY_EXCLUDE}")

# -------------------------------------------------------------
# Encode categoricals — one-hot on the whole frame before splitting
# -------------------------------------------------------------
# I apply get_dummies to the full dataframe rather than fitting
# separately on train and then transforming test. Fitting separately
# risks producing different column sets if a category level appears
# only in the test set — the model would then receive an unexpected
# number of columns at inference time. By encoding the whole frame
# first and then splitting, train, test, and score are guaranteed to
# have identical column layouts.
#
# gender is binary so I map it to 0/1 directly instead of one-hot,
# which would just create two perfectly anti-correlated columns.
df['gender'] = df['gender'].map({'female': 0, 'male': 1}).fillna(0).astype(int)
OHE = ['language', 'joinSeason', 'preferredShift', 'courseName']
df = pd.get_dummies(df, columns=OHE, prefix=OHE, dtype=int)
print(f"  → After one-hot: {len(df.columns)} cols")

# -------------------------------------------------------------
# Split rows by purpose
# -------------------------------------------------------------
# trainable rows have label ∈ {0, 1} — forward window fully observed.
# scoring rows have label = NaN — these are currently-active students
# whose outcome is unknown and who will be scored by predict.py.
# Censored rows are dropped at this point (they are in neither set).
trainable = df[df['rowSet'] == 'train'].copy()
score     = df[df['rowSet'] == 'scoring'].copy()
print(f"  → Trainable rows : {len(trainable):,}")
print(f"  → Scoring rows   : {len(score):,}")

# -------------------------------------------------------------
# 80/20 student-level split
# -------------------------------------------------------------
# I allocate 80% of unique students to training and 20% to test.
# This gives roughly 10,000 held-out students — enough to produce
# stable PR-AUC estimates on the test set. I use
# numpy.random.default_rng(42) rather than the legacy
# numpy.random.seed because the new Generator API is more
# reproducible across numpy versions (the legacy API's shuffle
# behaviour changed between numpy 1.x and 2.x).
students = np.array(trainable['studentId'].unique().tolist(), dtype=object)
rng = np.random.default_rng(42)
rng.shuffle(students)
cut = int(len(students) * 0.8)
train_ids, test_ids = set(students[:cut]), set(students[cut:])

train_set = trainable[trainable['studentId'].isin(train_ids)].copy()
test_set  = trainable[trainable['studentId'].isin(test_ids)].copy()

print(f"  → Train students/rows : {len(train_ids):,} / {len(train_set):,}  (label1 {train_set['label'].mean():.1%})")
print(f"  → Test  students/rows : {len(test_ids):,} / {len(test_set):,}  (label1 {test_set['label'].mean():.1%})")

for name, d in [('snap_train', train_set), ('snap_test', test_set), ('snap_score', score)]:
    d.to_parquet(PROCESSED / f"{name}.parquet", index=False)
    print(f"  ✅ {name}.parquet → {len(d):,} rows")
