# =============================================================
# encode_features.py
# =============================================================
# Feature Encoding for the Original (Pre-Snapshot) ML Pipeline
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why this script still exists:
#   This is the encoding step for the original single-row-per-student
#   design that pre-dates the snapshot pipeline. It reads master_ml.parquet
#   (one row per student-course, features computed over the student's full
#   history) and produces train_encoded / test_encoded / score_encoded.
#   I kept it because train_model.py (the LightGBM prototype) depends on
#   these files, and I include both scripts in the dissertation as evidence
#   of the iterative development process — from a leaky full-history design
#   to the leak-free monthly snapshot approach.
#
# Encoding decisions (summary):
#   - studentId    → kept as identifier only, never used as a feature
#   - courseName   → one-hot encoded, then original column dropped
#   - gender       → label encoded (binary: female=0, male=1)
#   - language     → one-hot encoded (uz, ru, en)
#   - joinMonth    → dropped (a "YYYY-MM" string that adds no signal
#                    beyond what joinYear and joinSeason already capture)
#   - joinSeason   → one-hot encoded (spring, summer, autumn, winter)
#   - academicYear → dropped (perfectly redundant with joinYear)
#   - preferredShift → one-hot encoded (morning, afternoon, evening)
#
# Sentinel fixes:
#   daysSinceLastRemoval_course = 9999 → wasEverRemoved = 0
#   daysSinceLastFreeze_course  = 9999 → wasEverFrozen  = 0
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import *

print("=" * 60)
print("ENCODING FEATURES FOR ML")
print("=" * 60)

df = pd.read_parquet(PROCESSED / "master_ml.parquet")
print(f"\nLoaded: {len(df):,} rows × {len(df.columns)} columns")

# =============================================================
# STEP 1 — DROP USELESS COLUMNS
# =============================================================
# joinMonth is stored as a "YYYY-MM" string — it carries no
# information that joinYear + joinSeason do not already cover,
# and feeding a raw string into a model would require further
# encoding with no payoff.
# academicYear is derived directly from the join date and duplicates
# joinYear exactly, so keeping both would just inflate the feature
# count without adding signal.
print("\nStep 1: Dropping useless columns...")

DROP = ['joinMonth', 'academicYear']
df = df.drop(columns=[c for c in DROP if c in df.columns])
print(f"  → Dropped: {DROP}")

# =============================================================
# STEP 2 — FIX 9999 SENTINEL VALUES
# =============================================================
# In build_master.py I used 9999 as a sentinel meaning "this event
# never happened for this student". I cannot leave 9999 in the
# feature matrix because the model would treat it as a real number
# and wrongly infer that a very large daysSinceLastRemoval means
# the student is at high risk. Instead I convert each sentinel into
# two pieces of information:
#   (1) a binary flag (wasEverRemoved / wasEverFrozen) that records
#       whether the event occurred at all
#   (2) the numeric column reset to 0 for students it never happened
#       to (0 days since an event that never occurred is a meaningful
#       neutral value)
print("\nStep 2: Fixing 9999 sentinel values...")

df['wasEverRemoved'] = (df['daysSinceLastRemoval_course'] < 9999).astype(int)
df['wasEverFrozen']  = (df['daysSinceLastFreeze_course']  < 9999).astype(int)

df['daysSinceLastRemoval_course'] = df['daysSinceLastRemoval_course'].replace(9999, 0)
df['daysSinceLastFreeze_course']  = df['daysSinceLastFreeze_course'].replace(9999, 0)

print(f"  → wasEverRemoved: {df['wasEverRemoved'].sum():,} students were ever removed")
print(f"  → wasEverFrozen : {df['wasEverFrozen'].sum():,} students were ever frozen")

# =============================================================
# STEP 3 — LABEL ENCODE BINARY COLUMNS
# =============================================================
# gender has exactly two values, so one-hot encoding would create
# two perfectly anti-correlated columns that carry identical
# information. A single 0/1 integer is simpler and equivalent.
# I fill any missing values with 0 (female) as the conservative
# default — there are very few nulls and they occur in records
# where gender was not collected.
print("\nStep 3: Label encoding binary columns...")

df['gender'] = df['gender'].map({'female': 0, 'male': 1}).fillna(0).astype(int)
print(f"  → gender encoded: female=0, male=1")

# =============================================================
# STEP 4 — ONE-HOT ENCODE CATEGORICAL COLUMNS
# =============================================================
# I set drop_first=False deliberately: I want all category levels
# present in the encoded matrix so that SHAP importance values are
# interpretable for every level. Dropping the first level to avoid
# perfect multicollinearity is only strictly necessary for linear
# models; my tree-based models are unaffected by it, and keeping
# all levels makes the feature names self-explanatory in the
# importance chart.
#
# fillna('Unknown') before encoding ensures that rows with a missing
# category get their own 'Unknown' column rather than producing all
# zeros, which would be indistinguishable from a valid category.
print("\nStep 4: One-hot encoding categorical columns...")

OHE_COLS = ['courseName', 'language', 'joinSeason', 'preferredShift']

for col in OHE_COLS:
    if col in df.columns:
        dummies = pd.get_dummies(
            df[col].fillna('Unknown'),
            prefix=col,
            drop_first=False,
            dtype=int
        )
        df = pd.concat([df, dummies], axis=1)
        print(f"  → {col}: {list(dummies.columns)}")

# drop the original text columns — the one-hot columns replace them.
# studentId is deliberately kept as an identifier for the student-level
# train/test split that follows.
df = df.drop(columns=['language', 'joinSeason', 'preferredShift'])
df = df.drop(columns=['courseName'])

# =============================================================
# STEP 5 — SEPARATE TRAINING AND SCORING SETS
# =============================================================
# In the old pipeline, dropout = -1 flags currently-active students
# whose outcome is unknown (the moderator's scoring cohort).
# dropout ∈ {0, 1} are students with a confirmed label.
# I separate them now so that the scoring set is never touched during
# the train/test split or the null-filling step below.
print("\nStep 5: Separating training and scoring sets...")

train = df[df['dropout'].isin([0, 1])].copy()
score = df[df['dropout'] == -1].copy()

print(f"  → Training set : {len(train):,} rows")
print(f"  → Scoring set  : {len(score):,} rows (active students)")

# =============================================================
# STEP 6 — TRAIN/TEST SPLIT BY STUDENT ID
# =============================================================
# I split by studentId so that all rows belonging to one student
# (one per course in this single-row-per-enrolment design) end up
# on the same side of the split. A row-level split would allow the
# same student to appear in both train and test — the model would
# effectively have seen that student during training, making test
# metrics overoptimistic.
print("\nStep 6: Train/test split by studentId (80/20)...")

unique_students = train['studentId'].unique().tolist()
np.random.seed(42)
np.random.shuffle(unique_students)
unique_students = np.array(unique_students)

split_idx   = int(len(unique_students) * 0.8)
train_ids   = unique_students[:split_idx]
test_ids    = unique_students[split_idx:]

train_set = train[train['studentId'].isin(train_ids)].copy()
test_set  = train[train['studentId'].isin(test_ids)].copy()

print(f"  → Train students : {len(train_ids):,}")
print(f"  → Test students  : {len(test_ids):,}")
print(f"  → Train rows     : {len(train_set):,}")
print(f"  → Test rows      : {len(test_set):,}")
print(f"  → Train dropout% : {train_set['dropout'].mean():.1%}")
print(f"  → Test dropout%  : {test_set['dropout'].mean():.1%}")

# =============================================================
# STEP 7 — DATA QUALITY CHECK
# =============================================================
# I check for remaining nulls before saving. Any that remain after
# the encoding steps above are filled with 0 — the neutral value
# for all numeric features in this pipeline. Leaving nulls would
# cause LightGBM and XGBoost to handle them internally (which they
# can do), but filling explicitly gives me a cleaner audit trail.
print("\nStep 7: Data quality check...")
nulls = train_set.isnull().sum().sum()
print(f"  → Total nulls in train set : {nulls}")
if nulls > 0:
    print(train_set.isnull().sum()[train_set.isnull().sum() > 0])

train_set = train_set.fillna(0)
test_set  = test_set.fillna(0)
score     = score.fillna(0)

# =============================================================
# STEP 8 — SAVE
# =============================================================
print("\nSaving encoded files...")

train_set.to_parquet(PROCESSED / "train_encoded.parquet", index=False)
test_set.to_parquet(PROCESSED  / "test_encoded.parquet",  index=False)
score.to_parquet(PROCESSED     / "score_encoded.parquet", index=False)

print(f"\n✅ train_encoded.parquet → {len(train_set):,} rows × {len(train_set.columns)} columns")
print(f"✅ test_encoded.parquet  → {len(test_set):,} rows × {len(test_set.columns)} columns")
print(f"✅ score_encoded.parquet → {len(score):,} rows × {len(score.columns)} columns")

print(f"\n  Final feature columns ({len(train_set.columns)} total):")
for i, col in enumerate(train_set.columns, 1):
    print(f"    {i:2}. {col}")
