# =============================================================
# clean_studentgroups.py
# =============================================================
# Clean the raw studentgroups export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# What this table contains:
#   Each row records one student's membership in one group —
#   including the per-lesson price they pay, when they joined,
#   when (if ever) they graduated, and their current state.
#   This is the primary join table between students and courses
#   in my pipeline.
#
# Why I drop deletedAt entirely here:
#   Unlike other tables where deletedAt is 0 or a timestamp,
#   in studentgroups ALL values are 0 (no rows are deleted).
#   Converting a column of zeros to a boolean would produce
#   an all-False column with no information value, so I drop
#   it completely.
#
# Why I flag isFreeOrTrial and isPriceOutlier:
#   price = 0 means a free or trial enrolment — these students
#   are not paying and their financial behaviour features would
#   be meaningless. I flag them so build_master.py can take this
#   into account when computing payment-based features.
#   price > 200,000 UZS is an outlier — the standard lesson
#   prices at Registan range from 15,000 to 150,000 UZS per lesson.
#   Values above 200,000 suggest a data-entry error.
#
# Input : data/raw/studentgroups.raw.csv
# Output: data/interim/studentgroups.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import STUDENTGROUPS as RAW, STUDENTGROUPS_CLEAN as OUT

print("Reading studentgroups.raw.csv ...")
df = pd.read_csv(RAW)

# ── step 1: rename columns ─────────────────────────────────────
df = df.rename(columns={"_id": "studentGroupId"})

# ── step 2: drop deletedAt (all zeros = no deletions) ──────────
# All rows have deletedAt = 0, so this column carries no information.
# Keeping an all-False boolean column would be misleading.
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse dates ────────────────────────────────────────
# graduatedAt is null for students who have not yet graduated.
# I parse it with errors="coerce" so nulls become NaT (not an error).
for col in ["joinedAt", "graduatedAt"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# ── step 4: fix price ──────────────────────────────────────────
# price is the per-lesson charge rate in UZS. Some values come in
# as strings or floats from the CSV export — I coerce to numeric
# and fill nulls with 0 (no price = free enrolment).
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype(int)
df["isFreeOrTrial"]  = df["price"] == 0
df["isPriceOutlier"] = df["price"] > 200000

# ── step 5: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── State distribution ──")
print(df["state"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print("\n── Price stats ──")
print(df["price"].describe())
print(f"\n  Free/trial enrollments : {df['isFreeOrTrial'].sum()}")
print(f"  Price outliers         : {df['isPriceOutlier'].sum()}")
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 6: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
