# =============================================================
# clean_lessons.py
# =============================================================
# Clean the raw lessons export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# What this table contains:
#   The lessons table records every scheduled lesson across all groups.
#   I use it primarily to link attendance records to specific lesson
#   dates and times, and to verify lesson counts against attendance
#   aggregations.
#
# Input : data/raw/lessons.raw.csv
# Output: data/interim/lessons.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import LESSONS as RAW, LESSONS_CLEAN as OUT

print("Reading lessons.raw.csv ...")
df = pd.read_csv(RAW, low_memory=False)

# ── step 1: rename ─────────────────────────────────────────────
df = df.rename(columns={"_id": "lessonId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
else:
    df["isDeleted"] = False
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse dates ────────────────────────────────────────
# I parse all three date fields with utc=True so that date arithmetic
# (e.g. lesson duration = endsAt - startsAt) works without timezone errors.
for col in ["date", "startsAt", "endsAt"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# ── step 4: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── State distribution ──")
print(df["state"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 5: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
