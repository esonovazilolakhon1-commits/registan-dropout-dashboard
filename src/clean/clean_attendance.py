# =============================================================
# clean_attendance.py
# =============================================================
# Clean the raw attendance export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why I clean this file separately from the others:
#   The attendance table is the most important data source in my
#   study — it is the foundation for the dropout label, the
#   attendance features, and the snapshot skeleton. Any cleaning
#   error here propagates through the entire pipeline, so I clean
#   it in isolation and verify the output before touching anything
#   else.
#
# Why the filename has a typo ('attandance'):
#   This is the exact filename from Registan's MongoDB export.
#   I preserved the original name rather than renaming it so that
#   if the data is ever re-exported I can match it immediately
#   without confusion. The cleaned output is saved under the
#   correctly spelled name (attendance.parquet).
#
# Input : data/raw/attandance_raw.csv
# Output: data/interim/attendance.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import ATTENDANCE as RAW, ATTENDANCE_CLEAN as OUT

print("Reading attandance_raw.csv ...")
df = pd.read_csv(RAW, low_memory=False)

# ── step 1: rename ─────────────────────────────────────────────
# The MongoDB _id field is the primary key. I rename it to
# attendanceId so downstream code does not have to handle the
# underscore-prefixed name, which looks like a private attribute
# in Python and can cause confusion.
df = df.rename(columns={"_id": "attendanceId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
# deletedAt is stored as 0 when the record has not been deleted,
# and as a timestamp when it has. I convert this to a clean boolean
# and drop the original column — keeping a column that mixes 0 and
# timestamps would be error-prone in any subsequent numeric operation.
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
else:
    df["isDeleted"] = False
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse date ─────────────────────────────────────────
# I parse with utc=True because Registan's timestamps are stored in
# UTC. The utc=True flag ensures that all datetimes are timezone-aware
# and consistent, which matters when I later compute date differences
# (e.g. days since last attendance) without timezone ambiguity.
df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

# ── step 4: fill isTrial nulls ─────────────────────────────────
# isTrial indicates whether a lesson was a free trial before formal
# enrolment. Null values mean the field was not recorded — these are
# regular paid lessons, not trials, so False is the correct fill.
df["isTrial"] = df["isTrial"].fillna(False).astype(bool)

# ── step 5: fill null states → 'unchecked' ─────────────────────
# The state field records whether a student attended, was absent
# (reasonable/unreasonable), or frozen. A null state means the
# teacher did not mark the lesson — functionally equivalent to
# 'unchecked' in Registan's system, which is the category for
# lessons where attendance was not recorded.
null_state_count = df["state"].isnull().sum()
df["state"] = df["state"].fillna("unchecked")
print(f"  → Filled {null_state_count} null states with 'unchecked'")

# ── step 6: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── State distribution ──")
print(df["state"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print("\n── Trial vs non-trial ──")
print(df["isTrial"].value_counts())
print("\n── Payment state ──")
print(df["isPaymentDone"].value_counts())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 7: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
