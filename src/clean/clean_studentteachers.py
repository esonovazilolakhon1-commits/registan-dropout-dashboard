# =============================================================
# clean_studentteachers.py
# =============================================================
# Clean the raw student-teachers export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# What this table contains:
#   Each row records one student's relationship with one teacher —
#   when they joined, when (if ever) they graduated or left, and
#   whether they were frozen. I use this table to link students to
#   their teachers in cases where the attendance file does not have
#   a teacherId attached.
#
# Input : data/raw/studentteachers.raw.csv
# Output: data/interim/studentteachers.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import STUDENTTEACHERS as RAW, STUDENTTEACHERS_CLEAN as OUT

if not RAW.exists():
    print(f"⚠  {RAW.name} not found (optional endpoint returned 0 records).")
    print("   Saving empty studentteachers.parquet so downstream scripts don't crash.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["studentTeacherId","studentId","teacherId","groupId",
                           "state","isDeleted","joinedAt","graduatedAt","leftAt","frozenAt"]).to_parquet(OUT, index=False)
    print(f"✅ Empty studentteachers saved → {OUT}")
    import sys; sys.exit(0)

print("Reading studentteachers.raw.csv ...")
df = pd.read_csv(RAW)

# ── step 1: rename ─────────────────────────────────────────────
df = df.rename(columns={"_id": "studentTeacherId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
else:
    df["isDeleted"] = False
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse dates ────────────────────────────────────────
# All four date fields are relevant: joinedAt and graduatedAt frame
# the relationship duration; leftAt and frozenAt are transition events.
for col in ["joinedAt", "graduatedAt", "leftAt", "frozenAt"]:
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
