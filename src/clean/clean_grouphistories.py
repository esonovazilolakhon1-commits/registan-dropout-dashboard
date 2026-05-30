# =============================================================
# clean_grouphistories.py
# =============================================================
# Clean the raw group histories export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# What this table contains:
#   The grouphistories table records every time a student joined or
#   left a group, including the timestamps for both events. I use it
#   primarily to track when students transferred between groups and
#   how long they stayed in each one.
#
# Input : data/raw/grouphistories.raw.csv
# Output: data/interim/grouphistories.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import GROUPHISTORIES as RAW, GROUPHISTORIES_CLEAN as OUT

print("Reading grouphistories.raw.csv ...")
df = pd.read_csv(RAW, low_memory=False)

# ── step 1: rename ─────────────────────────────────────────────
df = df.rename(columns={"_id": "groupHistoryId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
else:
    df["isDeleted"] = False
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse dates ────────────────────────────────────────
# joinedAt and leftAt are both needed to compute how long a student
# stayed in each group. utc=True for consistency across all tables.
for col in ["createdAt", "updatedAt", "joinedAt", "leftAt"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# ── step 4: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── Event type distribution ──")
print(df["type"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 5: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
