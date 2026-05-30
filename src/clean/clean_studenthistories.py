# =============================================================
# clean_studenthistories.py
# =============================================================
# Clean the raw student histories export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# What this table contains:
#   An event log of every administrative action taken on a student's
#   account — group transfers, freezes, unfreezes, price changes,
#   graduations, and moves to archive state. build_master.py and
#   build_snapshot_features.py aggregate these events by type to
#   create behavioural features.
#
# Why I preserve the 'fronzon' typo:
#   The raw MongoDB data contains the event type 'fronzen' (a
#   misspelling of 'frozen'). I preserve this typo deliberately
#   rather than correcting it here, because all downstream scripts
#   that query for freeze events use the same misspelled string.
#   If I corrected it here but forgot to update one downstream
#   script, that script would silently find zero freeze events —
#   a subtle and hard-to-detect bug. Keeping the original string
#   throughout the pipeline is the safer choice.
#
# Input : data/raw/studenthistories.raw.csv
# Output: data/interim/studenthistories.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import STUDENTHISTORIES as RAW, STUDENTHISTORIES_CLEAN as OUT

print("Reading studenthistories.raw.csv ...")
df = pd.read_csv(RAW, low_memory=False)

# ── step 1: rename ─────────────────────────────────────────────
df = df.rename(columns={"_id": "studentHistoryId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
else:
    df["isDeleted"] = False
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse dates ────────────────────────────────────────
# createdAt is the event timestamp — the only date field in this table.
df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

# ── step 4: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── Event type distribution ──")
print(df["type"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print("\n── Note: 'fronzen' typo kept as is intentionally ──")
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 5: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
