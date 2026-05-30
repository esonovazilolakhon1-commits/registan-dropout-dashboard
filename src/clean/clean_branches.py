# =============================================================
# clean_branches.py
# =============================================================
# Clean the raw branches export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why I clean branches at all:
#   Registan operates multiple branches across Tashkent. My dissertation
#   studies only the Chilonzor branch. I add an isChilonzor flag here so
#   that downstream scripts can filter to the correct branch using a simple
#   boolean column rather than repeating the ObjectId string every time.
#
# Input : data/raw/branches.raw.csv
# Output: data/interim/branches.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import BRANCHES as RAW, BRANCHES_CLEAN as OUT, CHILONZOR_BRANCH_ID

print("Reading branches.raw.csv ...")
df = pd.read_csv(RAW)

# ── step 1: rename ─────────────────────────────────────────────
# MongoDB exports the primary key as _id. I rename it to branchId
# so it reads as a proper column name in downstream joins.
df = df.rename(columns={"_id": "branchId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
# deletedAt is 0 when not deleted, a timestamp when deleted.
# The column may be absent in fresh API pulls that omit soft-delete fields.
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
    df = df.drop(columns=["deletedAt"])
else:
    df["isDeleted"] = False

# ── step 3: parse dates ────────────────────────────────────────
# utc=True ensures timezone-aware datetimes consistent with all other tables.
for col in ["createdAt", "updatedAt"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# ── step 4: flag Chilonzor branch ──────────────────────────────
# I add this flag so downstream scripts can filter with
# df[df["isChilonzor"]] rather than hard-coding the ObjectId string,
# which would be fragile if the config ever changes.
df["isChilonzor"] = df["branchId"] == CHILONZOR_BRANCH_ID

# ── step 5: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── Null counts ──")
print(df.isnull().sum())
print("\n── Chilonzor branch ──")
print(df[df["isChilonzor"]][["branchId","name","address"]].to_string())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 6: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
