# =============================================================
# clean_levels.py
# =============================================================
# Clean the raw levels export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why I flag test records:
#   The levels table contains a record with name "qweqwe" — clearly
#   a keyboard-mashing test entry made by a developer. I flag it with
#   isTestRecord rather than deleting it silently, so I have an audit
#   trail of what was in the raw data. Downstream scripts can filter
#   it out using df[~df["isTestRecord"]] if needed.
#
# Input : data/raw/levelsifneeded.raw.csv
# Output: data/interim/levels.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import LEVELS as RAW, LEVELS_CLEAN as OUT

if not RAW.exists():
    print(f"⚠  {RAW.name} not found (optional endpoint returned 0 records).")
    print("   Saving empty levels.parquet so downstream scripts don't crash.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["levelId","name","type","isDeleted","isTestRecord"]).to_parquet(OUT, index=False)
    print(f"✅ Empty levels saved → {OUT}")
    import sys; sys.exit(0)

print("Reading levelsifneeded.raw.csv ...")
df = pd.read_csv(RAW)

# ── step 1: rename ─────────────────────────────────────────────
df = df.rename(columns={"_id": "levelId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
else:
    df["isDeleted"] = False
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse dates ────────────────────────────────────────
for col in ["createdAt", "updatedAt"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# ── step 4: flag test records ──────────────────────────────────
# "qweqwe" is a developer test entry. I flag it rather than deleting
# silently so downstream code can decide whether to exclude it.
df["isTestRecord"] = df["name"].str.lower().str.strip() == "qweqwe"

# ── step 5: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── Type distribution ──")
print(df["type"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print("\n── Test records ──")
print(df[df["isTestRecord"]][["levelId","name","type"]].to_string())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 6: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
