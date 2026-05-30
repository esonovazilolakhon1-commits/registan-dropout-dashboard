# =============================================================
# clean_transactions.py
# =============================================================
# Clean the raw transactions export from Registan's MongoDB.
# Keeps only student transactions (studentId not null).
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why I read in chunks:
#   The transactions file is the largest raw file in the project
#   (several million rows). Reading it in chunks of 200,000 rows
#   keeps peak memory usage manageable and filters out non-student
#   transactions during loading rather than after, which further
#   reduces the final dataframe size.
#
# Why I keep only rows where studentId is not null:
#   Registan's transaction system records payments for teachers,
#   moderators, and other staff as well as students. Only student
#   transactions are relevant to my dropout model — teacher salary
#   payments and internal transfers add noise and would distort
#   the financial feature distributions.
#
# Why I remove transactions with amount > 10,000,000 UZS:
#   The standard lesson price at Registan is 15,000–150,000 UZS per
#   lesson. A top-up of 10 million UZS or more is not a plausible
#   student payment — it indicates a bulk internal transfer or a data
#   entry error. I remove these outliers to prevent them from
#   skewing statistics like avgPaymentAmount and currentBalance.
#
# Why I fix the 'vouncher' typo:
#   The raw data contains 'vouncher' as a transaction type. This is
#   a misspelling of 'voucher' (likely introduced by the developer).
#   I correct it here so downstream code that filters on transaction
#   type does not need to know about both spellings.
#
# Input : data/raw/transactions.raw.csv
# Output: data/interim/transactions.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
from src.config import TRANSACTIONS as RAW, TRANSACTIONS_CLEAN as OUT

print("Reading transactions.raw.csv in chunks ...")
chunks = []
for chunk in pd.read_csv(RAW, chunksize=200_000, low_memory=False):
    # filter to student transactions during loading to keep memory low
    chunk = chunk[chunk["studentId"].notna()]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"  → {len(df)} student transactions loaded")

# ── step 1: rename ─────────────────────────────────────────────
df = df.rename(columns={"_id": "transactionId"})

# ── step 2: deletedAt → isDeleted boolean ──────────────────────
if "deletedAt" in df.columns:
    df["isDeleted"] = df["deletedAt"] != 0
else:
    df["isDeleted"] = False
if "deletedAt" in df.columns:
    df = df.drop(columns=["deletedAt"])

# ── step 3: parse dates ────────────────────────────────────────
df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)

# ── step 4: fix amount type ────────────────────────────────────
# All three amount columns (transaction amount, before, after) must
# be numeric for financial feature computation. errors="coerce"
# converts any non-numeric values to NaN rather than raising an error.
df["amount"]       = pd.to_numeric(df["amount"],       errors="coerce")
df["beforeAmount"] = pd.to_numeric(df["beforeAmount"], errors="coerce")
df["afterAmount"]  = pd.to_numeric(df["afterAmount"],  errors="coerce")

# ── step 4b: remove outlier transactions ──────────────────────
before = len(df)
df = df[df["amount"] <= 10_000_000]
print(f"  → Removed {before - len(df)} outlier transactions (amount > 10M)")

# ── step 5: normalize type typos ──────────────────────────────
# 'vouncher' is a misspelling present in the raw data.
# I correct it to 'voucher' here so downstream code uses one name.
df["type"] = df["type"].replace({"vouncher": "voucher"})

# ── step 6: summary ────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── Type distribution ──")
print(df["type"].value_counts())
print("\n── Origin distribution ──")
print(df["origin"].value_counts())
print("\n── Amount stats ──")
print(df["amount"].describe())
print("\n── Null counts ──")
print(df.isnull().sum())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")
print(f"Columns       : {list(df.columns)}")

# ── step 7: save ───────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
