# =============================================================
# clean_courses.py
# =============================================================
# Clean the raw courses export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why I use ijson instead of json.load:
#   The courses JSON is a large nested file. ijson streams it object
#   by object rather than loading everything into memory at once.
#   This keeps memory usage constant regardless of file size — the
#   same approach I use for students, groups, and orders.
#
# Why I extract only the Chilonzor branch price:
#   Each course has a 'branches' array with separate price and
#   availability settings per branch. My dissertation covers only
#   Chilonzor, so I extract only the Chilonzor entry. Prices at other
#   branches are irrelevant to my model and would create confusion
#   if included.
#
# Input : data/raw/courses.raw.json
# Output: data/interim/courses.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ijson
import pandas as pd
from src.config import COURSES as RAW, COURSES_CLEAN as OUT, CHILONZOR_BRANCH_ID

def _oid(val):
    """Handle both MongoDB {'$oid': '...'} and plain string IDs from the API."""
    if isinstance(val, dict):
        return val.get("$oid")
    return val

print("Reading courses.raw.json ...")
records = []

with open(RAW, "rb") as f:
    for course in ijson.items(f, "item"):
        cid        = _oid(course.get("_id"))
        name       = course.get("name", "")
        deleted_at = course.get("deletedAt", 0)

        # find Chilonzor branch entry to extract its price and availability
        price     = None
        available = None
        for branch in course.get("branches") or []:
            bid = _oid(branch.get("branchId")) or ""
            if bid == CHILONZOR_BRANCH_ID:
                price     = branch.get("price")
                available = branch.get("available")
                break

        records.append({
            "courseId"  : cid,
            "name"      : name,
            "price"     : price,
            "available" : available,
            "isDeleted" : deleted_at != 0,
        })

print(f"  → {len(records)} courses loaded")

# ── build dataframe ────────────────────────────────────────────
df = pd.DataFrame(records)
# fillna(0) for price: courses with no Chilonzor branch entry have no
# standard price defined. 0 is the correct neutral value and prevents
# downstream numeric operations from failing on NaN.
df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype(int)

# ── summary ────────────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── All courses ──")
print(df[["courseId","name","price","available","isDeleted"]].to_string())
print("\n── Null counts ──")
print(df.isnull().sum())

# ── save ───────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
