# =============================================================
# clean_users.py
# =============================================================
# Clean the raw users export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# What this table contains:
#   Users in Registan's system are staff members — teachers, moderators,
#   administrators. I use this table for two purposes:
#   (1) to look up teacher and moderator names for dashboard display
#       (so the moderator sees "Teacher: Sardor" not a hex ID), and
#   (2) to compute teacher-level features (average attendance rate,
#       dropout rate) that appear in build_master.py.
#
# Why I extract only Chilonzor branch data for each user:
#   Like students, users have a 'branches' array with branch-specific
#   salary, balance, and state. I extract only the Chilonzor entry
#   because salary and availability at other branches are irrelevant
#   to my model.
#
# Input : data/raw/users.raw.json
# Output: data/interim/users.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ijson
import pandas as pd
from src.config import USERS as RAW, USERS_CLEAN as OUT, CHILONZOR_BRANCH_ID

print("Reading users.raw.json ...")
records = []

with open(RAW, "rb") as f:
    for user in ijson.items(f, "item"):

        uid        = (user.get("_id") or {}).get("$oid")
        type_      = user.get("type")
        first_name = (user.get("firstName") or "").strip()
        full_name  = (user.get("fullName") or "").strip()
        phone      = user.get("phoneNumber", "")
        is_boss    = user.get("isBoss", False)
        deleted_at = user.get("deletedAt", 0)

        # nested dates
        created_at = (user.get("createdAt") or {}).get("$date")
        updated_at = (user.get("updatedAt") or {}).get("$date")

        # extract Chilonzor branch data — salary, balance, and role
        balance             = None
        salary              = None
        bonus               = None
        punish              = None
        prepayment          = None
        unpaid              = None
        attendance_pct      = None
        branch_state        = None
        available           = None
        role_id             = None

        for branch in user.get("branches") or []:
            bid = (branch.get("branchId") or {}).get("$oid", "")
            if bid == CHILONZOR_BRANCH_ID:
                balance        = branch.get("balance")
                salary         = branch.get("salary")
                bonus          = branch.get("bonus")
                punish         = branch.get("punish")
                prepayment     = branch.get("prepayment")
                unpaid         = branch.get("unpaid")
                attendance_pct = branch.get("attendancePercentage")
                branch_state   = branch.get("state")
                available      = branch.get("available")
                role_id        = (branch.get("roleId") or {}).get("$oid")
                break

        records.append({
            "userId"             : uid,
            "type"               : type_,
            "firstName"          : first_name,
            "fullName"           : full_name,
            "phoneNumber"        : phone,
            "isBoss"             : is_boss,
            "createdAt"          : created_at,
            "updatedAt"          : updated_at,
            "isDeleted"          : deleted_at != 0,
            "branchBalance"      : balance,
            "branchSalary"       : salary,
            "branchBonus"        : bonus,
            "branchPunish"       : punish,
            "branchPrepayment"   : prepayment,
            "branchUnpaid"       : unpaid,
            "attendancePercentage": attendance_pct,
            "branchState"        : branch_state,
            "available"          : available,
            "roleId"             : role_id,
        })

print(f"  → {len(records)} users loaded")

# ── build dataframe ────────────────────────────────────────────
df = pd.DataFrame(records)

for col in ["createdAt", "updatedAt"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# financial and performance fields can be null for users with no
# Chilonzor branch entry — coerce to numeric and leave as NaN
for col in ["branchBalance","branchSalary","branchBonus",
            "branchPunish","branchPrepayment","branchUnpaid",
            "attendancePercentage"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ── summary ────────────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── Type distribution ──")
print(df["type"].value_counts())
print("\n── Branch state distribution ──")
print(df["branchState"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print("\n── Sample ──")
print(df[["userId","type","firstName","fullName","attendancePercentage","branchState"]].head(10).to_string())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")

# ── save ───────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
