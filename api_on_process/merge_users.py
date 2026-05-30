# merge_users.py
# ─────────────────────────────────────────────────────────────────────────────
# Smart users merge for the live pipeline.
#
# Logic:
#   1. Always start with the original data/interim/users.parquet as the base
#      (contains all historical staff — teachers, moderators, admins).
#   2. If data/live_raw/users.raw.json exists and has new records (newly added
#      staff since the last snapshot), parse them and merge on top.
#   3. Dedup by userId — keep the newest record (by updatedAt) for each user.
#   4. Save the merged result to data/live_interim/users.parquet.
#
# This ensures live predictions always see ALL current staff, not just newly
# added ones. If the API returns 0 new users, original data is used as-is.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.config import USERS as RAW, USERS_CLEAN as OUT, CHILONZOR_BRANCH_ID, ROOT

ORIGINAL_INTERIM = ROOT / "data" / "interim" / "users.parquet"

print("=" * 60)
print("MERGE USERS — live + historical")
print("=" * 60)

# ── Step 1: load original cleaned users (always available) ───────────────────
print(f"\nLoading original users from: {ORIGINAL_INTERIM}")
if not ORIGINAL_INTERIM.exists():
    print("  ✗  Original users.parquet not found — cannot continue.")
    sys.exit(1)

base_df = pd.read_parquet(ORIGINAL_INTERIM)
print(f"  → {len(base_df):,} historical users loaded")

# ── Step 2: check if API pulled any new users ────────────────────────────────
new_df = None

if RAW.exists():
    try:
        with open(RAW, "r", encoding="utf-8") as f:
            raw_records = json.load(f)

        if raw_records:
            print(f"\nFound {len(raw_records):,} new users from API — parsing...")
            new_records = []

            for user in raw_records:
                # API returns flat records (no $oid/$date nesting)
                uid        = user.get("_id", "")
                type_      = user.get("type")
                first_name = (user.get("firstName") or "").strip()
                full_name  = (user.get("fullName") or "").strip()
                phone      = user.get("phoneNumber", "")
                is_boss    = user.get("isBoss", False)
                created_at = user.get("createdAt")
                updated_at = user.get("updatedAt")
                deleted_at = user.get("deletedAt", 0)

                # extract Chilonzor branch data
                balance = salary = bonus = punish = None
                prepayment = unpaid = attendance_pct = None
                branch_state = available = role_id = None

                for branch in user.get("branches") or []:
                    bid = branch.get("branchId", "")
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
                        role_id        = branch.get("roleId")
                        break

                new_records.append({
                    "userId"              : uid,
                    "type"                : type_,
                    "firstName"           : first_name,
                    "fullName"            : full_name,
                    "phoneNumber"         : phone,
                    "isBoss"              : is_boss,
                    "createdAt"           : created_at,
                    "updatedAt"           : updated_at,
                    "isDeleted"           : deleted_at != 0 if deleted_at else False,
                    "branchBalance"       : balance,
                    "branchSalary"        : salary,
                    "branchBonus"         : bonus,
                    "branchPunish"        : punish,
                    "branchPrepayment"    : prepayment,
                    "branchUnpaid"        : unpaid,
                    "attendancePercentage": attendance_pct,
                    "branchState"         : branch_state,
                    "available"           : available,
                    "roleId"              : role_id,
                })

            new_df = pd.DataFrame(new_records)
            for col in ["createdAt", "updatedAt"]:
                new_df[col] = pd.to_datetime(new_df[col], errors="coerce", utc=True)
            for col in ["branchBalance","branchSalary","branchBonus",
                        "branchPunish","branchPrepayment","branchUnpaid",
                        "attendancePercentage"]:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")

            print(f"  → {len(new_df):,} new users parsed")
        else:
            print("\nAPI returned 0 new users — using historical data only")
    except Exception as e:
        print(f"\nWARNING: Could not parse live users file: {e}")
        print("  → Falling back to historical data only")
else:
    print("\nNo live users file found — using historical data only")

# ── Step 3: merge ────────────────────────────────────────────────────────────
if new_df is not None and len(new_df) > 0:
    print("\nMerging new users with historical data...")
    combined = pd.concat([base_df, new_df], ignore_index=True)
    # keep newest record per userId (new API data wins over old)
    combined = combined.sort_values("updatedAt", ascending=True, na_position="first")
    combined = combined.drop_duplicates(subset=["userId"], keep="last")
    combined = combined.reset_index(drop=True)
    print(f"  → Historical: {len(base_df):,} | New: {len(new_df):,} | Merged (deduped): {len(combined):,}")
else:
    combined = base_df.copy()
    print(f"\n  → Using {len(combined):,} historical users as-is")

# ── Step 4: save ─────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}  ({len(combined):,} users)")
