# =============================================================
# clean_orders.py
# =============================================================
# Clean the raw orders export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# What the orders table contains:
#   An order in Registan's system is a student's initial enrolment
#   request — created before the student is formally assigned to a
#   group. A cancelled order means the student expressed interest but
#   never joined. I use pre-enrolment cancellation counts as a
#   commitment signal in build_master.py: students who cancelled
#   multiple orders before finally enrolling may have lower commitment
#   to continuing their studies.
#
# Why I derive isReferred here:
#   referredByStudentId is non-null when a student was referred by an
#   existing student. I convert this to a clean boolean (isReferred)
#   because downstream code only needs to know whether a referral
#   occurred, not who made it.
#
# Why I derive shiftPeriod from startHour:
#   I use the same hour-based derivation as clean_groups.py so that
#   shiftPeriod is consistent across all tables that contain shift data.
#
# Input : data/raw/orders.raw.json
# Output: data/interim/orders.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ijson
import pandas as pd
from src.config import ORDERS as RAW, ORDERS_CLEAN as OUT

def _oid(val):
    """Handle both MongoDB {'$oid': '...'} and plain string IDs from the API."""
    if isinstance(val, dict):
        return val.get("$oid")
    return val

def _date(val):
    """Handle both MongoDB {'$date': '...'} and plain date strings from the API."""
    if isinstance(val, dict):
        return val.get("$date")
    return val

print("Reading orders.raw.json ...")
records = []

with open(RAW, "rb") as f:
    for order in ijson.items(f, "item"):

        order_id    = _oid(order.get("_id"))
        deleted_at  = order.get("deletedAt", 0)
        state       = order.get("state")
        comment     = order.get("comment", "")

        # nested $oid fields
        branch_id   = _oid(order.get("branchId"))
        student_id  = _oid(order.get("studentId"))
        course_id   = _oid(order.get("courseId"))
        moderator_id= _oid(order.get("moderatorId"))
        group_id    = _oid(order.get("groupId"))
        referred_by = _oid(order.get("referredByStudentId"))

        # nested $date fields
        created_at  = _date(order.get("createdAt"))
        come_date   = _date(order.get("comeDate"))

        # flatten courseShift into start/end hour strings
        shift       = order.get("courseShift") or {}
        shift_start = shift.get("startHour")
        shift_end   = shift.get("endHour")

        # derive shift period using the same hour thresholds as clean_groups.py
        shift_period = None
        if shift_start:
            try:
                hour = int(shift_start.split(":")[0])
                if hour < 12:
                    shift_period = "morning"
                elif hour < 17:
                    shift_period = "afternoon"
                else:
                    shift_period = "evening"
            except:
                pass

        # flatten courseDay — kept as raw title (no Uzbek decoding needed here)
        course_day  = order.get("courseDay") or {}
        day_title   = course_day.get("title")

        records.append({
            "orderId"            : order_id,
            "studentId"          : student_id,
            "branchId"           : branch_id,
            "courseId"           : course_id,
            "moderatorId"        : moderator_id,
            "groupId"            : group_id,
            "referredByStudentId": referred_by,
            "state"              : state,
            "comment"            : comment,
            "createdAt"          : created_at,
            "comeDate"           : come_date,
            "shiftStart"         : shift_start,
            "shiftEnd"           : shift_end,
            "shiftPeriod"        : shift_period,
            "courseDays"         : day_title,
            "isDeleted"          : deleted_at != 0,
            # isReferred is True when referredByStudentId is not null
            "isReferred"         : referred_by is not None,
        })

print(f"  → {len(records)} orders loaded")

# ── build dataframe ────────────────────────────────────────────
df = pd.DataFrame(records)

for col in ["createdAt", "comeDate"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# ── summary ────────────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── State distribution ──")
print(df["state"].value_counts())
print("\n── Shift period distribution ──")
print(df["shiftPeriod"].value_counts())
print("\n── Referred orders ──")
print(df["isReferred"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")

# ── save ───────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
