# =============================================================
# clean_groups.py
# =============================================================
# Clean the raw groups export from Registan's MongoDB.
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why this script is more complex than the other clean scripts:
#   The groups JSON has heavily nested fields (courseShift, courseDay,
#   branch-specific prices) that need to be flattened into a flat
#   dataframe. The most important non-trivial step is decoding the
#   Uzbek day-of-week abbreviations — Registan stores schedule days
#   as Uzbek abbreviations ('du', 'se', 'ch' etc.) which would be
#   opaque in downstream analysis. I decode them to English names.
#
# Why I derive shiftPeriod from startHour:
#   The raw data stores the shift as a start/end hour string
#   (e.g. "09:00", "14:00"). I convert this to a categorical
#   (morning / afternoon / evening) because:
#   (1) it is more interpretable for the dashboard, and
#   (2) build_master.py uses it as a categorical feature rather than
#       a numeric time, so the categories need to be consistent
#       across groups, orders, and students.
#
# Input : data/raw/groups.raw.json
# Output: data/interim/groups.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ijson
import pandas as pd
from src.config import GROUPS as RAW, GROUPS_CLEAN as OUT

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

# ── weekday decoder ────────────────────────────────────────────
# Registan stores lesson days as Uzbek two-letter abbreviations.
# I map these to English day names so downstream code is readable
# without knowledge of Uzbek.
WEEKDAY_MAP = {
    'ya': 'sunday',  'du': 'monday',  'se': 'tuesday',
    'ch': 'wednesday', 'pa': 'thursday', 'ju': 'friday',
    'sh': 'saturday'
}

# Some schedules are stored as full Uzbek phrases ('juft kunlar' =
# even days = Mon/Wed/Fri) rather than abbreviations. I decode these
# separately before trying the abbreviation lookup.
SPECIAL_DAYS = {
    'juft kunlar'  : 'mon,wed,fri',
    'toq kunlar'   : 'tue,thu,sat',
    'har kuni'     : 'mon,tue,wed,thu,fri,sat,sun',
    'dushanba'     : 'monday',
    'seshanba'     : 'tuesday',
    'chorshanba'   : 'wednesday',
    'payshanba'    : 'thursday',
    'juma'         : 'friday',
    'shanba'       : 'saturday',
    'yakshanba'    : 'sunday',
}

def decode_days(title):
    """Decode Uzbek day abbreviations to English day names."""
    if not title:
        return None
    t = title.strip().lower()

    # check special patterns first
    if t in SPECIAL_DAYS:
        return SPECIAL_DAYS[t]

    # split by comma and decode each abbreviation
    parts = [p.strip() for p in t.split(',')]
    decoded = []
    for p in parts:
        if p in WEEKDAY_MAP:
            decoded.append(WEEKDAY_MAP[p])
        else:
            decoded.append(p)
    return ','.join(decoded) if decoded else None

# ── stream and extract ─────────────────────────────────────────
print("Reading groups.raw.json ...")
records = []

with open(RAW, "rb") as f:
    for group in ijson.items(f, "item"):

        gid        = _oid(group.get("_id"))
        name       = group.get("name", "")
        state      = group.get("state")
        type_      = group.get("type")
        course_price = group.get("coursePrice", 0)
        deleted_at = group.get("deletedAt", 0)
        lessons_count = group.get("lessonsCount", 0)
        students   = group.get("students", 0)

        # nested $oid fields
        course_id  = _oid(group.get("courseId"))
        teacher_id = _oid(group.get("teacherId"))
        room_id    = _oid(group.get("roomId"))
        branch_id  = _oid(group.get("branchId"))
        lesson_id  = _oid(group.get("lessonId"))

        # nested $date fields
        starts_at  = _date(group.get("startsAt"))
        ends_at    = _date(group.get("endsAt"))

        # flatten courseShift into start/end hour strings
        shift      = group.get("courseShift") or {}
        shift_start = shift.get("startHour")
        shift_end   = shift.get("endHour")

        # flatten courseDay and decode Uzbek abbreviations to English
        course_day  = group.get("courseDay") or {}
        day_title   = course_day.get("title")
        days_decoded = decode_days(day_title)

        # derive shift period from the start hour:
        # before 12:00 = morning, 12:00–16:59 = afternoon, 17:00+ = evening
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

        records.append({
            "groupId"      : gid,
            "name"         : name,
            "state"        : state,
            "type"         : type_,
            "courseId"     : course_id,
            "teacherId"    : teacher_id,
            "roomId"       : room_id,
            "branchId"     : branch_id,
            "lessonId"     : lesson_id,
            "coursePrice"  : course_price,
            "lessonsCount" : lessons_count,
            "students"     : students,
            "startsAt"     : starts_at,
            "endsAt"       : ends_at,
            "shiftStart"   : shift_start,
            "shiftEnd"     : shift_end,
            "courseDays"   : days_decoded,
            "rawCourseDays": day_title,
            "shiftPeriod"  : shift_period,
            "isDeleted"    : deleted_at != 0,
        })

print(f"  → {len(records)} groups loaded")

# ── build dataframe ────────────────────────────────────────────
df = pd.DataFrame(records)

for col in ["startsAt", "endsAt"]:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# fill 0 for numeric fields where null means zero (no students, no lessons)
df["coursePrice"]   = pd.to_numeric(df["coursePrice"],   errors="coerce").fillna(0).astype(int)
df["lessonsCount"]  = pd.to_numeric(df["lessonsCount"],  errors="coerce").fillna(0).astype(int)
df["students"]      = pd.to_numeric(df["students"],      errors="coerce").fillna(0).astype(int)

# ── summary ────────────────────────────────────────────────────
print("\n── Shape ──")
print(df.shape)
print("\n── State distribution ──")
print(df["state"].value_counts())
print("\n── Type distribution ──")
print(df["type"].value_counts())
print("\n── Shift period distribution ──")
print(df["shiftPeriod"].value_counts())
print("\n── Null counts ──")
print(df.isnull().sum())
print("\n── Sample decoded days ──")
print(df[["rawCourseDays","courseDays","shiftPeriod"]].head(10).to_string())
print(f"\nTotal rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")

# ── save ───────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"\n✅ Saved → {OUT}")
