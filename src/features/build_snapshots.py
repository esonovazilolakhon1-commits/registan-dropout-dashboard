# =============================================================
# build_snapshots.py
# =============================================================
# Monthly Snapshot Skeleton + Leak-Free Dropout Label
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan
#               Chilonzor Branch using Machine Learning
#
# Why I wrote this script:
#   My original pipeline assigned one row per student-course and
#   computed every feature over the student's full history. That
#   produced a ROC-AUC of 0.9996 — which I immediately recognised
#   as too good to be real. After investigation I found the design
#   had data leakage: features like final attendance rate and
#   terminal balance were computed AFTER the dropout event, so they
#   were essentially encoding the outcome itself.
#
#   I switched to a monthly snapshot panel to fix this. Each row
#   represents one student-course observed at a specific calendar
#   month. Features are computed from data strictly before the end
#   of that month; the label looks strictly forward. No future
#   information can leak into past features.
#
#   This script handles only the skeleton (student-course-month
#   combinations) and the label. I deliberately kept feature
#   computation in a separate script so I can audit the label
#   distribution before building anything on top of it.
#
# Why I defined "dropout" through attendance, not system status:
#   Registan staff rarely update the student status field — only
#   933 of 24,044 students are marked as "graduated", which I know
#   from the data is a severe undercount. Rather than rely on
#   unreliable admin records, I define dropout behaviourally:
#   a student is labelled as dropping out in month m if they attend
#   no lesson in the following HORIZON_DAYS (30 days). This is
#   fully derived from the attendance log, which is well-maintained
#   because teachers record attendance for every lesson.
#
#   Crucially, the label is built from raw attendance DATES, never
#   from attendanceRate — which stays a model feature. There is
#   therefore no circularity between the label and the features.
#
# How I handle the right-censoring problem:
#   The attendance log ends at REFERENCE_DATE (the last date any
#   lesson was recorded). For a snapshot in, say, February 2026,
#   the forward window extends into March 2026 — beyond the data.
#   I cannot observe whether those students attended or not, so I
#   cannot assign a label. I separate these rows into two sets:
#     - "scoring": the LATEST month of a currently-active student-
#       course. These are the students my model will score for the
#       moderators — the real deployment output.
#     - "censored": other months whose window extends past the end
#       of data. I exclude these from training entirely.
#
# Output:
#   data/processed/snapshots_labeled.parquet
#     studentId, courseName, snapshotMonth, label (0/1/NaN), rowSet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import (
    ATTENDANCE_CLEAN, STUDENTGROUPS_CLEAN, PROCESSED,
    COURSE_NAME_MAP, DROPOUT_GAP_DAYS, HORIZON_DAYS, ACTIVE_WINDOW_DAYS,
)

print("=" * 60)
print("BUILDING MONTHLY SNAPSHOT SKELETON + LABEL")
print("=" * 60)

# =============================================================
# STEP 1: LOAD ATTENDANCE AND MAP TO COURSE NAMES
# =============================================================
# I define student activity using ATTENDED lessons only, not
# scheduled ones. A student who is scheduled but has not shown
# up for weeks has, in practice, already stopped coming —
# which is exactly the dropout signal I want to detect early.
# Using "attended" as the activity signal also means my label
# (no attendance in the next 30 days) is consistent with the
# features (attendance rates computed from attended lessons).
print("\nStep 1: Loading attendance, keeping attended lessons...")

att = pd.read_parquet(ATTENDANCE_CLEAN)
att['courseName'] = att['courseId'].map(COURSE_NAME_MAP)
att = att[att['courseName'].notna()].copy()

# I strip timezone info and round to day precision so that month
# arithmetic with pd.period_range works without timezone-related
# errors. The raw timestamps from MongoDB are stored in UTC, but
# all analysis is relative so the timezone offset does not matter.
att['date'] = pd.to_datetime(att['date'], errors='coerce', utc=True).dt.tz_localize(None)
att = att.dropna(subset=['date'])

attended = att[att['state'] == 'attended'][['studentId', 'courseName', 'date']].copy()
attended = attended.sort_values(['studentId', 'courseName', 'date'])

# I anchor "today" to the last date in the data rather than
# pd.Timestamp.now(). This makes the pipeline fully reproducible:
# running the script tomorrow would not change which students are
# labelled as active or which months are censored.
REFERENCE_DATE = attended['date'].max()
print(f"  → Attended-lesson rows : {len(attended):,}")
print(f"  → REFERENCE_DATE (data 'today') : {REFERENCE_DATE.date()}")
print(f"  → Student-course pairs : {attended.groupby(['studentId','courseName']).ngroups:,}")

# =============================================================
# STEP 2: IDENTIFY GRADUATED STUDENT-COURSES
# =============================================================
# Even though the status field is unreliable, the studentgroups
# table does contain graduation signals — either state='graduated'
# or a non-null graduatedAt timestamp. I use these to identify
# student-course pairs where the student completed the course.
#
# This matters because a graduating student's final month looks
# identical to a dropout's final month in the attendance log:
# both stop attending. Without this correction, I would label
# every completer's exit as a dropout and inflate the positive
# class rate with false positives.
print("\nStep 2: Identifying graduated student-courses (completion, not dropout)...")

sg = pd.read_parquet(STUDENTGROUPS_CLEAN)
sg['courseName'] = sg['courseId'].map(COURSE_NAME_MAP)
sg = sg[sg['courseName'].notna()].copy()

is_grad = (sg['state'] == 'graduated') | (sg['graduatedAt'].notna())
graduated_pairs = set(
    map(tuple, sg.loc[is_grad, ['studentId', 'courseName']].drop_duplicates().values)
)
print(f"  → Graduated student-courses : {len(graduated_pairs):,}")

# =============================================================
# STEP 3: BUILD MONTHLY SNAPSHOTS + FORWARD-LOOKING LABEL
# =============================================================
# For each student-course I iterate through every calendar month
# they were active (from their first attended lesson to their
# last). For each month m I define:
#
#   forward window = (last moment of month m,
#                     last moment of month m + HORIZON_DAYS]
#
#   label = 0  if any attended lesson falls in that window
#   label = 1  if no attended lesson falls in that window
#
# I use np.searchsorted on the sorted date array rather than a
# date-range filter because searchsorted runs in O(log n) per
# month, which keeps the whole loop fast even for students with
# several years of attendance history.
#
# Graduation correction: if a student-course is in graduated_pairs
# and its final month gets label=1 (no more attendance), I flip
# the label to 0. Stopping because you graduated is not dropping
# out, and mislabelling it would teach the model the wrong thing.
print("\nStep 3: Generating monthly snapshots and forward label...")

HORIZON = pd.Timedelta(days=HORIZON_DAYS)
ACTIVE_CUTOFF = REFERENCE_DATE - pd.Timedelta(days=ACTIVE_WINDOW_DAYS)

rows = []
for (sid, course), g in attended.groupby(['studentId', 'courseName'], sort=False):
    dates = g['date'].values  # sorted datetime64, ascending
    first_date = pd.Timestamp(dates[0])
    last_date  = pd.Timestamp(dates[-1])

    # I define "currently active" as having attended at least one
    # lesson within ACTIVE_WINDOW_DAYS of the reference date.
    # These students are the scoring cohort the model predicts for.
    is_active = last_date >= ACTIVE_CUTOFF
    is_grad   = (sid, course) in graduated_pairs

    months = pd.period_range(first_date, last_date, freq='M')
    for i, p in enumerate(months):
        m_end   = p.end_time          # last microsecond of month m
        win_end = m_end + HORIZON

        # searchsorted lets me count attended lessons in the window
        # without creating a boolean mask over the full date array
        lo = np.searchsorted(dates, np.datetime64(m_end), side='right')
        hi = np.searchsorted(dates, np.datetime64(win_end), side='right')
        attended_next = hi > lo

        is_last_month = (i == len(months) - 1)

        if win_end <= REFERENCE_DATE:
            # The forward window is fully within the data — I can
            # observe the label with certainty.
            label  = 0 if attended_next else 1
            # Flip the label for a graduated student's final exit
            # so completers are not counted as dropouts.
            if label == 1 and is_grad and is_last_month:
                label = 0
            row_set = 'train'
        else:
            # The forward window extends past the end of the data.
            # I cannot observe the label, so I must classify this
            # row as either scoring (to be predicted) or censored
            # (to be excluded from training entirely).
            if is_active and is_last_month:
                label, row_set = np.nan, 'scoring'
            else:
                label, row_set = np.nan, 'censored'

        rows.append((sid, course, str(p), label, row_set))

snap = pd.DataFrame(rows, columns=['studentId', 'courseName', 'snapshotMonth', 'label', 'rowSet'])
print(f"  → Total snapshot rows : {len(snap):,}")

# =============================================================
# STEP 4: SUMMARY + SAVE
# =============================================================
# I print the class balance and per-course breakdown before saving
# so I can verify the label is behaving as expected. I expect
# label=1 to be around 30–35% of training rows, which is a
# moderate imbalance I can handle with scale_pos_weight in XGBoost.
print("\nStep 4: Summary...")

train = snap[snap['rowSet'] == 'train']
score = snap[snap['rowSet'] == 'scoring']
cens  = snap[snap['rowSet'] == 'censored']

n_pos = int((train['label'] == 1).sum())
n_neg = int((train['label'] == 0).sum())
pos_rate = n_pos / max(len(train), 1)

print(f"  → Trainable snapshots : {len(train):,}")
print(f"      label=1 (dropped next month) : {n_pos:,}  ({pos_rate:.2%})")
print(f"      label=0 (retained)           : {n_neg:,}  ({1-pos_rate:.2%})")
print(f"  → Scoring snapshots (active now) : {len(score):,}")
print(f"  → Censored (excluded)            : {len(cens):,}")
print(f"  → Distinct students             : {snap['studentId'].nunique():,}")
print(f"  → Distinct student-courses      : {snap.groupby(['studentId','courseName']).ngroups:,}")

print("\n  Trainable label-1 rate by course:")
by_course = train.groupby('courseName')['label'].agg(['mean', 'size']).sort_values('mean', ascending=False)
print(by_course.to_string())

print("\n  Trainable snapshots by year:")
train_year = train.assign(year=train['snapshotMonth'].str[:4])
print(train_year.groupby('year')['label'].agg(['mean', 'size']).to_string())

PROCESSED.mkdir(parents=True, exist_ok=True)
out = PROCESSED / "snapshots_labeled.parquet"
snap.to_parquet(out, index=False)
print(f"\n✅ Saved snapshot skeleton → {out}")
print(f"   Rows    : {len(snap):,}")
print(f"   Columns : {list(snap.columns)}")
