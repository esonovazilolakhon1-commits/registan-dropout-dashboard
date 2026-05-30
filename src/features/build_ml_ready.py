# =============================================================
# build_ml_ready.py
# =============================================================
# ML-Ready Feature Table Builder
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan
#               Chilonzor Branch using Machine Learning
#
# Why I wrote this as a separate script from build_master.py:
#   build_master.py produces a single wide table that serves two
#   different downstream consumers: the dashboard and the model.
#   The dashboard needs identifier columns (teacher names, group IDs)
#   that would corrupt the model's learning if included as features.
#   The model needs a clean, correlation-checked feature set that the
#   dashboard does not care about. Rather than embed these conflicting
#   requirements in one script, I split them here — one output for
#   each consumer.
#
#   1. master_full.parquet
#      All 72 columns, including identifiers. Used by the dashboard
#      to show contextual information (who a student's teacher is,
#      which group they are in) when a moderator views an alert.
#
#   2. master_ml.parquet
#      Identifier columns removed. Redundant, highly correlated, and
#      leakage-risk features removed. 56 clean features in a
#      principled order from student-level to course-level context.
#      This is the file that train_model.py and encode_features.py
#      read for model training and evaluation.
#
# Column removal rationale (three categories):
#
#   Category 1 — Identifiers (MongoDB ObjectIds)
#   These are arbitrary hexadecimal strings. A gradient boosted tree
#   or logistic regression cannot learn anything useful from a random
#   identifier — it can only memorise specific IDs, which would fail
#   entirely on new students with new IDs.
#
#   Category 2 — Redundant features
#   When two features encode the same underlying construct, keeping
#   both splits the importance signal in SHAP analysis — each appears
#   half as important as it truly is. I remove the member of each
#   redundant pair that is less interpretable or less directly linked
#   to the dropout mechanism.
#
#   Category 3 — Leakage-risk features
#   n_toArchiveState is almost synonymous with the dropout label —
#   students are moved to archive state precisely when they drop out.
#   Including it would let the model predict dropout from information
#   derived from the outcome itself.
#
#   High-correlation removals (from correlation.py, threshold |r| > 0.85):
#   - totalPaid vs paymentMonths: r = 0.943 → remove totalPaid
#   - n_addedToGroup_course vs n_removedFromGroup_course: r = 0.872
#     → remove n_addedToGroup_course
#
# Output:
#   data/processed/master_full.parquet — 72 columns (dashboard)
#   data/processed/master_ml.parquet  — 56 columns (ML model)
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import *

print("=" * 60)
print("BUILDING ML-READY FEATURE TABLE")
print("=" * 60)

master = pd.read_parquet(PROCESSED / "master.parquet")
print(f"\nLoaded master table: {len(master):,} rows × {len(master.columns)} columns")
print(f"  Label distribution in master:")
print(master['dropout'].value_counts().to_string())

# =============================================================
# OUTPUT 1: master_full.parquet — complete table for dashboard
# =============================================================
# I keep all columns here, including lastTeacherId and
# lastModeratorId, because the dashboard uses them to display
# contextual information about each at-risk student — who their
# teacher is, which group they are in, and which moderator should
# follow up. Without these columns, the moderator would see a
# risk score with no actionable context.
print("\nSaving master_full.parquet (complete table for dashboard)...")
master_full = master.copy()
master_full.to_parquet(PROCESSED / "master_full.parquet", index=False)
print(f"✅ master_full.parquet saved → {len(master_full.columns)} columns, {len(master_full):,} rows")

# =============================================================
# OUTPUT 2: master_ml.parquet — clean feature table for ML
# =============================================================
print("\nBuilding ML-ready feature table...")

# -------------------------------------------------------------
# COLUMNS TO REMOVE
# -------------------------------------------------------------
# Every removal below is explicitly justified. I went through this
# list with my supervisor and documented the reason for each decision
# so that the feature selection is transparent and reproducible.
DROP_COLS = [

    # ── Category 1: Identifier columns ──────────────────────
    # MongoDB ObjectIds are arbitrary hex strings with no ordinal
    # or mathematical meaning. Including them would let the model
    # memorise specific teacher or group IDs, which would fail
    # completely on any new student who has never been seen before.
    # These columns are kept in master_full for the dashboard only.
    'lastGroupId',       # hex ID of last group — not a predictor
    'lastTeacherId',     # hex ID of last teacher — not a predictor
    'lastModeratorId',   # hex ID of last moderator — not a predictor

    # ── Category 2: Redundant features ──────────────────────

    # avgGroupAttendanceRate is the average of per-group attendance
    # rates across all groups a student studied. It is conceptually
    # very similar to attendanceRate (the simpler overall ratio),
    # and the difference is methodological rather than substantive.
    # I retain attendanceRate as it is more directly computed and
    # easier to explain to my supervisor in the results section.
    'avgGroupAttendanceRate',

    # trialLessons counts introductory lessons before formal
    # enrolment. For students who did enrol, these lessons are
    # already captured in attendanceRate. The marginal signal
    # from knowing how many trial lessons a student attended is
    # negligible and adds unnecessary noise.
    'trialLessons',

    # totalCoursesFrozen counts courses where a student was frozen.
    # n_fronzen_course provides the same information at the more
    # granular course level, which makes the student-level count
    # fully redundant when course-level data is present.
    'totalCoursesFrozen',

    # minLessonPrice and maxLessonPrice add only marginal signal
    # beyond avgLessonPrice combined with hasDiscount. A student
    # with a lower minimum price than the standard already has
    # hasDiscount = 1. The range adds complexity without benefit.
    'minLessonPrice',
    'maxLessonPrice',

    # negativeEpisodes is the raw count of months ending in debt.
    # debtRate (proportion of months in debt) conveys the same
    # information in normalised form, which is preferable because
    # a student who was in debt in 3 of 4 months is more at risk
    # than one in debt 3 of 12 months, but raw counts cannot
    # distinguish them.
    'negativeEpisodes',

    # paymentCount is the number of individual top-up transactions.
    # paymentMonths (distinct months with a payment) aligns better
    # with Registan's monthly payment cycle and is more interpretable:
    # a student who paid in 6 months is more committed than one who
    # made 6 payments in a single month. paymentMonths is retained.
    'paymentCount',

    # totalReturned records refunds. Refunds are rare at Registan
    # and do not carry a consistent directional signal — a refund
    # can mean the student is leaving (dropout) or that there was
    # a billing correction (not dropout). The ambiguity makes this
    # feature more likely to confuse the model than help it.
    'totalReturned',

    # n_unFrozen_course is the raw unfreeze count. freezeReturnRate_course
    # (unfreezes / freezes) already captures this information as a
    # ratio. Keeping the raw count alongside the ratio would
    # artificially split the importance signal between them in SHAP.
    'n_unFrozen_course',

    # hadCancelledOrderBefore is a binary flag derivable directly
    # from orderCancellationRate > 0. Keeping both is redundant —
    # any model that has orderCancellationRate does not need a
    # separate binary copy of the same information.
    'hadCancelledOrderBefore',

    # ── Category 3: Leakage-risk features ───────────────────
    # n_toArchiveState counts how many times a student was moved
    # to archive state in Registan's system. Archive state is the
    # primary mechanism for marking students as inactive — it is
    # almost directly synonymous with the dropout label (dropout = 1).
    # Including this would be target leakage: the model would learn
    # to predict dropout from information that is essentially a
    # delayed echo of the dropout outcome itself, producing metrics
    # that look impressive in training but would not generalise.
    'n_toArchiveState',

    # ── High correlation removals (|r| > 0.85) ──────────────
    # I identified these pairs using the correlation analysis in
    # src/analysis/correlation.py. The 0.85 threshold follows
    # Kuhn & Johnson (2013). For each pair I removed the feature
    # that is less interpretable or less directly tied to dropout.

    # totalPaid vs paymentMonths: r = 0.943
    # A student who paid in more months naturally paid more total.
    # The two features are nearly linearly dependent. paymentMonths
    # is retained as it directly reflects consistent engagement
    # with Registan's monthly payment cycle.
    'totalPaid',

    # n_addedToGroup_course vs n_removedFromGroup_course: r = 0.872
    # Every group addition (except the very first) is preceded by
    # a removal, making the two counts nearly identical. I retain
    # n_removedFromGroup_course because removals are a stronger
    # direct signal of disrupted enrolment than additions.
    'n_addedToGroup_course',
]

DROP_COLS = [c for c in DROP_COLS if c in master.columns]
master_ml = master.drop(columns=DROP_COLS).copy()

print(f"\n  Removed {len(DROP_COLS)} columns:")
for c in DROP_COLS:
    print(f"    ✗ {c}")

# =============================================================
# FINAL COLUMN ORDER
# =============================================================
# I order columns from student-level to course-level, and within
# each level from most predictive to most contextual. This ordering
# has no effect on model performance but makes the feature importance
# outputs and SHAP visualisations much easier to read in the
# dissertation — the most important features appear near the top
# of any sorted list rather than scattered randomly.
FINAL_COL_ORDER = [
    # ── Composite keys ──────────────────────────────────────
    'studentId',        # primary key (excluded as a feature in training)
    'courseName',       # secondary key — one row per course

    # ── Demographics — student level ────────────────────────
    # Background features that may correlate with dropout through
    # cultural, seasonal, or cohort effects. I kept these despite
    # relatively low SHAP values because removing them would require
    # strong evidence they add no signal, and I do not have that.
    'gender',
    'language',
    'joinMonth',
    'joinYear',
    'joinSeason',
    'academicYear',
    'isReferred',

    # ── Attendance ratios — student level ───────────────────
    # The most predictive feature group. All expressed as ratios
    # (0.0–1.0) for comparability across students with different
    # study durations.
    'attendanceRate',
    'unreasonableAbsenceRate',
    'reasonableAbsenceRate',
    'frozenLessonRate',
    'activeMonthsRate',
    'attendanceTrend',
    'attendanceLast30Days',
    'attendanceLast60Days',
    'consecutiveMissedLessons',
    'paymentDoneRate',

    # ── Enrollment features — student level ─────────────────
    # Captures commitment depth and pricing context
    'totalGroupsJoined',
    'totalCoursesGraduated',
    'avgLessonPrice',
    'hasDiscount',
    'groupSwitchCount',

    # ── Transaction features — student level ────────────────
    # Financial behaviour — at Registan, financial stress is one
    # of the two most common dropout causes (the other is time
    # pressure). I capture both balance health and payment regularity.
    'currentBalance',
    'minBalance',
    'maxBalance',
    'avgMonthlyEndBalance',
    'debtRate',
    'avgPaymentAmount',
    'paymentMonths',
    'paymentRegularity',
    'totalDebtAmount',
    'unpaidRate',

    # ── Order features — student level ──────────────────────
    # Pre-enrolment commitment signals
    'cancelledOrders',
    'orderCancellationRate',

    # ── Course context — course level ───────────────────────
    # Environmental risk signals — some courses are structurally
    # harder to retain students in regardless of the individual
    'totalCoursesStudied',
    'courseAttendanceRate',
    'courseDropoutRate',

    # ── History events — course level ───────────────────────
    # Administrative event counts within this specific course
    'n_removedFromGroup_course',
    'n_fronzen_course',
    'freezeReturnRate_course',
    'n_groupPriceChanged_course',
    'n_graduatedFromGroup_course',
    'daysSinceLastRemoval_course',
    'daysSinceLastFreeze_course',

    # ── Last group context — course level ───────────────────
    # Quality and risk signals for the student's current group
    'lastGroupAttendanceRate',
    'lastGroupDropoutRate',

    # ── Last teacher context — course level ─────────────────
    # Teacher quality signals based on all students they have taught
    'lastTeacherAvgAttendanceRate',
    'lastTeacherLastMonthAttRate',
    'lastTeacherAttendanceTrend',
    'lastTeacherDropoutRate',

    # ── Moderator context ────────────────────────────────────
    'moderatorDropoutRate',

    # ── Shift context ────────────────────────────────────────
    'preferredShift',
    'shiftDropoutRate',

    # ── Target variable ──────────────────────────────────────
    'dropout',
]

FINAL_COL_ORDER = [c for c in FINAL_COL_ORDER if c in master_ml.columns]
master_ml = master_ml[FINAL_COL_ORDER].copy()

# =============================================================
# DATA QUALITY CHECK
# =============================================================
# I check for nulls before saving. Any remaining nulls after all
# the fill logic in build_master.py indicate students whose records
# are so incomplete that I cannot meaningfully compute features
# for them — I drop them rather than impute, because imputing a
# feature like attendanceRate for a student with no attendance
# record would produce a misleading value.
print(f"\n  Data quality check:")
total_nulls = master_ml.isnull().sum().sum()
print(f"  → Total null values : {total_nulls}")
if total_nulls > 0:
    null_cols = master_ml.isnull().sum()
    print(null_cols[null_cols > 0].to_string())

before = len(master_ml)
master_ml = master_ml.dropna(subset=['attendanceRate', 'dropout'])
print(f"\n  Dropped {before - len(master_ml)} rows with no behavioral data")

# =============================================================
# SAVE ML-READY TABLE
# =============================================================
master_ml.to_parquet(PROCESSED / "master_ml.parquet", index=False)

print(f"\n✅ master_ml.parquet saved")
print(f"   Rows            : {len(master_ml):,}")
print(f"   Columns         : {len(master_ml.columns)}")

print(f"\n  Final column list:")
for i, col in enumerate(master_ml.columns, 1):
    print(f"    {i:2}. {col}")

print(f"\n  Label distribution:")
print(master_ml['dropout'].value_counts().to_string())

train  = master_ml[master_ml['dropout'].isin([0, 1])]
score  = master_ml[master_ml['dropout'] == -1]
d1     = (train['dropout'] == 1).sum()
d0     = (train['dropout'] == 0).sum()

print(f"\n  Training set : {len(train):,} rows")
print(f"    dropout = 1 : {d1:,} ({d1/len(train):.1%})")
print(f"    dropout = 0 : {d0:,} ({d0/len(train):.1%})")
print(f"  Scoring set  : {len(score):,} rows (active students)")
