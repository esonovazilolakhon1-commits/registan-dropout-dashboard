# config.py
#
# I centralise every path and tuning parameter here so that no script
# ever needs to know where it is on disk or hard-code a threshold.
# If the project root moves, or if I decide to change the dropout gap
# from 30 to 45 days, I only edit this one file — not twelve scripts.

import os
from pathlib import Path

# I use Path(__file__).parent.parent so the root resolves correctly
# regardless of which directory I launch Python from. Hard-coding an
# absolute path would break the moment I moved the project to another
# machine or submitted the replication package to my supervisor.
ROOT = Path(__file__).parent.parent

# I split the data into three layers following the standard data
# engineering convention that my supervisor suggested:
#   raw     → exactly what came out of Registan's MongoDB, untouched
#   interim → cleaned and typed, one table per collection
#   processed → merged, feature-engineered, model-ready
# This separation means I can always re-derive any output from raw
# without losing the intermediate cleaning steps.
#
# When the live pipeline runs (LIVE_PIPELINE=1), all three layers are
# redirected to data/live_raw, data/live_interim, data/live_processed
# so the original files are NEVER overwritten. The dashboard and the
# trained model always read from the original data/ directories.
if os.getenv("LIVE_PIPELINE") == "1":
    RAW        = ROOT / "data" / "live_raw"
    INTERIM    = ROOT / "data" / "live_interim"
    PROCESSED  = ROOT / "data" / "live_processed"
else:
    RAW        = ROOT / "data" / "raw"
    INTERIM    = ROOT / "data" / "interim"
    PROCESSED  = ROOT / "data" / "processed"
REFERENCE  = ROOT / "data" / "reference"

# Trained model artifacts — I version them here so predict.py and
# explain_model.py always load from the same location.
MODELS     = ROOT / "models"

# All figures are written here so they can be embedded directly in
# the dissertation without manual copy-pasting.
REPORTS    = ROOT / "reports"
FIGURES    = ROOT / "reports" / "figures"

# Raw collection exports from MongoDB. The attendance file was
# exported with a typo in the original filename ("attandance"), so
# I preserved the original name to avoid confusion when re-exporting.
STUDENTS        = RAW / "students.raw.json"
USERS           = RAW / "users.raw.json"
GROUPS          = RAW / "groups.raw.json"
COURSES         = RAW / "courses.raw.json"
ORDERS          = RAW / "orders.raw.json"
STUDENTGROUPS   = RAW / "studentgroups.raw.csv"
STUDENTTEACHERS = RAW / "studentteachers.raw.csv"
GROUPHISTORIES  = RAW / "grouphistories.raw.csv"
STUDENTHISTORIES= RAW / "studenthistories.raw.csv"
LESSONS         = RAW / "lessons.raw.csv"
ATTENDANCE      = RAW / "attandance_raw.csv"
TRANSACTIONS    = RAW / "transactions.raw.csv"
BRANCHES        = RAW / "branches.raw.csv"
LEVELS          = RAW / "levelsifneeded.raw.csv"

# Reference look-up tables for reasons, payment methods, statuses and
# bonuses. These are small CSV files I compiled manually from Registan's
# admin panel because they were not exported as part of the MongoDB dump.
REASONS         = REFERENCE / "reasons.ready.csv"
PAYMENTMETHODS  = REFERENCE / "paymentmethods.ready.csv"
STATUSES        = REFERENCE / "statues.ready.csv"
BONUSES         = REFERENCE / "bonuses.ready.csv"
REASONSANALYTICS= REFERENCE / "reasonsanalytics.ready.csv"

# Cleaned parquet files — one per source collection. I chose parquet
# over CSV here because it preserves dtypes (datetimes stay datetimes,
# integers stay integers) and loads about 5x faster, which matters
# when the feature builder reads several of these at once.
STUDENTS_CLEAN        = INTERIM / "students.parquet"
STUDENTGROUPS_CLEAN   = INTERIM / "studentgroups.parquet"
STUDENTTEACHERS_CLEAN = INTERIM / "studentteachers.parquet"
GROUPHISTORIES_CLEAN  = INTERIM / "grouphistories.parquet"
STUDENTHISTORIES_CLEAN= INTERIM / "studenthistories.parquet"
LESSONS_CLEAN         = INTERIM / "lessons.parquet"
ATTENDANCE_CLEAN      = INTERIM / "attendance.parquet"
TRANSACTIONS_CLEAN    = INTERIM / "transactions.parquet"
GROUPS_CLEAN          = INTERIM / "groups.parquet"
COURSES_CLEAN         = INTERIM / "courses.parquet"
ORDERS_CLEAN          = INTERIM / "orders.parquet"
USERS_CLEAN           = INTERIM / "users.parquet"
BRANCHES_CLEAN        = INTERIM / "branches.parquet"
LEVELS_CLEAN          = INTERIM / "levels.parquet"

# The final merged table before feature engineering.
MASTER = PROCESSED / "master.parquet"

# The Registan Chilonzor branch is the one site this dissertation
# studies. I filter to this branch ID early in the pipeline so that
# students from other branches never contaminate the analysis.
CHILONZOR_BRANCH_ID = "6266d9e35bbdd74734fddadd"

# Before 2022-05-01 Registan bulk-imported historical records that
# were not entered in real time. Those records have unreliable
# timestamps and would distort the attendance trend features, so I
# drop any enrolment that pre-dates this cutoff.
BULK_IMPORT_CUTOFF  = "2022-05-01"

# I map raw MongoDB ObjectId strings to human-readable subject names
# for two reasons: (1) the dashboard needs legible labels for charts,
# and (2) English is taught at nine different proficiency levels at
# Registan — treating them as nine separate courses would fragment
# the data and make course-level dropout rates unreliable. I collapse
# all English levels into a single 'English' subject to get enough
# samples for stable statistics.
#
# Courses not listed here — notably Turkish (n=38 students) — are
# excluded from modelling. Thirty-eight students is too small to train
# or evaluate a per-course dropout signal reliably, so I decided to
# drop them rather than introduce noise.
COURSE_NAME_MAP = {
    # English language — all levels unified into one subject
    '622f54b4475460a853fd9797': 'English',  # Ingliz tili (IELTS)
    '664051617db91d5e02eaf2ee': 'English',  # IELTS
    '66404b6fd1ebe463be646ef2': 'English',  # Beginner
    '66404e3bd1ebe463be646f28': 'English',  # Elementary
    '66404f7454f171b1c342f579': 'English',  # Pre-intermediate
    '664050544dcd8abb27fbd43b': 'English',  # Intermediate
    '66405bda4dcd8abb27fbd83a': 'English',  # CEFR
    '66405a8d7db91d5e02eaf5fc': 'English',  # English Grammar
    '664052794dcd8abb27fbd474': 'English',  # Level 2 kids
    '664051ac4dcd8abb27fbd460': 'English',  # Level 1 kids
    # Other subjects kept as individual courses
    '622f54b4475460a853fd97a4': 'Russian',
    '622f54b4475460a853fd9799': 'Math',
    '622f54b4475460a853fd9798': 'Uzbek',
    '622f54b4475460a853fd979e': 'Korean',
    '622f54b4475460a853fd97a3': 'Biology',
    '622f54b4475460a853fd97a0': 'Chinese',
    '622f54b4475460a853fd97a7': 'Chemistry',
    '622f54b4475460a853fd97a6': 'History',
    '62f4cc5215efadb00bf0c8f9': 'Law',
    '622f54b4475460a853fd979f': 'Physics',
}

# These three thresholds define the core methodology of the
# snapshot model. I derived them from the attendance log itself
# rather than assuming a value from the literature:
#
#   DROPOUT_GAP_DAYS = 30
#   During active study at Registan, lessons happen every 2–3 days
#   (the median inter-lesson gap is 2 days, the 99th percentile is
#   16 days). A gap of 30 or more days with no attended lesson occurs
#   in fewer than 0.5 % of active study spells, so I treat 30 days of
#   silence as a clean, conservative signal that a student has stopped
#   coming — not just taken a short break.
#
#   HORIZON_DAYS = 30
#   The model predicts whether a student will stop attending in the
#   next 30 days. I chose 30 days because it gives moderators enough
#   lead time to intervene (a phone call, a payment reminder) before
#   the student is already gone.
#
#   ACTIVE_WINDOW_DAYS = 30
#   A student is treated as currently active if they attended at least
#   one lesson within the last 30 days of the reference date. I anchored
#   the reference date to 2026-03-19 (the last date in the attendance
#   log) rather than pd.Timestamp.now() so that the definition of
#   "active" is reproducible and does not change every time the
#   scoring script is run.
DROPOUT_GAP_DAYS = 30   # >= this many days with no attended lesson = stopped
HORIZON_DAYS     = 30   # prediction target: will the student stop in next 30 days
ACTIVE_WINDOW_DAYS = 30 # attended within this many days of the reference date = currently active
