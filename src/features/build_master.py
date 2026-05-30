# =============================================================
# build_master.py
# =============================================================
# Master Feature Table Builder — Student Dropout Prediction
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan
#               Chilonzor Branch using Machine Learning
#
# Why I built this script:
#   This is the foundation of my original (pre-snapshot) pipeline.
#   It integrates 10 cleaned source files, engineers features at
#   both student and course level, and assigns a behaviourally-
#   defined dropout label. I kept it alongside the newer snapshot
#   pipeline to document the full development arc of my dissertation:
#   this script produced the single-row-per-student design that I
#   later discovered had data leakage — features like final attendance
#   rate were computed over the student's full history, including
#   the period after dropout. The snapshot approach in
#   build_snapshots.py was the fix.
#
# Key Design Decisions:
#   1. Composite key: studentId + courseName
#      One row per student per course. A student studying both
#      English and Math appears twice. This enables course-level risk
#      assessment, which is more actionable than a single student-level
#      score — a moderator can see that Marjona is high risk in English
#      but not Math and target her intervention accordingly.
#
#   2. Behavioural label definition:
#      The system state column alone is unreliable — Registan managers
#      frequently archive students without updating their status to
#      'graduated'. I define dropout using three behavioural conditions
#      (see Step 3 for full justification).
#
#   3. Ratios over raw counts:
#      I express all features as ratios where possible. This normalises
#      for different study durations — a student who attended 8 of 10
#      lessons is behaviourally equivalent to one who attended 80 of
#      100, but their raw counts differ by a factor of 10.
#
#   4. Course-level context features:
#      Teacher, group, moderator, and shift dropout rates are computed
#      from ALL students in that context, not just the individual. This
#      gives the model environmental risk signals that affect retention
#      independently of the student's own behaviour.
#
# Output:
#   data/processed/master.parquet — full master table (72 cols)
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import *

print("=" * 60)
print("BUILDING MASTER FEATURE TABLE")
print("=" * 60)

# =============================================================
# LOAD CLEANED SOURCE FILES
# =============================================================
# All source files were cleaned in Phase 1 (src/clean/).
# I chose parquet over CSV here because it preserves dtypes
# (datetimes stay datetimes) and loads significantly faster —
# important when 10 files are read at once on each pipeline run.
print("\nLoading cleaned source files...")
students = pd.read_parquet(STUDENTS_CLEAN)
sg       = pd.read_parquet(STUDENTGROUPS_CLEAN)
att      = pd.read_parquet(ATTENDANCE_CLEAN)
txn      = pd.read_parquet(TRANSACTIONS_CLEAN)
sh       = pd.read_parquet(STUDENTHISTORIES_CLEAN)
orders   = pd.read_parquet(ORDERS_CLEAN)
st       = pd.read_parquet(STUDENTTEACHERS_CLEAN)
users    = pd.read_parquet(USERS_CLEAN)
groups   = pd.read_parquet(GROUPS_CLEAN)
courses  = pd.read_parquet(COURSES_CLEAN)

print(f"  → Students      : {len(students):,}")
print(f"  → StudentGroups : {len(sg):,}")
print(f"  → Attendance    : {len(att):,}")
print(f"  → Transactions  : {len(txn):,}")
print(f"  → Histories     : {len(sh):,}")
print(f"  → Orders        : {len(orders):,}")

# =============================================================
# COURSE NAME MAPPING
# =============================================================
# Registan offers English at nine different proficiency levels.
# I treat them as a single subject because: (1) the pedagogy and
# dropout dynamics are the same at every level, and (2) splitting
# them would give each level only a few hundred students — far
# too few to compute reliable course-level dropout rates.
# Turkish is excluded for the same reason (n=38 students).
#
# This mapping is identical to the one in src/config.py; it is
# repeated here because build_master.py pre-dates the centralised
# config and I did not want to break the script by removing it.
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

sg['courseName'] = sg['courseId'].map(COURSE_NAME_MAP)
sg_mapped = sg[sg['courseName'].notna()].copy()

print(f"\n  → Valid course enrollments : {len(sg_mapped):,}")
print(f"  → Excluded enrollments     : {len(sg) - len(sg_mapped):,}")
print(f"\n  Course enrollment distribution:")
print(sg_mapped['courseName'].value_counts().to_string())

# =============================================================
# STEP 1: BASE TABLE — ENROLLED STUDENTS ONLY
# =============================================================
# I restrict to students who enrolled in at least one valid course.
# Students who registered but never enrolled have no attendance,
# transaction, or behavioural data — I cannot compute meaningful
# features for them, and including them would inflate the dataset
# with near-zero-feature rows that would distort training. In total
# 9,791 students (40.7%) are excluded on this basis.
print("\nStep 1: Filtering to enrolled students only...")
enrolled_ids = set(sg_mapped['studentId'].unique())
df = students[students['studentId'].isin(enrolled_ids)].copy()
print(f"  → Total registered students : {len(students):,}")
print(f"  → Never enrolled            : {len(students) - len(enrolled_ids):,} (excluded)")
print(f"  → Enrolled students kept    : {len(df):,}")

# =============================================================
# STEP 2: ATTENDANCE FEATURES
# =============================================================
# Attendance behaviour is the strongest predictor of dropout in
# the educational data mining literature (Tinto, 1975; Bean, 1980).
# At Registan, a student who stops attending is almost always about
# to leave — the median gap between last attendance and dropout
# confirmation is under two weeks.
#
# I express all attendance features as ratios rather than raw counts.
# This is essential for fair comparison across students with different
# study durations — a student who attended 70% of 10 lessons is
# behaviourally equivalent to one who attended 70% of 100 lessons,
# but their raw counts (7 vs 70) would mislead a model that does
# not normalise for duration.
print("\nStep 2: Computing attendance features...")

att['yearMonth'] = pd.to_datetime(att['date']).dt.to_period('M')

att_agg = att.groupby('studentId').agg(
    # primary metric: proportion of scheduled lessons attended
    attendanceRate    = ('state', lambda x: (x=='attended').sum() / len(x)),
    # date range used to compute study duration and active months
    firstAttendance   = ('date', 'min'),
    lastAttendance    = ('date', 'max'),
    # raw counts needed for ratio computation below
    totalLessons      = ('state', 'count'),
    countUnreasonable = ('state', lambda x: (x=='unreasonable').sum()),
    countReasonable   = ('state', lambda x: (x=='reasonable').sum()),
    countFrozen       = ('state', lambda x: (x=='frozen').sum()),
    # trial lessons indicate exploratory behaviour before commitment
    trialLessons      = ('isTrial', 'sum'),
    # proportion of lessons where payment was processed
    paymentDoneRate   = ('isPaymentDone', 'mean'),
).reset_index()

# studyDays is used only in the dropout label assignment (Step 3)
# as a filter to exclude very short enrolments. I do not use it as
# a model feature because it is correlated with many other features
# and inflates the apparent predictability of students who have
# simply been enrolled longer.
att_agg['studyDays'] = (
    att_agg['lastAttendance'] - att_agg['firstAttendance']
).dt.days.fillna(0)

# I decompose absence into two types because they carry different
# signals: unreasonable absences (unjustified, teacher-recorded)
# are a much stronger dropout warning than reasonable absences
# (documented illness, official leave).
att_agg['unreasonableAbsenceRate'] = (
    att_agg['countUnreasonable'] / att_agg['totalLessons']
)
att_agg['reasonableAbsenceRate'] = (
    att_agg['countReasonable'] / att_agg['totalLessons']
)
# A high frozen lesson rate may indicate financial difficulty —
# Registan uses lesson freezes as a short-term accommodation for
# students who cannot pay but want to preserve their place.
att_agg['frozenLessonRate'] = (
    att_agg['countFrozen'] / att_agg['totalLessons']
)

# I define an "active month" as one with at least 3 lessons. The
# threshold of 3 is deliberate: 1-2 lessons in a month often
# represent partial months at the start or end of enrolment rather
# than genuine engagement. 3 is the minimum that confirms the student
# was actually attending that month.
active_months_df = att.groupby(
    ['studentId', 'yearMonth']
).size().reset_index(name='lessonsInMonth')

active_months_count = active_months_df[
    active_months_df['lessonsInMonth'] >= 3
].groupby('studentId').size().reset_index(name='totalActiveMonths')

# I clip totalEnrolledMonths at 60 to remove data-entry outliers —
# some records show implausibly long enrolment histories that are
# clearly not real.
att_agg['totalEnrolledMonths'] = (
    (att_agg['lastAttendance'] - att_agg['firstAttendance']).dt.days / 30
).clip(lower=1, upper=60).round(0).astype(int)

att_agg = att_agg.merge(active_months_count, on='studentId', how='left')
att_agg['totalActiveMonths'] = att_agg['totalActiveMonths'].fillna(0).astype(int)

att_agg['activeMonthsRate'] = (
    att_agg['totalActiveMonths'] / att_agg['totalEnrolledMonths']
).clip(upper=1.0).round(4)

# attendanceTrend is the linear slope of monthly attendance rates
# over the student's full history. A negative slope signals declining
# engagement before the student actually stops attending — exactly the
# kind of early warning signal I want the model to pick up.
print("  → Computing attendance trend (linear slope)...")
att_monthly = att.groupby(['studentId', 'yearMonth']).apply(
    lambda x: (x['state'] == 'attended').sum() / len(x)
).reset_index(name='monthlyRate')

def get_trend(group):
    """
    Returns the slope of a linear fit to monthly attendance rates.
    Requires at least 2 data points; returns 0.0 for new students.
    Positive = improving attendance. Negative = declining.
    """
    if len(group) < 2:
        return 0.0
    x = np.arange(len(group))
    y = group['monthlyRate'].values
    return round(np.polyfit(x, y, 1)[0], 4)

trends = att_monthly.groupby('studentId').apply(
    get_trend
).reset_index()
trends.columns = ['studentId', 'attendanceTrend']

att_agg = att_agg.merge(trends, on='studentId', how='left')
att_agg['attendanceTrend'] = att_agg['attendanceTrend'].fillna(0)

# Recent attendance windows matter more than lifetime averages for
# near-term dropout prediction. A student whose attendance has
# collapsed in the last 30 days is at high risk even if their
# lifetime rate looks acceptable.
today     = pd.Timestamp.now(tz='UTC')
cutoff_30 = today - pd.Timedelta(days=30)
cutoff_60 = today - pd.Timedelta(days=60)

att_30 = att[att['date'] >= cutoff_30].groupby('studentId').agg(
    attendanceLast30Days = ('state', lambda x: (x=='attended').sum() / len(x))
).reset_index()

att_60 = att[att['date'] >= cutoff_60].groupby('studentId').agg(
    attendanceLast60Days = ('state', lambda x: (x=='attended').sum() / len(x))
).reset_index()

att_agg = att_agg.merge(att_30, on='studentId', how='left')
att_agg = att_agg.merge(att_60, on='studentId', how='left')

# consecutiveMissedLessons captures abrupt disengagement — a student
# who misses 5+ unjustified lessons in a row has almost certainly
# already decided to leave. I count only 'unreasonable' and 'unchecked'
# states because reasonable absences and frozen lessons have documented
# reasons that are distinct from disengagement.
print("  → Computing maximum consecutive missed lessons...")

def max_consecutive_missed(states):
    """
    Scans the ordered lesson sequence for the longest unbroken
    streak of unreasonable or unchecked absences.
    Reasonable absences and frozen lessons are not counted
    because they have documented justification.
    """
    max_streak = current = 0
    for s in states:
        if s in ['unreasonable', 'unchecked']:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak

consecutive = att.sort_values(['studentId', 'date']).groupby(
    'studentId'
)['state'].apply(max_consecutive_missed).reset_index()
consecutive.columns = ['studentId', 'consecutiveMissedLessons']

att_agg = att_agg.merge(consecutive, on='studentId', how='left')
att_agg['consecutiveMissedLessons'] = (
    att_agg['consecutiveMissedLessons'].fillna(0).astype(int)
)

df = df.merge(att_agg[[
    'studentId', 'attendanceRate', 'studyDays',
    'firstAttendance', 'lastAttendance',
    'trialLessons', 'paymentDoneRate',
    'unreasonableAbsenceRate', 'reasonableAbsenceRate', 'frozenLessonRate',
    'totalActiveMonths', 'totalEnrolledMonths', 'activeMonthsRate',
    'attendanceTrend', 'attendanceLast30Days', 'attendanceLast60Days',
    'consecutiveMissedLessons'
]], on='studentId', how='left')

# 169 students enrolled but never attended — I fill their rates with 0
# rather than NaN because 0 attendance is a meaningful signal
# (the model should learn that never attending is a risk factor).
att_fill_cols = [
    'attendanceRate', 'studyDays', 'trialLessons', 'paymentDoneRate',
    'unreasonableAbsenceRate', 'reasonableAbsenceRate', 'frozenLessonRate',
    'totalActiveMonths', 'totalEnrolledMonths', 'activeMonthsRate',
    'attendanceTrend', 'attendanceLast30Days', 'attendanceLast60Days',
    'consecutiveMissedLessons'
]
for col in att_fill_cols:
    df[col] = df[col].fillna(0)

print(f"  → Attendance features computed for {att_agg['studentId'].nunique():,} students")
print(f"  → Students with no attendance records : {df['attendanceRate'].eq(0).sum():,}")
print(f"  → Average attendance rate             : {df['attendanceRate'].mean():.2%}")
print(f"  → Average active months rate          : {df['activeMonthsRate'].mean():.2%}")
print(f"  → Average consecutive missed          : {df['consecutiveMissedLessons'].mean():.1f}")

# =============================================================
# STEP 3: DROPOUT LABEL ASSIGNMENT
# =============================================================
# Defining the target variable is the most methodologically critical
# step in this study. A naive approach — using the system 'state'
# column directly — would produce unreliable labels. Registan staff
# frequently archive students without updating their status to
# 'graduated', even when the student has genuinely completed their
# course. I counted only 933 graduation records in the system for
# 24,044 students — a severe undercount that I verified by cross-
# referencing with payment records and attendance histories.
#
# I define dropout behaviourally using three independent conditions.
# A student is a COMPLETER (dropout = 0) if ANY of the following:
#
#   Condition 1: state == 'graduated'
#   The system explicitly records the student as graduated.
#   Reliable when present but rare (n=931) due to inconsistent
#   data entry by Registan admin staff.
#
#   Condition 2: graduatedAt is not null in studentgroups
#   The student was formally graduated from at least one group,
#   even if their overall state was later set to 'archive'.
#   This catches completers correctly recorded at group level
#   but missed at student level (n=1,957).
#
#   Condition 3: attendanceRate >= 70% AND studyDays >= 30
#   The student attended at least 70% of lessons over at least
#   30 days (one full payment cycle). I chose 70% after reviewing
#   Registan's payment model: a student paying monthly and
#   attending 7 of 10 lessons is demonstrating genuine commitment
#   that is inconsistent with dropout behaviour. Students meeting
#   this condition but marked 'archive' are assumed to be
#   completers mislabelled by staff (n=5,124).
#
# A student is a TRUE DROPOUT (dropout = 1) if their state is
# 'archive' or 'pro-archive' AND none of the three conditions above.
#
# Students with state 'active' or 'new' get dropout = -1 (score
# only). Their outcome is unknown — they cannot be used for
# training but are scored by predict.py to generate risk alerts.
print("\nStep 3: Assigning dropout labels...")

grads_in_sg = set(sg[sg['graduatedAt'].notna()]['studentId'].unique())
df['graduatedInGroup'] = df['studentId'].isin(grads_in_sg)

cond_state_grad    = df['state'] == 'graduated'
cond_group_grad    = df['graduatedInGroup']
cond_att_threshold = (df['attendanceRate'] >= 0.70) & (df['studyDays'] >= 30)

print(f"  → Qualified by Condition 1 (state=graduated)      : {cond_state_grad.sum():,}")
print(f"  → Qualified by Condition 2 (graduatedInGroup)     : {cond_group_grad.sum():,}")
print(f"  → Qualified by Condition 3 (att≥70% + days≥30)   : {cond_att_threshold.sum():,}")
print(f"  → Overlap between C1 and C2 (counted once)        : {(cond_state_grad & cond_group_grad).sum():,}")

def assign_label(row):
    # active/new: still studying — outcome unknown
    if row['state'] in ['active', 'new']:
        return -1
    # condition 1: explicitly graduated
    if row['state'] == 'graduated':
        return 0
    # condition 2: graduated from at least one group
    if row['graduatedInGroup']:
        return 0
    # condition 3: behavioural completion signal
    if row['attendanceRate'] >= 0.70 and row['studyDays'] >= 30:
        return 0
    # true dropout: archived with no completion evidence
    return 1

df['dropout'] = df.apply(assign_label, axis=1)

d1 = (df['dropout'] == 1).sum()
d0 = (df['dropout'] == 0).sum()
dm = (df['dropout'] == -1).sum()
total_labeled = d1 + d0

print(f"\n  ── Label Distribution ──────────────────────────────")
print(f"  dropout = 1  (true dropout) : {d1:,}  ({d1/total_labeled:.1%})")
print(f"  dropout = 0  (completer)    : {d0:,}   ({d0/total_labeled:.1%})")
print(f"  dropout = -1 (score only)   : {dm:,}")
print(f"  Total labelled training set : {total_labeled:,}")

# =============================================================
# STEP 4: ENROLLMENT FEATURES FROM STUDENTGROUPS
# =============================================================
# The studentgroups table records every group a student has ever
# enrolled in. Students at Registan frequently transfer between
# groups within the same course (e.g. changing time slot or teacher),
# so I need to distinguish group-level from course-level counts.
#
# I aggregate graduation and freeze status at course level, not
# group level, because a student who progressed through three English
# groups (Beginner → Elementary → Intermediate) completed one course,
# not three. Counting group graduations would inflate the metric for
# diligent students and make it uninformative.
#
# The 'price' column in studentgroups represents the per-lesson charge
# rate in UZS, not the total amount paid. I compute total payments
# from the transactions file in Step 8B.
print("\nStep 4: Computing enrollment features from studentgroups...")

sg_counts = sg_mapped.groupby('studentId').agg(
    totalGroupsJoined = ('studentGroupId', 'count'),
).reset_index()

sg_course_states = sg_mapped.groupby(['studentId', 'courseName']).agg(
    courseGraduated = ('state', lambda x: 1 if 'graduated' in x.values else 0),
    courseFrozen    = ('state', lambda x: 1 if 'frozen' in x.values else 0),
).reset_index()

course_level = sg_course_states.groupby('studentId').agg(
    totalCoursesGraduated = ('courseGraduated', 'sum'),
    totalCoursesFrozen    = ('courseFrozen', 'sum'),
).reset_index()

att_per_group = att.groupby(['studentId', 'groupId']).agg(
    groupAttRate = ('state', lambda x: (x=='attended').sum() / len(x))
).reset_index()

avg_group_att = att_per_group.groupby('studentId').agg(
    avgGroupAttendanceRate = ('groupAttRate', 'mean')
).reset_index()

sg_price = sg_mapped.groupby('studentId')['price'].agg(
    avgLessonPrice = 'mean',
    minLessonPrice = 'min',
    maxLessonPrice = 'max',
).reset_index()

# I define a discount as confirmed when the student's actual lesson
# price is strictly below the standard course price in the groups
# table. The groupPriceChanged event in studenthistories only tells me
# a change occurred — not the direction — so I cannot use it alone to
# detect discounts.
sg_with_std = sg_mapped.merge(
    groups[['groupId', 'coursePrice']], on='groupId', how='left'
)
sg_with_std['isDiscounted'] = (
    (sg_with_std['price'] < sg_with_std['coursePrice']) &
    (sg_with_std['coursePrice'] > 0)
)
sg_discount = sg_with_std.groupby('studentId').agg(
    hasDiscount = ('isDiscounted', 'any'),
).reset_index()
sg_discount['hasDiscount'] = sg_discount['hasDiscount'].astype(int)

# groupSwitchCount counts groupTransferred events. A high switch
# count may reflect scheduling instability, dissatisfaction with
# the teacher, or financial negotiation — all of which are associated
# with higher dropout risk in the Registan context.
transfers = sh[sh['type'] == 'groupTransferred'].groupby(
    'studentId'
).size().reset_index(name='groupSwitchCount')

sg_features = sg_counts \
    .merge(course_level,  on='studentId', how='left') \
    .merge(avg_group_att, on='studentId', how='left') \
    .merge(sg_price,      on='studentId', how='left') \
    .merge(sg_discount,   on='studentId', how='left') \
    .merge(transfers,     on='studentId', how='left')

enroll_fill = [
    'totalGroupsJoined', 'totalCoursesGraduated', 'totalCoursesFrozen',
    'avgGroupAttendanceRate', 'avgLessonPrice', 'minLessonPrice',
    'maxLessonPrice', 'hasDiscount', 'groupSwitchCount'
]
for col in enroll_fill:
    sg_features[col] = sg_features[col].fillna(0)

df = df.merge(sg_features, on='studentId', how='left')

print(f"  → Average groups joined per student  : {df['totalGroupsJoined'].mean():.1f}")
print(f"  → Average courses graduated          : {df['totalCoursesGraduated'].mean():.1f}")
print(f"  → Students who received a discount   : {df['hasDiscount'].sum():,}")
print(f"  → Average group attendance rate      : {df['avgGroupAttendanceRate'].mean():.2%}")
print(f"  → Average group switch count         : {df['groupSwitchCount'].mean():.1f}")

# =============================================================
# STEP 5: COURSE CONTEXT FEATURES
# =============================================================
# These features describe the learning environment at course level.
# My supervisor asked me to include environmental risk factors
# separately from individual student behaviour — a student's dropout
# risk is partly determined by which course they are enrolled in,
# not just what they personally do. Some courses at Registan have
# structurally higher dropout rates (Korean, for instance, is harder
# and has fewer students who reach advanced levels).
#
# courseAttendanceRate: average attendance rate of ALL students
# who have ever studied the course. A low course-wide rate may
# indicate poor content design, scheduling issues, or teaching quality
# problems that affect everyone in that course.
#
# courseDropoutRate: proportion of labelled students in this course
# who are classified as dropouts. This gives the model information
# about the base risk associated with each subject.
print("\nStep 5: Computing course-level context features...")

distinct_courses = sg_mapped.groupby(
    'studentId'
)['courseName'].nunique().reset_index()
distinct_courses.columns = ['studentId', 'totalCoursesStudied']

att_with_course = att.merge(
    sg_mapped[['studentId', 'groupId', 'courseName']].drop_duplicates(),
    on=['studentId', 'groupId'], how='left'
)

course_level_att = att_with_course.groupby('courseName').agg(
    courseAttendanceRate = ('state', lambda x: (x=='attended').sum() / len(x))
).reset_index()

course_dropout_rate = df[df['dropout'].isin([0, 1])].merge(
    sg_mapped[['studentId', 'courseName']].drop_duplicates(),
    on='studentId', how='left'
).groupby('courseName').agg(
    courseDropoutRate = ('dropout', 'mean')
).reset_index()

# I define 'last' as the most recent joinedAt date — this represents
# the student's current or most recent learning context, which is
# the most relevant for predicting near-term dropout.
sg_sorted = sg_mapped.sort_values('joinedAt', ascending=False)
last_per_course = sg_sorted.groupby(
    ['studentId', 'courseName']
).first().reset_index()[[
    'studentId', 'courseName', 'groupId', 'teacherId', 'moderatorId'
]].copy()
last_per_course.columns = [
    'studentId', 'courseName',
    'lastGroupId', 'lastTeacherId', 'lastModeratorId'
]

group_att_rate = att_per_group.groupby('groupId').agg(
    lastGroupAttendanceRate = ('groupAttRate', 'mean')
).reset_index()

group_dropout = df[df['dropout'].isin([0,1])].merge(
    sg_mapped[['studentId','groupId']].drop_duplicates(),
    on='studentId', how='left'
).groupby('groupId').agg(
    lastGroupDropoutRate = ('dropout', 'mean')
).reset_index()

last_per_course = last_per_course \
    .merge(group_att_rate, left_on='lastGroupId', right_on='groupId', how='left') \
    .merge(group_dropout,  left_on='lastGroupId', right_on='groupId', how='left')

print(f"  → Course attendance rates computed for {len(course_level_att)} courses")
print(f"  → Course dropout rates computed for   {len(course_dropout_rate)} courses")

# =============================================================
# STEP 6: TEACHER CONTEXT FEATURES
# =============================================================
# Teacher quality is a well-established predictor of student
# retention (Pascarella & Terenzini, 2005). I operationalise
# teacher quality through the attendance behaviour of all students
# a teacher has taught — if their students consistently attend at
# high rates, that reflects engagement with the teacher's instruction.
#
# I use a two-stage averaging: group → monthly → lifetime. This
# prevents teachers who manage many large groups from dominating
# the metric simply because they have more students. Each teacher-
# month gets equal weight regardless of how many groups they taught.
#
# lastTeacherLastMonthAttRate captures recent performance, which
# is more relevant than a lifetime average for predicting current
# student dropout. A teacher whose attendance has declined recently
# presents a higher environmental risk for their current students.
#
# lastTeacherAttendanceTrend is the linear slope of the teacher's
# monthly average attendance over time. A negative trend may signal
# declining teaching quality or engagement issues in those classes.
print("\nStep 6: Computing teacher context features...")

att_with_teacher = att[att['teacherId'].notna()].copy()

# stage 1: monthly attendance rate per teacher per group
teacher_monthly = att_with_teacher.groupby(
    ['teacherId', 'groupId', 'yearMonth']
).agg(
    monthGroupRate = ('state', lambda x: (x=='attended').sum() / len(x))
).reset_index()

# stage 2: average across groups for each teacher per month
teacher_monthly_avg = teacher_monthly.groupby(
    ['teacherId', 'yearMonth']
).agg(
    monthAvgRate = ('monthGroupRate', 'mean')
).reset_index()

teacher_avg_att = teacher_monthly_avg.groupby('teacherId').agg(
    lastTeacherAvgAttendanceRate = ('monthAvgRate', 'mean')
).reset_index()

last_month = teacher_monthly_avg['yearMonth'].max()
teacher_last_month = teacher_monthly_avg[
    teacher_monthly_avg['yearMonth'] == last_month
][['teacherId', 'monthAvgRate']].rename(
    columns={'monthAvgRate': 'lastTeacherLastMonthAttRate'}
)

def teacher_trend(group):
    """
    Computes linear slope of a teacher's monthly average
    student attendance rates over time.
    Returns 0.0 if fewer than 2 months of data available.
    """
    if len(group) < 2:
        return 0.0
    x = np.arange(len(group))
    y = group['monthAvgRate'].values
    return round(np.polyfit(x, y, 1)[0], 4)

teacher_trends = teacher_monthly_avg.groupby('teacherId').apply(
    teacher_trend
).reset_index()
teacher_trends.columns = ['teacherId', 'lastTeacherAttendanceTrend']

teacher_students = sg_mapped[['studentId', 'teacherId']].drop_duplicates()
teacher_dropout = df[df['dropout'].isin([0,1])].merge(
    teacher_students, on='studentId', how='left'
).groupby('teacherId').agg(
    lastTeacherDropoutRate = ('dropout', 'mean')
).reset_index()

teacher_features = teacher_avg_att \
    .merge(teacher_last_month, on='teacherId', how='left') \
    .merge(teacher_trends,     on='teacherId', how='left') \
    .merge(teacher_dropout,    on='teacherId', how='left')

last_per_course = last_per_course.merge(
    teacher_features,
    left_on='lastTeacherId', right_on='teacherId', how='left'
)

print(f"  → Teacher features computed for {len(teacher_features)} teachers")

# =============================================================
# STEP 7: MODERATOR CONTEXT FEATURES
# =============================================================
# Moderators at Registan are the administrative staff responsible
# for student relationship management — they follow up on absences,
# process payments, and handle enrolment queries. My supervisor
# suggested including moderator-level context because different
# moderators may vary in how proactively they reach out to at-risk
# students. I capture this through the dropout rate among all
# students assigned to each moderator — a high rate may reflect
# either a high-risk student population or less effective follow-up.
print("\nStep 7: Computing moderator context features...")

# I take moderatorId from the students table (not studentgroups)
# to avoid the naming conflicts that arise when merging both tables
# and finding two 'moderatorId' columns.
mod_lookup = students[students['moderatorId'].notna()][
    ['studentId', 'moderatorId']
].drop_duplicates().rename(columns={'moderatorId': 'modId'})

moderator_dropout = df[df['dropout'].isin([0,1])].merge(
    mod_lookup, on='studentId', how='left'
).groupby('modId').agg(
    moderatorDropoutRate = ('dropout', 'mean')
).reset_index().rename(columns={'modId': 'lastModeratorId'})

print(f"  → Moderator dropout rates computed for {len(moderator_dropout)} moderators")

# =============================================================
# STEP 8: SHIFT CONTEXT FEATURES
# =============================================================
# Registan operates three shifts: morning, afternoon, evening.
# I included shift as a feature because evening students at Registan
# are often working adults — they face greater time pressure and
# tend to drop out more frequently than morning students (who are
# typically school-age children studying alongside their parents).
# This is a structural risk factor independent of the individual's
# attendance behaviour.
#
# I define preferredShift as the most frequently attended shift
# across a student's enrolments. For students whose shift data is
# missing from the filtered sg_mapped (because they only studied
# excluded courses like Turkish), I fall back to the full
# studentgroups table.
print("\nStep 8: Computing shift context features...")

sg_with_shift = sg_mapped.merge(
    groups[['groupId', 'shiftPeriod']], on='groupId', how='left'
)

def get_preferred_shift(x):
    """Returns the most common non-null shift for a student."""
    m = x.dropna().mode()
    return m.iloc[0] if len(m) > 0 else None

preferred_shift = sg_with_shift.groupby('studentId')['shiftPeriod'].agg(
    get_preferred_shift
).reset_index()
preferred_shift.columns = ['studentId', 'preferredShift']

# fall back to full studentgroups table for students whose shift
# data is missing from sg_mapped (e.g. Turkish-only students)
last_group_all = sg.sort_values(
    'joinedAt', ascending=False
).groupby('studentId').first().reset_index()[['studentId', 'groupId']]

group_shift_lookup = groups[['groupId', 'shiftPeriod']].dropna()

last_group_all = last_group_all.merge(
    group_shift_lookup, on='groupId', how='left'
).rename(columns={'shiftPeriod': 'shiftFill'})

preferred_shift = preferred_shift.merge(
    last_group_all[['studentId', 'shiftFill']],
    on='studentId', how='left'
)
preferred_shift['preferredShift'] = preferred_shift['preferredShift'].fillna(
    preferred_shift['shiftFill']
)
preferred_shift = preferred_shift.drop(columns=['shiftFill'])

# 475 students have no group data in any source. I fill with
# 'afternoon' (the modal shift) as a neutral imputation rather than
# introducing an 'unknown' category that the model would need to
# learn from only a tiny fraction of the training data.
preferred_shift['preferredShift'] = preferred_shift['preferredShift'].fillna('afternoon')

remaining_nulls = preferred_shift['preferredShift'].isna().sum()
print(f"  → Shift nulls after fallback fix: {remaining_nulls}")

shift_dropout = df[df['dropout'].isin([0,1])].merge(
    preferred_shift, on='studentId', how='left'
).groupby('preferredShift').agg(
    shiftDropoutRate = ('dropout', 'mean')
).reset_index()

df = df \
    .merge(preferred_shift, on='studentId', how='left') \
    .merge(shift_dropout,   on='preferredShift', how='left')

print(f"  → Shift distribution:")
print(df['preferredShift'].value_counts().to_string())

# =============================================================
# STEP 8B: TRANSACTION FEATURES
# =============================================================
# Financial behaviour is a critical dropout predictor in fee-paying
# educational settings. At Registan, students maintain a personal
# balance from which per-lesson charges are deducted daily. They
# top up periodically (payIn / income transactions). When the balance
# goes negative, the student incurs debt but may still attend.
# Persistent debt without repayment is one of the strongest leading
# indicators of impending dropout — I observed this pattern
# repeatedly when reviewing individual student records.
#
# I compute balance, debt, and payment behaviour separately because
# they represent distinct financial risk mechanisms:
#   - Balance features: how much financial headroom the student has
#   - Debt features: how often and how deeply they go into deficit
#   - Payment features: the regularity and size of top-ups
print("\nStep 8B: Computing transaction features...")

current_balance = txn.sort_values('createdAt').groupby(
    'studentId'
)['afterAmount'].last().reset_index()
current_balance.columns = ['studentId', 'currentBalance']

balance_stats = txn.groupby('studentId').agg(
    minBalance = ('afterAmount', 'min'),
    maxBalance = ('afterAmount', 'max'),
).reset_index()

txn['txnMonth'] = pd.to_datetime(txn['createdAt']).dt.to_period('M')
monthly_end_balance = txn.sort_values('createdAt').groupby(
    ['studentId', 'txnMonth']
)['afterAmount'].last().reset_index()
monthly_end_balance.columns = ['studentId', 'txnMonth', 'monthEndBalance']

avg_monthly_balance = monthly_end_balance.groupby('studentId').agg(
    avgMonthlyEndBalance = ('monthEndBalance', 'mean')
).reset_index()

monthly_end_balance['isNegativeMonth'] = monthly_end_balance['monthEndBalance'] < 0
debt_rate = monthly_end_balance.groupby('studentId').agg(
    debtRate         = ('isNegativeMonth', 'mean'),
    negativeEpisodes = ('isNegativeMonth', 'sum'),
).reset_index()

# I use payIn and income as the payment types — these are the two
# transaction types that represent a student topping up their balance.
payments = txn[txn['type'].isin(['payIn', 'income'])].copy()

payment_stats = payments.groupby('studentId').agg(
    totalPaid        = ('amount', 'sum'),
    paymentCount     = ('transactionId', 'count'),
    avgPaymentAmount = ('amount', 'mean'),
).reset_index()

payment_months = payments.groupby(
    'studentId'
)['txnMonth'].nunique().reset_index()
payment_months.columns = ['studentId', 'paymentMonths']

# paymentRegularity is the standard deviation of days between payments.
# A student who tops up at irregular intervals (high std) is more likely
# to eventually miss a payment entirely and stop attending.
payments_sorted = payments.sort_values(['studentId', 'createdAt'])
payments_sorted['prevDate'] = payments_sorted.groupby(
    'studentId'
)['createdAt'].shift(1)
payments_sorted['daysBetween'] = (
    payments_sorted['createdAt'] - payments_sorted['prevDate']
).dt.days

payment_regularity = payments_sorted.groupby('studentId').agg(
    paymentRegularity = ('daysBetween', 'std')
).reset_index()
payment_regularity['paymentRegularity'] = (
    payment_regularity['paymentRegularity'].fillna(0)
)

unpaid = txn[txn['type'] == 'unpaidAttendance']
debt_stats = unpaid.groupby('studentId').agg(
    totalDebtAmount = ('amount', 'sum'),
    unpaidLessons   = ('transactionId', 'count'),
).reset_index()

total_att_txn = txn[txn['type'].isin([
    'attendance', 'paidAttendance', 'unpaidAttendance'
])].groupby('studentId').size().reset_index(name='totalAttTxn')

debt_stats = debt_stats.merge(total_att_txn, on='studentId', how='left')
debt_stats['unpaidRate'] = (
    debt_stats['unpaidLessons'] / debt_stats['totalAttTxn']
).round(4)

returns = txn[txn['type'] == 'return'].groupby('studentId').agg(
    totalReturned = ('amount', 'sum')
).reset_index()

txn_features = current_balance \
    .merge(balance_stats,       on='studentId', how='outer') \
    .merge(avg_monthly_balance, on='studentId', how='outer') \
    .merge(debt_rate,           on='studentId', how='outer') \
    .merge(payment_stats,       on='studentId', how='outer') \
    .merge(payment_months,      on='studentId', how='outer') \
    .merge(payment_regularity,  on='studentId', how='outer') \
    .merge(debt_stats[['studentId', 'totalDebtAmount', 'unpaidRate']],
           on='studentId', how='outer') \
    .merge(returns,             on='studentId', how='outer')

txn_fill_cols = [
    'currentBalance', 'minBalance', 'maxBalance',
    'avgMonthlyEndBalance', 'debtRate', 'negativeEpisodes',
    'totalPaid', 'paymentCount', 'avgPaymentAmount',
    'paymentMonths', 'paymentRegularity',
    'totalDebtAmount', 'unpaidRate', 'totalReturned'
]
for col in txn_fill_cols:
    txn_features[col] = txn_features[col].fillna(0)

print(f"  → Transaction features for {len(txn_features):,} students")
print(f"  → Average current balance    : {txn_features['currentBalance'].mean():,.0f} UZS")
print(f"  → Average debt rate          : {txn_features['debtRate'].mean():.2%}")
print(f"  → Average payment months     : {txn_features['paymentMonths'].mean():.1f}")
print(f"  → Average unpaid lesson rate : {txn_features['unpaidRate'].mean():.2%}")

# =============================================================
# STEP 8C: STUDENT HISTORY EVENT FEATURES (COURSE LEVEL)
# =============================================================
# The studenthistories table is an event log recording every state
# change and administrative action on a student's account. I
# aggregate event counts at COURSE level (not overall student level)
# because a student's behaviour can differ substantially across
# courses — someone frequently removed from groups in Russian but
# never in English presents a different risk profile, and I want
# the model to capture that distinction.
#
# Note on the 'fronzen' typo: this is the spelling used in the raw
# MongoDB data. I preserved it rather than correcting it so that
# the code matches the source schema exactly — correcting it here
# but not in the source would create a silent mismatch if the data
# is ever re-exported.
#
# freezeReturnRate: the proportion of freeze events followed by an
# unfreeze in the same course. A rate of 0 means the student never
# returned after any freeze — a strong dropout signal. I clip at 1.0
# to handle rare data-entry errors where unfreeze > freeze counts.
#
# daysSinceLastRemoval / daysSinceLastFreeze: I use 9999 as a
# sentinel for "this event never occurred". This is deliberate —
# it separates "very old event" (which could be, say, 800 days)
# from "event never happened" and is handled in encode_features.py
# by converting 9999 to a binary flag + resetting to 0.
print("\nStep 8C: Computing student history event features (course level)...")

sh_with_course = sh.merge(
    sg_mapped[['studentId', 'groupId', 'courseName']].drop_duplicates(),
    on=['studentId', 'groupId'], how='left'
)

event_types_needed = [
    'removedFromGroup', 'addedToGroup', 'fronzen',
    'unFrozen', 'groupPriceChanged', 'graduatedFromGroup'
]

sh_course_events = sh_with_course[
    sh_with_course['type'].isin(event_types_needed) &
    sh_with_course['courseName'].notna()
].groupby(['studentId', 'courseName', 'type']).size().reset_index(name='count')

sh_pivot = sh_course_events.pivot_table(
    index=['studentId', 'courseName'],
    columns='type',
    values='count',
    fill_value=0
).reset_index()

sh_pivot.columns.name = None
rename_map = {
    'removedFromGroup'   : 'n_removedFromGroup_course',
    'addedToGroup'       : 'n_addedToGroup_course',
    'fronzen'            : 'n_fronzen_course',
    'unFrozen'           : 'n_unFrozen_course',
    'groupPriceChanged'  : 'n_groupPriceChanged_course',
    'graduatedFromGroup' : 'n_graduatedFromGroup_course',
}
sh_pivot = sh_pivot.rename(columns=rename_map)

for col in rename_map.values():
    if col not in sh_pivot.columns:
        sh_pivot[col] = 0

sh_pivot['freezeReturnRate_course'] = (
    sh_pivot['n_unFrozen_course'] /
    sh_pivot['n_fronzen_course'].replace(0, np.nan)
).fillna(0).clip(upper=1.0).round(4)

today_ts = pd.Timestamp.now(tz='UTC')

last_removal = sh_with_course[
    (sh_with_course['type'] == 'removedFromGroup') &
    (sh_with_course['courseName'].notna())
].groupby(['studentId', 'courseName'])['createdAt'].max().reset_index()
last_removal.columns = ['studentId', 'courseName', 'lastRemovalDate']
last_removal['daysSinceLastRemoval_course'] = (
    today_ts - last_removal['lastRemovalDate']
).dt.days

last_freeze = sh_with_course[
    (sh_with_course['type'] == 'fronzen') &
    (sh_with_course['courseName'].notna())
].groupby(['studentId', 'courseName'])['createdAt'].max().reset_index()
last_freeze.columns = ['studentId', 'courseName', 'lastFreezeDate']
last_freeze['daysSinceLastFreeze_course'] = (
    today_ts - last_freeze['lastFreezeDate']
).dt.days

# toArchiveState is student-level — archive events are not
# course-specific in Registan's data model.
archive_events = sh[
    sh['type'] == 'toArchiveState'
].groupby('studentId').size().reset_index(name='n_toArchiveState')

sh_features = sh_pivot \
    .merge(last_removal[['studentId', 'courseName',
                          'daysSinceLastRemoval_course']],
           on=['studentId', 'courseName'], how='left') \
    .merge(last_freeze[['studentId', 'courseName',
                         'daysSinceLastFreeze_course']],
           on=['studentId', 'courseName'], how='left')

# 9999 sentinel: event never occurred for this student-course
sh_features['daysSinceLastRemoval_course'] = (
    sh_features['daysSinceLastRemoval_course'].fillna(9999).astype(int)
)
sh_features['daysSinceLastFreeze_course'] = (
    sh_features['daysSinceLastFreeze_course'].fillna(9999).astype(int)
)

# pre-enrolment cancellation behaviour from the orders table.
# Students who cancelled multiple orders before enrolling may
# exhibit lower commitment — I include this as a student-level
# risk signal even though it pre-dates the actual enrolment.
orders_agg = orders.groupby('studentId').agg(
    totalOrders     = ('orderId', 'count'),
    cancelledOrders = ('state', lambda x: (x=='cancelled').sum()),
    completedOrders = ('state', lambda x: (x=='completed').sum()),
).reset_index()

orders_agg['orderCancellationRate'] = (
    orders_agg['cancelledOrders'] / orders_agg['totalOrders']
).round(4)

orders_agg['hadCancelledOrderBefore'] = (
    orders_agg['cancelledOrders'] > 0
).astype(int)

print(f"  → History features for {len(sh_features):,} student-course pairs")
print(f"  → Archive state events for {len(archive_events):,} students")
print(f"  → Order features for {len(orders_agg):,} students")
print(f"  → Average removals per course    : {sh_features['n_removedFromGroup_course'].mean():.2f}")
print(f"  → Average freezes per course     : {sh_features['n_fronzen_course'].mean():.2f}")
print(f"  → Average freeze return rate     : {sh_features['freezeReturnRate_course'].mean():.2%}")

# =============================================================
# STEP 9: COMPOSITE KEY TABLE (studentId + courseName)
# =============================================================
# The final table uses a composite key of studentId + courseName.
# A student studying both English and Math appears as two rows.
# I chose this design after discussing with my supervisor: a single
# student-level score would not tell a moderator WHICH course to
# address. With course-level rows, she can see "Marjona has 73%
# dropout risk in English but only 30% in Math" and target her
# call accordingly.
#
# All student-level features (demographics, attendance, transactions)
# are duplicated across a student's course rows. Course-specific
# features (last group, last teacher, course dropout rate) vary per row.
print("\nStep 9: Assembling composite key table (studentId + courseName)...")

student_courses = sg_mapped[['studentId', 'courseName']].drop_duplicates()
composite = student_courses.merge(df, on='studentId', how='left')

composite = composite \
    .merge(course_level_att,    on='courseName', how='left') \
    .merge(course_dropout_rate, on='courseName', how='left') \
    .merge(distinct_courses,    on='studentId',  how='left') \
    .merge(last_per_course,     on=['studentId', 'courseName'], how='left') \
    .merge(moderator_dropout,   on='lastModeratorId', how='left') \
    .merge(txn_features,        on='studentId', how='left') \
    .merge(sh_features,         on=['studentId', 'courseName'], how='left') \
    .merge(archive_events,      on='studentId', how='left') \
    .merge(orders_agg[['studentId', 'cancelledOrders',
                        'orderCancellationRate', 'hadCancelledOrderBefore']],
           on='studentId', how='left')

context_fill = [
    'courseAttendanceRate', 'courseDropoutRate', 'totalCoursesStudied',
    'lastGroupAttendanceRate', 'lastGroupDropoutRate',
    'lastTeacherAvgAttendanceRate', 'lastTeacherLastMonthAttRate',
    'lastTeacherAttendanceTrend', 'lastTeacherDropoutRate',
    'moderatorDropoutRate', 'shiftDropoutRate',
    'currentBalance', 'minBalance', 'maxBalance',
    'avgMonthlyEndBalance', 'debtRate', 'negativeEpisodes',
    'totalPaid', 'totalReturned', 'paymentCount',
    'avgPaymentAmount', 'paymentMonths', 'paymentRegularity',
    'totalDebtAmount', 'unpaidRate',
    'n_removedFromGroup_course', 'n_addedToGroup_course',
    'n_fronzen_course', 'n_unFrozen_course',
    'freezeReturnRate_course', 'n_groupPriceChanged_course',
    'n_graduatedFromGroup_course', 'n_toArchiveState',
    'cancelledOrders', 'orderCancellationRate', 'hadCancelledOrderBefore',
]
for col in context_fill:
    if col in composite.columns:
        composite[col] = composite[col].fillna(0)

composite['totalCoursesStudied'] = composite['totalCoursesStudied'].fillna(1)
# restore the 9999 sentinel for students with no removal/freeze history
composite['daysSinceLastRemoval_course'] = (
    composite['daysSinceLastRemoval_course'].fillna(9999)
)
composite['daysSinceLastFreeze_course'] = (
    composite['daysSinceLastFreeze_course'].fillna(9999)
)

print(f"  → Total rows (student-course pairs) : {len(composite):,}")
print(f"  → Unique students                   : {composite['studentId'].nunique():,}")
print(f"  → Unique courses                    : {composite['courseName'].nunique():,}")
print(f"  → Average courses per student       : {len(composite)/composite['studentId'].nunique():.1f}")

# =============================================================
# STEP 10: SELECT AND ORDER FINAL COLUMNS
# =============================================================
# I retain IDs (lastGroupId, lastTeacherId, lastModeratorId) in
# master.parquet for the dashboard, but exclude them from the ML
# feature set in build_ml_ready.py — they are identifiers, not
# behavioural signals, and keeping them as features would let the
# model memorise individual teachers rather than learn patterns.
print("\nStep 10: Selecting and ordering final columns...")

KEEP_COLS = [
    # composite keys
    'studentId', 'courseName',
    # demographics
    'gender', 'language', 'joinMonth', 'joinYear',
    'joinSeason', 'academicYear', 'isReferred',
    # attendance ratios — student level
    'attendanceRate', 'unreasonableAbsenceRate', 'reasonableAbsenceRate',
    'frozenLessonRate', 'activeMonthsRate', 'attendanceTrend',
    'attendanceLast30Days', 'attendanceLast60Days',
    'consecutiveMissedLessons', 'paymentDoneRate', 'trialLessons',
    # enrollment — student level
    'totalGroupsJoined', 'totalCoursesGraduated', 'totalCoursesFrozen',
    'avgGroupAttendanceRate', 'avgLessonPrice', 'minLessonPrice',
    'maxLessonPrice', 'hasDiscount', 'groupSwitchCount',
    # transaction features — student level
    'currentBalance', 'minBalance', 'maxBalance',
    'avgMonthlyEndBalance', 'debtRate', 'negativeEpisodes',
    'totalPaid', 'totalReturned', 'paymentCount',
    'avgPaymentAmount', 'paymentMonths', 'paymentRegularity',
    'totalDebtAmount', 'unpaidRate',
    # order features — student level
    'cancelledOrders', 'orderCancellationRate', 'hadCancelledOrderBefore',
    # course context — course level
    'totalCoursesStudied', 'courseAttendanceRate', 'courseDropoutRate',
    # history events — course level
    'n_removedFromGroup_course', 'n_addedToGroup_course',
    'n_fronzen_course', 'n_unFrozen_course',
    'freezeReturnRate_course', 'n_groupPriceChanged_course',
    'n_graduatedFromGroup_course', 'n_toArchiveState',
    'daysSinceLastRemoval_course', 'daysSinceLastFreeze_course',
    # last group context — course level
    'lastGroupId', 'lastGroupAttendanceRate', 'lastGroupDropoutRate',
    # last teacher context — course level
    'lastTeacherId', 'lastTeacherAvgAttendanceRate',
    'lastTeacherLastMonthAttRate', 'lastTeacherAttendanceTrend',
    'lastTeacherDropoutRate',
    # moderator context
    'lastModeratorId', 'moderatorDropoutRate',
    # shift context
    'preferredShift', 'shiftDropoutRate',
    # target variable
    'dropout',
]

KEEP_COLS = [c for c in KEEP_COLS if c in composite.columns]
master = composite[KEEP_COLS].copy()

print(f"  → Final columns : {len(master.columns)}")
print(f"  → Final rows    : {len(master):,}")
print(f"\n  Label distribution:")
print(master['dropout'].value_counts().to_string())

# =============================================================
# ADD TEACHER AND MODERATOR NAMES
# =============================================================
# I join real names from the users file for dashboard display.
# The dashboard needs human-readable names (not MongoDB ObjectIds)
# so a moderator can see "Teacher: Sardor Toshmatov" rather than
# a 24-character hex string.
users_lookup = users[['userId', 'fullName']].drop_duplicates()
master = master.merge(
    users_lookup.rename(columns={'userId': 'lastTeacherId', 'fullName': 'lastTeacherName'}),
    on='lastTeacherId', how='left'
)
master = master.merge(
    users_lookup.rename(columns={'userId': 'lastModeratorId', 'fullName': 'lastModeratorName'}),
    on='lastModeratorId', how='left'
)
master['lastTeacherName']   = master['lastTeacherName'].fillna('Unknown')
master['lastModeratorName'] = master['lastModeratorName'].fillna('Unknown')
print(f"  → Teacher names matched   : {(master['lastTeacherName'] != 'Unknown').sum():,}")
print(f"  → Moderator names matched : {(master['lastModeratorName'] != 'Unknown').sum():,}")

# =============================================================
# SAVE MASTER TABLE
# =============================================================
PROCESSED.mkdir(parents=True, exist_ok=True)
master.to_parquet(PROCESSED / "master.parquet", index=False)
print(f"\n✅ Master table saved → {PROCESSED}/master.parquet")
print(f"   Rows    : {len(master):,}")
print(f"   Columns : {len(master.columns)}")
print(f"   Columns : {list(master.columns)}")
