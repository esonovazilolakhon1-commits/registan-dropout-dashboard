# =============================================================
# build_snapshot_features.py
# =============================================================
# As-Of-Month Feature Engineering for the Snapshot Panel
#
# Author     : Zilolakhon Esonova
# University : Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan
#               Chilonzor Branch using Machine Learning
#
# What this script does and why I wrote it separately:
#   After build_snapshots.py creates the (student, course, month)
#   skeleton with labels, this script attaches features to every
#   row. I kept them in separate scripts deliberately — it lets me
#   audit the label distribution before committing to any feature
#   engineering design.
#
# The core technique: "as-of-month" aggregation
#   Every feature must describe what was observable about a student
#   at the END of the snapshot month — nothing from later months
#   is allowed in. I achieve this by first collapsing each raw
#   source into monthly buckets, then taking CUMULATIVE (expanding)
#   sums or averages down the timeline of each student-course.
#   The cumulative value at month m is exactly "everything known
#   up to the end of month m".
#
#   Months where a student has no new activity (e.g. a gap month)
#   inherit the previous cumulative value via forward-fill, because
#   the balance, debt rate, and attendance rate from last month are
#   still the best estimate of the student's current state.
#   Short-window signals like last-30-day attendance are zero-filled
#   on empty months, because silence in the window IS signal.
#
# What I deliberately did NOT build here:
#   Context features like course-level dropout rate, teacher dropout
#   rate, and moderator dropout rate are means of the target label.
#   If I computed them here using all rows, they would leak label
#   information into the features. I compute those only in the
#   encode step, after the train/test split, using training-set
#   means only.
#
# Output:
#   data/processed/snapshots_features.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
from src.config import (
    ATTENDANCE_CLEAN, TRANSACTIONS_CLEAN, STUDENTHISTORIES_CLEAN,
    STUDENTGROUPS_CLEAN, GROUPS_CLEAN, STUDENTS_CLEAN, ORDERS_CLEAN,
    PROCESSED, COURSE_NAME_MAP,
)

print("=" * 60)
print("BUILDING AS-OF-MONTH SNAPSHOT FEATURES")
print("=" * 60)

# I use this helper everywhere a rate is computed from a count.
# Some students have months with zero scheduled lessons (gap months,
# or the first partial month), and dividing by zero would produce
# NaN or inf that would poison the model. Returning 0.0 for an
# empty denominator is the safest neutral value for a rate feature.
def safe_div(num, den):
    return np.where(den > 0, num / np.where(den == 0, 1, den), 0.0)

# =============================================================
# STEP 0: LOAD SNAPSHOT SKELETON
# =============================================================
# I convert snapshotMonth strings back to Period objects here so
# that merging on 'ym' works correctly for all steps below.
print("\nStep 0: Loading snapshot skeleton...")
snap = pd.read_parquet(PROCESSED / "snapshots_labeled.parquet")
snap['ym'] = pd.PeriodIndex(snap['snapshotMonth'], freq='M')
print(f"  → Snapshot rows : {len(snap):,}")

# =============================================================
# STEP 1: ATTENDANCE FEATURES (student-course, as-of-month)
# =============================================================
# I use every attendance state here — not just 'attended' — because
# the MODEL needs to know about unreasonable absences, frozen lessons,
# and payment status, which only appear in the other states. The label
# in build_snapshots.py was built from 'attended' rows only; here I
# use all rows to compute the full feature set.
print("\nStep 1: Attendance features...")
att = pd.read_parquet(ATTENDANCE_CLEAN)
att['courseName'] = att['courseId'].map(COURSE_NAME_MAP)
att = att[att['courseName'].notna()].copy()
att['date'] = pd.to_datetime(att['date'], errors='coerce', utc=True).dt.tz_localize(None)
att = att.dropna(subset=['date'])
att['ym'] = att['date'].dt.to_period('M')

# I create binary indicator columns before grouping because vectorised
# operations on int columns are much faster than using lambda in .agg().
# On the full attendance table (2+ million rows) this matters.
att['is_att'] = (att['state'] == 'attended').astype('int32')
att['is_unr'] = (att['state'] == 'unreasonable').astype('int32')
att['is_rea'] = (att['state'] == 'reasonable').astype('int32')
att['is_fro'] = (att['state'] == 'frozen').astype('int32')
att['is_pay'] = att['isPaymentDone'].fillna(False).astype(bool).astype('int32')

m = att.groupby(['studentId', 'courseName', 'ym'], observed=True).agg(
    n_total=('is_att', 'size'),
    n_att=('is_att', 'sum'),
    n_unr=('is_unr', 'sum'),
    n_rea=('is_rea', 'sum'),
    n_fro=('is_fro', 'sum'),
    n_pay=('is_pay', 'sum'),
).reset_index().sort_values(['studentId', 'courseName', 'ym'])

g = m.groupby(['studentId', 'courseName'], observed=True)
m['monthIdx'] = g.cumcount()
m['elapsed']  = m['monthIdx'] + 1

# Cumulative sums are the "as-of-month" values I need. cumsum() within
# each student-course group gives me a running total that grows as new
# months arrive, which is exactly what the model would see in deployment.
for c in ['n_total', 'n_att', 'n_unr', 'n_rea', 'n_fro', 'n_pay']:
    m['cum_' + c] = g[c].cumsum()

m['attendanceRate']          = safe_div(m['cum_n_att'], m['cum_n_total'])
m['unreasonableAbsenceRate'] = safe_div(m['cum_n_unr'], m['cum_n_total'])
m['reasonableAbsenceRate']   = safe_div(m['cum_n_rea'], m['cum_n_total'])
m['frozenLessonRate']        = safe_div(m['cum_n_fro'], m['cum_n_total'])
m['paymentDoneRate']         = safe_div(m['cum_n_pay'], m['cum_n_total'])

# I define an "active month" as having 3 or more scheduled lessons.
# Months with 1–2 lessons are typically partial start or end months
# where the student just joined or the group ended mid-month; they
# do not represent genuine engagement and would distort the rate.
m['active'] = (m['n_total'] >= 3).astype('int32')
m['cum_active'] = g['active'].cumsum()
m['activeMonthsRate'] = safe_div(m['cum_active'], m['elapsed'])
m['tenureMonths'] = m['elapsed']

# Monthly attendance rate — I need this separately from the cumulative
# rate to compute trend and recent-window features.
m['mr'] = safe_div(m['n_att'], m['n_total'])

# attendanceTrend: slope of the student's monthly attendance rate over
# all months seen so far. I wanted a feature that captures whether a
# student is improving or declining, not just their overall average.
# I compute it as an expanding OLS slope using running sums of x, x²,
# xy, and y. This avoids applying a regression function row-by-row
# (which would be very slow on 60,000+ rows) and keeps the whole
# step fully vectorised.
m['x']  = m['monthIdx'].astype(float)
m['x2'] = m['x'] ** 2
m['xy'] = m['x'] * m['mr']
m['sx']  = g['x'].cumsum()
m['sxx'] = g['x2'].cumsum()
m['sy']  = g['mr'].cumsum()
m['sxy'] = g['xy'].cumsum()
n = m['elapsed']
denom = n * m['sxx'] - m['sx'] ** 2
m['attendanceTrend'] = np.where(denom != 0, (n * m['sxy'] - m['sx'] * m['sy']) / np.where(denom == 0, 1, denom), 0.0)

# Recent-window attendance is the model's single strongest predictor
# (Pearson r = –0.52 with dropout). I approximate the 30-day window
# using this month's rate and the 60-day window using this month plus
# last month combined. This is valid because each snapshot month is
# already a calendar month, so "this month" ≈ "last 30 days" at the
# time of the snapshot.
m['prev_att']   = g['n_att'].shift(1).fillna(0)
m['prev_total'] = g['n_total'].shift(1).fillna(0)
m['attendanceLast30Days'] = m['mr']
m['attendanceLast60Days'] = safe_div(m['n_att'] + m['prev_att'], m['n_total'] + m['prev_total'])

# consecutiveMissedLessons: longest streak of missed (unreasonable or
# unchecked) lessons accumulated up to and including the snapshot month.
# I wanted a feature that captures sudden complete disengagement, which
# a rolling average would smooth over. cummax() gives the worst streak
# seen so far, which is an as-of-month value and never leaks forward.
a = att.sort_values(['studentId', 'courseName', 'date']).copy()
a['miss'] = a['state'].isin(['unreasonable', 'unchecked'])
a['blk'] = (~a['miss']).groupby([a['studentId'], a['courseName']], observed=True).cumsum()
a['streak'] = a.groupby(['studentId', 'courseName', 'blk'], observed=True).cumcount() + 1
a.loc[~a['miss'], 'streak'] = 0
a['runmax'] = a.groupby(['studentId', 'courseName'], observed=True)['streak'].cummax()
cm = a.groupby(['studentId', 'courseName', 'ym'], observed=True)['runmax'].max().reset_index(name='consecutiveMissedLessons')
m = m.merge(cm, on=['studentId', 'courseName', 'ym'], how='left')

# Features that should be forward-filled vs zero-filled on empty months.
# Cumulative rates persist across gap months (nothing changed, so the
# last known value is still correct). Recent-window features become 0
# on gap months because the student genuinely had no attendance in that
# window — and zero is the correct, informative value for the model.
ATT_FFILL = ['attendanceRate', 'unreasonableAbsenceRate', 'reasonableAbsenceRate',
             'frozenLessonRate', 'paymentDoneRate', 'activeMonthsRate', 'tenureMonths',
             'attendanceTrend', 'consecutiveMissedLessons']
ATT_ZERO  = ['attendanceLast30Days', 'attendanceLast60Days']
att_feats = m[['studentId', 'courseName', 'ym'] + ATT_FFILL + ATT_ZERO]
print(f"  → Attendance monthly rows : {len(att_feats):,}")

# =============================================================
# STEP 2: TRANSACTION FEATURES (student level, as-of-month)
# =============================================================
# I compute financial features at the STUDENT level, not the
# student-course level, because at Registan the balance account is
# shared across all of a student's courses. A student who is
# struggling to pay for their English class will have a low balance
# that affects their Math enrolment too. Aggregating at course level
# would artificially duplicate or split the same financial signal.
print("\nStep 2: Transaction features...")
txn = pd.read_parquet(TRANSACTIONS_CLEAN)
txn['createdAt'] = pd.to_datetime(txn['createdAt'], errors='coerce', utc=True).dt.tz_localize(None)
txn = txn.dropna(subset=['createdAt'])
txn['ym'] = txn['createdAt'].dt.to_period('M')
txn['is_pay']     = txn['type'].isin(['payIn', 'income']).astype('int32')
txn['is_unpaid']  = (txn['type'] == 'unpaidAttendance').astype('int32')
txn['is_atttxn']  = txn['type'].isin(['attendance', 'paidAttendance', 'unpaidAttendance']).astype('int32')
txn['pay_amt']    = np.where(txn['is_pay'] == 1, txn['amount'], 0.0)
txn['unpaid_amt'] = np.where(txn['is_unpaid'] == 1, txn['amount'], 0.0)

txn = txn.sort_values(['studentId', 'createdAt'])
tm = txn.groupby(['studentId', 'ym'], observed=True).agg(
    end_balance=('afterAmount', 'last'),
    min_after=('afterAmount', 'min'),
    max_after=('afterAmount', 'max'),
    sum_pay=('pay_amt', 'sum'),
    n_pay=('is_pay', 'sum'),
    n_unpaid=('is_unpaid', 'sum'),
    sum_unpaid=('unpaid_amt', 'sum'),
    n_atttxn=('is_atttxn', 'sum'),
).reset_index().sort_values(['studentId', 'ym'])

gt = tm.groupby('studentId', observed=True)
tm['monthIdx'] = gt.cumcount()
tm['elapsed']  = tm['monthIdx'] + 1
# currentBalance is the month-end balance — the most recent snapshot
# of how much credit a student has. minBalance and maxBalance track
# the lowest and highest the balance has ever been as-of this month,
# which captures whether the student has ever been in financial stress.
tm['currentBalance']      = tm['end_balance']
tm['minBalance']          = gt['min_after'].cummin()
tm['maxBalance']          = gt['max_after'].cummax()
tm['cum_end']             = gt['end_balance'].cumsum()
tm['avgMonthlyEndBalance'] = tm['cum_end'] / tm['elapsed']
# debtRate: proportion of months so far where the student ended the
# month in a negative balance (i.e. they owe money). I found this
# more stable than the raw balance because it is normalised by tenure.
tm['neg_month']           = (tm['end_balance'] < 0).astype('int32')
tm['debtRate']            = gt['neg_month'].cumsum() / tm['elapsed']
tm['totalPaid']           = gt['sum_pay'].cumsum()
tm['paymentCount']        = gt['n_pay'].cumsum()
tm['avgPaymentAmount']    = safe_div(tm['totalPaid'], tm['paymentCount'])
tm['pay_month']           = (tm['n_pay'] > 0).astype('int32')
tm['paymentMonths']       = gt['pay_month'].cumsum()
tm['totalDebtAmount']     = gt['sum_unpaid'].cumsum()
tm['unpaidRate']          = safe_div(gt['n_unpaid'].cumsum(), gt['n_atttxn'].cumsum())

TXN_FFILL = ['currentBalance', 'minBalance', 'maxBalance', 'avgMonthlyEndBalance',
             'debtRate', 'totalPaid', 'paymentCount', 'avgPaymentAmount',
             'paymentMonths', 'totalDebtAmount', 'unpaidRate']
txn_feats = tm[['studentId', 'ym'] + TXN_FFILL]
print(f"  → Transaction monthly rows : {len(txn_feats):,}")

# =============================================================
# STEP 3: HISTORY EVENT FEATURES (student-course, as-of-month)
# =============================================================
# The studenthistories table records lifecycle events: group freezes,
# unfreezes, group switches, price changes, and removals. I included
# these because freeze/unfreeze behaviour is a signal of a student
# trying to pause rather than leave — students who freeze and never
# return (freezeReturnRate = 0) may be effectively pre-dropouts.
# Group switches (moving to a different class time) can indicate
# dissatisfaction or scheduling conflicts.
print("\nStep 3: History event features...")
sh = pd.read_parquet(STUDENTHISTORIES_CLEAN)
sh['createdAt'] = pd.to_datetime(sh['createdAt'], errors='coerce', utc=True).dt.tz_localize(None)
sh = sh.dropna(subset=['createdAt'])
sh['ym'] = sh['createdAt'].dt.to_period('M')

sg = pd.read_parquet(STUDENTGROUPS_CLEAN)
sg['courseName'] = sg['courseId'].map(COURSE_NAME_MAP)
sg_map = sg[sg['courseName'].notna()][['studentId', 'groupId', 'courseName']].drop_duplicates()

shc = sh.merge(sg_map, on=['studentId', 'groupId'], how='left')
shc = shc[shc['courseName'].notna()]
EVT = {'removedFromGroup': 'n_removedFromGroup_course',
       'fronzen': 'n_fronzen_course',
       'unFrozen': 'n_unFrozen_course',
       'groupPriceChanged': 'n_groupPriceChanged_course'}
shc = shc[shc['type'].isin(EVT.keys())].copy()
for raw, _ in EVT.items():
    shc[raw] = (shc['type'] == raw).astype('int32')

hm = shc.groupby(['studentId', 'courseName', 'ym'], observed=True)[list(EVT.keys())].sum().reset_index()
hm = hm.sort_values(['studentId', 'courseName', 'ym'])
gh = hm.groupby(['studentId', 'courseName'], observed=True)
for raw, out in EVT.items():
    hm[out] = gh[raw].cumsum()
# freezeReturnRate: what fraction of freezes did the student come back
# from? I clip it to [0, 1] because in rare cases the data has an
# unfreeze event without a matching freeze (a data entry artefact).
hm['freezeReturnRate_course'] = np.clip(safe_div(hm['n_unFrozen_course'], hm['n_fronzen_course']), 0, 1)
hm['wasEverRemoved'] = (hm['n_removedFromGroup_course'] > 0).astype('int32')
hm['wasEverFrozen']  = (hm['n_fronzen_course'] > 0).astype('int32')

HIST_FFILL = ['n_removedFromGroup_course', 'n_fronzen_course', 'n_unFrozen_course',
              'n_groupPriceChanged_course', 'freezeReturnRate_course',
              'wasEverRemoved', 'wasEverFrozen']
hist_feats = hm[['studentId', 'courseName', 'ym'] + HIST_FFILL]

# groupSwitchCount is at student level because groupTransferred events
# in the history table do not carry a courseId. I attach it student-wide
# and let the model learn whether frequent class changes are predictive
# regardless of which course the switch happened in.
gtr = sh[sh['type'] == 'groupTransferred'].groupby(['studentId', 'ym'], observed=True).size().reset_index(name='sw')
gtr = gtr.sort_values(['studentId', 'ym'])
gtr['groupSwitchCount'] = gtr.groupby('studentId', observed=True)['sw'].cumsum()
sw_feats = gtr[['studentId', 'ym', 'groupSwitchCount']]
print(f"  → History monthly rows : {len(hist_feats):,}")

# =============================================================
# STEP 4: STATIC FEATURES (known at/before enrolment)
# =============================================================
# These features describe the student as a person rather than their
# in-course behaviour. They do not change over time, so I attach them
# to every snapshot without any as-of-month treatment.
#
# I include gender, language, and enrolment season because I wanted
# to check whether demographic patterns exist in Registan dropout data.
# After model training I found these features have low SHAP values,
# which suggests enrolment demographics are less predictive than
# attendance and payment behaviour — but I kept them in rather than
# removing them without evidence.
#
# preferredShift: I define this as the modal (most common) shift
# period across all groups the student has been in. The mode handles
# cases where a student switched shifts mid-enrolment.
print("\nStep 4: Static demographic / shift / order features...")
stu = pd.read_parquet(STUDENTS_CLEAN)
demo = stu[['studentId', 'gender', 'language', 'joinSeason', 'joinYear', 'isReferred']].drop_duplicates('studentId')

groups = pd.read_parquet(GROUPS_CLEAN)
shift_lk = sg.merge(groups[['groupId', 'shiftPeriod']], on='groupId', how='left')
def mode_shift(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s.mode()) else 'afternoon'
pref = shift_lk.groupby('studentId')['shiftPeriod'].agg(mode_shift).reset_index()
pref.columns = ['studentId', 'preferredShift']

# I include order cancellation rate because a student who frequently
# cancels lesson orders before attending may be showing early
# disengagement before it shows up in the attendance log.
orders = pd.read_parquet(ORDERS_CLEAN)
od = orders.groupby('studentId').agg(
    totalOrders=('orderId', 'count'),
    cancelledOrders=('state', lambda x: (x == 'cancelled').sum()),
).reset_index()
od['orderCancellationRate'] = safe_div(od['cancelledOrders'], od['totalOrders'])
od = od[['studentId', 'cancelledOrders', 'orderCancellationRate']]

# =============================================================
# STEP 5: MERGE EVERYTHING ONTO THE SNAPSHOT SKELETON
# =============================================================
# I merge in a specific order: attendance first (most important,
# most rows), then history events, then transactions, then static.
# Left-joins throughout because the snapshot skeleton is the
# authoritative list of rows — I never want to add extra rows.
print("\nStep 5: Merging features onto snapshots...")
out = snap.merge(att_feats,  on=['studentId', 'courseName', 'ym'], how='left')
out = out.merge(hist_feats,  on=['studentId', 'courseName', 'ym'], how='left')
out = out.merge(txn_feats,   on=['studentId', 'ym'], how='left')
out = out.merge(sw_feats,    on=['studentId', 'ym'], how='left')
out = out.merge(demo,        on='studentId', how='left')
out = out.merge(pref,        on='studentId', how='left')
out = out.merge(od,          on='studentId', how='left')

# Forward-fill cumulative (as-of) features so that gap months carry
# forward the last known value. I forward-fill within each
# student-course group for attendance/history features, and within
# each student for transaction features, matching the aggregation
# level I used in steps 1–3.
out = out.sort_values(['studentId', 'courseName', 'ym'])
out[ATT_FFILL + HIST_FFILL] = out.groupby(['studentId', 'courseName'])[ATT_FFILL + HIST_FFILL].ffill()
out = out.sort_values(['studentId', 'ym'])
out[TXN_FFILL + ['groupSwitchCount']] = out.groupby('studentId')[TXN_FFILL + ['groupSwitchCount']].ffill()

# After forward-fill, remaining NaN means "no activity of this type
# has ever occurred for this student". Zero is the correct default
# for a count or rate where nothing has happened yet.
NUM_ZERO = (ATT_FFILL + ATT_ZERO + HIST_FFILL + TXN_FFILL +
            ['groupSwitchCount', 'cancelledOrders', 'orderCancellationRate', 'isReferred'])
for c in NUM_ZERO:
    out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0)

# For categorical static fields I use sensible defaults. 'afternoon'
# is the most common shift at Registan, so it is the least-surprise
# fallback when a student has no shift history yet.
out['gender']        = out['gender'].fillna('unknown')
out['language']      = out['language'].fillna('unknown')
out['joinSeason']    = out['joinSeason'].fillna('unknown')
out['preferredShift'] = out['preferredShift'].fillna('afternoon')
out['joinYear']      = pd.to_numeric(out['joinYear'], errors='coerce').fillna(out['ym'].dt.year)

# =============================================================
# STEP 6: SANITY CHECKS + SAVE
# =============================================================
# Before saving I verify that all rate features are within [0, 1].
# Any value outside this range would indicate a bug in the cumulative
# arithmetic (e.g. numerator exceeding denominator due to a join
# creating duplicate rows). I also check for remaining nulls, because
# a single unexpected null in a feature column would silently cause
# XGBoost to use its internal NaN handling rather than the intended
# value, which could bias the model.
print("\nStep 6: Sanity checks...")
out['snapshotMonth'] = out['ym'].astype(str)
out = out.drop(columns=['ym'])

rate_cols = ['attendanceRate', 'unreasonableAbsenceRate', 'reasonableAbsenceRate',
             'frozenLessonRate', 'activeMonthsRate', 'paymentDoneRate',
             'attendanceLast30Days', 'attendanceLast60Days', 'debtRate', 'unpaidRate']
bad = {c: (float(out[c].min()), float(out[c].max())) for c in rate_cols if out[c].min() < -1e-9 or out[c].max() > 1.0 + 1e-9}
print(f"  → Rate columns out of [0,1] range : {bad if bad else 'none ✅'}")
print(f"  → Total nulls remaining            : {int(out.isnull().sum().sum())}")
print(f"  → Final shape                      : {out.shape[0]:,} rows × {out.shape[1]} cols")

train = out[out['rowSet'] == 'train']
print(f"\n  Trainable label-1 rate         : {train['label'].mean():.2%}")
print(f"  Mean attendanceRate by label:")
print(train.groupby('label')['attendanceRate'].mean().to_string())
print(f"  Mean attendanceTrend by label:")
print(train.groupby('label')['attendanceTrend'].mean().to_string())

PROCESSED.mkdir(parents=True, exist_ok=True)
dst = PROCESSED / "snapshots_features.parquet"
out.to_parquet(dst, index=False)
print(f"\n✅ Saved → {dst}")
print(f"   Columns ({len(out.columns)}): {list(out.columns)}")
