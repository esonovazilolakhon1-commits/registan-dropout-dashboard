# =============================================================
# predict.py
# =============================================================
# Score Currently-Active Students for Dropout Risk
#
# Author : Zilolakhon Esonova — Westminster International University Tashkent
# Dissertation: Predicting Student Dropout at Registan (Chilonzor)
#
# Why this script exists separately from train_models.py:
#   Training produces a static artifact (best_model.pkl). This script
#   is the deployment step — it takes currently-active students whose
#   label is unknown (because their forward window extends past the
#   data) and produces a ranked at-risk list a moderator can act on
#   today. Keeping scoring separate means a moderator can re-run
#   predictions monthly without re-training the model every time.
#
# Why I include human-readable "topFactors" rather than raw SHAP values:
#   The output of this script goes directly into the dashboard that
#   Registan's moderators use. A moderator who sees "dropout_probability
#   = 0.82" without context will not know whether to call the student,
#   check their payment, or speak to their teacher. By translating the
#   three strongest risk-increasing SHAP values into plain phrases
#   ("low attendance this month", "frequent negative balance"), I give
#   her an immediately actionable reason to reach out.
#
# Why raw probabilities rather than a binary flag:
#   I deliberately output the continuous probability rather than a
#   0/1 prediction because different moderators and different situations
#   call for different thresholds. A moderator with capacity to call
#   10 students might set her own cutoff at 0.80; one with capacity
#   for 50 might use 0.50. Using a fixed cutoff inside this script
#   would make that decision for her. The risk_level column (High /
#   Medium / Low) is a convenience grouping for the dashboard, not a
#   substitute for the probability.
#
# Inputs : models/best_model.pkl, data/processed/snap_score.parquet
# Output : data/processed/predictions.parquet
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import pickle
import shap

from src.config import PROCESSED, MODELS

print("=" * 60)
print("SCORING ACTIVE STUDENTS")
print("=" * 60)

# I translate internal feature names into plain phrases that a moderator
# can read without knowing anything about the model. The phrasing is
# intentionally action-oriented ("low attendance this month" rather than
# "attendanceLast30Days = 0.3") because the goal is to tell the moderator
# what to address, not to explain the model.
FRIENDLY = {
    "attendanceLast30Days": "low attendance this month",
    "attendanceLast60Days": "low attendance recently",
    "attendanceRate": "low overall attendance",
    "attendanceTrend": "declining attendance trend",
    "consecutiveMissedLessons": "consecutive missed lessons",
    "unreasonableAbsenceRate": "unexcused absences",
    "activeMonthsRate": "inconsistent monthly activity",
    "paymentDoneRate": "lessons left unpaid",
    "unpaidRate": "high unpaid-lesson rate",
    "debtRate": "frequent negative balance",
    "currentBalance": "low current balance",
    "minBalance": "deep debt at some point",
    "totalDebtAmount": "accumulated debt",
    "tenureMonths": "short tenure so far",
    "groupSwitchCount": "frequent group switching",
    "frozenLessonRate": "frozen lessons",
    "reasonableAbsenceRate": "excused absences",
    "currentBalance": "low current balance",
    "avgMonthlyEndBalance": "low monthly balance",
    "maxBalance": "balance pattern",
    "minBalance": "deep debt at some point",
    "totalPaid": "payment history",
    "avgPaymentAmount": "payment size",
    "paymentCount": "payment frequency",
    "paymentMonths": "few months paid",
    "joinYear": "enrolment cohort",
    "isReferred": "referral status",
    "n_fronzen_course": "freeze events",
    "n_groupPriceChanged_course": "price changes",
}

# For one-hot encoded features I strip the prefix and add a readable
# label so the moderator sees "course: English" rather than "courseName_English".
OHE_PREFIX = {"courseName_": "course", "language_": "language",
              "joinSeason_": "joined in", "preferredShift_": "shift"}

def friendly(f):
    if f in FRIENDLY:
        return FRIENDLY[f]
    for pre, lab in OHE_PREFIX.items():
        if f.startswith(pre):
            return f"{lab}: {f[len(pre):]}"
    # Fallback for any feature not explicitly mapped — at least replace
    # underscores so it does not look like raw code to the moderator.
    return f.replace("_", " ")

# =============================================================
# STEP 1: LOAD MODEL + SCORING SET
# =============================================================
# I load from snap_score.parquet, not snap_test.parquet. The test set
# is for evaluation (students whose outcome is already known). The
# scoring set contains only the students whose forward window extends
# past the reference date — their label is genuinely unknown and these
# are the students the moderator needs to act on right now.
print("\nStep 1: Loading model and active students...")
with open(MODELS / "best_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model, features, name = bundle["model"], bundle["features"], bundle["model_name"]

score = pd.read_parquet(PROCESSED / "snap_score.parquet")
X = score[features]
print(f"  → Model: {name} | Active student-courses to score: {len(score):,}")

# I reconstruct courseName from the one-hot columns so the output table
# has a readable course label rather than a row of 0s and 1s.
course_cols = [c for c in score.columns if c.startswith("courseName_")]
score["courseName"] = (
    score[course_cols].idxmax(axis=1).str.replace("courseName_", "", regex=False)
)

# =============================================================
# STEP 2: PREDICT PROBABILITIES + RISK LEVEL
# =============================================================
# I chose 0.70 and 0.40 as the High/Medium/Low boundaries after
# inspecting the probability distribution on the test set. At 0.70
# the model's precision was high enough that most flagged students
# were genuine dropout risks. At 0.40 the recall was high enough to
# catch the majority of eventual dropouts. These are reporting
# thresholds for the dashboard — the raw probability column lets
# the moderator apply her own cutoff if she wants.
print("\nStep 2: Predicting dropout probability...")
score["dropout_probability"] = model.predict_proba(X)[:, 1]

def risk_level(p):
    return "High" if p > 0.70 else ("Medium" if p >= 0.40 else "Low")
score["risk_level"] = score["dropout_probability"].apply(risk_level)

# =============================================================
# STEP 3: PER-STUDENT TOP RISK FACTORS (SHAP)
# =============================================================
# For each active student I take the 3 features with the largest
# POSITIVE SHAP contribution — those that are pushing the model
# toward dropout for that specific student. I filter to positive
# contributions only because a feature that reduces the risk score
# is not something the moderator needs to worry about; she needs to
# know what is driving the alert, not what is holding it down.
print("\nStep 3: Computing per-student top factors (SHAP)...")
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)

top_factors = []
for i in range(len(X)):
    contrib = sv[i]
    order = np.argsort(contrib)[::-1][:3]          # most risk-increasing first
    picks = [friendly(features[j]) for j in order if contrib[j] > 0]
    top_factors.append(", ".join(picks) if picks else "no strong risk factors")
score["topFactors"] = top_factors

# =============================================================
# STEP 4: SAVE RANKED PREDICTIONS
# =============================================================
# I sort by dropout_probability descending so the moderator's dashboard
# shows the highest-risk students first. I keep only the columns that
# are meaningful for the deployment context — stripping the 50+ raw
# feature columns keeps the file small and the dashboard query fast.
out = score[["studentId", "courseName", "snapshotMonth",
             "dropout_probability", "risk_level", "topFactors"]].copy()
out = out.sort_values("dropout_probability", ascending=False).reset_index(drop=True)

PROCESSED.mkdir(parents=True, exist_ok=True)
out.to_parquet(PROCESSED / "predictions.parquet", index=False)

print("\n  Risk-level distribution:")
print(out["risk_level"].value_counts().to_string())
print("\n  Top 10 at-risk active students:")
print(out.head(10).to_string(index=False))
print(f"\n✅ Predictions saved → {PROCESSED / 'predictions.parquet'}  ({len(out):,} students)")
