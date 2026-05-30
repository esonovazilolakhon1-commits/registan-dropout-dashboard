# predict_live.py
# ─────────────────────────────────────────────────────────────────────────────
# Score active students using freshly pulled API data.
# Identical logic to src/models/predict.py but saves to:
#   data/processed/live_predictions.parquet
#
# The original predictions.parquet (from the local pipeline) is NEVER touched.
#
# How to run:
#   python3 api_on_process/predict_live.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import pickle
import shap
from datetime import datetime

from src.config import PROCESSED, MODELS, ROOT

# The live predictions file must always land in the ORIGINAL data/processed/
# so the dashboard can find it, regardless of the LIVE_PIPELINE path redirect.
ORIGINAL_PROCESSED = ROOT / "data" / "processed"

print("=" * 60)
print("LIVE SCORING — ACTIVE STUDENTS (API DATA)")
print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

FRIENDLY = {
    "attendanceLast30Days":     "low attendance this month",
    "attendanceLast60Days":     "low attendance recently",
    "attendanceRate":           "low overall attendance",
    "attendanceTrend":          "declining attendance trend",
    "consecutiveMissedLessons": "consecutive missed lessons",
    "unreasonableAbsenceRate":  "unexcused absences",
    "activeMonthsRate":         "inconsistent monthly activity",
    "paymentDoneRate":          "lessons left unpaid",
    "unpaidRate":               "high unpaid-lesson rate",
    "debtRate":                 "frequent negative balance",
    "currentBalance":           "low current balance",
    "minBalance":               "deep debt at some point",
    "totalDebtAmount":          "accumulated debt",
    "tenureMonths":             "short tenure so far",
    "groupSwitchCount":         "frequent group switching",
    "frozenLessonRate":         "frozen lessons",
    "reasonableAbsenceRate":    "excused absences",
    "avgMonthlyEndBalance":     "low monthly balance",
    "maxBalance":               "balance pattern",
    "totalPaid":                "payment history",
    "avgPaymentAmount":         "payment size",
    "paymentCount":             "payment frequency",
    "paymentMonths":            "few months paid",
    "joinYear":                 "enrolment cohort",
    "isReferred":               "referral status",
    "n_fronzen_course":         "freeze events",
    "n_groupPriceChanged_course": "price changes",
}
OHE_PREFIX = {"courseName_": "course", "language_": "language",
              "joinSeason_": "joined in", "preferredShift_": "shift"}

def friendly(f):
    if f in FRIENDLY:
        return FRIENDLY[f]
    for pre, lab in OHE_PREFIX.items():
        if f.startswith(pre):
            return f"{lab}: {f[len(pre):]}"
    return f.replace("_", " ")

# ── Step 1: load model + scoring set ─────────────────────────────────────────
print("\nStep 1: Loading model and active students...")
with open(MODELS / "best_model.pkl", "rb") as f:
    bundle = pickle.load(f)
model, features, name = bundle["model"], bundle["features"], bundle["model_name"]

score = pd.read_parquet(PROCESSED / "snap_score.parquet")

# The live snapshot may not contain every one-hot category the model was
# trained on (e.g. no Physics students in the live window). Add any missing
# feature columns as 0 so the model receives exactly the features it expects.
missing_cols = [f for f in features if f not in score.columns]
if missing_cols:
    print(f"  → Adding {len(missing_cols)} missing feature columns as 0: {missing_cols}")
    for col in missing_cols:
        score[col] = 0

X = score[features]
print(f"  → Model: {name} | Active students to score: {len(score):,}")

course_cols = [c for c in score.columns if c.startswith("courseName_")]
score["courseName"] = (
    score[course_cols].idxmax(axis=1).str.replace("courseName_", "", regex=False)
)

# ── Step 2: predict probabilities + risk level ────────────────────────────────
print("\nStep 2: Predicting dropout probability...")
score["dropout_probability"] = model.predict_proba(X)[:, 1]

def risk_level(p):
    return "High" if p > 0.70 else ("Medium" if p >= 0.40 else "Low")

score["risk_level"] = score["dropout_probability"].apply(risk_level)

# ── Step 3: top risk factors via SHAP ────────────────────────────────────────
print("\nStep 3: Computing per-student top factors (SHAP)...")
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X)
if isinstance(sv, list):
    sv = sv[1]
sv = np.asarray(sv)

top_factors = []
for i in range(len(X)):
    contrib = sv[i]
    order = np.argsort(contrib)[::-1][:3]
    picks = [friendly(features[j]) for j in order if contrib[j] > 0]
    top_factors.append(", ".join(picks) if picks else "no strong risk factors")
score["topFactors"] = top_factors

# ── Step 3b: save features for next month's outcome labeling ─────────────────
# check_outcomes.py needs the actual feature values used this month so it can
# create new training rows once we know which students actually dropped out.
print("\nStep 3b: Saving feature snapshot for future retraining...")
feat_snapshot = score[['studentId', 'courseName', 'snapshotMonth'] + features].copy()
feat_out_path = ORIGINAL_PROCESSED / "live_features.parquet"
feat_snapshot.to_parquet(feat_out_path, index=False)
print(f"  → Feature snapshot saved → {feat_out_path}  ({len(feat_snapshot):,} rows)")

# ── Step 4: enrich with student names, phones, teacher & moderator names ──────
print("\nStep 4: Enriching with student and staff names...")

# Build name lookup directly from raw JSON — no branch filter, no parquet.
# The cleaned parquets filter to Chilonzor branch which excludes students who
# attend Chilonzor but are registered under a different branch in the CRM.
# Reading the raw JSON ensures we resolve names for all active students.
import json as _json

def _load_students_raw(path):
    """Return {studentId: (fullName, phoneNumber)} from a raw students JSON."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = _json.load(fh)
    items = data if isinstance(data, list) else data.get("data", [])
    out = {}
    for s in items:
        sid_raw = s.get("_id", "")
        sid = sid_raw.get("$oid", "") if isinstance(sid_raw, dict) else str(sid_raw)
        if not sid:
            continue
        name  = s.get("fullName") or (
            (s.get("firstName", "") + " " + s.get("lastName", "")).strip()
        )
        phone = str(s.get("phoneNumber", "") or "")
        out[sid] = (name, phone)
    return out

def _load_from_orders(path):
    """Extract {studentId: (fullName, phoneNumber)} from embedded studentInfo in orders."""
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = _json.load(fh)
    items = data if isinstance(data, list) else data.get("data", [])
    out = {}
    for o in items:
        sid = str(o.get("studentId", "") or "")
        if not sid:
            continue
        si = o.get("studentInfo") or {}
        name  = si.get("fullName") or (
            (si.get("firstName","") + " " + si.get("lastName","")).strip()
        )
        phone = str(si.get("phoneNumber", "") or "")
        if sid not in out and name:   # first occurrence wins
            out[sid] = (name, phone)
    return out

# Layer 1: historical MongoDB dump (24k students, no branch filter)
raw_lookup = _load_students_raw(ROOT / "data" / "raw"      / "students.raw.json")
# Layer 2: live students endpoint (students registered since March 19)
live_lookup = _load_students_raw(ROOT / "data" / "live_raw" / "students.raw.json")
# Layer 3: orders embed studentInfo for every active student who placed an order
orders_lookup = _load_from_orders(ROOT / "data" / "live_raw" / "orders.raw.json")

# Merge: orders wins (most current), then live students, then historical
merged = {**raw_lookup, **orders_lookup, **live_lookup}

print(f"  → Student lookup: {len(raw_lookup):,} from historical dump "
      f"+ {len(orders_lookup):,} from orders + {len(live_lookup):,} new students "
      f"= {len(merged):,} unique")

students_lookup = pd.DataFrame([
    {"studentId": sid, "studentName": name, "phoneNumber": phone}
    for sid, (name, phone) in merged.items()
])

# Build teacher/moderator lookup from live master (already has resolved names).
live_master_path = ROOT / "data" / "live_processed" / "master.parquet"
orig_master_path = ROOT / "data" / "processed"      / "master_full.parquet"

ctx_cols = ["studentId", "courseName", "lastTeacherName", "lastModeratorName"]
ctx = None
for mpath in [live_master_path, orig_master_path]:
    if mpath.exists():
        m = pd.read_parquet(mpath)
        available = [c for c in ctx_cols if c in m.columns]
        if "studentId" in available and "courseName" in available:
            ctx = m[available].drop_duplicates(["studentId", "courseName"])
            print(f"  → Staff names from: {mpath.name}")
            break

# ── Step 5: save to live_predictions.parquet (NOT predictions.parquet) ────────
out = score[["studentId", "courseName", "snapshotMonth",
             "dropout_probability", "risk_level", "topFactors"]].copy()
out = out.sort_values("dropout_probability", ascending=False).reset_index(drop=True)
out["retrievedAt"] = datetime.now().strftime("%Y-%m-%d %H:%M")

# merge student names + phones
out = out.merge(students_lookup, on="studentId", how="left")
out["studentName"]  = out["studentName"].fillna("Unregistered")
out["phoneNumber"]  = out["phoneNumber"].fillna("").astype(str).str.replace("None", "", regex=False)

# merge teacher + moderator names
if ctx is not None:
    merge_cols = [c for c in ["studentId", "courseName", "lastTeacherName", "lastModeratorName"] if c in ctx.columns]
    out = out.merge(ctx[merge_cols], on=["studentId", "courseName"], how="left")
if "lastTeacherName"   not in out.columns: out["lastTeacherName"]   = "Unknown"
if "lastModeratorName" not in out.columns: out["lastModeratorName"] = "Unknown"
out["lastTeacherName"]   = out["lastTeacherName"].fillna("Unknown")
out["lastModeratorName"] = out["lastModeratorName"].fillna("Unknown")

named = (out["studentName"] != "Unknown").sum()
print(f"  → Names resolved: {named:,}/{len(out):,} students "
      f"({named/len(out):.0%})")
print(f"  → Teacher names : {(out['lastTeacherName'] != 'Unknown').sum():,}/{len(out):,}")

ORIGINAL_PROCESSED.mkdir(parents=True, exist_ok=True)
out_path = ORIGINAL_PROCESSED / "live_predictions.parquet"
out.to_parquet(out_path, index=False)

print("\n  Risk-level distribution:")
print(out["risk_level"].value_counts().to_string())
print(f"\n✅ Live predictions saved → {out_path}  ({len(out):,} students)")
print("   Original predictions.parquet was NOT modified.")
