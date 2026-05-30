# Registan Dropout Prediction — Complete Project Documentation

A stage-by-stage account of everything that was built to turn a leaking 99.9% model
into an honest, deployable next-month dropout predictor — what was done, **why**, and
how each piece works. All numbers come from actual pipeline runs on the project data.

---

## 0. The goal and the starting point

**Goal.** For every *currently-active* student-course, predict the probability the
student stops attending **next month**, with reasons a moderator can act on.

**Starting point.** An existing pipeline (`build_master.py` → `build_ml_ready.py` →
`encode_features.py` → `train_model.py`) produced one lifetime row per student-course
and a LightGBM model scoring **ROC-AUC 0.9996 / accuracy 0.987**. That score is
impossible for real dropout prediction (good education models reach ≈0.75–0.85), so the
first job was to find out *why* it was so high.

---

## 1. Environment setup

The project was synced into the workspace from the student's Mac. The Linux environment
had no Python data stack, so `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`,
`xgboost`, `shap`, and `matplotlib` were installed. Everything was then run directly
against the cleaned parquet files in `data/interim/` and `data/processed/`.

**Why:** running the real scripts on the real data (rather than reasoning on paper) is
what let us measure the leakage and verify each fix with concrete numbers.

---

## 2. Data inspection (before changing anything)

Before touching the model, the raw cleaned data was inspected. Key findings:

- **Scale:** 554,651 attendance records, 1,221,987 transactions, 708,370 history
  events, 24,044 students, 26,422 group enrolments.
- **Graduation is barely recorded.** `students.state` = `graduated` for only **933**
  students versus **17,199 archive + 4,888 pro-archive**. The `students.graduatedAt`
  field was **100% empty (0/24,044)**. Reliable graduation exists only in
  `studentgroups` (2,359 `graduated` rows).
- **The data ends 2026-03-19**, but the old code computed "last 30 days" relative to
  `pd.Timestamp.now()` (~2026-05), so those recency features were empty for everyone.
- **Attendance rhythm:** during active study, lessons occur every 2–3 days (median gap
  2 days, 99th percentile 16 days). Gaps over 30 days occur in only **0.51%** of active
  study spells.
- **Currently active population:** only ~1,400 student-courses attended within 30 days
  of the data end; ~91% had already stopped.

**Why this mattered:** these facts directly shaped the redesign — the 30-day dropout
threshold (from the gap distribution), the decision to anchor "today" to the data, and
the choice to define the label from attendance behaviour rather than the broken
`graduated`/`graduatedAt` status.

---

## 3. The leakage diagnosis

Four leakage mechanisms and two bugs were identified in the original pipeline:

1. **Label–feature circularity (the main cause).** The completer label rule included
   `attendanceRate ≥ 0.70 AND studyDays ≥ 30`, but `attendanceRate` was also fed to the
   model. The conditions `state == graduated` / `graduatedAt` were mirrored by the
   features `totalCoursesGraduated` and `n_graduatedFromGroup_course`. The model could
   literally reconstruct the label from its own inputs.
2. **Outcome encoding via lifetime aggregation.** Features computed over the *whole*
   history (final balance, terminal attendance collapse, total debt) encode what
   happened *at/after* the dropout.
3. **Target-encoded context features.** `courseDropoutRate`, `lastGroupDropoutRate`,
   `lastTeacherDropoutRate`, `moderatorDropoutRate`, `shiftDropoutRate` were the mean of
   the label over all labelled students, merged onto every row **before** the train/test
   split — leaking test labels into training features.
4. **Random split + lifetime features** — the test set was just the past reshuffled.

Plus the two bugs from §2 (empty `graduatedAt`; `now()`-based recency).

**Why the redesign rather than patching:** even fixing the label, lifetime features and a
random split still leak. The only design that removes all four at once — and matches the
real "predict next month for an active student" task — is a time-indexed snapshot panel.

---

## 4. `config.py` — shared constants

Added a single shared `COURSE_NAME_MAP` (English levels unified; low-volume courses like
Turkish excluded) so the master builder and the new snapshot builder use one definition
instead of duplicated inline copies. Added data-derived constants:

- `DROPOUT_GAP_DAYS = 30` — from the gap analysis (>30-day gaps are 0.5% of active study).
- `HORIZON_DAYS = 30` — the prediction target window (next month).
- `ACTIVE_WINDOW_DAYS = 30` — attended within 30 days of the reference = currently active.

**Why:** thresholds are justified by the data and centralised, so an examiner can see
they were chosen, not assumed, and the pipeline stays consistent.

---

## 5. Stage 1 — `build_snapshots.py` (the label)

**What it does.** Produces the snapshot *skeleton*: one row per
`(studentId, courseName, snapshotMonth)` with a leak-free label, **no features yet**.

**How it works, step by step:**
1. Load attendance, map `courseId → courseName`, keep `state == 'attended'` rows
   (these define "showing up"). Convert dates to tz-naive.
2. `REFERENCE_DATE` = the latest attended date in the data (2026-03-19) — "today" is
   anchored to the data, never `now()`.
3. Identify graduated student-courses from `studentgroups` (the only reliable signal):
   2,284 pairs.
4. For each student-course, walk month-by-month from the first to the last attended
   month. For each month *m*:
   - Look at the forward window `(end of m, end of m + 30 days]`.
   - **Label = 0** if any attended lesson falls in that window (the student kept coming),
     else **label = 1** (they stopped next month).
   - If the student-course graduated and this is its final month, the stop is *completion*,
     so the label is set to 0, not 1.
   - **Censoring:** if the forward window runs past `REFERENCE_DATE`, the label can't be
     observed. The latest month of a currently-active student-course becomes a **scoring**
     row (what we predict in deployment); other unobservable months are excluded.

**Result:** 63,655 snapshots over 13,800 students / 16,617 student-courses; 61,001
trainable rows with a **32.4% positive rate**; 1,403 scoring rows; 1,251 censored.

**Why this label is correct:**
- It is built from attendance **dates**, while the model uses attendance **rates** — so
  there is no circularity (fixes leak #1).
- It is behavioural, so it does not depend on the unreliable `graduated` status.
- It looks strictly forward, matching the real question "will they leave next month".
- The healthy 32% balance (not 0% or 100%, and spread sensibly across courses) is itself
  evidence the label is not leaking.

---

## 6. Stage 2 — `build_snapshot_features.py` (as-of-month features)

**What it does.** Attaches, to every snapshot row, features computed using **only data up
to the end of that month**. This is what makes every feature backward-looking.

**The core technique (why it is leak-free):** each source is aggregated into monthly
buckets, then **cumulative (expanding)** sums/means are taken down the months of each
student-course. The cumulative value at month *m* is exactly "everything known by the end
of *m*" — it cannot contain future information. Ratios are formed from these cumulative
counts.

**Feature groups built:**
- **Attendance (student-course):** `attendanceRate`, `unreasonable/reasonable/frozen`
  rates, `activeMonthsRate`, `paymentDoneRate`, `tenureMonths` — all cumulative-to-month.
  - `attendanceTrend`: slope of monthly attendance rate via an **expanding least-squares**
    fit, maintained with running sums (Σx, Σy, Σxy, Σx²) so it stays vectorised.
  - `attendanceLast30Days` / `Last60Days`: anchored to the snapshot month (this month /
    this + previous month) — now genuinely meaningful, unlike the old `now()` version.
  - `consecutiveMissedLessons`: longest run of unexcused/unchecked absences up to month *m*,
    computed as an expanding max of a per-lesson streak.
- **Transactions (student level):** `currentBalance`, `min/maxBalance`,
  `avgMonthlyEndBalance`, `debtRate`, `totalPaid`, `paymentCount`, `avgPaymentAmount`,
  `paymentMonths`, `totalDebtAmount`, `unpaidRate` — cumulative month-end values.
- **History events (student-course):** cumulative counts of freeze / unfreeze /
  price-change events and `freezeReturnRate`; `groupSwitchCount` at student level.
- **Static features:** `gender`, `language`, `joinSeason`, `joinYear`, `isReferred`,
  `preferredShift`, order-cancellation features — known at enrolment, safe to attach as-is.

**Filling rule:** months with no new activity **forward-fill** the previous cumulative
value (nothing changed); genuine current-month signals (last-30/60-day) are **zero-filled**
on empty months.

**Deliberately deferred:** the context dropout-rate features were **not** built here,
because they are target-encoded and must be computed train-only after the split to avoid
leak #3.

**Sanity checks (why we trust it):** every rate column lies within [0,1]; the only nulls
are the (intended) blank labels on scoring/censored rows; and the discriminating signal is
real but not deterministic — dropouts average `attendanceRate` 0.68 vs 0.86 for stayers,
with a more negative trend. Predictive, not circular.

---

## 7. Stage 3 — `encode_snapshots.py` (encode + split)

**What it does.** Encodes categoricals and splits into train / test / score.

- **Encoding:** `gender` → 0/1; one-hot for `language`, `joinSeason`, `preferredShift`,
  `courseName` (done on the whole frame so all splits share identical columns).
- **Excludes the borderline-leaky removal features** (`n_removedFromGroup_course`,
  `wasEverRemoved`) — see §11.
- **Split by `studentId`, 80/20.** All snapshots of one student go to the same side.
  - Result: train 48,448 rows (10,645 students), test 12,553 rows (2,662 students),
    score 1,403 active rows.

**Why split by student, not by row:** one student contributes many monthly snapshots. If
some of their months were in train and others in test, the model could memorise that
specific student and the test score would be inflated. Splitting by student makes the test
set genuinely *unseen people*.

---

## 8. Stage 4 — `train_models.py` (model comparison)

### 8.1 Why these four models

- **Logistic Regression** — a linear baseline; interpretable coefficients; needed to show
  whether non-linearities matter. Wrapped in a `StandardScaler` pipeline because linear
  models are sensitive to feature magnitude (balances in the millions vs rates in [0,1]).
- **Random Forest** — a robust bagged-tree ensemble; handles non-linearity and feature
  interactions; no scaling required.
- **LightGBM** and **XGBoost** — gradient-boosted trees, the state of the art for tabular
  data; typically top performers and standard in dropout/churn literature.

Comparing a linear model, a bagging ensemble, and two boosting implementations on the
**same split with the same metrics** is a fair, conventional benchmark and lets us claim
the winner is genuinely best rather than cherry-picked.

### 8.2 How imbalance is handled (32% positive)

Imbalance was addressed at three levels:

1. **Cost re-weighting inside each model.**
   - Logistic Regression, Random Forest, LightGBM use `class_weight='balanced'`, which
     weights each class inversely to its frequency, so the 32% minority (dropouts) is not
     ignored in the loss.
   - XGBoost uses `scale_pos_weight = (#negatives / #positives) ≈ 2.07`, the equivalent
     mechanism for boosting.
2. **The right evaluation metric.** **PR-AUC (average precision)** is the primary metric,
   not accuracy. With a 32% positive rate, a model that predicts "no dropout" for everyone
   scores 68% accuracy while being useless; PR-AUC measures performance on the minority
   (dropout) class directly, so it cannot be gamed that way. The winner is chosen by
   held-out PR-AUC.
3. **An intervention-appropriate operating point.** At the 0.5 threshold the model gives
   **recall 0.79, precision 0.70** on dropouts. Because a false positive costs only a
   check-in call while a false negative is a lost student, the recall-oriented behaviour is
   desirable, and the threshold can be lowered further if outreach capacity allows.

### 8.3 How it is validated

- **Grouped 5-fold cross-validation** (`StratifiedGroupKFold`, grouped by `studentId`):
  stratified to preserve class balance, grouped so no student appears in both a training
  and validation fold.
- **Held-out test set** (different 20% of students).
- **Temporal hold-out** (train on months ≤ 2024, test on 2025+): the strongest test of
  real deployment, because in production the model always predicts the future from the past.

### 8.4 Results (clean headline model)

| Model | CV PR-AUC | Test ROC-AUC | Test PR-AUC | Temporal PR-AUC |
|-------|-----------|--------------|-------------|-----------------|
| **XGBoost (winner)** | 0.796 | **0.890** | **0.796** | 0.751 |
| LightGBM | 0.795 | 0.887 | 0.790 | 0.753 |
| Random Forest | 0.786 | 0.881 | 0.784 | 0.755 |
| Logistic Regression | 0.731 | 0.853 | 0.732 | 0.704 |

The three tree models cluster within 0.012 PR-AUC — the signal is in the *features*, not a
lucky algorithm. XGBoost wins on PR-AUC and is saved as `models/best_model.pkl`. Dropout
recall 0.79, precision 0.70, accuracy 0.83.

---

## 9. Proving the new result is NOT leakage

This is the most important validation for a dissertation, since the whole project began
with a leaked 0.999. Three independent checks:

1. **Temporal stability.** Under the time-based split the model holds at ROC-AUC ~0.88 /
   PR-AUC ~0.75 — a small, expected dip, not the collapse a leaked model shows.
2. **No single dominant feature.** The strongest feature alone reaches only ROC-AUC 0.83
   (`attendanceLast30Days`), 0.73 (`attendanceRate`), 0.71 (`currentBalance`). Leakage
   usually appears as one near-perfect feature; there is none.
3. **Ablation insensitivity.** Removing *all* financial features changed temporal PR-AUC by
   ≈0 (0.79 → 0.79), proving no financial feature is secretly encoding the outcome.

---

## 10. SHAP explainability — `explain_model.py`

**Why SHAP:** raw tree "importance" only counts splits; SHAP attributes each prediction to
its features additively and with direction, which is far more trustworthy and gives the
per-student "why" the dashboard needs.

**What it showed (clean model):** `attendanceLast30Days` dominates, followed by
`currentBalance`, `totalPaid`, and `tenureMonths`. Interpretation:
- **Recent attendance beats lifetime averages** — disengagement is recent and visible
  shortly before leaving. This validates the snapshot design.
- **Finance is secondary but real** — low balance / irregular payment raise risk.
- **Early tenure is the vulnerable period** — concentrate retention effort in the first
  months.

Outputs: `reports/figures/shap_summary.png` (beeswarm) and `shap_importance_bar.png`.

---

## 11. The removal-feature decision

SHAP initially flagged `n_removedFromGroup_course` as the single most influential feature
(~0.046 PR-AUC). A group removal is often the *administrative act of leaving*, recorded in
the same month — so it is a borderline proxy for the outcome, and a student already removed
is past the point of useful intervention. It was therefore **excluded** from the headline
model. The variant keeping it (ROC 0.90 / PR 0.80) is documented as a **sensitivity
analysis**. This trades ~0.04 PR-AUC for an unambiguously clean, actionable model — the
right call for a dissertation.

---

## 12. Scoring active students — `predict.py`

**What it does.** Loads `best_model.pkl`, scores the 1,403 active student-courses, and
writes `predictions.parquet` with:
- `dropout_probability` — chance of stopping next month.
- `risk_level` — High >70%, Medium 40–70%, Low <40% (**194 / 273 / 936** students).
- `topFactors` — the three features with the largest *positive* SHAP contribution for that
  student, translated into plain language (e.g. "low attendance this month, low current
  balance, payment history").

**Why per-student SHAP factors:** a probability alone is not actionable; the top factors
tell the moderator *what kind* of intervention fits (attendance collapse vs debt).

---

## 13. The dashboard — Predictions page

A 7th page was added to the existing Streamlit dashboard. It shows risk metrics
(High/Medium/Low counts), filters (risk level, course, **"my students" by moderator**, name
search), a risk-by-course chart, a **colour-coded ranked table** (student, course, risk %,
risk, top factors, teacher, moderator), and a **CSV export**. Student/teacher/moderator
names are joined from the cleaned files (verified 100% match).

**Why:** this is the deployment artefact — the "one-click" tool a moderator uses to triage
and act, which is how the system is demonstrated.

---

## 14. Repository artefacts

- `README.md` rewritten to document the leak-free method, the run order, and the results.
- `requirements.txt` updated (`xgboost`, `matplotlib`, `seaborn`, `numpy`).
- `.gitignore` confirmed to exclude all student data, `venv/`, and model files (PII safe).
- Push is left to the student's machine (their GitHub account); commands provided.

---

## 15. End-to-end data flow

```
attendance / transactions / studenthistories / studentgroups / students / orders
        │
        ▼
build_snapshots.py            → snapshots_labeled.parquet      (skeleton + label)
        ▼
build_snapshot_features.py    → snapshots_features.parquet     (as-of-month features)
        ▼
encode_snapshots.py           → snap_train / snap_test / snap_score .parquet
        ▼
train_models.py               → models/best_model.pkl + comparison + figures
        ▼
explain_model.py              → SHAP figures
        ▼
predict.py                    → predictions.parquet
        ▼
eda_dashboard.py (Predictions page) → ranked, explainable at-risk list
```

**Run order:**
```bash
python3 src/features/build_snapshots.py
python3 src/features/build_snapshot_features.py
python3 src/features/encode_snapshots.py
python3 src/models/train_models.py
python3 src/models/explain_model.py
python3 src/models/predict.py
streamlit run src/dashboard/eda_dashboard.py
```

---

## 16. Summary — what changed and why it matters

| Aspect | Before | After |
|--------|--------|-------|
| Unit of analysis | one lifetime row per student-course | one row per student-course-**month** |
| Label | partly defined by a model feature (circular) | forward-looking attendance behaviour |
| Features | lifetime aggregates (see the future) | as-of-month cumulative (strictly backward) |
| Context rates | target-encoded, pre-split (leaked) | excluded (deferred, train-only if added) |
| Split | random | by student + temporal validation |
| Imbalance | accuracy-focused | class-weighting + PR-AUC + recall operating point |
| Score | ROC-AUC 0.9996 (leaked, useless) | ROC-AUC 0.890 / PR-AUC 0.796 (honest, deployable) |

The headline reduction from 0.9996 to a temporally-validated 0.89 is the project's central
result: an artefact was replaced with an honest, explainable, and actionable model.

---

## 17. Limitations

- A 30-day absence defines dropout, so the rare students who return after longer breaks are
  labelled at-risk during the gap.
- Graduation is inferred from group-level signals because the student-level field is empty.
- Evaluation is retrospective; proving the model *reduces* dropout requires a prospective
  study where predictions drive interventions and outcomes are measured.
