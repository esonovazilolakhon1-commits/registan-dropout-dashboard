# Registan Student Dropout Prediction

Master's dissertation project — predicting student dropout at **Registan (Chilonzor branch)**, a learning center in Tashkent, using machine learning.

**Author:** Zilolakhon Esonova — Westminster International University Tashkent

## What the system does

For every **currently-active** student-course, it predicts the probability that the
student will **stop attending next month**, and surfaces the top reasons — so a
moderator can intervene early. The predictions are served through a Streamlit
dashboard (the "Predictions" page).

## Method (and why it is leak-free)

The model uses a **monthly snapshot panel**: one row per `(student, course, month)`.
Every feature is computed using **only data up to the end of that month**, and the
label looks **strictly forward** — `1` if the student attends no lesson in the next
30 days, else `0`. Graduations (the only reliable completion signal, from
`studentgroups`) are treated as completion, not dropout. This design eliminates the
target leakage that affected an earlier lifetime-aggregate design (which scored an
unrealistic ~0.999 AUC).

**Headline result (honest, temporally validated):**

| Model | Test ROC-AUC | Test PR-AUC | Temporal PR-AUC |
|-------|-------------|-------------|-----------------|
| **XGBoost** (winner) | **0.890** | **0.796** | 0.751 |
| LightGBM | 0.887 | 0.790 | 0.753 |
| Random Forest | 0.881 | 0.784 | 0.755 |
| Logistic Regression | 0.853 | 0.732 | 0.704 |

Winner chosen by PR-AUC (appropriate for the ~32% positive class). Recall on dropouts ≈ 0.79.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # plus: shap, xgboost
```

## Pipeline (run in order)

```bash
# 1. clean raw exports → data/interim/   (src/clean/*)
# 2. build the leak-free snapshot panel + features
python3 src/features/build_snapshots.py
python3 src/features/build_snapshot_features.py
python3 src/features/encode_snapshots.py
# 3. train + compare models, explain, and score active students
python3 src/models/train_models.py      # 4-model comparison → models/best_model.pkl
python3 src/models/explain_model.py      # SHAP global plots
python3 src/models/predict.py            # → data/processed/predictions.parquet
# 4. launch the dashboard
streamlit run src/dashboard/eda_dashboard.py
```

## Project structure

- `data/raw/` — original MongoDB exports (git-ignored; contains student PII)
- `data/interim/` — one cleaned parquet per source
- `data/processed/` — snapshot panel, splits, predictions
- `src/clean/` — per-source cleaning scripts
- `src/features/` — `build_snapshots.py`, `build_snapshot_features.py`, `encode_snapshots.py`
- `src/models/` — `train_models.py`, `explain_model.py`, `predict.py`
- `src/dashboard/` — Streamlit app (6 EDA pages + Predictions page)
- `models/` — saved best model (git-ignored)
- `reports/figures/` — model comparison + SHAP figures

## Notes

- The legacy lifetime-aggregate scripts (`build_master.py`, `build_ml_ready.py`,
  `train_model.py`) are retained for the dissertation's leakage discussion but are
  **superseded** by the snapshot pipeline above.
- All student data is excluded from version control via `.gitignore`.
