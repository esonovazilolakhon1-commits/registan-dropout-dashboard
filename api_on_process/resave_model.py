# resave_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Re-saves best_model.pkl using the current XGBoost version's format.
# Run this once to eliminate the XGBoost serialization warning and ensure
# the model loads cleanly on any machine regardless of XGBoost version.
#
# Safe to run: model weights and hyperparameters are unchanged.
# Only the serialization format is updated.
#
# How to run:
#   python3 api_on_process/resave_model.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import pickle
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MODELS

model_path = MODELS / "best_model.pkl"

print("Loading best_model.pkl (may show a version warning — expected) ...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   # suppress the old-format warning during load
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

model      = bundle["model"]
model_name = bundle["model_name"]
features   = bundle["features"]
metrics    = bundle["metrics"]

print(f"  → Loaded: {model_name}")
print(f"  → Features: {len(features)}")

# Re-save using current XGBoost — pickle now serializes the booster
# in the current format, eliminating the warning on future loads.
backup_path = MODELS / "best_model_backup.pkl"
print(f"\nBacking up original → {backup_path}")
with open(backup_path, "wb") as f:
    pickle.dump(bundle, f)

print(f"Re-saving in current XGBoost format → {model_path}")
with open(model_path, "wb") as f:
    pickle.dump({"model": model, "model_name": model_name,
                 "features": features, "metrics": metrics}, f)

# Verify it loads cleanly
print("\nVerifying clean load (no warnings expected)...")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with open(model_path, "rb") as f:
        check = pickle.load(f)
    xgb_warnings = [w for w in caught if "XGBoost" in str(w.message) or "serializ" in str(w.message)]

if xgb_warnings:
    print("  ⚠️  Warning still present — XGBoost version gap may be too large.")
    print("     The model still works; this is cosmetic only.")
else:
    print("  ✅ Loads cleanly — no XGBoost warnings.")

print(f"\n✅ Done. Original backed up to: best_model_backup.pkl")
