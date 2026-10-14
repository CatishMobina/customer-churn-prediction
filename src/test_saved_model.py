
from pathlib import Path
import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "saved_models"

model = joblib.load(MODEL_DIR / "xgb_model.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
threshold = joblib.load(MODEL_DIR / "threshold.pkl")

print("Model loaded:", type(model).__name__)
print("Number of features:", len(feature_columns))
print("Threshold:", threshold)

# Check that the saved model and feature list agree.
assert model.n_features_in_ == len(feature_columns)

print("Saved model checks passed.")
