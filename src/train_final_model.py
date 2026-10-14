from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)
from xgboost import XGBClassifier

# =========================
# Paths and settings
# =========================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data.csv"
MODEL_DIR = PROJECT_DIR / "saved_models"

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.38

# =========================
# Load and clean data
# =========================

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce",
)

df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isnull()),
    "TotalCharges",
] = 0

df = df.drop(columns=["customerID"])

# This column was created only for EDA.
if "tenure_group" in df.columns:
    df = df.drop(columns=["tenure_group"])

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1,
})

if df["Churn"].isnull().any():
    raise ValueError("Unexpected values found in the Churn column.")

X = df.drop(columns=["Churn"])
y = df["Churn"].astype(int)

# =========================
# Encode categorical features
# =========================

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True,
    dtype=int,
)

# =========================
# Train / Validation / Test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train,
)

# =========================
# Final training data
# =========================

X_train_final = pd.concat(
    [X_train, X_val],
    axis=0,
)

y_train_final = pd.concat(
    [y_train, y_val],
    axis=0,
)


# =========================
# Train final XGBoost model
# =========================

final_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_model.fit(
    X_train_final,
    y_train_final,
)


# =========================
# Final test evaluation
# =========================

y_test_proba = final_model.predict_proba(X_test)[:, 1]

y_test_pred = (
    y_test_proba >= FINAL_THRESHOLD
).astype(int)

print("\nFinal XGBoost Results")
print("Threshold:", FINAL_THRESHOLD)
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print(
    "Precision:",
    precision_score(y_test, y_test_pred, zero_division=0),
)
print(
    "Recall   :",
    recall_score(y_test, y_test_pred, zero_division=0),
)
print(
    "F1 Score :",
    f1_score(y_test, y_test_pred, zero_division=0),
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
    )
)

# =========================
# Feature importance
# =========================

feature_importance = pd.DataFrame({
    "Feature": X_train_final.columns,
    "Importance": final_model.feature_importances_,
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False,
)

print("\nTop 15 Features:")
print(feature_importance.head(15).to_string(index=False))


# =========================
# Save model and settings
# =========================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    final_model,
    MODEL_DIR / "xgb_model.pkl",
)

joblib.dump(
    X_train_final.columns.tolist(),
    MODEL_DIR / "feature_columns.pkl",
)

joblib.dump(
    FINAL_THRESHOLD,
    MODEL_DIR / "threshold.pkl",
)

print("\nModel and settings saved successfully.")
print("Saved to:", MODEL_DIR)
