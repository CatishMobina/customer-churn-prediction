import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier , ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import classification_report

df = pd.read_csv("data.csv")

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.duplicated().sum())

print(df["Churn"].value_counts())

print(df.nunique())

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isnull()),
    "TotalCharges"
] = 0

churn_rate = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

churn_rate["Yes"].plot(kind="bar")

plt.ylabel("Churn Rate (%)")
plt.xlabel("Contract")
plt.title("Churn Rate by Contract")
plt.show()

internet_churn = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print(internet_churn)

payment_churn = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print(payment_churn)

techsupport_churn = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print(techsupport_churn)

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

tenure_churn = pd.crosstab(
    df["tenure_group"],
    df["Churn"],
    normalize="index"
) * 100

print(tenure_churn)

tenure_churn["Yes"].plot(kind="bar")

plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Tenure Group")
plt.xticks(rotation=0)

plt.show()

numeric_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

correlation = df[numeric_cols].corr()

print(correlation)

categorical_cols = [
    "gender",
    "Partner",
    "Dependents",
    "Contract",
    "InternetService",
    "PaymentMethod",
    "PaperlessBilling",
    "TechSupport"
]

for col in categorical_cols:
    churn_rate = pd.crosstab(
        df[col],
        df["Churn"],
        normalize="index"
    ) * 100

    print(f"\n--- {col} ---")
    print(churn_rate)

# =========================
# Feature Engineering
# =========================

df = df.drop(columns=["customerID"])

df = df.drop(columns=["tenure_group"])

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn"])
y = df["Churn"]

# result
print("X shape:", X.shape)
print("\nX dtypes:")
print(X.dtypes)

print("\ny shape:", y.shape)
print("\nChurn distribution:")
print(y.value_counts())

# =========================
# Encoding Categorical Features
# =========================

categorical_cols = X.select_dtypes(include=["object", "string"]).columns

print("Categorical columns:")
print(categorical_cols.tolist())

# One-Hot Encoding
X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)

# result
print("\nX shape after encoding:", X.shape)

print("\nX dtypes after encoding:")
print(X.dtypes)

print("\nFirst 5 rows:")
print(X.head())

# =========================
# Train / Validation / Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# result
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nTrain Churn:")
print(y_train.value_counts(normalize=True))

print("\nValidation Churn:")
print(y_val.value_counts(normalize=True))

print("\nTest Churn:")
print(y_test.value_counts(normalize=True))

# =========================
# Scaling for Logistic Regression
# =========================

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

scaler.fit(X_train[numeric_features])

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.transform(
    X_train[numeric_features]
)

X_val_scaled[numeric_features] = scaler.transform(
    X_val[numeric_features]
)

X_test_scaled[numeric_features] = scaler.transform(
    X_test[numeric_features]
)

# result
print("Train after scaling:")
print(X_train_scaled[numeric_features].describe())

print("\nValidation after scaling:")
print(X_val_scaled[numeric_features].describe())

print("\nTest after scaling:")
print(X_test_scaled[numeric_features].describe())

# =========================
# Logistic Regression
# =========================

model = LogisticRegression(
    max_iter=10,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

y_val_pred = model.predict(
    X_val_scaled
)

print("First 20 predictions:")
print(y_val_pred[:20])

print("\nFirst 20 actual values:")
print(y_val.iloc[:20].values)

accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

y_val_proba = model.predict_proba(X_val_scaled)

churn_proba = y_val_proba[:, 1]

print("First 10 churn probabilities:")
print(churn_proba[:10])

thresholds = [0.5, 0.4, 0.3, 0.25, 0.2]

for threshold in thresholds:
    y_pred_threshold = (churn_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Accuracy :", accuracy_score(y_val, y_pred_threshold))
    print("Precision:", precision_score(y_val, y_pred_threshold, zero_division=0))
    print("Recall   :", recall_score(y_val, y_pred_threshold, zero_division=0))
    print("F1 Score :", f1_score(y_val, y_pred_threshold, zero_division=0))
    
for threshold in [0.30, 0.20]:
    y_pred_threshold = (churn_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print(confusion_matrix(y_val, y_pred_threshold))
    
# =========================
# Decision Tree
# =========================

tree_model = DecisionTreeClassifier(
    random_state=42
)

tree_model.fit(
    X_train,
    y_train
)

y_val_tree_pred = tree_model.predict(
    X_val
)

print("Decision Tree Results")
print("Accuracy :", accuracy_score(y_val, y_val_tree_pred))
print("Precision:", precision_score(y_val, y_val_tree_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_tree_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_tree_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_val_tree_pred))

for depth in [3, 5, 7, 10, None]:
    tree_model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_val)

    print(f"\nMax Depth: {depth}")
    print("Precision:", precision_score(y_val, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_val, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_val, y_pred, zero_division=0))
    
for weight in [None, "balanced"]:
    tree_model = DecisionTreeClassifier(
        max_depth=5,
        class_weight=weight,
        random_state=42
    )

    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_val)

    print(f"\nClass Weight: {weight}")
    print("Precision:", precision_score(y_val, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_val, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_val, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
# =========================
# Random Forest
# =========================

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_val_rf_pred = rf_model.predict(X_val)

print("Random Forest Results")
print("Precision:", precision_score(y_val, y_val_rf_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_rf_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_rf_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_rf_pred))

y_val_rf_proba = rf_model.predict_proba(X_val)[:, 1]

thresholds = [0.5, 0.4, 0.3, 0.25, 0.2]

for threshold in thresholds:
    y_pred = (y_val_rf_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(y_val, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_val, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_val, y_pred, zero_division=0))
    
y_pred_rf_020 = (y_val_rf_proba >= 0.20).astype(int)

print(confusion_matrix(y_val, y_pred_rf_020))

# =========================
# Gradient Boosting
# =========================

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)

y_val_gb_pred = gb_model.predict(X_val)

print("Gradient Boosting Results")
print("Precision:", precision_score(y_val, y_val_gb_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_gb_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_gb_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_gb_pred))

y_val_gb_proba = gb_model.predict_proba(X_val)[:, 1]

thresholds = [0.5, 0.4, 0.3, 0.25, 0.2]

for threshold in thresholds:
    y_pred = (y_val_gb_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(y_val, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_val, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_val, y_pred, zero_division=0))
    
# =========================
# Gradient Boosting
# =========================

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

y_val_xgb_pred = xgb_model.predict(X_val)

print("XGBoost Results")
print("Precision:", precision_score(y_val, y_val_xgb_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_xgb_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_xgb_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_xgb_pred))


y_val_xgb_proba = xgb_model.predict_proba(X_val)[:, 1]

thresholds = [0.5, 0.4, 0.3, 0.25, 0.2]

for threshold in thresholds:
    y_pred = (y_val_xgb_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(y_val, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_val, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_val, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    
# =========================
# SVM
# =========================

svm_model = SVC(
    kernel="rbf",
    C=1.0,
    class_weight="balanced",
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

y_val_svm_pred = svm_model.predict(X_val_scaled)

print("SVM Results")
print("Precision:", precision_score(y_val, y_val_svm_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_svm_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_svm_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_svm_pred))

# =========================
# KNN
# =========================

knn_model = KNeighborsClassifier(
    n_neighbors=15,
    weights="distance",
    p=2,
    n_jobs=-1
)

knn_model.fit(X_train_scaled, y_train)

y_val_knn_pred = knn_model.predict(X_val_scaled)

print("KNN Results")
print("Precision:", precision_score(y_val, y_val_knn_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_knn_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_knn_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_knn_pred))

# =========================
# GaussianNB
# =========================

nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)

y_val_nb_pred = nb_model.predict(X_val_scaled)

print("Naive Bayes Results")
print("Precision:", precision_score(y_val, y_val_nb_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_nb_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_nb_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_nb_pred))

# ==========================
#  Extra Trees
# ==========================

et_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

et_model.fit(X_train, y_train)

y_val_et_pred = et_model.predict(X_val)

print("Extra Trees Results")
print("Precision:", precision_score(y_val, y_val_et_pred, zero_division=0))
print("Recall   :", recall_score(y_val, y_val_et_pred, zero_division=0))
print("F1 Score :", f1_score(y_val, y_val_et_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_et_pred))

#
models = {
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model
}

results = []

for model_name, model in models.items():
    probabilities = model.predict_proba(X_val)[:, 1]

    for threshold in np.arange(0.30, 0.501, 0.01):
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_val, predictions, zero_division=0
        )
        recall = recall_score(
            y_val, predictions, zero_division=0
        )
        f1 = f1_score(
            y_val, predictions, zero_division=0
        )

        results.append({
            "Model": model_name,
            "Threshold": round(threshold, 2),
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

results_df = pd.DataFrame(results)

best_results = (
    results_df.sort_values("F1", ascending=False)
    .groupby("Model", sort=False)
    .head(5)
    .sort_values(["Model", "F1"], ascending=[True, False])
)

print(best_results.to_string(index=False))

#

X_train_final = pd.concat([X_train, X_val], axis=0)
y_train_final = pd.concat([y_train, y_val], axis=0)

final_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_final, y_train_final)

y_test_proba = final_model.predict_proba(X_test)[:, 1]

final_threshold = 0.38
y_test_pred = (y_test_proba >= final_threshold).astype(int)

print("Final XGBoost Results")
print("Threshold:", final_threshold)
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_test_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_test_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

#

feature_importance = pd.DataFrame({
    "Feature": X_train_final.columns,
    "Importance": final_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importance.head(15))

top_features = feature_importance.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features - XGBoost")
plt.tight_layout()
plt.show()

#

import joblib
import os

os.makedirs("saved_models", exist_ok=True)

joblib.dump(final_model, "saved_models/xgb_model.pkl")

joblib.dump(
    X_train_final.columns.tolist(),
    "saved_models/feature_columns.pkl"
)

joblib.dump(0.38, "saved_models/threshold.pkl")

print("Model and settings saved successfully!")



import joblib

loaded_model = joblib.load("saved_models/xgb_model.pkl")
loaded_columns = joblib.load("saved_models/feature_columns.pkl")
loaded_threshold = joblib.load("saved_models/threshold.pkl")

print("Model loaded:", type(loaded_model).__name__)
print("Number of features:", len(loaded_columns))
print("Threshold:", loaded_threshold)


sample = X_test.iloc[[0]]

probability = loaded_model.predict_proba(sample)[0, 1]
prediction = int(probability >= loaded_threshold)

print("Churn probability:", round(probability, 3))
print("Predicted churn:", prediction)