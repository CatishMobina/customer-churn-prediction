# Customer Churn Prediction

## Overview
This project uses machine learning to predict customer churn for a telecommunications company. The goal is to identify customers who are more likely to leave the service and help businesses make better customer-retention decisions.

## Problem Statement
Customer churn can affect business revenue and long-term growth. By analyzing customer demographics, service subscriptions, contract types, and billing information, this project builds a classification model to identify customers at risk of leaving.

## Dataset
The project uses a Telco Customer Churn dataset containing customer information, subscribed services, contract details, billing information, and churn labels.

The target variable is `Churn`:
- `0`: No churn
- `1`: Churn

## Workflow
1. Data cleaning and preparation
2. Exploratory Data Analysis (EDA)
3. Categorical feature encoding
4. Stratified train, validation, and test splitting
5. Training and comparing multiple classification models
6. Threshold selection using validation data
7. Final model training and test evaluation
8. Saving the trained model and prediction threshold

## Model
The final model is **XGBoost Classifier**.

- Number of estimators: 200
- Learning rate: 0.05
- Maximum depth: 3
- Subsample: 0.8
- Column subsample: 0.8
- Classification threshold: 0.38

The threshold was selected during validation to balance precision and recall. The test set was reserved for final evaluation.

## Test Results

| Metric | Score |
|---|---:|
| Accuracy | 78.57% |
| Precision | 58.07% |
| Recall | 69.25% |
| F1-score | 63.17% |

### Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 848 | 187 |
| Actual Churn | 115 | 259 |

The model identified 259 of the 374 customers who churned in the test set. It missed 115 churn cases, so its predictions should support—not replace—business decision-making.

## Feature Importance
The model assigned high importance to features including:
- Internet service type
- Contract type
- Backup and security service categories
- Payment method
- Customer tenure

Feature importance describes the model's predictive use of these variables; it does not establish causation.

## Project Structure

```text
customer-churn-prediction/
├── src/
│   ├── train_final_model.py
│   └── test_saved_model.py
├── notebooks/
├── saved_models/
├── data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Training

Run the final training and evaluation script from the project root:

```bash
python src/train_final_model.py
```

The script saves the trained model, feature column names, and classification threshold in the `saved_models/` directory.

## Verify Saved Artifacts

```bash
python src/test_saved_model.py
```

## Limitations
- Model performance depends on the dataset and may not generalize to other companies or time periods.
- False positives and missed churn cases can have different business costs.
- Feature importance should not be interpreted as causal evidence.
- The model is intended as a learning and decision-support project, not as a guarantee of customer behavior.

## Future Improvements
- Build a reusable preprocessing and prediction pipeline.
- Tune hyperparameters using cross-validation.
- Evaluate probability calibration and business-specific costs.
- Add automated tests and a reproducible environment.
- Develop a prediction interface after validating the model pipeline.

## Technologies
Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, and joblib.

## Disclaimer
This project is for educational and portfolio purposes.
