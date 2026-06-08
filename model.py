import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import json
from datetime import datetime

# Import the preprocessor function we built earlier
from preprocess import load_and_preprocess_data

print("🚀 Starting Model Training Pipeline...")

# 1. Load and split the data
DATA_PATH = "data/train.csv"
X, y = load_and_preprocess_data(DATA_PATH)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Scale numeric variables (Critical for Logistic Regression)
scaler = StandardScaler()
numeric_cols = ["LTV", "PERFORM_CNS_SCORE", "DISBURSED_AMOUNT", "PRI_CURRENT_BALANCE", "NO_OF_INQUIRIES"]

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

# -------------------------------------------------------------
# MODEL 1: Logistic Regression (Interpretable Credit Standard)
# -------------------------------------------------------------
print("\n--- Training Logistic Regression ---")
lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Evaluate Logistic Regression
lr_preds = lr_model.predict(X_val_scaled)
lr_probs = lr_model.predict_proba(X_val_scaled)[:, 1]

print("\n[Logistic Regression Results]")
print(f"ROC-AUC Score: {roc_auc_score(y_val, lr_probs):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, lr_preds))
print("\nClassification Report:")
print(classification_report(y_val, lr_preds))

# -------------------------------------------------------------
# MODEL 2: XGBoost (For Maximum Predictive Performance)
# -------------------------------------------------------------
print("\n--- Training XGBoost Classifier ---")
# Calculate scale_pos_weight to handle class imbalance: (negative instances / positive instances)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = neg_count / pos_count

xgb_model = XGBClassifier(
    scale_pos_weight=scale_weight,
    random_state=42,
    eval_metric='logloss',
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate XGBoost
xgb_preds = xgb_model.predict(X_val_scaled)
xgb_probs = xgb_model.predict_proba(X_val_scaled)[:, 1]

print("\n[XGBoost Results]")
print(f"ROC-AUC Score: {roc_auc_score(y_val, xgb_probs):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, xgb_preds))
print("\nClassification Report:")
print(classification_report(y_val, xgb_preds))

# -------------------------------------------------------------
# MODEL 3: LightGBM (Challenger Model)
# -------------------------------------------------------------
print("\n--- Training LightGBM Classifier ---")
lgbm = LGBMClassifier(
    scale_pos_weight=scale_weight,
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)
# Train using the scaled data for consistency
lgbm.fit(X_train_scaled, y_train)
lgbm_preds = lgbm.predict_proba(X_val_scaled)[:, 1]

print("\n[LightGBM Results]")
lgbm_auc = roc_auc_score(y_val, lgbm_preds)
print(f"ROC-AUC Score: {lgbm_auc:.4f}")

# -------------------------------------------------------------
# LOGGING: Save metrics for model monitoring
# -------------------------------------------------------------
# 1. Save the baseline AUC scores into variables
lr_auc = roc_auc_score(y_val, lr_probs)
xgb_auc = roc_auc_score(y_val, xgb_probs)

# 2. Log model performance for monitoring
model_log = {
    "timestamp": datetime.now().isoformat(),
    "logistic_regression_auc": round(lr_auc, 4),
    "xgboost_auc": round(xgb_auc, 4),
    "lightgbm_auc": round(lgbm_auc, 4),
    "val_size": len(y_val),
    "default_rate_val": round(y_val.mean(), 4)
}

with open("data/model_performance_log.json", "w") as f:
    json.dump(model_log, f, indent=2)

print("\n✅ Model performance logged to data/model_performance_log.json")