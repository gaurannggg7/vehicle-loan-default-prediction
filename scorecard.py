import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from preprocess import load_and_preprocess_data

print("🎯 Creating GLS Credit Scorecard Engine...")

# 1. Reload and re-train our baseline Logistic Regression model
DATA_PATH = "data/train.csv"
X, y = load_and_preprocess_data(DATA_PATH)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
numeric_cols = ["LTV", "PERFORM_CNS_SCORE", "DISBURSED_AMOUNT", "PRI_CURRENT_BALANCE", "NO_OF_INQUIRIES"]

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])

lr_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 2. Scorecard Scaling Parameters
# Target score of 600 at 1:1 odds, and odds double every 20 points (PDO)
factor = 20 / np.log(2)
offset = 600 - factor * np.log(1)  # log(1) is 0, so offset = 600

# 3. Calculate Scores for the Validation Set
# Extract the log-odds (logit) directly from the model decision function
log_odds = lr_model.decision_function(X_val_scaled)

# Convert log-odds to the 300-850 scale
# Note: Since higher decision_function means higher probability of default (1), 
# we subtract it so higher credit scores represent LOWER risk of default.
val_scores = offset - (factor * log_odds)

# Clip scores to ensure they stay strictly within the classic 300 - 850 range
val_scores = np.clip(val_scores, 300, 850)

# 4. Create a Summary Results DataFrame
scorecard_df = X_val.copy()
scorecard_df['ACTUAL_DEFAULT'] = y_val
scorecard_df['CREDIT_SCORE'] = val_scores.astype(int)

# 5. Bin Applicants into Subprime-Optimized Risk Tiers
def assign_risk_tier(score):
    if score >= 620: return "Tier 1 (Subprime Elite - Low Risk)"
    elif score >= 600: return "Tier 2 (Subprime Standard)"
    elif score >= 580: return "Tier 3 (Subprime Hardened)"
    else: return "Tier 4 (Deep Subprime - High Risk)"

scorecard_df['RISK_TIER'] = scorecard_df['CREDIT_SCORE'].apply(assign_risk_tier)

# 6. Print the Underwriting Business Summary
print("\n==================================================")
print("        GLS OPTIMIZED SUBPRIME PORTFOLIO SUMMARY   ")
print("==================================================")
tier_summary = scorecard_df.groupby('RISK_TIER', observed=False).agg(
    applicant_count=('CREDIT_SCORE', 'count'),
    average_score=('CREDIT_SCORE', 'mean'),
    observed_default_rate=('ACTUAL_DEFAULT', 'mean')
).sort_values(by='average_score', ascending=False)

tier_summary['observed_default_rate'] = (tier_summary['observed_default_rate'] * 100).round(2).astype(str) + '%'
print(tier_summary)
# 7. EXPORT UNDERWRITING DECISION LOG FOR BUSINESS OPERATIONS
# Map risk tiers to specific automated business actions
action_map = {
    "Tier 1 (Subprime Elite - Low Risk)": "AUTO-APPROVE (Standard Rate)",
    "Tier 2 (Subprime Standard)": "AUTO-APPROVE (Add 1.5% Risk Premium)",
    "Tier 3 (Subprime Hardened)": "MANUAL REVIEW (Require Proof of Income)",
    "Tier 4 (Deep Subprime - High Risk)": "AUTO-DECLINE (Exceeds Risk Tolerance)"
}

scorecard_df['UNDERWRITING_ACTION'] = scorecard_df['RISK_TIER'].map(action_map)

# Select clean columns for the business team to read
final_export = scorecard_df[[
    "DISBURSED_AMOUNT", "LTV", "PERFORM_CNS_SCORE", 
    "NO_OF_INQUIRIES", "CREDIT_SCORE", "RISK_TIER", "UNDERWRITING_ACTION"
]]

output_path = "data/gls_underwriting_decisions.csv"
final_export.to_csv(output_path, index=False)

print(f"\n✅ Success! Business-ready decision log exported to: {output_path}")
print("   Ready for cross-functional review with Risk and Dealership Relations teams.")
print("==================================================")