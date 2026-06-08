# Vehicle Loan Default Prediction & Credit Risk Scorecard

End-to-end credit risk scoring pipeline for subprime auto lending. Ingests raw applicant data, engineers credit features, trains predictive models, and scales outputs into an operational 300–850 scorecard with automated underwriting decisions.

**Dataset:** [LT Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction) — 233,154 loan records, 41 features

---

## Key Findings

- **Thin-file concentration:** 50% of applicants (116,950) had no prior credit bureau history — isolated via engineered `NO_BUREAU_HISTORY` binary flag to prevent model conflation with poor credit
- **Non-linear LTV risk:** Default rates peak at the 80–90% LTV bracket (25.9%) rather than scaling linearly with leverage
- **Class imbalance:** 21.7% baseline default rate handled via balanced class weighting (Logistic Regression) and dynamic `scale_pos_weight` (XGBoost)
- **Scorecard tiers:** Custom subprime bins yielding observed default rates from 11.1% (Subprime Elite) to 32.9% (Deep Subprime)

---

## Pipeline
eda.py          → Exploratory analysis, distribution plots, LTV bucketing
preprocess.py   → Null imputation, feature engineering, encoding, scaling
model.py        → Logistic Regression baseline + XGBoost champion model
scorecard.py    → Log-odds → 300–850 score scaling via PDO logic
visuals.py      → Executive dashboard + underwriting decision log export
---

## Model Performance

| Model | ROC-AUC | Default Recall |
|---|---|---|
| Logistic Regression | 0.600 | 0.62 |
| XGBoost | 0.624 | 0.64 |

---

## Scorecard Tiers

| Tier | Score Range | Observed Default Rate | Decision |
|---|---|---|---|
| Subprime Elite | ≥ 620 | 11.1% | Auto-Approve |
| Subprime Standard | 600–619 | ~18% | Risk Premium Pricing |
| Subprime Hardened | 580–599 | ~26% | Manual Verification |
| Deep Subprime | < 580 | 32.9% | Auto-Decline |

---

## Scorecard Scaling

Uses industry-standard **Points to Double the Odds (PDO)** methodology:
- Base score: 600 (1:1 odds of default)
- PDO factor: 20
- Score range: 300–850

---

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn

---

## Setup

```bash
git clone https://github.com/gaurannggg7/vehicle-loan-default-prediction.git
cd vehicle-loan-default-prediction
pip install -r requirements.txt
```

Download `train.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/avikpaul4u/vehicle-loan-default-prediction) and place it in the `archive/` directory, then run each script in order.
