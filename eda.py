import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/gaurangmohan/Desktop/AUto lOan credit /data/train.csv")


# 1. basic shape
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# 2. target distribution
print(df["LOAN_DEFAULT"].value_counts(normalize=True))

# 3. key numeric distributions
cols = ["LTV", "PERFORM_CNS_SCORE", "DISBURSED_AMOUNT", "PRI_CURRENT_BALANCE", "NO_OF_INQUIRIES"]
df[cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

# 4. default rate by LTV bucket
df["ltv_bucket"] = pd.cut(df["LTV"], bins=[0,60,80,90,100,120,200])
print(df.groupby("ltv_bucket")["LOAN_DEFAULT"].mean())