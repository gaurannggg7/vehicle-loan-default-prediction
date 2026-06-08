import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scorecard import scorecard_df  # Import the scored dataframe directly

print("📊 Generating Executive Risk Dashboard...")

# Set the plotting style for a clean, corporate look
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})

# Create a 1-row, 2-column layout for our dashboard
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -------------------------------------------------------------
# VISUAL 1: Distribution of GLS Credit Scores (Volume)
# -------------------------------------------------------------
sns.histplot(
    data=scorecard_df, 
    x="CREDIT_SCORE", 
    hue="RISK_TIER", 
    element="step", 
    stat="count", 
    palette="viridis",
    ax=axes[0],
    legend=True
)
axes[0].set_title("GLS Applicant Volume by Credit Score", fontweight='bold', pad=15)
axes[0].set_xlabel("Custom GLS Credit Score (300 - 850)")
axes[0].set_ylabel("Number of Applicants")

# -------------------------------------------------------------
# VISUAL 2: Observed Default Rate by Risk Tier (Risk Profile)
# -------------------------------------------------------------
# Calculate the percentage default rates per tier
tier_data = scorecard_df.groupby('RISK_TIER', observed=False)['ACTUAL_DEFAULT'].mean().reset_index()
tier_data['DEFAULT_RATE_PCT'] = tier_data['ACTUAL_DEFAULT'] * 100

# Sort from lowest risk to highest risk
tier_order = [
    "Tier 1 (Subprime Elite - Low Risk)", 
    "Tier 2 (Subprime Standard)", 
    "Tier 3 (Subprime Hardened)", 
    "Tier 4 (Deep Subprime - High Risk)"
]

sns.barplot(
    data=tier_data, 
    x="DEFAULT_RATE_PCT", 
    y="RISK_TIER", 
    order=tier_order,
    palette="Reds_r", 
    ax=axes[1]
)

# Add value labels to the ends of the bars
for index, row in tier_data.iterrows():
    # Find the corresponding position in the ordered list
    try:
        y_pos = tier_order.index(row['RISK_TIER'])
        axes[1].text(
            row['DEFAULT_RATE_PCT'] + 0.5, 
            y_pos, 
            f"{row['DEFAULT_RATE_PCT']:.1f}%", 
            va='center', 
            fontweight='bold'
        )
    except ValueError:
        continue

axes[1].set_title("Observed Default Rate by Risk Tier", fontweight='bold', pad=15)
axes[1].set_xlabel("Default Rate (%)")
axes[1].set_ylabel("")
axes[1].set_xlim(0, 40) # Give room for the text labels

# Final adjustments and save
plt.tight_layout()
output_image = "data/gls_risk_dashboard.png"
plt.savefig(output_image, dpi=300)
print(f"✅ Success! Executive dashboard saved as an image to: {output_image}")
plt.show()