import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path):
    # 1. Load Data
    df = pd.read_csv(data_path)

    # Clean up column case issues if any (e.g., disbursed_amount vs DISBURSED_AMOUNT)
    df.columns = [col.upper() for col in df.columns]
    
    # 2. Handle Missing Data
    # Fill missing employment types with 'Unknown'
    df['EMPLOYMENT_TYPE'] = df['EMPLOYMENT_TYPE'].fillna('Unknown')
    
    # 3. Feature Engineering
    # Create a binary flag for people with no credit history (Score of 0)
    df['NO_BUREAU_HISTORY'] = np.where(df['PERFORM_CNS_SCORE'] == 0, 1, 0)
    
    # 4. Select Target and Core Features
    y = df['LOAN_DEFAULT']
    
    numeric_cols = ["LTV", "PERFORM_CNS_SCORE", "DISBURSED_AMOUNT", "PRI_CURRENT_BALANCE", "NO_OF_INQUIRIES"]
    X = df[numeric_cols].copy()
    X['NO_BUREAU_HISTORY'] = df['NO_BUREAU_HISTORY']
    
    # 5. One-Hot Encoding for Categorical Data (Employment Type)
    employment_dummies = pd.get_dummies(df['EMPLOYMENT_TYPE'], prefix='emp', drop_first=True)
    X = pd.concat([X, employment_dummies], axis=1)
    
    # Convert booleans from get_dummies to integers (0/1) for compatibility
    X = X.astype(float)
    
    return X, y

if __name__ == "__main__":
    # Using the relative path that worked perfectly for you
    DATA_PATH = "train.csv"
    
    print("Testing preprocessing pipeline...")
    X, y = load_and_preprocess_data(DATA_PATH)
    
    # Split into train/validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale continuous variables so Logistic Regression behaves nicely
    scaler = StandardScaler()
    numeric_cols = ["LTV", "PERFORM_CNS_SCORE", "DISBURSED_AMOUNT", "PRI_CURRENT_BALANCE", "NO_OF_INQUIRIES"]
    
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols] = scaler.transform(X_val[numeric_cols])
    
    print("\n--- Preprocessing Complete ---")
    print(f"Training features shape: {X_train_scaled.shape}")
    print(f"Validation features shape: {X_val_scaled.shape}")