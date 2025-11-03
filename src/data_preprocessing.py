"""
data_preprocessing.py
---------------------
Cleans and preprocesses the raw Glassdoor salary dataset.

Input:  data/raw/eda_data.csv
Output: data/processed/cleaned_salary_data.csv
Logs:   logs/data_cleaning.log
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime

# ----------------------------------------------------------
# 0. Setup Logging
# ----------------------------------------------------------
os.makedirs("logs", exist_ok=True)
log_file = f"logs/data_cleaning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)


# ----------------------------------------------------------
# 1. Load Data
# ----------------------------------------------------------
def load_data(path='data/raw/eda_data.csv'):
    logging.info(f"Loading raw dataset from {path}")
    df = pd.read_csv(path)
    logging.info(f"✅ Loaded dataset with {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ----------------------------------------------------------
# 2. Cleaning Helpers
# ----------------------------------------------------------
def size_to_num(x):
    """Convert company size text to approximate employee count."""
    if isinstance(x, str):
        x = x.lower()
        if 'to' in x:
            try:
                low, high = x.split('to')
                return (int(low.strip()) + int(high.split()[0])) // 2
            except:
                return np.nan
        elif '10000+' in x:
            return 10000
    return np.nan


def rev_to_num(x):
    """Convert revenue ranges like '$1 to $2 billion' → numeric millions."""
    if not isinstance(x, str):
        return np.nan
    x = x.lower()
    try:
        if 'billion' in x:
            nums = [float(s.replace('$','').replace('billion','')) 
                    for s in x.split('to') if '$' in s]
            return np.mean(nums)*1000  # billion → million
        elif 'million' in x:
            nums = [float(s.replace('$','').replace('million','')) 
                    for s in x.split('to') if '$' in s]
            return np.mean(nums)
    except:
        return np.nan
    return np.nan


# ----------------------------------------------------------
# 3. Cleaning & Feature Engineering
# ----------------------------------------------------------
def clean_data(df: pd.DataFrame):
    df = df.copy()
    logging.info("🔧 Starting data cleaning...")

    # Drop irrelevant columns
    drop_cols = [
        'Unnamed: 0', 'Salary Estimate', 'Job Description',
        'Competitors', 'company_txt', 'Company Name'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')
    logging.info(f"Dropped irrelevant columns: {drop_cols}")

    # Normalize text and replace -1 with NaN
    df.replace(-1, np.nan, inplace=True)
    df = df.applymap(lambda s: s.lower().strip() if isinstance(s, str) else s)

    # Company size numeric
    if 'size' in df.columns:
        df['company_size'] = df['size'].apply(size_to_num)
        df.drop('size', axis=1, inplace=True)
        logging.info("Converted 'size' → 'company_size' numeric midpoint")

    # Company age
    if 'founded' in df.columns:
        df['company_age'] = 2025 - df['founded']
        df.drop('founded', axis=1, inplace=True)
        logging.info("Created 'company_age' from 'founded'")

    # Revenue numeric
    if 'revenue' in df.columns:
        df['revenue_million'] = df['revenue'].apply(rev_to_num)
        df.drop('revenue', axis=1, inplace=True)
        logging.info("Converted 'revenue' → numeric 'revenue_million' (in millions)")

    # Missing summary before encoding
    missing_summary = df.isna().sum()
    logging.info(f"Missing values before encoding:\n{missing_summary[missing_summary>0]}")

    # One-hot encode categorical columns
    cat_cols = [
        'job_simp', 'seniority', 'job_state',
        'type_of_ownership', 'industry', 'sector'
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    logging.info(f"One-hot encoded categorical features: {cat_cols}")

    # Scale numeric features
    num_cols = [
        c for c in ['rating', 'age', 'company_age', 'company_size',
                    'revenue_million', 'desc_len', 'num_comp',
                    'min_salary', 'max_salary']
        if c in df.columns
    ]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    os.makedirs('app', exist_ok=True)
    joblib.dump(scaler, 'app/scaler.pkl')
    logging.info(f"Scaled numeric columns and saved scaler.pkl to /app")

    # Log dataset overview
    logging.info(f"✅ Cleaned data shape: {df.shape}")
    logging.info(f"Numeric columns scaled: {num_cols}")
    logging.info(f"Binary skill flags kept: {[c for c in df.columns if c.endswith('_yn')]}")

    return df


# ----------------------------------------------------------
# 4. Main Runner
# ----------------------------------------------------------
def main():
    raw_path = 'data/raw/eda_data.csv'
    processed_path = 'data/processed/cleaned_salary_data.csv'
    os.makedirs('data/processed', exist_ok=True)

    df = load_data(raw_path)
    df_cleaned = clean_data(df)
    df_cleaned.to_csv(processed_path, index=False)
    logging.info(f"✅ Cleaned data saved to {processed_path}")
    logging.info("Data preprocessing completed successfully!")


if __name__ == '__main__':
    main()

