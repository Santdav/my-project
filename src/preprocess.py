"""
preprocess.py
===========
Feature engineering pipeline for the Telco Customer Churn dataset.

Usage:
    python preprocess.py

Output:
    data/churn_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_PATH       = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR  = Path("data")
OUTPUT_PATH    = PROCESSED_DIR / "churn_features.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  Raw shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix TotalCharges — stored as string with whitespace for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Rows with tenure == 0 have NaN TotalCharges → impute with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop customerID — not a feature
    df = df.drop(columns=["customerID"])

    # Encode target: Yes → 1, No → 0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    print(f"[clean] Shape after cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 3. Engineer Features
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- 3.1 Numerical interactions ---

    # Monthly charges relative to tenure — high ratio = expensive & new
    df["charge_per_month_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Difference between monthly charges and average total-per-month
    # Captures whether customer is paying more than their historical average
    df["monthly_vs_avg"] = df["MonthlyCharges"] - df["charge_per_month_ratio"]

    # --- 3.2 Tenure features ---

    df["is_new_customer"]      = (df["tenure"] <= 12).astype(int)
    df["is_long_term_customer"] = (df["tenure"] >= 48).astype(int)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],   # ordinal encoding
        include_lowest=True,
    ).astype(int)

    # --- 3.3 Service count ---
    # How many optional add-on services does the customer have active?
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # No internet → no services are possible; flag separately
    df["has_internet"] = (df["InternetService"] != "No").astype(int)

    # --- 3.4 High-risk flags ---
    df["is_month_to_month"]    = (df["Contract"] == "Month-to-month").astype(int)
    df["is_electronic_check"]  = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_fiber_optic"]       = (df["InternetService"] == "Fiber optic").astype(int)
    df["no_online_security"]   = (df["OnlineSecurity"] == "No").astype(int)
    df["no_tech_support"]      = (df["TechSupport"] == "No").astype(int)

    # Composite high-risk score (sum of risk flags)
    risk_flags = [
        "is_month_to_month", "is_electronic_check", "is_fiber_optic",
        "no_online_security", "no_tech_support",
    ]
    df["risk_score"] = df[risk_flags].sum(axis=1)

    # --- 3.5 Paperless billing ---
    df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)

    print(f"[engineer] Shape after feature engineering: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. Encode Categoricals
# ---------------------------------------------------------------------------

# Binary yes/no columns → 0/1
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines",
]

# Nominal columns → one-hot encoded
OHE_COLS = [
    "InternetService", "Contract", "PaymentMethod",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binary mappings
    binary_map = {"Yes": 1, "No": 0, "Female": 1, "Male": 0,
                  "No phone service": 0, "No internet service": 0}
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(binary_map).fillna(df[col])

    # One-hot encode nominal columns
    df = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)

    # Cast all bool columns (from get_dummies) to int
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[encode] Shape after encoding: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 5. Final Validation
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> None:
    print("\n[validate] --- Final Dataset Report ---")
    print(f"  Shape        : {df.shape}")
    print(f"  Churn rate   : {df['Churn'].mean():.2%}")
    print(f"  Nulls        : {df.isna().sum().sum()}")
    print(f"  Dtypes       :\n{df.dtypes.value_counts()}")

    assert df.isna().sum().sum() == 0, "Dataset still has null values — check clean() or encode()."
    assert "Churn" in df.columns, "'Churn' target column missing."
    assert df["Churn"].nunique() == 2, "'Churn' should be binary (0/1)."

    print("\n[validate] All checks passed.\n")


# ---------------------------------------------------------------------------
# 6. Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> pd.DataFrame:
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    df = encode(df)
    validate(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[save]  Processed data saved to: {OUTPUT_PATH}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_pipeline()
    print(df.head())