import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows based on 'datetime' column, keeping the last occurrence.
    """
    if df.empty or "datetime" not in df.columns:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=["datetime"], keep="last")
    removed = before - len(df)
    if removed > 0:
        print(f"🧹 Removed {removed} duplicate rows.")
    return df.reset_index(drop=True)


def cap_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    Caps outliers using IQR-based method on numeric columns.
    Values below Q1 - factor*IQR are clipped to the lower bound.
    Values above Q3 + factor*IQR are clipped to the upper bound.

    Skips time-derived columns (unix_time, hour, day, month, weekday)
    since those are deterministic and should not be capped.
    """
    if df.empty:
        return df

    df = df.copy()

    # Columns to skip (time-derived + identifiers)
    skip_cols = {"unix_time", "hour", "day", "month", "weekday", "datetime"}
    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if col not in skip_cols
    ]

    capped_count = 0
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        capped_count += outliers

        df[col] = df[col].clip(lower=lower, upper=upper)

    if capped_count > 0:
        print(f"📊 Capped {capped_count} outlier values across {len(numeric_cols)} columns.")

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fills then back-fills remaining NaN values in numeric columns.
    """
    if df.empty:
        return df

    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns

    missing_before = df[numeric_cols].isna().sum().sum()
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    missing_after = df[numeric_cols].isna().sum().sum()

    filled = missing_before - missing_after
    if filled > 0:
        print(f"🔧 Filled {filled} missing values.")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Remove duplicates (by datetime)
      2. Cap outliers (IQR-based)
      3. Fill remaining missing values
    Returns cleaned DataFrame.
    """
    if df.empty:
        return df

    print("🔄 Running preprocessing pipeline...")
    df = remove_duplicates(df)
    df = cap_outliers(df)
    df = fill_missing(df)

    print("✅ Preprocessing complete.")
    return df


def fit_scaler(X_train, save_path: str = "models/scaler.pkl"):
    """
    Fits a StandardScaler on the training data and saves it to disk.
    Returns (scaler, X_train_scaled).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Ensure no NaNs or Infs
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("⚠️ X_train contains NaNs or Infs. Filling/Clipping...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=None, neginf=None)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, save_path)
    print(f"⚖️ Scaler fitted and saved to {save_path}")
    return scaler, X_train_scaled


def scale_features(X, scaler):
    """Transforms feature array using a previously fitted scaler."""
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        # print("⚠️ X contains NaNs or Infs during scaling. Filling/Clipping...")
        X = np.nan_to_num(X, nan=0.0, posinf=None, neginf=None)
        
    return scaler.transform(X)
