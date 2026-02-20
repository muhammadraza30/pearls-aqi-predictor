import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features from the datetime column:
      - unix_time  (int64, seconds since epoch)
      - hour       (0-23)
      - day        (1-31)
      - month      (1-12)
      - weekday    (0=Monday, 6=Sunday)

    Also forward/back-fills any remaining NaN values in numeric columns.
    """
    if df.empty:
        return df

    df = df.copy()
    
    # Ensure datetime is proper type
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Unix timestamp in seconds
    df["unix_time"] = (df["datetime"].astype("int64") // 10**9).astype("int64")
    df = df.sort_values("unix_time")
    # Time-based features
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.dayofweek
     # Lagged features
    df["aqi_lag_1"] = df["aqi"].shift(1)
    df["aqi_lag_3"] = df["aqi"].shift(3)
    df["aqi_lag_6"] = df["aqi"].shift(6)
    df["aqi_lag_24"] = df["aqi"].shift(24)

    # Target: Next hour AQI
    df["target"] = df["aqi"].shift(-1)
    # Fill missing values in numeric columns
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    return df


# def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Creates lag features for time-series forecasting (used by training pipeline).

#       - aqi_lag_1   (1-hour lag)
#       - aqi_lag_3   (3-hour lag)
#       - aqi_lag_6   (6-hour lag)
#       - aqi_lag_24  (24-hour lag)
#       - target      (next-hour AQI)

#     Drops rows with NaN introduced by shifting.
#     """
#     if df.empty:
#         return df

#     df = df.copy()
#     df = df.sort_values("unix_time")

#     # Lagged features
#     df["aqi_lag_1"] = df["aqi"].shift(1)
#     df["aqi_lag_3"] = df["aqi"].shift(3)
#     df["aqi_lag_6"] = df["aqi"].shift(6)
#     df["aqi_lag_24"] = df["aqi"].shift(24)

#     # Target: Next hour AQI
#     df["target"] = df["aqi"].shift(-1)

#     return df.dropna()
