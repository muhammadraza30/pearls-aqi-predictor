import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import requests

# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.features.data_fetching import fetch_current_data, fetch_historical_data, fetch_recent_data
from src.features.feature_engineering import engineer_features
from src.features.preprocessing import preprocess
from src.hopsworks_api import get_project

LAT = float(os.getenv("LATITUDE"))
LON = float(os.getenv("LONGITUDE"))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "data.csv")


def _enforce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct dtypes before Hopsworks insert."""
    # Convert to datetime and strip timezone (naive timestamp for Hopsworks)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df["unix_time"] = df["unix_time"].astype("int64")

    # Round all float columns to 1 decimal place
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].round(1)

    return df


def _cast_to_fg_schema(df: pd.DataFrame, fg) -> pd.DataFrame:
    """
    Casts DataFrame columns to match the Feature Group's expected types.
    Handles 'float' (32-bit) vs 'double' (64-bit) mismatches.
    """
    df_casted = df.copy()
    
    # Map Hopsworks types to Numpy/Pandas types
    type_mapping = {
        "float": np.float32,
        "double": np.float64,
        "bigint": np.int64,
        "int": np.int32,
        "smallint": np.int16,
        "tinyint": np.int8,
        "boolean": bool,
        "string": str,
    }

    print("🔧 Syncing DataFrame types with Feature Group schema...")
    for feature in fg.features:
        col_name = feature.name
        hs_type = feature.type.lower()
        
        if col_name in df_casted.columns:
            target_type = type_mapping.get(hs_type)
            
            if target_type:
                try:
                    # Special handling for nullable integers in float columns
                    if "int" in str(target_type) and df_casted[col_name].isnull().any():
                        print(f"⚠️ Column '{col_name}' contains NaNs, cannot cast to {target_type}. Filling with 0.")
                        df_casted[col_name].fillna(0, inplace=True)
                        
                    current_type = df_casted[col_name].dtype
                    # Cast if types look different (e.g. float64 vs float32)
                    if current_type != target_type:
                        df_casted[col_name] = df_casted[col_name].astype(target_type)
                except Exception as e:
                    print(f"⚠️ Failed to cast '{col_name}' to {target_type}: {e}")
            else:
                pass # Unknown type, skip casting
                
    return df_casted


def _save_to_csv(df: pd.DataFrame):
    """Append new rows to local CSV, avoiding duplicates."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # Format datetime as clean string for CSV (no tz, no microseconds)
    save_df = df.copy()
    save_df["datetime"] = pd.to_datetime(save_df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        # Remove rows already in CSV (by unix_time)
        new_rows = save_df[~save_df["unix_time"].isin(existing_df["unix_time"].values)]
        if new_rows.empty:
            print("📄 All records already exist in CSV. Skipping.")
            return
        new_rows.to_csv(CSV_PATH, mode="a", header=False, index=False)
        print(f"📄 Appended {len(new_rows)} new rows to CSV.")
    else:
        save_df.to_csv(CSV_PATH, mode="w", header=True, index=False)
        print(f"📄 Created CSV with {len(save_df)} rows.")


def run_feature_pipeline():
    """Main orchestrator: fetch → engineer → preprocess → upload."""

    # ──────────────── 1. CONNECT TO HOPSWORKS ────────────────
    project = get_project()
    if not project:
        print("❌ Could not connect to Hopsworks. Aborting.")
        return

    fs = project.get_feature_store()

    # Check if feature group already exists
    # Check if feature group exists
    try:
        aqi_fg = fs.get_feature_group("aqi_features", version=1)
        # Check if schema matches current dataframe columns
        # We need to fetch one row or check features to see if lags exist
        existing_features = [f.name for f in aqi_fg.features]
        required_features = ["aqi_lag_1", "target"] # Check key new features
        
        missing_features = [f for f in required_features if f not in existing_features]
        
        if missing_features:
            print(f"⚠️ Schema mismatch! Missing: {missing_features}")
            print("🗑️ Deleting old Feature Group to recreate with new schema...")
            aqi_fg.delete()
            fg_exists = False
            aqi_fg = None
        else:
            fg_exists = True
            print("✅ Feature Group 'aqi_features' found and schema matches.")
            
    except Exception:
        aqi_fg = None
        fg_exists = False
        print("ℹ️  Feature Group not found. Will backfill.")

    # ──────────────── 2. FETCH DATA ────────────────
    if fg_exists:
        # Fetch last 72 hours to ensure lags can be computed
        print(f"\n📡 Fetching RECENT data for ({LAT}, {LON})...")
        raw_df = fetch_recent_data(LAT, LON, past_days=3)
        if raw_df.empty:
            print("❌ Failed to fetch recent data. Aborting.")
            return
    else:
        # Backfill from 2025-08-01 to last complete hour
        print(f"\n📡 Fetching HISTORICAL data for ({LAT}, {LON})...")
        raw_df = fetch_historical_data(LAT, LON, start_date="2025-08-01")
        if raw_df.empty:
            print("❌ Failed to fetch historical data. Aborting.")
            return

    print(f"   Raw data shape: {raw_df.shape}")

    # ──────────────── 3. FEATURE ENGINEERING ────────────────
    print("\n⚙️  Applying feature engineering...")
    featured_df = engineer_features(raw_df)
    print(f"   Engineered data shape: {featured_df.shape}")

    # ──────────────── 4. PREPROCESSING ────────────────
    print("\n🧹 Applying preprocessing...")
    clean_df = preprocess(featured_df)
    clean_df = _enforce_types(clean_df)
    print(f"   Clean data shape: {clean_df.shape}")

    # ──────────────── 5. UPLOAD TO HOPSWORKS ────────────────
    if not fg_exists:
        # Create new feature group
        print("\n☁️  Creating Feature Group and inserting backfill data...")
        aqi_fg = fs.create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["unix_time"],
            event_time="datetime",
            online_enabled=True,
            description="AQI and Weather features for Karachi",
        )
        try:
            aqi_fg.insert(clean_df, write_options={"wait_for_job": True})
            print(f"✅ Backfill complete — {len(clean_df)} rows inserted.")
        except requests.exceptions.ConnectionError:
            print("⚠️  Insert triggered but connection dropped. Verify in Hopsworks UI.")
    else:
        # Insert current data into existing FG
        print("\n☁️  Inserting current data into Feature Group...")
        
        # Cast types to match schema (fix for float32/float64 mismatch)
        clean_df = _cast_to_fg_schema(clean_df, aqi_fg)
        
        try:
            aqi_fg.insert(clean_df, write_options={"wait_for_job": True})
            print(f"✅ Current data inserted — {len(clean_df)} rows.")
        except requests.exceptions.ConnectionError:
            print("⚠️  Insert triggered but connection dropped. Verify in Hopsworks UI.")

    # ──────────────── 6. SAVE TO LOCAL CSV ────────────────
    _save_to_csv(clean_df)

    print("\n🎉 Feature pipeline complete!")


if __name__ == "__main__":
    run_feature_pipeline()
