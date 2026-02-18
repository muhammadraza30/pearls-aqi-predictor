"""
Feature pipeline to fetch weather and AQI data,
process it, and insert into Hopsworks Feature Store.
"""

import sys
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import requests
# Ensure project root in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.utils import (
    fetch_weather_data,
    fetch_historical_weather,
    process_data,
    engineer_features,
)

from src.hopsworks_api import get_project

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

LAT = float(os.getenv("LATITUDE"))
LON = float(os.getenv("LONGITUDE"))


def run_feature_pipeline():
    csv_path = "data/data.csv"  # ← move here
    os.makedirs("data", exist_ok=True)
    print(f"Fetching CURRENT data for location: {LAT}, {LON}...")
    current_df = pd.DataFrame()

    # ------------------- CURRENT DATA -------------------
    try:
        weather_data, aqi_data = fetch_weather_data(LAT, LON)

        if weather_data and aqi_data:
            current_df = process_data(weather_data, aqi_data)
            current_df = engineer_features(current_df)

            # Enforce types
            current_df["datetime"] = pd.to_datetime(current_df["datetime"])
            current_df["unix_time"] = current_df["unix_time"].astype("int64")

            print(f"Current data prepared. Shape: {current_df.shape}")
        else:
            print("⚠️ Failed to fetch current data.")

    except Exception as e:
        print(f"❌ Error processing current data: {e}")

    # ------------------- HOPSWORKS -------------------
    project = get_project()
    if not project:
        return

    fs = project.get_feature_store()

    # Try to get existing FG
    try:
        aqi_fg = fs.get_feature_group("aqi_features", version=1)
        print("✅ Feature Group loaded.")
    except:
        aqi_fg = None

    # ------------------- BACKFILL -------------------
    if aqi_fg is None:
        print("🔄 Feature Group not found. Creating and backfilling from 1 Aug 2025...")

        start_date = "2025-08-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        hist_df = fetch_historical_weather(LAT, LON)

        # If your util uses days param, override manually:
        if hist_df.empty:
            print("❌ Historical fetch failed.")
            return

        hist_df = engineer_features(hist_df)

        hist_df["datetime"] = pd.to_datetime(hist_df["datetime"])
        hist_df["unix_time"] = hist_df["unix_time"].astype("int64")

        # Save historical data to CSV
        hist_df.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"✅ Saved {len(hist_df)} historical rows to CSV.")

        # Create Feature Group using dataframe schema
        aqi_fg = fs.create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["unix_time"],
            event_time="datetime",
            online_enabled=True,
            description="AQI and Weather features for Karachi",
        )

        print(f"Inserting {len(hist_df)} historical rows...")
        try:
            aqi_fg.insert(hist_df, write_options={"wait_for_job": False})
            print("✅ Backfill complete.")
        except requests.exceptions.ConnectionError:
            print("⚠️ Insert triggered but connection dropped. Verify in UI.")

    # ------------------- INSERT CURRENT -------------------
    if not current_df.empty:
        print("Inserting current data...")
        try:
            aqi_fg.insert(current_df, write_options={"wait_for_job": False})
            print("✅ Current data inserted.")
        except requests.exceptions.ConnectionError:
            print("⚠️ Insert triggered but connection dropped. Verify in UI.")

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if current_df['unix_time'].iloc[0] in existing_df['unix_time'].values:
            print("Record already exists. Skipping.")
        else:
            current_df.to_csv(csv_path, mode='a', header=False, index=False)
            print("Record added.")
    else:
        # File doesn't exist yet, write with header
        current_df.to_csv(csv_path, mode='w', header=True, index=False)
        print("CSV created and record added.")

if __name__ == "__main__":
    run_feature_pipeline()
