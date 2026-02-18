"""
Feature pipeline to fetch weather and AQI data, process it, and insert into Hopsworks Feature Store.
Includes robust error handling, data type enforcement, auto-backfill logic, and singleton connection.
"""
import sys
import os
import time
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.features.utils import fetch_weather_data, fetch_historical_weather, process_data, engineer_features
from src.hopsworks_api import get_project

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")
LAT = os.getenv("LATITUDE")
LON = os.getenv("LONGITUDE")

if not LAT or not LON:
    raise ValueError("LATITUDE and LONGITUDE environment variables must be set")

try:
    LAT = float(LAT)
    LON = float(LON)
except ValueError:
    raise ValueError("LATITUDE and LONGITUDE must be valid numeric values")

def run_feature_pipeline():
    print(f"Fetching CURRENT data for location: {LAT}, {LON}...")
    current_df = pd.DataFrame()
    try:
        # Fetch raw data
        weather_data, aqi_data = fetch_weather_data(LAT, LON)
        if weather_data and aqi_data:
            current_df = process_data(weather_data, aqi_data)
            current_df = engineer_features(current_df)
            
            # Enforce types
            if "datetime" in current_df.columns:
                current_df["datetime"] = pd.to_datetime(current_df["datetime"])
            if "unix_time" in current_df.columns:
                current_df["unix_time"] = current_df["unix_time"].astype(int)

            print(f"Current data prepared. Shape: {current_df.shape}")
        else:
            print("⚠️ Failed to fetch current data.")
    except Exception as e:
        print(f"❌ Error processing current data: {e}")

    # Connect to Hopsworks using Singleton
    project = get_project()
    if not project:
        return

    try:
        from hsfs.feature import Feature
        fs = project.get_feature_store()

        # Get or Create Feature Group
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_features",
            version=1,
            primary_key=["unix_time"],
            description="AQI and Weather features for Karachi",
            online_enabled=True,
            event_time="datetime",
            features=[Feature("unix_time", "bigint"),]
        )
        print("✅ Feature Group retrieved/created.")

        # Check for Auto-Backfill necessity
        backfill_needed = False
        try:
            # Check existing count by reading a small sample
            # This is safer than count() which might be slow on large datasets
            # If FG is empty, read() returns empty DF immediately.
            try:
                # Try reading using Hive (offline) to check total volume if possible, or usually just check online?
                # Online is faster but only has latest.
                # Use query to read 10 rows.
                query = aqi_fg.select(["unix_time"])
                sample = query.read(read_options={"limit": 10}) # Defaults to offline?
                if len(sample) < 10:
                    print(f"⚠️ Feature Group has only {len(sample)} rows. Triggering Auto-Backfill.")
                    backfill_needed = True
                else:
                    print(f"✅ Feature Group has sufficient data ({len(sample)}+ rows).")
            except:
                # If read fails, assume empty or issues
                print("⚠️ Could not read existing data. Triggering Auto-Backfill.")
                backfill_needed = True

        except Exception as e:
            print(f"Error checking FG status: {e}")
            backfill_needed = True

        # Perform Backfill if needed
        if backfill_needed:
            print("🔄 Fetching 90 days of historical data...")
            hist_df = fetch_historical_weather(LAT, LON, days=90)
            if not hist_df.empty:
                hist_df = engineer_features(hist_df)
                # Deduplicate current from history if overlap
                if not current_df.empty:
                    # history ends normally 5 days ago (archive), but standard API gives recent history.
                    # fetch_historical_weather ends "yesterday".
                    # current_df is "now".
                    # Should be fine to insert both.
                    pass
                
                print(f"Inserting {len(hist_df)} historical rows...")
                aqi_fg.insert(hist_df, write_options={"wait_for_job": True})
                print("✅ Backfill complete.")
            else:
                print("❌ Failed to fetch history.")

        # Insert Current Data
        if not current_df.empty:
            print("Inserting current data...")
            aqi_fg.insert(current_df, write_options={"wait_for_job": True})
            print("✅ Current data inserted.")

    except Exception as e:
        print(f"❌ Error interacting with Hopsworks: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_feature_pipeline()
