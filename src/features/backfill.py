import sys
import os

# Add 'src' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import hopsworks
from dotenv import load_dotenv
from datetime import datetime
from src.features.utils import engineer_features

# Load environment variables
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

LAT = os.getenv("LATITUDE")
LON = os.getenv("LONGITUDE")

if not LAT or not LON:
    raise ValueError("LATITUDE and LONGITUDE environment variables must be set")

LAT = float(LAT)
LON = float(LON)


def fetch_historical_data(start_date, end_date):
    """
    Fetch historical weather + AQI data from Open-Meteo
    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
        ],
    }

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi",
        ],
    }

    weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
    aqi_responses = openmeteo.weather_api(aqi_url, params=aqi_params)

    # Weather processing
    response = weather_responses[0]
    hourly = response.Hourly()

    datetime_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    weather_df = pd.DataFrame({
        "datetime": datetime_index,
        "temp": hourly.Variables(0).ValuesAsNumpy(),
        "humidity": hourly.Variables(1).ValuesAsNumpy(),
        "pressure": hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed": hourly.Variables(3).ValuesAsNumpy(),
        "wind_deg": hourly.Variables(4).ValuesAsNumpy(),
        "clouds": hourly.Variables(5).ValuesAsNumpy(),
    })

    # AQI processing
    response = aqi_responses[0]
    hourly = response.Hourly()

    datetime_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    aqi_df = pd.DataFrame({
        "datetime": datetime_index,
        "pm10": hourly.Variables(0).ValuesAsNumpy(),
        "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
        "co": hourly.Variables(2).ValuesAsNumpy(),
        "no2": hourly.Variables(3).ValuesAsNumpy(),
        "so2": hourly.Variables(4).ValuesAsNumpy(),
        "o3": hourly.Variables(5).ValuesAsNumpy(),
        "aqi": hourly.Variables(6).ValuesAsNumpy(),
    })

    df = pd.merge(weather_df, aqi_df, on="datetime")
    df = df.ffill().bfill()

    df = engineer_features(df)

    # IMPORTANT: enforce correct types
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["unix_time"] = df["unix_time"].astype("int64")

    return df


def backfill_feature_store():
    start_date = "2025-08-01"
    end_date = datetime.now().strftime("%Y-%m-%d")

    df = fetch_historical_data(start_date, end_date)

    if df.empty:
        print("No data fetched.")
        return

    print(f"Fetched {len(df)} rows")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )

    fs = project.get_feature_store()

    # DELETE existing feature group completely (schema reset)
    try:
        fg = fs.get_feature_group("aqi_features", version=1)
        fg.delete()
        print("Deleted existing feature group.")
    except:
        pass

    # Create new feature group (schema inferred from df)
    aqi_fg = fs.create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["unix_time"],
        event_time="datetime",
        online_enabled=False,
        description="AQI and Weather features",
    )

    print("Inserting data...")
    aqi_fg.insert(df, write_options={"wait_for_job": True})
    print("Backfill complete.")


if __name__ == "__main__":
    backfill_feature_store()
