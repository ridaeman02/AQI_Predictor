import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta, timezone


CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}


URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------
# Historical period
# ---------------------------------------------------------
# Match the AQI historical dataset:
#
# AQI:
# 2026-07-24 00:00 UTC
# through
# 2026-08-23 23:00 UTC
#
# Open-Meteo archive uses calendar dates in the requested
# timezone. We request one additional day so that the final
# UTC hours are definitely available.
# ---------------------------------------------------------

START_DATE = datetime(2026, 7, 24)
END_DATE = datetime(2026, 8, 24)


all_data = []


for city, coordinates in CITIES.items():

    latitude, longitude = coordinates

    print(f"\nCollecting historical weather for {city}...")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE.strftime("%Y-%m-%d"),
        "end_date": END_DATE.strftime("%Y-%m-%d"),
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "wind_speed_10m"
        ),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }

    try:

        response = requests.get(
            URL,
            params=params,
            timeout=60
        )

        if response.status_code != 200:

            print(
                f"ERROR: {city}: "
                f"HTTP {response.status_code}"
            )

            print(response.text)

            continue

        data = response.json()

        hourly = data["hourly"]

        city_data = pd.DataFrame({
            "city": city,
            "timestamp": hourly["time"],
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "wind_speed": hourly["wind_speed_10m"],
        })

        # Convert timestamps to UTC-aware timestamps
        city_data["timestamp"] = pd.to_datetime(
            city_data["timestamp"],
            utc=True
        )

        # Keep exactly the same period as the AQI dataset
        start_timestamp = pd.Timestamp(
            "2026-07-24 00:00:00",
            tz="UTC"
        )

        end_timestamp = pd.Timestamp(
            "2026-08-23 23:00:00",
            tz="UTC"
        )

        city_data = city_data[
            (city_data["timestamp"] >= start_timestamp)
            & (city_data["timestamp"] <= end_timestamp)
        ]

        all_data.append(city_data)

        print(
            f"✓ {city}: "
            f"{len(city_data)} records collected"
        )

    except Exception as error:

        print(
            f"ERROR: {city}: "
            f"{error}"
        )

    time.sleep(1)


# ---------------------------------------------------------
# Validate collection
# ---------------------------------------------------------

if not all_data:

    print("\nNo weather data was collected.")
    raise SystemExit(1)


df = pd.concat(
    all_data,
    ignore_index=True
)


# Sort data
df = df.sort_values(
    ["city", "timestamp"]
).reset_index(drop=True)


# Remove accidental duplicates
df = df.drop_duplicates(
    subset=["city", "timestamp"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

os.makedirs("data", exist_ok=True)

output_file = "data/historical_weather.csv"

df.to_csv(
    output_file,
    index=False
)


# ---------------------------------------------------------
# Final validation
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("HISTORICAL WEATHER COLLECTION COMPLETED")
print("=" * 60)

print(
    f"Date range: "
    f"{df['timestamp'].min()} "
    f"to "
    f"{df['timestamp'].max()}"
)

print(f"Total records: {len(df)}")

print(f"Saved to: {output_file}")

print("\nRecords by city:")
print(df["city"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nColumns:")
print(df.columns.tolist())