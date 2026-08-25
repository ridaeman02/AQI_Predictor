import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv


# Load API key from .env
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY was not found in .env")


# Five major cities of Pakistan
CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}


def fetch_weather(city, latitude, longitude):
    """Fetch current weather data for a city."""

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": API_KEY,
        "units": "metric",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    return {
        "city": city,
        "timestamp": datetime.now().isoformat(),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
    }


def fetch_air_quality(city, latitude, longitude):
    """Fetch current air-quality data for a city."""

    url = "https://api.openweathermap.org/data/2.5/air_pollution"

    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    air_data = data["list"][0]

    return {
        "aqi": air_data["main"]["aqi"],
        "pm2_5": air_data["components"]["pm2_5"],
        "pm10": air_data["components"]["pm10"],
        "co": air_data["components"]["co"],
        "no2": air_data["components"]["no2"],
        "o3": air_data["components"]["o3"],
        "so2": air_data["components"]["so2"],
    }


def collect_data():
    """Collect weather and air-quality data for all five cities."""

    records = []

    for city, (latitude, longitude) in CITIES.items():

        print(f"Collecting data for {city}...")

        try:
            weather = fetch_weather(city, latitude, longitude)
            air_quality = fetch_air_quality(city, latitude, longitude)

            record = {
                **weather,
                **air_quality,
                "latitude": latitude,
                "longitude": longitude,
            }

            records.append(record)

            print(f"✓ {city} data collected successfully")

        except requests.RequestException as error:
            print(f"✗ Error collecting data for {city}: {error}")

    return records


def save_data(records):
    """Save collected data to a CSV file."""

    if not records:
        print("No data was collected.")
        return

    df = pd.DataFrame(records)

    # Make sure the data folder exists
    os.makedirs("data", exist_ok=True)

    output_file = "data/raw_data.csv"

    df.to_csv(output_file, index=False)

    print()
    print(f"Data saved to: {output_file}")
    print(f"Number of records: {len(df)}")


if __name__ == "__main__":

    print("=" * 50)
    print("AQI DATA COLLECTION PIPELINE")
    print("=" * 50)

    records = collect_data()

    save_data(records)

    print()
    print("Data collection completed!")