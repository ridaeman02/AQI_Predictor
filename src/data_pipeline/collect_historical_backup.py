import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Five major cities of Pakistan
CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}

OUTPUT_FILE = "data/historical_data.csv"

# Get the last 24 hours
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=24)

start_timestamp = int(start_time.timestamp())
end_timestamp = int(end_time.timestamp())

URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"


def collect_historical_data():
    all_records = []

    for city, coordinates in CITIES.items():

        latitude, longitude = coordinates

        print(f"\nCollecting historical data for {city}...")

        params = {
            "lat": latitude,
            "lon": longitude,
            "start": start_timestamp,
            "end": end_timestamp,
            "appid": API_KEY,
        }

        try:
            response = requests.get(
                URL,
                params=params,
                timeout=30
            )

            if response.status_code == 200:

                data = response.json()

                print(
                    f"✓ {city}: "
                    f"{len(data.get('list', []))} hourly records received"
                )

                for record in data.get("list", []):

                    components = record["components"]

                    all_records.append({
                        "city": city,
                        "timestamp": datetime.fromtimestamp(
                            record["dt"],
                            timezone.utc
                        ).isoformat(),

                        "aqi": record["main"]["aqi"],

                        "pm2_5": components.get("pm2_5"),
                        "pm10": components.get("pm10"),
                        "co": components.get("co"),
                        "no2": components.get("no2"),
                        "o3": components.get("o3"),
                        "so2": components.get("so2"),
                        "nh3": components.get("nh3"),
                        "no": components.get("no"),
                    })

            else:
                print(
                    f"✗ {city}: API error "
                    f"{response.status_code}"
                )
                print(response.text)

        except requests.exceptions.RequestException as error:
            print(f"✗ {city}: Request failed")
            print(error)

        # Small delay between API requests
        time.sleep(1)

    # Convert collected data into a DataFrame
    if all_records:

        df = pd.DataFrame(all_records)

        # Make sure the data folder exists
        os.makedirs("data", exist_ok=True)

        # Save to CSV
        df.to_csv(OUTPUT_FILE, index=False)

        print("\n" + "=" * 50)
        print("HISTORICAL DATA COLLECTION COMPLETE")
        print("=" * 50)

        print(f"Total records: {len(df)}")
        print(f"Cities: {df['city'].nunique()}")
        print(f"Saved to: {OUTPUT_FILE}")

        print("\nRecords per city:")
        print(df["city"].value_counts())

    else:
        print("\nNo historical data was collected.")


if __name__ == "__main__":
    collect_historical_data()