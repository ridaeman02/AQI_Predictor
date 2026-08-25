import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}


URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

OUTPUT_FILE = "data/historical_data.csv"


# Match the historical weather period
START_DATE = "2026-07-24"
END_DATE = "2026-08-23"


# ---------------------------------------------------------
# Convert date to UTC timestamp
# ---------------------------------------------------------

def date_to_timestamp(date_string, end_of_day=False):

    if end_of_day:

        dt = datetime.strptime(
            date_string + " 23:59:59",
            "%Y-%m-%d %H:%M:%S"
        )

    else:

        dt = datetime.strptime(
            date_string + " 00:00:00",
            "%Y-%m-%d %H:%M:%S"
        )

    dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp())


# ---------------------------------------------------------
# Collect AQI data
# ---------------------------------------------------------

def collect_historical_data():

    if not API_KEY:

        raise ValueError(
            "OPENWEATHER_API_KEY was not found in .env"
        )


    start_timestamp = date_to_timestamp(
        START_DATE
    )

    end_timestamp = date_to_timestamp(
        END_DATE,
        end_of_day=True
    )


    print("=" * 60)
    print("HISTORICAL AQI DATA COLLECTION")
    print("=" * 60)

    print(f"Date range: {START_DATE} → {END_DATE}")
    print(f"Cities: {len(CITIES)}")


    all_records = []


    for city, coordinates in CITIES.items():

        latitude, longitude = coordinates

        print(f"\nCollecting historical AQI for {city}...")


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
                timeout=60
            )


            if response.status_code != 200:

                print(
                    f"✗ {city}: API error "
                    f"{response.status_code}"
                )

                print(response.text)

                continue


            data = response.json()

            records = data.get("list", [])


            print(
                f"✓ {city}: "
                f"{len(records)} hourly records received"
            )


            for record in records:

                components = record.get(
                    "components",
                    {}
                )


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


        except requests.exceptions.RequestException as error:

            print(
                f"✗ {city}: Request failed"
            )

            print(error)


        # Avoid sending requests too quickly
        time.sleep(2)


    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    if not all_records:

        print("\n✗ No historical AQI data was collected.")

        return


    df = pd.DataFrame(all_records)


    # -----------------------------------------------------
    # Remove duplicate city/timestamp records
    # -----------------------------------------------------

    df = df.drop_duplicates(
        subset=["city", "timestamp"]
    )


    # -----------------------------------------------------
    # Sort data
    # -----------------------------------------------------

    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)


    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    os.makedirs(
        "data",
        exist_ok=True
    )


    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("HISTORICAL AQI DATA COLLECTION COMPLETED")
    print("=" * 60)

    print(f"Date range: {START_DATE} → {END_DATE}")

    print(
        f"Total records: {len(df)}"
    )

    print(
        f"Cities: {df['city'].nunique()}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


    print("\nRecords by city:")

    print(
        df["city"].value_counts()
    )


    print("\nColumns:")

    print(
        df.columns.tolist()
    )


if __name__ == "__main__":

    collect_historical_data()