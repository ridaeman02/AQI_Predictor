import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LATITUDE = 31.5204
LONGITUDE = 74.3587

URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(days=7)

params = {
    "lat": LATITUDE,
    "lon": LONGITUDE,
    "start": int(start_time.timestamp()),
    "end": int(end_time.timestamp()),
    "appid": API_KEY,
}

print("=" * 60)
print("TESTING OPENWEATHER HISTORICAL AQI RANGE")
print("=" * 60)

print(f"Start: {start_time.isoformat()}")
print(f"End:   {end_time.isoformat()}")

try:

    response = requests.get(
        URL,
        params=params,
        timeout=60
    )

    print(f"\nStatus code: {response.status_code}")

    if response.status_code == 200:

        data = response.json()

        records = data.get("list", [])

        print("✓ Historical AQI request successful!")
        print(f"Records received: {len(records)}")

        if records:

            first = datetime.fromtimestamp(
                records[0]["dt"],
                timezone.utc
            )

            last = datetime.fromtimestamp(
                records[-1]["dt"],
                timezone.utc
            )

            print(f"First record: {first.isoformat()}")
            print(f"Last record:  {last.isoformat()}")

    else:

        print("✗ OpenWeather API returned an error.")
        print(response.text)

except Exception as error:

    print("✗ Request failed:")
    print(error)