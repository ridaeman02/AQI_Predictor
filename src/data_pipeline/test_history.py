import os
import time
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LATITUDE = 31.5204
LONGITUDE = 74.3587


# Get the last 24 hours
end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=24)

start_timestamp = int(start_time.timestamp())
end_timestamp = int(end_time.timestamp())


url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

params = {
    "lat": LATITUDE,
    "lon": LONGITUDE,
    "start": start_timestamp,
    "end": end_timestamp,
    "appid": API_KEY,
}

response = requests.get(url, params=params, timeout=30)

print("Status code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    print("Historical data received successfully!")
    print("Number of hourly records:", len(data["list"]))

    if data["list"]:
        first_record = data["list"][0]

        print()
        print("Example record:")
        print("AQI:", first_record["main"]["aqi"])
        print("PM2.5:", first_record["components"]["pm2_5"])
        print("PM10:", first_record["components"]["pm10"])

else:

    print("Historical API error:")
    print(response.text)