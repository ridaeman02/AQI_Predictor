import os
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

LATITUDE = 31.5204
LONGITUDE = 74.3587

end_time = datetime.now(timezone.utc)
start_time = end_time - timedelta(hours=24)

start_timestamp = int(start_time.timestamp())
end_timestamp = int(end_time.timestamp())

URL = "https://api.openweathermap.org/data/3.0/onecall/timemachine"

params = {
    "lat": LATITUDE,
    "lon": LONGITUDE,
    "dt": end_timestamp,
    "appid": API_KEY,
    "units": "metric",
}

response = requests.get(
    URL,
    params=params,
    timeout=30
)

print("Status code:", response.status_code)

if response.status_code == 200:
    data = response.json()

    print("Historical weather data received!")

    if "data" in data:
        print("Number of records:", len(data["data"]))

        if data["data"]:
            weather = data["data"][0]

            print("Temperature:", weather.get("temp"))
            print("Humidity:", weather.get("humidity"))
            print("Wind speed:", weather.get("wind_speed"))
else:
    print("Weather API error:")
    print(response.text)