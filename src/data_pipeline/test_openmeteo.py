import requests

LATITUDE = 31.5204
LONGITUDE = 74.3587

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": "2026-08-10",
    "end_date": "2026-08-11",
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "timezone": "Asia/Karachi",
    "wind_speed_unit": "ms",
}

response = requests.get(
    url,
    params=params,
    timeout=30
)

print("Status code:", response.status_code)

if response.status_code == 200:

    data = response.json()

    print("Historical weather received successfully!")

    print(
        "Number of hourly records:",
        len(data["hourly"]["time"])
    )

    print("\nFirst record:")

    print(
        "Time:",
        data["hourly"]["time"][0]
    )

    print(
        "Temperature:",
        data["hourly"]["temperature_2m"][0],
        "°C"
    )

    print(
        "Humidity:",
        data["hourly"]["relative_humidity_2m"][0],
        "%"
    )

    print(
        "Wind speed:",
        data["hourly"]["wind_speed_10m"][0],
        "m/s"
    )

else:

    print("Open-Meteo error:")
    print(response.text)