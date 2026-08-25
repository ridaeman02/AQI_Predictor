import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


if not API_KEY:
    print("ERROR: OPENWEATHER_API_KEY was not found in .env")
    exit()


for city, (lat, lon) in CITIES.items():

    print("\n" + "=" * 50)
    print(f"City: {city}")
    print("=" * 50)

    # Get weather
    weather_params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }

    weather_response = requests.get(
        WEATHER_URL,
        params=weather_params
    )

    # Get air pollution
    air_params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }

    air_response = requests.get(
        AIR_URL,
        params=air_params
    )

    print("Weather status:", weather_response.status_code)
    print("Air quality status:", air_response.status_code)

    if weather_response.status_code == 200:
        weather = weather_response.json()

        print("Temperature:", weather["main"]["temp"], "°C")
        print("Humidity:", weather["main"]["humidity"], "%")
        print("Wind speed:", weather["wind"]["speed"], "m/s")

    else:
        print("Weather API error:", weather_response.text)

    if air_response.status_code == 200:
        air = air_response.json()

        current_air = air["list"][0]

        print("OpenWeather AQI:", current_air["main"]["aqi"])
        print("PM2.5:", current_air["components"]["pm2_5"], "µg/m³")
        print("PM10:", current_air["components"]["pm10"], "µg/m³")
        print("CO:", current_air["components"]["co"], "µg/m³")
        print("NO2:", current_air["components"]["no2"], "µg/m³")
        print("O3:", current_air["components"]["o3"], "µg/m³")
        print("SO2:", current_air["components"]["so2"], "µg/m³")

    else:
        print("Air quality API error:", air_response.text)