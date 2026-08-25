from src.utils.aqi_categories import get_aqi_category
import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "outputs/models/best_aqi_model.joblib"
FEATURE_DATA_PATH = "data/features.csv"
OUTPUT_DIR = "outputs/predictions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "forecast_24h.csv")

FORECAST_HOURS = 24

CITIES = [
    "Islamabad",
    "Karachi",
    "Lahore",
    "Peshawar",
    "Quetta"
]

# ============================================================
# MAIN FORECASTING PIPELINE
# ============================================================

print("=" * 60)
print("AQI PREDICTOR - 24 HOUR AQI FORECAST")
print("=" * 60)


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print("✓ Model loaded")


# ------------------------------------------------------------
# Load feature data
# ------------------------------------------------------------

print("\nLoading feature data...")

df = pd.read_csv(FEATURE_DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

print(f"✓ Records loaded: {len(df)}")


# ------------------------------------------------------------
# Prepare output directory
# ------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "city",
    "pm2_5",
    "pm10",
    "co",
    "no2",
    "o3",
    "so2",
    "nh3",
    "no",
    "temperature",
    "humidity",
    "wind_speed",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_24",
    "aqi_rolling_3",
    "aqi_rolling_6",
    "aqi_rolling_24",
]


# ------------------------------------------------------------
# Forecast each city
# ------------------------------------------------------------

all_forecasts = []


for city in CITIES:

    print("\n" + "-" * 60)
    print(f"Forecasting: {city}")
    print("-" * 60)

    city_df = df[df["city"] == city].copy()

    city_df = city_df.sort_values("timestamp").reset_index(drop=True)

    if city_df.empty:
        print(f"⚠ No data found for {city}")
        continue

    # --------------------------------------------------------
    # Get latest historical record
    # --------------------------------------------------------

    latest = city_df.iloc[-1]

    latest_timestamp = latest["timestamp"]

    print(f"Last historical timestamp: {latest_timestamp}")

    # --------------------------------------------------------
    # Maintain AQI history for lag/rolling features
    # --------------------------------------------------------

    aqi_history = city_df["aqi"].astype(float).tolist()

    # --------------------------------------------------------
    # Use latest environmental conditions
    #
    # Future weather values are not available in the current
    # dataset, so we use the latest available measurements as
    # the baseline for the 24-hour forecast.
    # --------------------------------------------------------

    latest_weather = {
        "pm2_5": latest["pm2_5"],
        "pm10": latest["pm10"],
        "co": latest["co"],
        "no2": latest["no2"],
        "o3": latest["o3"],
        "so2": latest["so2"],
        "nh3": latest["nh3"],
        "no": latest["no"],
        "temperature": latest["temperature"],
        "humidity": latest["humidity"],
        "wind_speed": latest["wind_speed"],
    }


    # --------------------------------------------------------
    # Generate 24 future predictions
    # --------------------------------------------------------

    for hour_ahead in range(1, FORECAST_HOURS + 1):

        future_timestamp = latest_timestamp + pd.Timedelta(
            hours=hour_ahead
        )

        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------

        hour = future_timestamp.hour
        day_of_week = future_timestamp.dayofweek
        day_of_month = future_timestamp.day
        month = future_timestamp.month

        # ----------------------------------------------------
        # Lag features
        # ----------------------------------------------------

        aqi_lag_1 = aqi_history[-1]

        aqi_lag_3 = (
            aqi_history[-3]
            if len(aqi_history) >= 3
            else aqi_history[0]
        )

        aqi_lag_6 = (
            aqi_history[-6]
            if len(aqi_history) >= 6
            else aqi_history[0]
        )

        aqi_lag_24 = (
            aqi_history[-24]
            if len(aqi_history) >= 24
            else aqi_history[0]
        )

        # ----------------------------------------------------
        # Rolling features
        # ----------------------------------------------------

        aqi_rolling_3 = np.mean(
            aqi_history[-3:]
        )

        aqi_rolling_6 = np.mean(
            aqi_history[-6:]
        )

        aqi_rolling_24 = np.mean(
            aqi_history[-24:]
        )

        # ----------------------------------------------------
        # Build prediction row
        # ----------------------------------------------------

        prediction_row = pd.DataFrame([{
            "city": city,

            "pm2_5": latest_weather["pm2_5"],
            "pm10": latest_weather["pm10"],
            "co": latest_weather["co"],
            "no2": latest_weather["no2"],
            "o3": latest_weather["o3"],
            "so2": latest_weather["so2"],
            "nh3": latest_weather["nh3"],
            "no": latest_weather["no"],

            "temperature": latest_weather["temperature"],
            "humidity": latest_weather["humidity"],
            "wind_speed": latest_weather["wind_speed"],

            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "month": month,

            "aqi_lag_1": aqi_lag_1,
            "aqi_lag_3": aqi_lag_3,
            "aqi_lag_6": aqi_lag_6,
            "aqi_lag_24": aqi_lag_24,

            "aqi_rolling_3": aqi_rolling_3,
            "aqi_rolling_6": aqi_rolling_6,
            "aqi_rolling_24": aqi_rolling_24,
        }])

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        predicted_aqi = float(
            model.predict(prediction_row[FEATURE_COLUMNS])[0]
        )

        # Keep prediction within model's observed AQI range
        predicted_aqi = np.clip(
            predicted_aqi,
            2,
            5
        )

        category = get_aqi_category(predicted_aqi)

        # ----------------------------------------------------
        # Add prediction to history
        #
        # This is important because the next prediction will
        # use this predicted AQI as a lag value.
        # ----------------------------------------------------

        aqi_history.append(predicted_aqi)

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        all_forecasts.append({
            "city": city,
            "timestamp": future_timestamp,
            "hours_ahead": hour_ahead,
            "predicted_aqi": round(predicted_aqi, 3),
            "category": category,
        })


# ============================================================
# SAVE FORECAST
# ============================================================

forecast_df = pd.DataFrame(all_forecasts)

forecast_df = forecast_df.sort_values(
    ["city", "timestamp"]
).reset_index(drop=True)

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("24-HOUR AQI FORECAST")
print("=" * 60)

for city in CITIES:

    city_forecast = forecast_df[
        forecast_df["city"] == city
    ]

    if city_forecast.empty:
        continue

    print(f"\nCity: {city}")

    for _, row in city_forecast.head(5).iterrows():

        print(
            f"{row['timestamp']} | "
            f"AQI: {row['predicted_aqi']:.3f} | "
            f"{row['category']}"
        )

    if len(city_forecast) > 5:
        print("...")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FORECAST COMPLETED")
print("=" * 60)

print(f"Cities forecasted: {forecast_df['city'].nunique()}")
print(f"Forecast hours per city: {FORECAST_HOURS}")
print(f"Total predictions: {len(forecast_df)}")

print(f"\nForecast saved to:")
print(OUTPUT_PATH)

print("=" * 60)