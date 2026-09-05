import os
import sys
import requests
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Add project root to sys.path if not present
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.aqi_categories import get_aqi_category
from src.prediction.predict import load_trained_models, FEATURES, MODEL_DIR, DATA_FILE

try:
    from src.explainability.shap_analysis import generate_local_explanation
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES_COORDS = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}


def _fetch_weather(lat, lon):
    weather_dict = {}
    try:
        w_url = "https://api.openweathermap.org/data/2.5/forecast"
        w_params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": "metric"}
        w_res = requests.get(w_url, params=w_params, timeout=2)
        if w_res.status_code == 200:
            w_data = w_res.json()
            for item in w_data.get("list", []):
                dt = pd.to_datetime(item["dt"], unit="s", utc=True)
                weather_dict[dt] = {
                    "temperature": float(item["main"]["temp"]),
                    "humidity": float(item["main"]["humidity"]),
                    "wind_speed": float(item["wind"]["speed"]),
                }
    except Exception:
        pass
    return weather_dict


def _fetch_pollutants(lat, lon):
    pollutants_dict = {}
    try:
        p_url = "https://api.openweathermap.org/data/2.5/air_pollution/forecast"
        p_params = {"lat": lat, "lon": lon, "appid": API_KEY}
        p_res = requests.get(p_url, params=p_params, timeout=2)
        if p_res.status_code == 200:
            p_data = p_res.json()
            for item in p_data.get("list", []):
                dt = pd.to_datetime(item["dt"], unit="s", utc=True)
                comp = item.get("components", {})
                pollutants_dict[dt] = {
                    "pm2_5": float(comp.get("pm2_5", 0.0)),
                    "pm10": float(comp.get("pm10", 0.0)),
                    "co": float(comp.get("co", 0.0)),
                    "no2": float(comp.get("no2", 0.0)),
                    "o3": float(comp.get("o3", 0.0)),
                    "so2": float(comp.get("so2", 0.0)),
                    "nh3": float(comp.get("nh3", 0.0)),
                    "no": float(comp.get("no", 0.0)),
                }
    except Exception:
        pass
    return pollutants_dict


def fetch_openweather_forecast(city):
    """
    Fetches future weather and air pollution forecasts concurrently from OpenWeather API.
    Returns dictionaries of weather and pollutant features indexed by timestamp.
    """
    if not API_KEY or city not in CITIES_COORDS:
        return None, None

    lat, lon = CITIES_COORDS[city]

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_w = executor.submit(_fetch_weather, lat, lon)
            fut_p = executor.submit(_fetch_pollutants, lat, lon)
            weather_dict = fut_w.result()
            pollutants_dict = fut_p.result()
    except Exception as e:
        print(f"Warning: Live forecast API call failed for {city}: {e}")
        return None, None

    return (weather_dict if weather_dict else None), (pollutants_dict if pollutants_dict else None)


def forecast_next_72_hours(city, hours=72, data_file=DATA_FILE, model_dir=MODEL_DIR, include_shap=False, live_api=True):
    """
    Generates a genuine 72-hour recursive AQI forecast for the specified city.
    Uses local files directly to ensure fast execution without remote Hopsworks calls.
    Uses recursive lag propagation without future target data leakage.
    """
    # 1. Load trained models directly from local disk
    models = load_trained_models(model_dir=model_dir, from_hopsworks=False)

    # 2. Load historical feature data directly from local CSV
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Required local data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    city_data = df[df["city"].str.lower() == city.lower()].sort_values("timestamp").reset_index(drop=True)

    if city_data.empty:
        raise ValueError(f"No data available for city: {city}")

    if len(city_data) < 3:
        raise ValueError(f"Not enough historical records for {city} to compute lag features.")

    # 3. Extract baseline history
    latest_rows = city_data.iloc[-3:].copy()
    
    # If live_api is True, we want the forecast to start from the current actual time
    # so it correctly aligns with live OpenWeather forecasts instead of old CSV data.
    if live_api:
        # Determine the timezone of the existing data, if any, or default to UTC
        csv_tz = latest_rows.iloc[-1]["timestamp"].tzinfo
        base_t0 = pd.Timestamp.now(tz=csv_tz).floor("h")
    else:
        base_t0 = latest_rows.iloc[-1]["timestamp"]
    
    # aqi history queue: [AQI(t-2), AQI(t-1), AQI(t-0)]
    aqi_history = [
        float(latest_rows.iloc[-3]["aqi"]),
        float(latest_rows.iloc[-2]["aqi"]),
        float(latest_rows.iloc[-1]["aqi"])
    ]

    # Baseline weather & pollutant values from latest observation
    baseline_weather = {
        "temperature": float(latest_rows.iloc[-1].get("temperature", 25.0)),
        "humidity": float(latest_rows.iloc[-1].get("humidity", 60.0)),
        "wind_speed": float(latest_rows.iloc[-1].get("wind_speed", 2.0))
    }
    
    pollutant_keys = ["pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3", "no"]
    baseline_pollutants = {
        k: float(latest_rows.iloc[-1].get(k, 0.0)) for k in pollutant_keys
    }

    # 4. Attempt to fetch live OpenWeather forecast data
    live_weather_dict, live_pollutants_dict = (None, None)
    if live_api:
        live_weather_dict, live_pollutants_dict = fetch_openweather_forecast(city)

    weather_source = "OpenWeather API 5-Day Forecast" if live_weather_dict else "Historical Baseline Estimation"
    pollutant_source = "OpenWeather API Air Pollution Forecast" if live_pollutants_dict else "Historical Baseline Estimation"

    # Initialize rolling scaled history buffer for LSTM
    scaled_history_seq = []
    if "LSTM" in models and "scaler" in models:
        try:
            scaler = models["scaler"]
            if len(city_data) >= 24:
                last_24 = city_data.iloc[-24:].copy()
                scaled_history_seq = list(scaler.transform(last_24[FEATURES]))
            else:
                scaled_history_seq = list(np.zeros((24, len(FEATURES))))
        except Exception as e:
            print(f"Warning: Failed to initialize LSTM history sequence: {e}")

    forecast_results = []

    # 5. Fast Recursive 72-Hour Forecasting Loop
    for k in range(1, hours + 1):
        step_time = base_t0 + pd.Timedelta(hours=k)

        # Lags & Rolling Features (Leakage-free recursive state)
        lag_1 = aqi_history[-1]
        lag_2 = aqi_history[-2]
        change = lag_1 - lag_2
        rolling_mean_3 = float(np.mean(aqi_history[-3:]))

        # Time features
        hour_val = float(step_time.hour)
        day_val = float(step_time.day)
        month_val = float(step_time.month)
        day_of_week_val = float(step_time.dayofweek)

        # Get Weather features for step_time
        curr_weather = None
        if live_weather_dict:
            matching_dt = min(live_weather_dict.keys(), key=lambda d: abs(d - step_time))
            if abs(matching_dt - step_time) <= pd.Timedelta(hours=3):
                curr_weather = live_weather_dict[matching_dt]

        if not curr_weather:
            # Diurnal temperature/humidity estimation strategy
            diurnal_temp = baseline_weather["temperature"] + 3.0 * np.sin(2 * np.pi * (hour_val - 9) / 24.0)
            diurnal_hum = max(10.0, min(100.0, baseline_weather["humidity"] - 5.0 * np.sin(2 * np.pi * (hour_val - 9) / 24.0)))
            curr_weather = {
                "temperature": diurnal_temp,
                "humidity": diurnal_hum,
                "wind_speed": baseline_weather["wind_speed"]
            }

        # Get Pollutant features for step_time
        curr_pollutants = None
        if live_pollutants_dict:
            matching_dt = min(live_pollutants_dict.keys(), key=lambda d: abs(d - step_time))
            if abs(matching_dt - step_time) <= pd.Timedelta(hours=2):
                curr_pollutants = live_pollutants_dict[matching_dt]

        if not curr_pollutants:
            curr_pollutants = baseline_pollutants.copy()

        # Construct 19-feature DataFrame with names
        row_vals = [
            curr_weather["temperature"],
            curr_weather["humidity"],
            curr_weather["wind_speed"],
            curr_pollutants["pm2_5"],
            curr_pollutants["pm10"],
            curr_pollutants["co"],
            curr_pollutants["no2"],
            curr_pollutants["o3"],
            curr_pollutants["so2"],
            curr_pollutants["nh3"],
            curr_pollutants["no"],
            hour_val,
            day_val,
            month_val,
            day_of_week_val,
            lag_1,
            lag_2,
            change,
            rolling_mean_3
        ]
        X_step_df = pd.DataFrame([row_vals], columns=FEATURES)

        # Predict using models
        model_preds = {}
        # 1. Predictions for static models
        for m_name in ["Random Forest", "Ridge Regression", "XGBoost"]:
            if m_name in models:
                model_preds[m_name] = float(models[m_name].predict(X_step_df)[0])

        # 2. Prediction for LSTM sequential model
        if "LSTM" in models and "scaler" in models and len(scaled_history_seq) > 0:
            try:
                scaler = models["scaler"]
                curr_scaled = scaler.transform(X_step_df)[0]
                scaled_history_seq.append(curr_scaled)
                
                # Fetch sequence input window
                lstm_seq_input = np.array(scaled_history_seq[-24:]).reshape(1, 24, len(FEATURES))
                lstm_pred = float(models["LSTM"](lstm_seq_input, training=False)[0][0])
                model_preds["LSTM"] = lstm_pred
            except Exception as e:
                print(f"Warning: LSTM step prediction failed: {e}")
                model_preds["LSTM"] = np.nan

        # 3. Ensemble computation
        preds_list = [val for name, val in model_preds.items() if name in ["Random Forest", "Ridge Regression", "XGBoost", "LSTM"] and not pd.isna(val)]
        ensemble_pred = float(np.mean(preds_list)) if preds_list else 0.0

        # Update recursive state for next step
        aqi_history.append(ensemble_pred)

        # Optional SHAP explanation (deferred from critical UI path)
        shap_json = None
        if include_shap and k == 1 and SHAP_AVAILABLE and "Random Forest" in models:
            try:
                shap_json = generate_local_explanation(models["Random Forest"], X_step_df.values[0], FEATURES)
            except Exception:
                shap_json = None

        forecast_results.append({
            "step": k,
            "timestamp": step_time.isoformat(),
            "random_forest": model_preds.get("Random Forest", np.nan),
            "ridge": model_preds.get("Ridge Regression", np.nan),
            "xgboost": model_preds.get("XGBoost", np.nan),
            "lstm": model_preds.get("LSTM", np.nan),
            "ensemble": ensemble_pred,
            "predicted_aqi": ensemble_pred,
            "category": get_aqi_category(ensemble_pred),
            "temperature": curr_weather.get("temperature", np.nan),
            "humidity": curr_weather.get("humidity", np.nan),
            "wind_speed": curr_weather.get("wind_speed", np.nan),
            "weather_source": weather_source,
            "pollutant_source": pollutant_source,
            "feature_vector": X_step_df.iloc[0].to_dict(),
            "shap_explanation": shap_json if k == 1 else None
        })

    return pd.DataFrame(forecast_results)


def get_forecast_shap_explanation(city, data_file=DATA_FILE, model_dir=MODEL_DIR):
    """
    On-demand SHAP calculation for the first forecast step.
    Allows Streamlit to compute SHAP lazily when expander is opened.
    """
    if not SHAP_AVAILABLE:
        return None
    try:
        models = load_trained_models(model_dir=model_dir, from_hopsworks=False)
        if "Random Forest" not in models or not os.path.exists(data_file):
            return None
        
        df = pd.read_csv(data_file)
        city_data = df[df["city"].str.lower() == city.lower()].sort_values("timestamp")
        if city_data.empty:
            return None
            
        latest = city_data.iloc[-1]
        feature_vals = [latest[f] for f in FEATURES]
        return generate_local_explanation(models["Random Forest"], np.array(feature_vals, dtype=np.float32), FEATURES)
    except Exception as e:
        print(f"Error computing SHAP: {e}")
        return None


if __name__ == "__main__":
    print("Testing local 72-hour AQI Forecast for Lahore...")
    df_fc = forecast_next_72_hours("Lahore", hours=72)
    print(f"Retrieved {len(df_fc)} forecast steps.")
    print("Head:")
    print(df_fc[["step", "timestamp", "ensemble", "category"]].head())
    print("\nTail:")
    print(df_fc[["step", "timestamp", "ensemble", "category"]].tail())
