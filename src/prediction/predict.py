import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path if not present
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.aqi_categories import get_aqi_category

try:
    from src.feature_store.fetch_model import get_model_from_hopsworks
    from src.explainability.shap_analysis import generate_local_explanation
    MLOPS_AVAILABLE = True
except ImportError:
    MLOPS_AVAILABLE = False

# ==========================================
# FILE PATHS
# ==========================================
MODEL_DIR = os.path.join(str(BASE_DIR), "models")
DATA_FILE = os.path.join(str(BASE_DIR), "data", "processed_features.csv")

# ==========================================
# FEATURES
# ==========================================
FEATURES = [
    "temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co",
    "no2", "o3", "so2", "nh3", "no", "hour", "day", "month",
    "day_of_week", "aqi_lag_1", "aqi_lag_2", "aqi_change", "aqi_rolling_mean_3",
]


def load_trained_models(model_dir=MODEL_DIR):
    """Load all available trained models from model directory or Hopsworks."""
    model_configs = {
        "Random Forest": {"local": os.path.join(model_dir, "aqi_random_forest.pkl"), "hw_name": "aqi_random_forest"},
        "Ridge Regression": {"local": os.path.join(model_dir, "aqi_ridge.pkl"), "hw_name": "aqi_ridge"},
        "XGBoost": {"local": os.path.join(model_dir, "aqi_xgboost.pkl"), "hw_name": "aqi_xgboost"},
    }

    loaded_models = {}
    for model_name, cfg in model_configs.items():
        model = None
        # Try Hopsworks first
        if MLOPS_AVAILABLE:
            model = get_model_from_hopsworks(cfg["hw_name"])
            
        # Fallback to local
        if model is None and os.path.exists(cfg["local"]):
            model = joblib.load(cfg["local"])
            
        if model is not None:
            loaded_models[model_name] = model

    if not loaded_models:
        raise FileNotFoundError(f"No trained models found locally or in Hopsworks.")

    return loaded_models


def get_next_hour_predictions(data_file=DATA_FILE, model_dir=MODEL_DIR):
    """
    Generate next-hour AQI predictions for all cities in the dataset.

    Returns:
        pd.DataFrame: Contains current AQI, timestamps, individual model predictions,
                      ensemble prediction, and categories.
    """
    models = load_trained_models(model_dir)

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Processed features data file '{data_file}' not found.")

    df = pd.read_csv(data_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required prediction features: {missing_features}")

    cities = ["Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta"]
    available_cities = [c for c in cities if c in df["city"].unique()]
    if not available_cities:
        available_cities = sorted(df["city"].unique())

    results = []

    for city in available_cities:
        city_data = df[df["city"] == city].copy()
        if city_data.empty:
            continue

        latest = city_data.sort_values("timestamp").iloc[-1]
        current_timestamp = latest["timestamp"]
        prediction_timestamp = current_timestamp + pd.Timedelta(hours=1)

        X = pd.DataFrame([[latest[feature] for feature in FEATURES]], columns=FEATURES)

        model_preds = {}
        for model_name, model in models.items():
            model_preds[model_name] = float(model.predict(X)[0])

        preds_list = list(model_preds.values())
        ensemble_pred = sum(preds_list) / len(preds_list) if preds_list else 0.0

        current_aqi_val = float(latest["aqi"])
        current_cat = get_aqi_category(current_aqi_val)
        next_hour_cat = get_aqi_category(ensemble_pred)
        
        # Generate SHAP optionally
        shap_json = None
        if MLOPS_AVAILABLE and "Random Forest" in models:
            shap_json = generate_local_explanation(models["Random Forest"], X.values[0], FEATURES)

        results.append({
            "city": city,
            "current_aqi": current_aqi_val,
            "current_category": current_cat,
            "current_timestamp": current_timestamp,
            "prediction_timestamp": prediction_timestamp,
            "random_forest": model_preds.get("Random Forest", np.nan),
            "ridge": model_preds.get("Ridge Regression", np.nan),
            "xgboost": model_preds.get("XGBoost", np.nan),
            "ensemble": ensemble_pred,
            "next_hour_aqi": ensemble_pred,
            "next_hour_category": next_hour_cat,
            "pollutants": {
                "pm2_5": float(latest["pm2_5"]) if "pm2_5" in latest and not pd.isna(latest["pm2_5"]) else None,
                "pm10": float(latest["pm10"]) if "pm10" in latest and not pd.isna(latest["pm10"]) else None,
                "co": float(latest["co"]) if "co" in latest and not pd.isna(latest["co"]) else None,
                "no2": float(latest["no2"]) if "no2" in latest and not pd.isna(latest["no2"]) else None,
                "o3": float(latest["o3"]) if "o3" in latest and not pd.isna(latest["o3"]) else None,
                "so2": float(latest["so2"]) if "so2" in latest and not pd.isna(latest["so2"]) else None,
                "nh3": float(latest["nh3"]) if "nh3" in latest and not pd.isna(latest["nh3"]) else None,
                "no": float(latest["no"]) if "no" in latest and not pd.isna(latest["no"]) else None,
            },
            "weather": {
                "temperature": float(latest["temperature"]) if "temperature" in latest and not pd.isna(latest["temperature"]) else None,
                "humidity": float(latest["humidity"]) if "humidity" in latest and not pd.isna(latest["humidity"]) else None,
                "wind_speed": float(latest["wind_speed"]) if "wind_speed" in latest and not pd.isna(latest["wind_speed"]) else None,
            },
            "shap_explanation": shap_json
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Loading models...")
    models = load_trained_models(MODEL_DIR)
    for m_name in models:
        print(f"{m_name} loaded successfully.")

    df = pd.read_csv(DATA_FILE)
    print("Data loaded successfully!")
    print(f"Using {len(FEATURES)} features.")

    print("\n" + "=" * 70)
    print("NEXT-HOUR AQI PREDICTIONS")
    print("=" * 70)

    predictions_df = get_next_hour_predictions(DATA_FILE, MODEL_DIR)

    for _, row in predictions_df.iterrows():
        print(f"\nCity: {row['city']}")
        print(f"Current AQI: {row['current_aqi']:.2f}")
        print(f"Current Timestamp: {row['current_timestamp']}")
        print(f"Prediction Timestamp: {row['prediction_timestamp']}")
        print("Predicted AQI for next hour:")
        if not pd.isna(row["random_forest"]):
            print(f"Random Forest: {row['random_forest']:.2f}")
        if not pd.isna(row["ridge"]):
            print(f"Ridge Regression: {row['ridge']:.2f}")
        if not pd.isna(row["xgboost"]):
            print(f"XGBoost: {row['xgboost']:.2f}")
        print(f"Ensemble Prediction: {row['ensemble']:.2f}")

    print("\n" + "=" * 70)
    print("Prediction completed successfully!")
    print("=" * 70)