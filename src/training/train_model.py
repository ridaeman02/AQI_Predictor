import os
import sys
import joblib
import json
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.feature_store.fetch_features import get_training_data
    from src.feature_store.hopsworks_connection import get_hopsworks_project
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False

from src.models.lstm_model import create_lstm_sequences, build_lstm_model, train_lstm

INPUT_FILE = "data/processed_features.csv"
MODEL_DIR = "models"

RANDOM_FOREST_FILE = os.path.join(MODEL_DIR, "aqi_random_forest.pkl")
RIDGE_FILE = os.path.join(MODEL_DIR, "aqi_ridge.pkl")
XGBOOST_FILE = os.path.join(MODEL_DIR, "aqi_xgboost.pkl")
LSTM_FILE = os.path.join(MODEL_DIR, "aqi_lstm.keras")
SCALER_FILE = os.path.join(MODEL_DIR, "aqi_lstm_scaler.pkl")
PERFORMANCE_FILE = os.path.join(MODEL_DIR, "model_performance.json")

FEATURES = [
    "temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co",
    "no2", "o3", "so2", "nh3", "no", "hour", "day", "month",
    "day_of_week", "aqi_lag_1", "aqi_lag_2", "aqi_change", "aqi_rolling_mean_3"
]

TARGET = "target_aqi"


def evaluate_model(name, model, X, y, split_name="Test"):
    predictions = model.predict(X)
    mae = float(mean_absolute_error(y, predictions))
    rmse = float(mean_squared_error(y, predictions) ** 0.5)
    r2 = float(r2_score(y, predictions))

    print(f"[{split_name}] {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def register_model_in_hopsworks(name, metrics, model_file):
    """Registers a locally saved model to Hopsworks Model Registry"""
    if not HOPSWORKS_AVAILABLE:
        print(f"Hopsworks not available. Skipping model registry for {name}.")
        return

    try:
        project = get_hopsworks_project()
        mr = project.get_model_registry()

        # Link Feature View to model registry for schema inference and provenance tracking
        fv = None
        try:
            from src.feature_store.hopsworks_connection import get_feature_store
            fs = get_feature_store()
            fv = fs.get_feature_view(name="aqi_features_view", version=1)
            print(f"Linked Feature View to model registry metadata for {name}.")
        except Exception as fv_err:
            print(f"Warning: Could not fetch Feature View to link during model registration: {fv_err}")

        # Define model metadata
        hs_model = mr.python.create_model(
            name=name.replace(" ", "_").lower(), 
            metrics=metrics,
            description=f"{name} model predicting {TARGET}",
            feature_view=fv,
            training_dataset_version=1 if fv is not None else None
        )
        
        # Save model to registry without removing local file
        hs_model.save(model_file, keep_original_files=True)
        print(f"Successfully registered {name} in Hopsworks Model Registry.")
    except Exception as e:
        print(f"Warning: Failed to register {name} in Hopsworks: {e}")

def train_model():
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("=" * 70)
    print("LOADING PROCESSED DATA")
    print("=" * 70)

    if HOPSWORKS_AVAILABLE:
        try:
            df = get_training_data(fallback_csv_path=INPUT_FILE)
        except Exception as e:
            print(f"Could not use Hopsworks data fetching ({e}). Using direct local fallback.")
            df = pd.read_csv(INPUT_FILE)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
    else:
        print("Training data source: Local CSV fallback")
        df = pd.read_csv(INPUT_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    df = df.sort_values(["timestamp", "city"]).reset_index(drop=True)
    print(f"Loaded {len(df)} records.")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # ==========================================
    # 2. VERIFY FEATURES
    # ==========================================
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    # ==========================================
    # 3. THREE-WAY CHRONOLOGICAL TRAIN / VAL / TEST SPLIT
    # ==========================================
    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    n_samples = len(df)
    train_end = int(n_samples * 0.70)
    val_end = int(n_samples * 0.85)

    train_df = df.iloc[:train_end].copy().dropna(subset=FEATURES + [TARGET])
    val_df = df.iloc[train_end:val_end].copy().dropna(subset=FEATURES + [TARGET])
    test_df = df.iloc[val_end:].copy().dropna(subset=FEATURES + [TARGET])

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_val, y_val = val_df[FEATURES], val_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print(f"Training records:   {len(X_train)} (approx 70%)")
    print(f"Validation records: {len(X_val)} (approx 15%)")
    print(f"Testing records:    {len(X_test)} (approx 15%)")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==========================================
    # 4. RANDOM FOREST
    # ==========================================
    print("\n" + "=" * 60 + "\nTRAINING RANDOM FOREST\n" + "=" * 60)
    random_forest = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    rf_val_metrics = evaluate_model("Random Forest", random_forest, X_val, y_val, "Validation")
    rf_test_metrics = evaluate_model("Random Forest", random_forest, X_test, y_test, "Test")
    joblib.dump(random_forest, RANDOM_FOREST_FILE)
    print(f"Random Forest saved: {RANDOM_FOREST_FILE}")
    register_model_in_hopsworks("aqi_random_forest", rf_test_metrics, RANDOM_FOREST_FILE)

    # ==========================================
    # 5. RIDGE REGRESSION
    # ==========================================
    print("\n" + "=" * 60 + "\nTRAINING RIDGE REGRESSION\n" + "=" * 60)
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    ridge_val_metrics = evaluate_model("Ridge Regression", ridge, X_val, y_val, "Validation")
    ridge_test_metrics = evaluate_model("Ridge Regression", ridge, X_test, y_test, "Test")
    joblib.dump(ridge, RIDGE_FILE)
    print(f"Ridge Regression saved: {RIDGE_FILE}")
    register_model_in_hopsworks("aqi_ridge", ridge_test_metrics, RIDGE_FILE)

    # ==========================================
    # 6. XGBOOST
    # ==========================================
    print("\n" + "=" * 60 + "\nTRAINING XGBOOST\n" + "=" * 60)
    xgboost_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, objective="reg:squarederror", n_jobs=-1)
    xgboost_model.fit(X_train, y_train)
    xgb_val_metrics = evaluate_model("XGBoost", xgboost_model, X_val, y_val, "Validation")
    xgb_test_metrics = evaluate_model("XGBoost", xgboost_model, X_test, y_test, "Test")
    joblib.dump(xgboost_model, XGBOOST_FILE)
    print(f"XGBoost saved: {XGBOOST_FILE}")
    register_model_in_hopsworks("aqi_xgboost", xgb_test_metrics, XGBOOST_FILE)

    # ==========================================
    # 7. LSTM TIME-SERIES MODEL
    # ==========================================
    print("\n" + "=" * 60 + "\nTRAINING LSTM TIME-SERIES MODEL\n" + "=" * 60)
    try:
        # Fit scaler strictly on training features
        scaler = StandardScaler()
        scaler.fit(train_df[FEATURES])
        joblib.dump(scaler, SCALER_FILE)
        print(f"StandardScaler saved: {SCALER_FILE}")
        
        train_df_scaled = train_df.copy()
        train_df_scaled[FEATURES] = scaler.transform(train_df[FEATURES])
        
        val_df_scaled = val_df.copy()
        val_df_scaled[FEATURES] = scaler.transform(val_df[FEATURES])

        test_df_scaled = test_df.copy()
        test_df_scaled[FEATURES] = scaler.transform(test_df[FEATURES])

        # Prepare sequential windows
        X_train_seq, y_train_seq = create_lstm_sequences(train_df_scaled, FEATURES, TARGET, sequence_length=24)
        X_val_seq, y_val_seq = create_lstm_sequences(val_df_scaled, FEATURES, TARGET, sequence_length=24)
        X_test_seq, y_test_seq = create_lstm_sequences(test_df_scaled, FEATURES, TARGET, sequence_length=24)
        
        print(f"LSTM Training sequences:   {X_train_seq.shape}")
        print(f"LSTM Validation sequences: {X_val_seq.shape}")
        print(f"LSTM Testing sequences:    {X_test_seq.shape}")

        lstm_model = build_lstm_model(input_shape=(24, len(FEATURES)))
        
        # Train LSTM with validation split
        train_lstm(lstm_model, X_train_seq, y_train_seq, epochs=30, batch_size=32, validation_data=(X_val_seq, y_val_seq))
        
        # Evaluate LSTM - Validation
        lstm_val_preds = lstm_model(X_val_seq, training=False).numpy().flatten()
        lstm_val_mae = float(mean_absolute_error(y_val_seq, lstm_val_preds))
        lstm_val_rmse = float(mean_squared_error(y_val_seq, lstm_val_preds) ** 0.5)
        lstm_val_r2 = float(r2_score(y_val_seq, lstm_val_preds))
        
        # Evaluate LSTM - Test
        lstm_test_preds = lstm_model(X_test_seq, training=False).numpy().flatten()
        lstm_test_mae = float(mean_absolute_error(y_test_seq, lstm_test_preds))
        lstm_test_rmse = float(mean_squared_error(y_test_seq, lstm_test_preds) ** 0.5)
        lstm_test_r2 = float(r2_score(y_test_seq, lstm_test_preds))
        
        print("\n" + "=" * 50)
        print("LSTM Performance")
        print("=" * 50)
        print(f"[Validation] MAE: {lstm_val_mae:.4f}, RMSE: {lstm_val_rmse:.4f}, R2: {lstm_val_r2:.4f}")
        print(f"[Test]       MAE: {lstm_test_mae:.4f}, RMSE: {lstm_test_rmse:.4f}, R2: {lstm_test_r2:.4f}")
        
        lstm_val_metrics = {"MAE": lstm_val_mae, "RMSE": lstm_val_rmse, "R2": lstm_val_r2}
        lstm_test_metrics = {"MAE": lstm_test_mae, "RMSE": lstm_test_rmse, "R2": lstm_test_r2}
        
        lstm_model.save(LSTM_FILE)
        print(f"LSTM model saved: {LSTM_FILE}")
        
        register_model_in_hopsworks("aqi_lstm", lstm_test_metrics, LSTM_FILE)
        LSTM_AVAILABLE = True
    except Exception as e:
        print(f"LSTM training failed: {e}")
        LSTM_AVAILABLE = False
        lstm_val_metrics = {"MAE": 999.0, "RMSE": 999.0, "R2": -999.0}
        lstm_test_metrics = {"MAE": 999.0, "RMSE": 999.0, "R2": -999.0}

    # ==========================================
    # 8. COMPARE MODELS AND GENERATE PERFORMANCE SUMMARY (Using Validation RMSE for selection)
    # ==========================================
    perf_data = [
        {
            "Model": "Ridge Regression", 
            "MAE": float(ridge_test_metrics["MAE"]), 
            "RMSE": float(ridge_test_metrics["RMSE"]), 
            "R²": float(ridge_test_metrics["R2"]), 
            "Val_RMSE": float(ridge_val_metrics["RMSE"]),
            "Status": "Active"
        },
        {
            "Model": "XGBoost", 
            "MAE": float(xgb_test_metrics["MAE"]), 
            "RMSE": float(xgb_test_metrics["RMSE"]), 
            "R²": float(xgb_test_metrics["R2"]), 
            "Val_RMSE": float(xgb_val_metrics["RMSE"]),
            "Status": "Active"
        },
        {
            "Model": "Random Forest", 
            "MAE": float(rf_test_metrics["MAE"]), 
            "RMSE": float(rf_test_metrics["RMSE"]), 
            "R²": float(rf_test_metrics["R2"]), 
            "Val_RMSE": float(rf_val_metrics["RMSE"]),
            "Status": "Active"
        },
    ]
    if LSTM_AVAILABLE:
        perf_data.append({
            "Model": "LSTM", 
            "MAE": lstm_test_metrics["MAE"], 
            "RMSE": lstm_test_metrics["RMSE"], 
            "R²": lstm_test_metrics["R2"], 
            "Val_RMSE": lstm_val_metrics["RMSE"],
            "Status": "Active"
        })

    # Selection based strictly on Validation RMSE
    best_model = min(perf_data, key=lambda x: x["Val_RMSE"])
    best_model["Status"] += " (Best Individual)"
    
    with open(PERFORMANCE_FILE, "w") as f:
        json.dump(perf_data, f, indent=2)
    print(f"\nPerformance summary written to: {PERFORMANCE_FILE}")

    print("\n" + "=" * 60 + "\nALL MODELS TRAINED SUCCESSFULLY\n" + "=" * 60)
    print("\nTraining completed using chronological validation.")


if __name__ == "__main__":
    train_model()