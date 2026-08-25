import os
import sys
import joblib
import pandas as pd

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


INPUT_FILE = "data/processed_features.csv"
MODEL_DIR = "models"

RANDOM_FOREST_FILE = os.path.join(MODEL_DIR, "aqi_random_forest.pkl")
RIDGE_FILE = os.path.join(MODEL_DIR, "aqi_ridge.pkl")
XGBOOST_FILE = os.path.join(MODEL_DIR, "aqi_xgboost.pkl")

FEATURES = [
    "temperature", "humidity", "wind_speed", "pm2_5", "pm10", "co",
    "no2", "o3", "so2", "nh3", "no", "hour", "day", "month",
    "day_of_week", "aqi_lag_1", "aqi_lag_2", "aqi_change", "aqi_rolling_mean_3"
]

TARGET = "target_aqi"


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 50)
    print(f"{name} Performance")
    print("=" * 50)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def register_model_in_hopsworks(name, metrics, model_file):
    """Registers a locally saved model to Hopsworks Model Registry"""
    if not HOPSWORKS_AVAILABLE:
        print(f"Hopsworks not available. Skipping model registry for {name}.")
        return

    try:
        project = get_hopsworks_project()
        mr = project.get_model_registry()

        # Define model metadata
        hs_model = mr.python.create_model(
            name=name.replace(" ", "_").lower(), 
            metrics=metrics,
            description=f"{name} model predicting {TARGET}",
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
    # 3. CHRONOLOGICAL SPLIT
    # ==========================================
    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TRAIN / TEST SPLIT")
    print("=" * 70)

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy().dropna(subset=FEATURES + [TARGET])
    test_df = df.iloc[split_index:].copy().dropna(subset=FEATURES + [TARGET])

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ==========================================
    # 4. RANDOM FOREST
    # ==========================================
    print("\n" + "=" * 60 + "\nTRAINING RANDOM FOREST\n" + "=" * 60)
    random_forest = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    random_forest.fit(X_train, y_train)
    rf_metrics = evaluate_model("Random Forest", random_forest, X_test, y_test)
    joblib.dump(random_forest, RANDOM_FOREST_FILE)
    print(f"Random Forest saved: {RANDOM_FOREST_FILE}")
    register_model_in_hopsworks("aqi_random_forest", rf_metrics, RANDOM_FOREST_FILE)

    # ==========================================
    # 5. RIDGE REGRESSION
    # ==========================================
    print("\n" + "=" * 60 + "\nTRAINING RIDGE REGRESSION\n" + "=" * 60)
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    ridge_metrics = evaluate_model("Ridge Regression", ridge, X_test, y_test)
    joblib.dump(ridge, RIDGE_FILE)
    print(f"Ridge Regression saved: {RIDGE_FILE}")
    register_model_in_hopsworks("aqi_ridge", ridge_metrics, RIDGE_FILE)

    # ==========================================
    # 6. XGBOOST
    # ==========================================
    print("\n" + "=" * 60 + "\nTRAINING XGBOOST\n" + "=" * 60)
    xgboost_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, objective="reg:squarederror", n_jobs=-1)
    xgboost_model.fit(X_train, y_train)
    xgb_metrics = evaluate_model("XGBoost", xgboost_model, X_test, y_test)
    joblib.dump(xgboost_model, XGBOOST_FILE)
    print(f"XGBoost saved: {XGBOOST_FILE}")
    register_model_in_hopsworks("aqi_xgboost", xgb_metrics, XGBOOST_FILE)

    print("\n" + "=" * 60 + "\nALL MODELS TRAINED SUCCESSFULLY\n" + "=" * 60)
    print("\nTraining completed using chronological evaluation.")

if __name__ == "__main__":
    train_model()