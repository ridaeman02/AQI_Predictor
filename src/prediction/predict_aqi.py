from src.utils.aqi_categories import get_aqi_category
import pandas as pd
import numpy as np
import joblib


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_FILE = "outputs/models/best_aqi_model.joblib"
DATA_FILE = "data/features.csv"
OUTPUT_FILE = "outputs/predictions/latest_predictions.csv"

# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("AQI PREDICTOR - CURRENT AQI PREDICTION")
print("=" * 60)

print("\nLoading trained model...")

model = joblib.load(
    MODEL_FILE
)

print("✓ Model loaded")


# =========================================================
# LOAD FEATURE DATA
# =========================================================

print("\nLoading feature data...")

df = pd.read_csv(
    DATA_FILE
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

print(
    f"✓ Records loaded: {len(df)}"
)


# =========================================================
# GET LATEST RECORD FOR EACH CITY
# =========================================================

print("\nFinding latest records...")

latest_records = (
    df.sort_values(
        ["city", "timestamp"]
    )
    .groupby(
        "city",
        as_index=False
    )
    .tail(1)
    .copy()
)

latest_records = latest_records.sort_values(
    "city"
)

print(
    f"✓ Latest records found: "
    f"{len(latest_records)}"
)


# =========================================================
# PREPARE FEATURES
# =========================================================

print("\nPreparing prediction features...")

X = latest_records.drop(
    columns=[
        "aqi",
        "timestamp"
    ]
)

print(
    f"✓ Feature shape: {X.shape}"
)


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X
)

print("✓ Predictions generated")


# =========================================================
# CREATE OUTPUT
# =========================================================

results = latest_records[
    [
        "city",
        "timestamp",
        "aqi"
    ]
].copy()

results["predicted_aqi"] = predictions

results["predicted_aqi_rounded"] = (
    results["predicted_aqi"]
    .round()
    .astype(int)
)

results["category"] = (
    results["predicted_aqi_rounded"]
    .apply(get_aqi_category)
)


# =========================================================
# CALCULATE DIFFERENCE
# =========================================================

results["difference"] = (
    results["predicted_aqi"]
    - results["aqi"]
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 60)
print("LATEST AQI PREDICTIONS")
print("=" * 60)

for _, row in results.iterrows():

    print(
        f"\nCity: {row['city']}"
    )

    print(
        f"Timestamp: {row['timestamp']}"
    )

    print(
        f"Actual AQI: {row['aqi']}"
    )

    print(
        f"Predicted AQI: "
        f"{row['predicted_aqi']:.3f}"
    )

    print(
        f"Category: {row['category']}"
    )

    print(
        f"Difference: "
        f"{row['difference']:.3f}"
    )


# =========================================================
# SAVE RESULTS
# =========================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\n✓ Predictions saved to:"
)

print(
    OUTPUT_FILE
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("PREDICTION PIPELINE COMPLETED")
print("=" * 60)

print(
    f"Cities predicted: "
    f"{len(results)}"
)

print(
    f"Output file: "
    f"{OUTPUT_FILE}"
)