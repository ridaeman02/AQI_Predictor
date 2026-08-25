import pandas as pd
import os


INPUT_FILE = "data/combined_historical_data.csv"
OUTPUT_FILE = "data/features.csv"


print("=" * 60)
print("AQI PREDICTOR - FEATURE ENGINEERING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"✓ Dataset loaded: {len(df)} records")


# ---------------------------------------------------------
# 2. Convert timestamp
# ---------------------------------------------------------

print("\nProcessing timestamps...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)


# ---------------------------------------------------------
# 3. Sort data
# ---------------------------------------------------------

df = df.sort_values(
    ["city", "timestamp"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 4. Create time-based features
# ---------------------------------------------------------

print("Creating time features...")

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["day_of_month"] = df["timestamp"].dt.day
df["month"] = df["timestamp"].dt.month


# ---------------------------------------------------------
# 5. Create AQI lag features
# ---------------------------------------------------------

print("Creating AQI lag features...")

df["aqi_lag_1"] = (
    df.groupby("city")["aqi"]
    .shift(1)
)

df["aqi_lag_3"] = (
    df.groupby("city")["aqi"]
    .shift(3)
)

df["aqi_lag_6"] = (
    df.groupby("city")["aqi"]
    .shift(6)
)

df["aqi_lag_24"] = (
    df.groupby("city")["aqi"]
    .shift(24)
)


# ---------------------------------------------------------
# 6. Create rolling AQI features
# ---------------------------------------------------------

print("Creating rolling AQI features...")

df["aqi_rolling_3"] = (
    df.groupby("city")["aqi"]
    .transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
)

df["aqi_rolling_6"] = (
    df.groupby("city")["aqi"]
    .transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
)

df["aqi_rolling_24"] = (
    df.groupby("city")["aqi"]
    .transform(
        lambda x: x.shift(1).rolling(24).mean()
    )
)


# ---------------------------------------------------------
# 7. Remove rows with unavailable lag values
# ---------------------------------------------------------

print("\nRemoving rows without sufficient historical data...")

before = len(df)

df = df.dropna().reset_index(drop=True)

after = len(df)

print(f"✓ Rows before: {before}")
print(f"✓ Rows after:  {after}")
print(f"✓ Rows removed: {before - after}")


# ---------------------------------------------------------
# 8. Save engineered dataset
# ---------------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 9. Final report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(f"\nFinal dataset shape: {df.shape}")

print("\nFeatures:")

print(df.columns.tolist())

print("\nMissing values:")

print(df.isnull().sum())

print("\nRecords by city:")

print(df.groupby("city").size())

print(f"\nSaved to: {OUTPUT_FILE}")