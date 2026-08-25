import pandas as pd
import os


INPUT_FILE = "data/features.csv"

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"


print("=" * 60)
print("AQI PREDICTOR - TRAIN/TEST DATA PREPARATION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

print("\nLoading feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"✓ Dataset loaded: {len(df)} records")


# ---------------------------------------------------------
# 2. Convert timestamp
# ---------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)


# ---------------------------------------------------------
# 3. Sort chronologically
# ---------------------------------------------------------

print("\nSorting data chronologically...")

df = df.sort_values(
    ["city", "timestamp"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 4. Time-based train/test split
# ---------------------------------------------------------

print("\nCreating time-based train/test split...")

# Use the first 80% of each city's timeline for training
# and the final 20% for testing.

train_parts = []
test_parts = []

for city, city_df in df.groupby("city"):

    city_df = city_df.sort_values("timestamp")

    split_index = int(len(city_df) * 0.80)

    train_city = city_df.iloc[:split_index]
    test_city = city_df.iloc[split_index:]

    train_parts.append(train_city)
    test_parts.append(test_city)


train = pd.concat(
    train_parts,
    ignore_index=True
)

test = pd.concat(
    test_parts,
    ignore_index=True
)


# ---------------------------------------------------------
# 5. Save datasets
# ---------------------------------------------------------

os.makedirs("data", exist_ok=True)

train.to_csv(
    TRAIN_FILE,
    index=False
)

test.to_csv(
    TEST_FILE,
    index=False
)


# ---------------------------------------------------------
# 6. Report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN/TEST SPLIT COMPLETED")
print("=" * 60)

print(f"\nTotal records: {len(df)}")
print(f"Training records: {len(train)}")
print(f"Testing records: {len(test)}")

print("\nTraining records by city:")
print(train.groupby("city").size())

print("\nTesting records by city:")
print(test.groupby("city").size())

print("\nTraining date range:")
print(
    train["timestamp"].min(),
    "to",
    train["timestamp"].max()
)

print("\nTesting date range:")
print(
    test["timestamp"].min(),
    "to",
    test["timestamp"].max()
)

print(f"\nTraining data saved to: {TRAIN_FILE}")
print(f"Testing data saved to: {TEST_FILE}")