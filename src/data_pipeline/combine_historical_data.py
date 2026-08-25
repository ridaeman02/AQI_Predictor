import pandas as pd


AQI_FILE = "data/historical_data.csv"
WEATHER_FILE = "data/historical_weather.csv"
OUTPUT_FILE = "data/combined_historical_data.csv"


print("=" * 60)
print("COMBINING HISTORICAL AQI + WEATHER DATA")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

print("\nLoading AQI data...")

aqi = pd.read_csv(AQI_FILE)

print(f"✓ AQI records loaded: {len(aqi)}")


print("\nLoading weather data...")

weather = pd.read_csv(WEATHER_FILE)

print(f"✓ Weather records loaded: {len(weather)}")


# ---------------------------------------------------------
# 2. Normalize timestamps
# ---------------------------------------------------------

print("\nNormalizing timestamps...")

# AQI timestamps contain UTC timezone information.
aqi["timestamp"] = pd.to_datetime(
    aqi["timestamp"],
    utc=True
)

# Weather timestamps are already stored as UTC.
# Do NOT use tz_localize("Asia/Karachi") here.
weather["timestamp"] = pd.to_datetime(
    weather["timestamp"],
    utc=True
)

print("✓ Timestamp normalization completed")


# ---------------------------------------------------------
# 3. Check city names
# ---------------------------------------------------------

print("\nAQI cities:")
print(sorted(aqi["city"].unique()))

print("\nWeather cities:")
print(sorted(weather["city"].unique()))


# ---------------------------------------------------------
# 4. Merge datasets
# ---------------------------------------------------------

print("\nMerging datasets...")

combined = pd.merge(
    aqi,
    weather,
    on=["city", "timestamp"],
    how="inner"
)

print(f"✓ Matching records: {len(combined)}")


# ---------------------------------------------------------
# 5. Verify all AQI records matched
# ---------------------------------------------------------

if len(combined) != len(aqi):

    missing = len(aqi) - len(combined)

    print(
        f"\nWARNING: {missing} AQI records "
        "did not have matching weather data."
    )

else:

    print(
        "\n✓ All AQI records have matching "
        "weather data."
    )


# ---------------------------------------------------------
# 6. Check records by city
# ---------------------------------------------------------

print("\nCombined records by city:")

print(
    combined.groupby("city").size()
)


# ---------------------------------------------------------
# 7. Check missing values
# ---------------------------------------------------------

print("\nMissing values:")

print(
    combined.isnull().sum()
)


# ---------------------------------------------------------
# 8. Sort data
# ---------------------------------------------------------

combined = combined.sort_values(
    ["city", "timestamp"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 9. Save combined dataset
# ---------------------------------------------------------

combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# 10. Final report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("HISTORICAL DATA COMBINATION COMPLETED")
print("=" * 60)

print(f"Total combined records: {len(combined)}")

print(f"Saved to: {OUTPUT_FILE}")

print("\nFinal columns:")

print(
    combined.columns.tolist()
)

print("\nFirst 5 combined records:")

print(
    combined.head()
)