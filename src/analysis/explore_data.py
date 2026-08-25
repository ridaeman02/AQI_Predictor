import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# AQI PREDICTOR - EXPLORATORY DATA ANALYSIS
# ============================================================

INPUT_FILE = "data/combined_historical_data.csv"
OUTPUT_DIR = "outputs/plots"


os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 60)
print("AQI PREDICTOR - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"✓ Dataset loaded: {len(df)} records")


# ------------------------------------------------------------
# 2. Basic information
# ------------------------------------------------------------

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)


# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\nMissing values:")

print(df.isnull().sum())


# ------------------------------------------------------------
# 4. Duplicate records
# ------------------------------------------------------------

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nDuplicate city/timestamps:")
print(
    df.duplicated(
        ["city", "timestamp"]
    ).sum()
)


# ------------------------------------------------------------
# 5. Cities
# ------------------------------------------------------------

print("\nCities:")
print(sorted(df["city"].unique()))

print("\nRecords by city:")
print(df.groupby("city").size())


# ------------------------------------------------------------
# 6. AQI distribution
# ------------------------------------------------------------

print("\nAQI distribution:")

print(
    df["aqi"]
    .value_counts()
    .sort_index()
)


print("\nAQI statistics:")

print(
    df["aqi"].describe()
)


# ------------------------------------------------------------
# 7. Numeric statistics
# ------------------------------------------------------------

print("\nNumeric statistics:")

print(
    df.select_dtypes("number")
    .describe()
    .round(2)
)


# ------------------------------------------------------------
# 8. AQI by city
# ------------------------------------------------------------

print("\nAverage AQI by city:")

print(
    df.groupby("city")["aqi"]
    .mean()
    .sort_values(ascending=False)
    .round(2)
)


# ------------------------------------------------------------
# 9. AQI distribution plot
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

df["aqi"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "aqi_distribution.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 10. Average AQI by city
# ------------------------------------------------------------

city_aqi = (
    df.groupby("city")["aqi"]
    .mean()
    .sort_values(ascending=False)
)


plt.figure(figsize=(8, 5))

city_aqi.plot(
    kind="bar"
)

plt.title("Average AQI by City")
plt.xlabel("City")
plt.ylabel("Average AQI")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "average_aqi_by_city.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 11. PM2.5 vs AQI
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["pm2_5"],
    df["aqi"],
    alpha=0.4
)

plt.title("PM2.5 vs AQI")
plt.xlabel("PM2.5")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pm25_vs_aqi.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 12. PM10 vs AQI
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["pm10"],
    df["aqi"],
    alpha=0.4
)

plt.title("PM10 vs AQI")
plt.xlabel("PM10")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "pm10_vs_aqi.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 13. Temperature vs AQI
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["temperature"],
    df["aqi"],
    alpha=0.4
)

plt.title("Temperature vs AQI")
plt.xlabel("Temperature")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "temperature_vs_aqi.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 14. Humidity vs AQI
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["humidity"],
    df["aqi"],
    alpha=0.4
)

plt.title("Humidity vs AQI")
plt.xlabel("Humidity")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "humidity_vs_aqi.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 15. Wind speed vs AQI
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["wind_speed"],
    df["aqi"],
    alpha=0.4
)

plt.title("Wind Speed vs AQI")
plt.xlabel("Wind Speed")
plt.ylabel("AQI")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "wind_speed_vs_aqi.png"
    )
)

plt.close()


# ------------------------------------------------------------
# 16. Correlation analysis
# ------------------------------------------------------------

print("\nCorrelation with AQI:")

correlations = (
    df.select_dtypes("number")
    .corr()["aqi"]
    .sort_values(ascending=False)
)

print(
    correlations.round(3)
)


# ------------------------------------------------------------
# 17. Save correlation results
# ------------------------------------------------------------

correlations.to_csv(
    "outputs/aqi_correlations.csv"
)


# ------------------------------------------------------------
# 18. Final report
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nPlots saved to:")
print(OUTPUT_DIR)

print("\nCorrelation results saved to:")
print("outputs/aqi_correlations.csv")