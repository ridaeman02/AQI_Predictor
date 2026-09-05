import os
import pandas as pd


INPUT_FILE = "data/combined_historical_data.csv"
OUTPUT_FILE = "data/processed_features.csv"


def create_features():

    # ==========================================
    # 1. LOAD DATA
    # ==========================================

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} records.")

    # ==========================================
    # 2. CONVERT TIMESTAMP
    # ==========================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    # Remove invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # Sort chronologically by city
    df = df.sort_values(
        ["city", "timestamp"]
    ).reset_index(drop=True)

    # ==========================================
    # 3. REQUIRED WEATHER FEATURES
    # ==========================================

    weather_features = [
        "temperature",
        "humidity",
        "wind_speed"
    ]

    print("\nWeather features:")

    for feature in weather_features:

        if feature in df.columns:
            missing = df[feature].isna().sum()

            if missing > 0:
                print(
                    f"- {feature}: "
                    f"{missing} missing values -> filling"
                )

                # Fill missing values within each city
                df[feature] = (
                    df.groupby("city")[feature]
                    .transform(
                        lambda x: x.interpolate(
                            method="linear",
                            limit_direction="both"
                        )
                    )
                )

                # Fallback to global median
                df[feature] = df[feature].fillna(
                    df[feature].median()
                )

            else:
                print(f"- {feature}: available")

        else:
            raise ValueError(
                f"Required weather feature '{feature}' "
                f"is missing from historical_data.csv"
            )

    # ==========================================
    # 4. CHECK POLLUTANT FEATURES
    # ==========================================

    pollutant_features = [
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "o3",
        "so2",
        "nh3",
        "no"
    ]

    for feature in pollutant_features:

        if feature not in df.columns:
            raise ValueError(
                f"Required pollutant feature "
                f"'{feature}' is missing."
            )

        # Convert to numeric
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        # Fill missing values per city
        df[feature] = (
            df.groupby("city")[feature]
            .transform(
                lambda x: x.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )
        )

        # Final fallback
        df[feature] = df[feature].fillna(
            df[feature].median()
        )

    # ==========================================
    # 5. CONVERT TO EPA CONTINUOUS AQI (0-500)
    # ==========================================
    from src.utils.aqi_categories import calculate_epa_aqi
    
    print("\nConverting categorical API index to continuous EPA AQI (0-500 scale)...")
    df["aqi"] = df["pm2_5"].apply(calculate_epa_aqi)

    df = df.dropna(
        subset=["aqi"]
    )

    # ==========================================
    # 6. TIME FEATURES
    # ==========================================

    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df[
        "timestamp"
    ].dt.dayofweek

    # ==========================================
    # 7. LAG FEATURES
    # ==========================================

    df["aqi_lag_1"] = (
        df.groupby("city")["aqi"]
        .shift(1)
    )

    df["aqi_lag_2"] = (
        df.groupby("city")["aqi"]
        .shift(2)
    )

    # ==========================================
    # 8. AQI CHANGE
    # ==========================================

    df["aqi_change"] = (
        df.groupby("city")["aqi"]
        .diff()
    )

    # ==========================================
    # 9. ROLLING AQI
    # ==========================================

    df["aqi_rolling_mean_3"] = (
        df.groupby("city")["aqi"]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=3
            ).mean()
        )
    )

    # ==========================================
    # 10. NEXT-HOUR TARGET
    # ==========================================

    df["target_aqi"] = (
        df.groupby("city")["aqi"]
        .shift(-1)
    )

    # ==========================================
    # 11. REMOVE INVALID ROWS
    # ==========================================

    # We need:
    # - lag 1
    # - lag 2
    # - rolling mean
    # - next-hour target

    df = df.dropna(
        subset=[
            "aqi_lag_1",
            "aqi_lag_2",
            "aqi_rolling_mean_3",
            "target_aqi"
        ]
    )

    # ==========================================
    # 12. CREATE DATA DIRECTORY
    # ==========================================

    os.makedirs(
        "data",
        exist_ok=True
    )

    # ==========================================
    # 13. SAVE
    # ==========================================

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ==========================================
    # 14. SUMMARY
    # ==========================================

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 60)

    print(
        f"Processed records: {len(df)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nFeatures created:")

    features = [
        "temperature",
        "humidity",
        "wind_speed",
        "pm2_5",
        "pm10",
        "co",
        "no2",
        "o3",
        "so2",
        "nh3",
        "no",
        "hour",
        "day",
        "month",
        "day_of_week",
        "aqi_lag_1",
        "aqi_lag_2",
        "aqi_change",
        "aqi_rolling_mean_3"
    ]

    for feature in features:
        print(f"- {feature}")

    print("\nTarget:")
    print("- target_aqi (next-hour AQI)")

    print("\nTarget AQI distribution:")
    print(
        df["target_aqi"].describe()
    )


if __name__ == "__main__":
    create_features()