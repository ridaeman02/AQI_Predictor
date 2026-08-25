import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


TEST_FILE = "data/test.csv"
MODEL_FILE = "outputs/models/best_aqi_model.joblib"

PREDICTIONS_FILE = "outputs/evaluation/test_predictions.csv"

ACTUAL_VS_PREDICTED_PLOT = "outputs/evaluation/actual_vs_predicted.png"
RESIDUAL_PLOT = "outputs/evaluation/residuals.png"


print("=" * 60)
print("AQI PREDICTOR - MODEL EVALUATION")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load testing data
# ---------------------------------------------------------

print("\nLoading testing data...")

df = pd.read_csv(TEST_FILE)

print(f"✓ Testing records: {len(df)}")


# ---------------------------------------------------------
# 2. Load trained model
# ---------------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("✓ Best model loaded")


# ---------------------------------------------------------
# 3. Prepare features
# ---------------------------------------------------------

target = "aqi"

drop_columns = [
    "aqi",
    "timestamp"
]

X_test = df.drop(
    columns=drop_columns
)

y_test = df[target]


print("\nTesting feature shape:")
print(X_test.shape)


# ---------------------------------------------------------
# 4. Generate predictions
# ---------------------------------------------------------

print("\nGenerating predictions...")

predictions = model.predict(X_test)

print("✓ Predictions generated")


# ---------------------------------------------------------
# 5. Calculate metrics
# ---------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "-" * 60)
print("MODEL PERFORMANCE")
print("-" * 60)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# ---------------------------------------------------------
# 6. Save predictions
# ---------------------------------------------------------

results = df[
    ["city", "timestamp", "aqi"]
].copy()

results["predicted_aqi"] = predictions

results["error"] = (
    results["aqi"] -
    results["predicted_aqi"]
)

results["absolute_error"] = (
    results["error"].abs()
)

results.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"\n✓ Predictions saved to: "
    f"{PREDICTIONS_FILE}"
)


# ---------------------------------------------------------
# 7. Actual vs predicted plot
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")

plt.title(
    "Actual vs Predicted AQI"
)

plt.tight_layout()

plt.savefig(
    ACTUAL_VS_PREDICTED_PLOT,
    dpi=150
)

plt.close()

print(
    f"✓ Plot saved to: "
    f"{ACTUAL_VS_PREDICTED_PLOT}"
)


# ---------------------------------------------------------
# 8. Residual plot
# ---------------------------------------------------------

residuals = (
    y_test -
    predictions
)

plt.figure(figsize=(8, 6))

plt.scatter(
    predictions,
    residuals,
    alpha=0.5
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted AQI")
plt.ylabel("Residual")

plt.title(
    "AQI Prediction Residuals"
)

plt.tight_layout()

plt.savefig(
    RESIDUAL_PLOT,
    dpi=150
)

plt.close()

print(
    f"✓ Plot saved to: "
    f"{RESIDUAL_PLOT}"
)


# ---------------------------------------------------------
# 9. Evaluation by city
# ---------------------------------------------------------

print("\nPerformance by city:")

for city in sorted(df["city"].unique()):

    city_mask = (
        df["city"] == city
    )

    city_actual = y_test[city_mask]

    city_predicted = predictions[city_mask]

    city_mae = mean_absolute_error(
        city_actual,
        city_predicted
    )

    city_rmse = np.sqrt(
        mean_squared_error(
            city_actual,
            city_predicted
        )
    )

    city_r2 = r2_score(
        city_actual,
        city_predicted
    )

    print(
        f"{city:12s} "
        f"MAE={city_mae:.4f} "
        f"RMSE={city_rmse:.4f} "
        f"R²={city_r2:.4f}"
    )


# ---------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print(
    "\nEvaluation outputs saved in:"
)

print("outputs/evaluation/")