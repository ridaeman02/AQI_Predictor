import pandas as pd
import joblib
import matplotlib.pyplot as plt


MODEL_FILE = "outputs/models/best_aqi_model.joblib"

OUTPUT_CSV = "outputs/feature_importance.csv"
OUTPUT_PLOT = "outputs/plots/feature_importance.png"


print("=" * 60)
print("AQI PREDICTOR - FEATURE IMPORTANCE")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("✓ Model loaded")


# ---------------------------------------------------------
# 2. Inspect pipeline
# ---------------------------------------------------------

print("\nPipeline steps:")

for name, step in model.named_steps.items():
    print(f"  {name}: {type(step).__name__}")


# ---------------------------------------------------------
# 3. Find the preprocessing step
# ---------------------------------------------------------

preprocessor = None
estimator = None

for name, step in model.named_steps.items():

    if hasattr(step, "get_feature_names_out"):
        if "preprocess" in name.lower() or "transform" in name.lower():
            preprocessor = step

    if hasattr(step, "feature_importances_"):
        estimator = step


if preprocessor is None:
    print("\nERROR: Could not find preprocessing step.")
    print("Available steps:")

    for name in model.named_steps:
        print(f" - {name}")

    raise SystemExit(1)


if estimator is None:
    print("\nERROR: Could not find a model with feature_importances_.")
    print("Available steps:")

    for name in model.named_steps:
        print(f" - {name}")

    raise SystemExit(1)


print(
    f"\n✓ Preprocessor found: "
    f"{type(preprocessor).__name__}"
)

print(
    f"✓ Estimator found: "
    f"{type(estimator).__name__}"
)


# ---------------------------------------------------------
# 4. Get feature names
# ---------------------------------------------------------

feature_names = (
    preprocessor.get_feature_names_out()
)

importance = estimator.feature_importances_


# ---------------------------------------------------------
# 5. Verify lengths
# ---------------------------------------------------------

print("\nFeature information:")

print(
    f"Feature names: {len(feature_names)}"
)

print(
    f"Importance values: {len(importance)}"
)


if len(feature_names) != len(importance):

    raise ValueError(
        "Feature names and feature importance "
        "lengths do not match."
    )


# ---------------------------------------------------------
# 6. Create importance dataframe
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# 7. Save CSV
# ---------------------------------------------------------

importance_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\n✓ Feature importance saved to: "
    f"{OUTPUT_CSV}"
)


# ---------------------------------------------------------
# 8. Display top features
# ---------------------------------------------------------

print("\nTop 15 features:")

print(
    importance_df.head(15).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 9. Plot top 15 features
# ---------------------------------------------------------

top_features = (
    importance_df
    .head(15)
    .sort_values(
        "importance"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.title(
    "Top 15 Features for AQI Prediction"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=150
)

plt.close()


print(
    f"\n✓ Plot saved to: "
    f"{OUTPUT_PLOT}"
)


# ---------------------------------------------------------
# 10. Complete
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 60)