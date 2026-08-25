import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"

MODEL_DIR = "outputs/models"
RESULT_FILE = "outputs/model_results.csv"


print("=" * 60)
print("AQI PREDICTOR - MODEL TRAINING")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

print("\nLoading training data...")

train = pd.read_csv(TRAIN_FILE)

print(f"✓ Training records: {len(train)}")


print("\nLoading testing data...")

test = pd.read_csv(TEST_FILE)

print(f"✓ Testing records: {len(test)}")


# ---------------------------------------------------------
# 2. Remove timestamp
# ---------------------------------------------------------

train = train.drop(columns=["timestamp"])
test = test.drop(columns=["timestamp"])


# ---------------------------------------------------------
# 3. Separate target and features
# ---------------------------------------------------------

TARGET = "aqi"

X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]

X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]


print("\nTarget variable:")
print(TARGET)

print("\nTraining feature shape:")
print(X_train.shape)

print("\nTesting feature shape:")
print(X_test.shape)


# ---------------------------------------------------------
# 4. Identify categorical columns
# ---------------------------------------------------------

categorical_features = ["city"]

numeric_features = [
    column
    for column in X_train.columns
    if column not in categorical_features
]


# ---------------------------------------------------------
# 5. Preprocessing
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "city",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ---------------------------------------------------------
# 6. Define models
# ---------------------------------------------------------

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ---------------------------------------------------------
# 7. Train and evaluate models
# ---------------------------------------------------------

results = []

best_model = None
best_model_name = None
best_rmse = float("inf")


for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training {name}...")
    print("-" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

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

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    if rmse < best_rmse:

        best_rmse = rmse
        best_model = pipeline
        best_model_name = name


# ---------------------------------------------------------
# 8. Save results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs(
    "outputs",
    exist_ok=True
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ---------------------------------------------------------
# 9. Save best model
# ---------------------------------------------------------

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model_path = os.path.join(
    MODEL_DIR,
    "best_aqi_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)


# ---------------------------------------------------------
# 10. Final report
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nBest model:")
print(best_model_name)

print(f"\nBest RMSE: {best_rmse:.4f}")

print(f"\nResults saved to:")
print(RESULT_FILE)

print("\nBest model saved to:")
print(model_path)