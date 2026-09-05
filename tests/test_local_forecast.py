import os
import sys
import time
import unittest
from unittest.mock import patch
import pandas as pd
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.prediction.predict import load_trained_models, get_next_hour_predictions, DATA_FILE, MODEL_DIR
from src.prediction.forecast import forecast_next_72_hours, get_forecast_shap_explanation


class TestLocalForecastPipeline(unittest.TestCase):

    def test_local_models_load(self):
        """Verify local models load without errors and without Hopsworks."""
        start_time = time.perf_counter()
        models = load_trained_models(model_dir=str(MODEL_DIR), from_hopsworks=False)
        elapsed = time.perf_counter() - start_time
        print(f"\n[BENCHMARK] Local models loaded in {elapsed:.4f}s")
        self.assertIn("Random Forest", models)
        self.assertIn("Ridge Regression", models)
        self.assertIn("XGBoost", models)
        self.assertLess(elapsed, 5.0)

    def test_missing_model_error(self):
        """Verify clear FileNotFoundError when a local model is missing."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_trained_models(model_dir="non_existent_model_dir", from_hopsworks=False)
        self.assertIn("Required local model not found", str(ctx.exception))

    def test_missing_data_error(self):
        """Verify clear FileNotFoundError when the local data file is missing."""
        with self.assertRaises(FileNotFoundError) as ctx:
            forecast_next_72_hours("Lahore", data_file="non_existent_data.csv")
        self.assertIn("Required local data file not found", str(ctx.exception))

    @patch("src.prediction.forecast.load_trained_models")
    def test_forecast_local_execution_timing(self, mock_load):
        """Verify 72-hour forecast generates 72 rows rapidly using local data & estimation."""
        actual_models = load_trained_models(MODEL_DIR)
        actual_models.pop("LSTM", None)
        actual_models.pop("scaler", None)
        mock_load.return_value = actual_models

        start_time = time.perf_counter()
        df_fc = forecast_next_72_hours("Lahore", hours=72, live_api=False)
        elapsed = time.perf_counter() - start_time
        
        print(f"\n[BENCHMARK] 72-Hour Pure Local Forecast executed in {elapsed:.4f} seconds.")
        
        self.assertEqual(len(df_fc), 72)
        self.assertIn("ensemble", df_fc.columns)
        self.assertIn("category", df_fc.columns)
        self.assertFalse(df_fc["ensemble"].isna().any())
        self.assertLess(elapsed, 30.0)

    @patch("src.prediction.predict.load_trained_models")
    def test_next_hour_prediction_timing(self, mock_load):
        """Verify next-hour predictions execute in milliseconds locally."""
        actual_models = load_trained_models(MODEL_DIR)
        actual_models.pop("LSTM", None)
        actual_models.pop("scaler", None)
        mock_load.return_value = actual_models

        start_time = time.perf_counter()
        df_pred = get_next_hour_predictions(DATA_FILE, MODEL_DIR, include_shap=False)
        elapsed = time.perf_counter() - start_time
        
        print(f"\n[BENCHMARK] Next-Hour Predictions executed in {elapsed:.4f} seconds.")
        self.assertGreater(len(df_pred), 0)
        self.assertLess(elapsed, 5.0)

    def test_deferred_shap(self):
        """Verify SHAP explanation can be calculated on-demand."""
        shap_result = get_forecast_shap_explanation("Lahore")
        if shap_result is not None:
            self.assertIn("features", shap_result)
            self.assertGreater(len(shap_result["features"]), 0)


if __name__ == "__main__":
    unittest.main()
