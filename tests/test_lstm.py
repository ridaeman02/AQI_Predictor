import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

from src.models.lstm_model import create_lstm_sequences, build_lstm_model
from src.prediction.predict import load_trained_models
from src.training.train_model import FEATURES, TARGET

class TestLSTMModel(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset with 2 cities and 50 sequential rows each
        data = []
        for city in ["Lahore", "Karachi"]:
            for i in range(50):
                row = {
                    "city": city,
                    "timestamp": pd.Timestamp("2026-08-27") + pd.Timedelta(hours=i),
                    "target_aqi": float(100 + i * 2)
                }
                for f in FEATURES:
                    # Sequential feature values
                    row[f] = float(i)
                data.append(row)
        self.df = pd.DataFrame(data)

    def test_sequence_creation_shape(self):
        """Test sequence creation returns correct shapes for Lahore and Karachi."""
        seq_len = 24
        X_seq, y_seq = create_lstm_sequences(self.df, FEATURES, TARGET, sequence_length=seq_len)
        
        # Total rows = 50 per city.
        # For each city, we construct sequences from index 24 to 49 (which is 26 sequences).
        # Total sequences = 26 * 2 = 52.
        self.assertEqual(X_seq.shape, (52, seq_len, len(FEATURES)))
        self.assertEqual(y_seq.shape, (52,))

    def test_no_future_leakage_in_sequences(self):
        """Test that y_seq contains strictly future target values relative to X_seq windows."""
        seq_len = 24
        X_seq, y_seq = create_lstm_sequences(self.df, FEATURES, TARGET, sequence_length=seq_len)
        
        # Verify that for the first sequence, the features are from steps 0..23, and target is from step 24
        # Since step i has feature values equal to i, X_seq[0, -1, 0] (last step of first sequence) should be 23.0
        # The target y_seq[0] should be 100 + 24 * 2 = 148.0
        self.assertEqual(X_seq[0, -1, 0], 23.0)
        self.assertEqual(y_seq[0], 148.0)

    def test_scaling_split_discipline(self):
        """Test scaler is fit ONLY on training features to prevent scale leakage."""
        split_index = int(len(self.df) * 0.8)
        train_df = self.df.iloc[:split_index].copy()
        test_df = self.df.iloc[split_index:].copy()
        
        scaler = StandardScaler()
        # Scale training
        X_train_scaled = scaler.fit_transform(train_df[FEATURES])
        # Transform testing (no fit!)
        X_test_scaled = scaler.transform(test_df[FEATURES])
        
        # Check that scaler mean/variance were calculated ONLY on training data
        self.assertAlmostEqual(scaler.mean_[0], train_df[FEATURES].iloc[:, 0].mean())

    def test_model_construction(self):
        """Test LSTM model compiles with correct input shape."""
        input_shape = (24, len(FEATURES))
        model = build_lstm_model(input_shape)
        
        # Verify layer structure
        self.assertEqual(model.input_shape, (None, 24, len(FEATURES)))
        self.assertEqual(model.layers[0].name.startswith("lstm"), True)
        self.assertEqual(model.layers[-1].units, 1)

    def test_prediction_output_shape(self):
        """Test that model runs prediction and returns expected shape."""
        input_shape = (24, len(FEATURES))
        model = build_lstm_model(input_shape)
        
        # Generate dummy input sequence
        dummy_input = np.random.rand(1, 24, len(FEATURES)).astype(np.float32)
        preds = model.predict(dummy_input, verbose=0)
        
        self.assertEqual(preds.shape, (1, 1))

    def test_city_isolation(self):
        """Test that sequences generated are city-isolated and do not mix data between cities."""
        # Create distinct features for each city
        data = []
        for city_val, city_name in enumerate(["Lahore", "Karachi"]):
            for i in range(30):
                row = {
                    "city": city_name,
                    "timestamp": pd.Timestamp("2026-08-27") + pd.Timedelta(hours=i),
                    "target_aqi": float(i)
                }
                for f in FEATURES:
                    row[f] = float(city_val + 1)  # Lahore features are 1.0, Karachi are 2.0
                data.append(row)
        df_isolated = pd.DataFrame(data)

        X_seq, _ = create_lstm_sequences(df_isolated, FEATURES, TARGET, sequence_length=5)
        
        # Check each generated sequence to make sure it contains only 1.0s or only 2.0s
        for seq in X_seq:
            first_val = seq[0, 0]
            self.assertTrue(first_val in [1.0, 2.0])
            for step in range(len(seq)):
                for feat in range(len(FEATURES)):
                    self.assertEqual(seq[step, feat], first_val)

    @patch("os.path.exists")
    def test_missing_model_fallback(self, mock_exists):
        """Verify load_trained_models continues working and ignores LSTM if it is missing."""
        def side_effect(path):
            if "lstm" in str(path):
                return False
            return True
        mock_exists.side_effect = side_effect
        
        with patch("joblib.load") as mock_load:
            mock_load.return_value = "dummy_model"
            models = load_trained_models(model_dir="dummy_dir", from_hopsworks=False)
            
            self.assertIn("Random Forest", models)
            self.assertIn("Ridge Regression", models)
            self.assertIn("XGBoost", models)
            self.assertNotIn("LSTM", models)
