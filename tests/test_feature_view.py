import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd

from src.feature_store.fetch_features import get_training_data
from src.prediction.predict import FEATURES

class TestFeatureView(unittest.TestCase):

    @patch("src.feature_store.fetch_features.get_feature_store")
    def test_existing_feature_view(self, mock_get_store):
        """Test 1: If Feature View exists, retrieve it and call get_batch_data."""
        mock_fs = MagicMock()
        mock_fv = MagicMock()
        mock_fv.name = "aqi_features_view"
        mock_fv.version = 1
        
        # Setup dummy returned dataframe
        dummy_df = pd.DataFrame([{
            "city": "Lahore",
            "timestamp": "2026-08-26T23:00:00",
            "temperature": 32.5,
            "humidity": 65.0,
            "wind_speed": 3.0,
            "target_aqi": 3
        }])
        mock_fv.get_batch_data.return_value = dummy_df
        mock_fs.get_feature_view.return_value = mock_fv
        mock_get_store.return_value = mock_fs

        df = get_training_data(fallback_csv_path="dummy.csv")
        
        mock_fs.get_feature_view.assert_called_with(name="aqi_features_view", version=1)
        mock_fv.get_batch_data.assert_called_once()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["city"], "Lahore")

    @patch("src.feature_store.fetch_features.get_feature_store")
    def test_feature_view_missing_created_idempotently(self, mock_get_store):
        """Test 2 & 5: If Feature View is missing, create it idempotently from verified Feature Group."""
        mock_fs = MagicMock()
        mock_fg = MagicMock()
        mock_fv = MagicMock()
        mock_fv.name = "aqi_features_view"
        mock_fv.version = 1

        # Simulate get_feature_view raising exception (not found)
        mock_fs.get_feature_view.side_effect = Exception("Not found")
        mock_fs.get_feature_group.return_value = mock_fg
        mock_fs.create_feature_view.return_value = mock_fv
        
        dummy_df = pd.DataFrame([{
            "city": "Lahore",
            "timestamp": "2026-08-26T23:00:00",
            "temperature": 32.5,
            "humidity": 65.0,
            "wind_speed": 3.0,
            "target_aqi": 3
        }])
        mock_fv.get_batch_data.return_value = dummy_df
        mock_get_store.return_value = mock_fs

        df = get_training_data(fallback_csv_path="dummy.csv")

        mock_fs.get_feature_view.assert_called_with(name="aqi_features_view", version=1)
        mock_fs.get_feature_group.assert_called_with(name="aqi_features", version=3)
        mock_fs.create_feature_view.assert_called_once_with(
            name="aqi_features_view",
            query=mock_fg.select_all(),
            version=1
        )
        mock_fv.get_batch_data.assert_called_once()
        self.assertEqual(len(df), 1)

    @patch("src.feature_store.fetch_features.get_feature_store")
    @patch("os.path.exists")
    @patch("pandas.read_csv")
    def test_local_fallback_on_unavailability(self, mock_read_csv, mock_exists, mock_get_store):
        """Test 3: If Hopsworks is completely unavailable, log warning and use local CSV fallback."""
        mock_get_store.side_effect = Exception("Connection Timeout")
        mock_exists.return_value = True
        
        dummy_df = pd.DataFrame([{
            "city": "Lahore",
            "timestamp": "2026-08-26T23:00:00",
            "target_aqi": 3
        }])
        mock_read_csv.return_value = dummy_df

        df = get_training_data(fallback_csv_path="data/processed_features.csv")
        
        mock_read_csv.assert_called_with("data/processed_features.csv")
        self.assertEqual(len(df), 1)

    def test_no_target_leakage_in_features(self):
        """Test 4: Verify that the prediction features list does not contain the prediction target."""
        self.assertNotIn("target_aqi", FEATURES)
        self.assertNotIn("aqi", FEATURES)
