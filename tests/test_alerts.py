import unittest
from unittest.mock import patch, MagicMock
import json
from pathlib import Path
import pandas as pd
import shutil

from src.alerts.alert_service import (
    load_alert_state,
    save_alert_state,
    check_and_trigger_alerts,
    CATEGORY_SEVERITY
)

TEST_DIR = Path("tests/scratch_alerts_test")
TEST_STATE_FILE = TEST_DIR / "alert_state.json"

class TestAQIAlerts(unittest.TestCase):
    
    def setUp(self):
        # Create test sandbox directory
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        if TEST_STATE_FILE.exists():
            TEST_STATE_FILE.unlink()

    def tearDown(self):
        # Cleanup test sandbox directory
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)

    def test_load_state_missing_file(self):
        """Test 7: Load state when file is missing should return empty dict."""
        state = load_alert_state(TEST_STATE_FILE)
        self.assertEqual(state, {})

    def test_load_state_corrupt_file(self):
        """Test 8: Load state when file is corrupt should handle safely and return empty dict."""
        with open(TEST_STATE_FILE, "w") as f:
            f.write("{invalid_json: 123")
        state = load_alert_state(TEST_STATE_FILE)
        self.assertEqual(state, {})

    @patch("src.alerts.alert_service.send_aqi_email")
    def test_alert_flow(self, mock_send_email):
        """
        Verify alert levels, initial warning, duplicate filtering,
        escalation, recovery, and re-triggering.
        """
        # Set environment threshold to Unhealthy
        with patch.dict("os.environ", {"AQI_ALERT_LEVEL": "Unhealthy", "ALERT_EMAIL_ENABLED": "true"}):
            
            # --- TEST 1: Below Threshold ---
            predictions_below = pd.DataFrame([{
                "city": "Lahore",
                "next_hour_aqi": 90.0,
                "next_hour_category": "Moderate",
                "prediction_timestamp": "2026-08-26T23:00:00"
            }])
            check_and_trigger_alerts(predictions_below, TEST_STATE_FILE)
            mock_send_email.assert_not_called()
            
            state = load_alert_state(TEST_STATE_FILE)
            self.assertFalse(state.get("Lahore", {}).get("active", False))

            # --- TEST 2: First Threshold Crossing ---
            predictions_at = pd.DataFrame([{
                "city": "Lahore",
                "next_hour_aqi": 180.0,
                "next_hour_category": "Unhealthy",
                "prediction_timestamp": "2026-08-26T23:00:00"
            }])
            check_and_trigger_alerts(predictions_at, TEST_STATE_FILE)
            self.assertEqual(mock_send_email.call_count, 1)
            mock_send_email.assert_called_with(
                city="Lahore",
                aqi=180.0,
                category="Unhealthy",
                timestamp="2026-08-26T23:00:00",
                recommendation=unittest.mock.ANY,
                threshold="Unhealthy"
            )
            
            state = load_alert_state(TEST_STATE_FILE)
            self.assertTrue(state["Lahore"]["active"])
            self.assertEqual(state["Lahore"]["last_alerted_category"], "Unhealthy")
            
            # Reset mock for duplicate check
            mock_send_email.reset_mock()

            # --- TEST 3: Duplicate Alert ---
            predictions_dup = pd.DataFrame([{
                "city": "Lahore",
                "next_hour_aqi": 190.0,
                "next_hour_category": "Unhealthy",
                "prediction_timestamp": "2026-08-27T00:00:00"
            }])
            check_and_trigger_alerts(predictions_dup, TEST_STATE_FILE)
            mock_send_email.assert_not_called() # Should not duplicate email alert

            # --- TEST 4: Severity Escalation ---
            predictions_escalate = pd.DataFrame([{
                "city": "Lahore",
                "next_hour_aqi": 280.0,
                "next_hour_category": "Very Unhealthy",
                "prediction_timestamp": "2026-08-27T01:00:00"
            }])
            check_and_trigger_alerts(predictions_escalate, TEST_STATE_FILE)
            self.assertEqual(mock_send_email.call_count, 1)
            mock_send_email.assert_called_with(
                city="Lahore",
                aqi=280.0,
                category="Very Unhealthy",
                timestamp="2026-08-27T01:00:00",
                recommendation=unittest.mock.ANY,
                threshold="Unhealthy"
            )
            
            state = load_alert_state(TEST_STATE_FILE)
            self.assertEqual(state["Lahore"]["last_alerted_category"], "Very Unhealthy")
            
            # Reset mock for recovery test
            mock_send_email.reset_mock()

            # --- TEST 5: Recovery ---
            predictions_recovery = pd.DataFrame([{
                "city": "Lahore",
                "next_hour_aqi": 80.0,
                "next_hour_category": "Good",
                "prediction_timestamp": "2026-08-27T02:00:00"
            }])
            check_and_trigger_alerts(predictions_recovery, TEST_STATE_FILE)
            mock_send_email.assert_not_called() # No recovery email required in this phase
            
            state = load_alert_state(TEST_STATE_FILE)
            self.assertFalse(state["Lahore"]["active"])
            self.assertIsNone(state["Lahore"]["last_alerted_category"])

            # --- TEST 6: New Event ---
            predictions_new = pd.DataFrame([{
                "city": "Lahore",
                "next_hour_aqi": 210.0,
                "next_hour_category": "Unhealthy",
                "prediction_timestamp": "2026-08-27T03:00:00"
            }])
            check_and_trigger_alerts(predictions_new, TEST_STATE_FILE)
            self.assertEqual(mock_send_email.call_count, 1)
            
            state = load_alert_state(TEST_STATE_FILE)
            self.assertTrue(state["Lahore"]["active"])
            self.assertEqual(state["Lahore"]["last_alerted_category"], "Unhealthy")
