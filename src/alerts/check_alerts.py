import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.prediction.predict import get_next_hour_predictions
from src.alerts.alert_service import check_and_trigger_alerts

def main():
    print("=" * 60)
    # 1. Generate predictions using existing pipeline logic
    try:
        print("Generating next-hour AQI predictions...")
        pred_df = get_next_hour_predictions()
        print("Predictions generated successfully.")
    except Exception as e:
        print(f"Error: Failed to generate next-hour predictions: {e}")
        sys.exit(1)

    # 2. Check alert threshold and trigger alerts
    try:
        print("Evaluating predictions against alert threshold...")
        check_and_trigger_alerts(pred_df)
        print("Alert check complete.")
    except Exception as e:
        print(f"Error: AQI alert evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
