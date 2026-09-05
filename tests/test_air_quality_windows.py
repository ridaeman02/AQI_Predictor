import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.recommendation.air_quality_windows import (
    analyze_forecast_windows, 
    calculate_hourly_score,
    analyze_all_activities,
    ACTIVITY_PROFILES
)

class TestAirQualityWindows(unittest.TestCase):

    def setUp(self):
        now = datetime(2026, 9, 6, 8, 0, 0)
        
        self.mock_data = []
        for i in range(12):
            hour_time = now + timedelta(hours=i)
            
            # 0-2: BEST (Low AQI, perfect temp)
            if i < 3:
                aqi = 30
                temp = 22
            # 3-5: GOOD (Moderate AQI, good temp)
            elif i < 6:
                aqi = 65
                temp = 25
            # 6-8: CAUTION (High AQI)
            elif i < 9:
                aqi = 120
                temp = 28
            # 9-11: AVOID (Hazardous AQI)
            else:
                aqi = 200
                temp = 35
                
            self.mock_data.append({
                "timestamp": hour_time.isoformat(),
                "ensemble": aqi,
                "temperature": temp,
                "humidity": 50,
                "wind_speed": 10
            })
            
        self.df = pd.DataFrame(self.mock_data)

    def test_calculate_hourly_score_best(self):
        row = {"ensemble": 30, "temperature": 20, "humidity": 50, "wind_speed": 10}
        score, classification = calculate_hourly_score(row, "General Outdoor")
        self.assertEqual(classification, "BEST")
        self.assertEqual(score, 100.0)

    def test_calculate_hourly_score_activity_differences(self):
        # AQI 120 is CAUTION for General, but AVOID for Running (threshold 100)
        row = {"ensemble": 120, "temperature": 20, "humidity": 50, "wind_speed": 10}
        
        gen_score, gen_cls = calculate_hourly_score(row, "General Outdoor")
        run_score, run_cls = calculate_hourly_score(row, "Running")
        
        self.assertEqual(gen_cls, "CAUTION")
        self.assertEqual(run_cls, "AVOID")
        
        # Wind 22 is OK for General, heavily penalized for Cycling
        row_wind = {"ensemble": 40, "temperature": 20, "humidity": 50, "wind_speed": 22}
        
        gen_wind_score, gen_wind_cls = calculate_hourly_score(row_wind, "General Outdoor")
        cyc_wind_score, cyc_wind_cls = calculate_hourly_score(row_wind, "Cycling")
        
        self.assertEqual(gen_wind_cls, "BEST")
        # Cycling penalty: (22 - 18) * 2 = 8. Base 100 - 8 = 92. Should still be BEST but score is lower.
        self.assertEqual(cyc_wind_cls, "BEST")
        self.assertTrue(cyc_wind_score < gen_wind_score)

    def test_analyze_forecast_windows(self):
        result = analyze_forecast_windows(self.df, "General Outdoor")
        
        best_window = result["best_window"]
        self.assertIsNotNone(best_window)
        self.assertEqual(best_window["classification"], "BEST")
        self.assertEqual(best_window["duration_hours"], 3)
        self.assertEqual(best_window["average_aqi"], 30.0)
        
        alt_windows = result["alternative_windows"]
        self.assertEqual(len(alt_windows), 1)
        self.assertEqual(alt_windows[0]["classification"], "GOOD")
        
        worst_window = result["worst_window"]
        self.assertIsNotNone(worst_window)
        self.assertEqual(worst_window["classification"], "AVOID")
        self.assertEqual(worst_window["duration_hours"], 3)
        self.assertEqual(worst_window["average_aqi"], 200.0)
        
    def test_analyze_all_activities(self):
        results = analyze_all_activities(self.df)
        self.assertIn("General Outdoor", results)
        self.assertIn("Running", results)
        
    def test_horizons_24_48_72(self):
        # Create a 72-hour dataset
        now = datetime(2026, 9, 6, 12, 0, 0)
        data = []
        for i in range(72):
            data.append({
                "timestamp": (now + timedelta(hours=i)).isoformat(),
                "ensemble": 50, # always good
                "temperature": 25,
                "humidity": 50,
                "wind_speed": 10
            })
        df_72 = pd.DataFrame(data)
        
        # Test 24-hour horizon
        res_24 = analyze_forecast_windows(df_72, horizon_hours=24)
        self.assertEqual(res_24["best_window"]["duration_hours"], 24)
        
        # Test 48-hour horizon
        res_48 = analyze_forecast_windows(df_72, horizon_hours=48)
        self.assertEqual(res_48["best_window"]["duration_hours"], 48)
        
        # Test 72-hour horizon (full set)
        res_72 = analyze_forecast_windows(df_72, horizon_hours=72)
        self.assertEqual(res_72["best_window"]["duration_hours"], 72)
        
        # Test insufficient data (request 72 on a 24h dataframe)
        res_insufficient = analyze_forecast_windows(df_72.head(24), horizon_hours=72)
        self.assertEqual(res_insufficient["best_window"]["duration_hours"], 24)
        
    def test_midnight_crossing(self):
        # Starts at 22:00, crosses midnight to 02:00 next day (5 hours total)
        start_time = datetime(2026, 9, 6, 22, 0, 0)
        data = []
        for i in range(5):
            data.append({
                "timestamp": (start_time + timedelta(hours=i)).isoformat(),
                "ensemble": 40,
                "temperature": 18,
            })
        df_night = pd.DataFrame(data)
        
        res = analyze_forecast_windows(df_night)
        best = res["best_window"]
        
        # Verify it groups as a single continuous 5-hour window
        self.assertEqual(best["duration_hours"], 5)
        self.assertEqual(best["start_time"], "2026-09-06T22:00:00")
        self.assertEqual(best["end_time"], "2026-09-07T03:00:00") # end time is +1 hr from last timestamp

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        result = analyze_forecast_windows(empty_df)
        self.assertIsNone(result["best_window"])
        self.assertEqual(len(result["alternative_windows"]), 0)

if __name__ == '__main__':
    unittest.main()
