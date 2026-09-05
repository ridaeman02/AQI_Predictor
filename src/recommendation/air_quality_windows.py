import pandas as pd
import numpy as np

ACTIVITY_PROFILES = {
    "General Outdoor": {
        "aqi_multiplier": 1.0,
        "temp_min": 10.0, "temp_max": 32.0, "temp_penalty_factor": 2.5,
        "hum_max": 75.0, "hum_penalty_factor": 0.5,
        "wind_max": 25.0, "wind_penalty_factor": 1.0,
        "aqi_avoid_threshold": 150,
    },
    "Walking": {
        "aqi_multiplier": 1.1,
        "temp_min": 10.0, "temp_max": 30.0, "temp_penalty_factor": 2.5,
        "hum_max": 70.0, "hum_penalty_factor": 0.6,
        "wind_max": 25.0, "wind_penalty_factor": 1.0,
        "aqi_avoid_threshold": 150,
    },
    "Running": {
        "aqi_multiplier": 1.5,
        "temp_min": 5.0, "temp_max": 26.0, "temp_penalty_factor": 3.0,
        "hum_max": 65.0, "hum_penalty_factor": 0.8,
        "wind_max": 20.0, "wind_penalty_factor": 1.2,
        "aqi_avoid_threshold": 100, # High exertion requires stricter air quality
    },
    "Cycling": {
        "aqi_multiplier": 1.4,
        "temp_min": 8.0, "temp_max": 28.0, "temp_penalty_factor": 3.0,
        "hum_max": 70.0, "hum_penalty_factor": 0.7,
        "wind_max": 18.0, "wind_penalty_factor": 2.0, # High wind sensitivity
        "aqi_avoid_threshold": 100,
    },
    "Outdoor Work": {
        "aqi_multiplier": 1.2,
        "temp_min": 5.0, "temp_max": 30.0, "temp_penalty_factor": 2.0,
        "hum_max": 75.0, "hum_penalty_factor": 0.6,
        "wind_max": 30.0, "wind_penalty_factor": 0.8,
        "aqi_avoid_threshold": 150,
    }
}

def generate_explanation(classification, aqi, temp, activity):
    """Generates a human-readable environmental suitability explanation."""
    act_name = activity.lower()
    if classification == "BEST":
        return f"Ideal environmental conditions for {act_name}. Air quality is clean and weather is pleasant."
    elif classification == "GOOD":
        return f"Suitable environmental conditions for {act_name}."
    elif classification == "CAUTION":
        if pd.isna(aqi):
            return "Proceed with caution. Incomplete air quality data."
        elif aqi > 100:
            return f"Environmental conditions are less than ideal for {act_name} due to elevated AQI."
        else:
            return f"Air quality is fair, but weather conditions may be less comfortable for {act_name}."
    else:  # AVOID
        if not pd.isna(aqi) and aqi > ACTIVITY_PROFILES[activity]["aqi_avoid_threshold"]:
            return f"Unfavorable environmental conditions for {act_name} due to poor air quality."
        else:
            return f"Unfavorable environmental conditions for {act_name} (extreme weather or elevated pollution)."

def calculate_hourly_score(row, activity="General Outdoor"):
    """Calculates outdoor suitability score (0-100) and classification based on activity profile."""
    prof = ACTIVITY_PROFILES.get(activity, ACTIVITY_PROFILES["General Outdoor"])
    
    aqi = row.get("ensemble", row.get("predicted_aqi", np.nan))
    temp = row.get("temperature", np.nan)
    hum = row.get("humidity", np.nan)
    wind = row.get("wind_speed", np.nan)
    
    if pd.isna(aqi):
        return 0.0, "AVOID"
        
    base_score = 100.0
    
    # 1. AQI Penalty
    if aqi <= 50:
        aqi_penalty = 0.0
    elif aqi <= 100:
        aqi_penalty = (aqi - 50) * 0.4 * prof["aqi_multiplier"]
    elif aqi <= 150:
        aqi_penalty = (20.0 + (aqi - 100) * 0.8) * prof["aqi_multiplier"]
    else:
        aqi_penalty = (60.0 + (aqi - 150) * 1.5) * prof["aqi_multiplier"]
        
    # 2. Temperature Penalty
    temp_penalty = 0.0
    if not pd.isna(temp):
        if temp < prof["temp_min"]:
            temp_penalty = (prof["temp_min"] - temp) * prof["temp_penalty_factor"]
        elif temp > prof["temp_max"]:
            temp_penalty = (temp - prof["temp_max"]) * prof["temp_penalty_factor"]
            
    # 3. Humidity Penalty
    hum_penalty = 0.0
    if not pd.isna(hum):
        if hum > prof["hum_max"]:
            hum_penalty = (hum - prof["hum_max"]) * prof["hum_penalty_factor"]
            
    # 4. Wind Penalty
    wind_penalty = 0.0
    if not pd.isna(wind):
        if wind > prof["wind_max"]:
            wind_penalty = (wind - prof["wind_max"]) * prof["wind_penalty_factor"]
            
    final_score = max(0.0, min(100.0, base_score - aqi_penalty - temp_penalty - hum_penalty - wind_penalty))
    
    # Classification Overrides
    if aqi > prof["aqi_avoid_threshold"] or final_score < 35:
        classification = "AVOID"
    elif aqi > 100 or final_score < 65:
        classification = "CAUTION"
    elif final_score >= 85 and aqi <= 60:
        classification = "BEST"
    else:
        classification = "GOOD"
        
    return final_score, classification

def analyze_forecast_windows(forecast_df, activity="General Outdoor", horizon_hours=None):
    """
    Analyzes forecast records and returns contiguous recommended windows for a specific activity.
    Allows filtering by a specific time horizon (e.g., 24, 48, 72 hours).
    """
    if forecast_df is None or forecast_df.empty:
        return {"best_window": None, "alternative_windows": [], "worst_window": None}
        
    df = forecast_df.copy()
    if "timestamp_dt" not in df.columns:
        if "timestamp" in df.columns:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        else:
            raise ValueError("Input dataframe must contain a 'timestamp' column.")
            
    df = df.sort_values("timestamp_dt").reset_index(drop=True)
    
    if horizon_hours is not None and horizon_hours > 0:
        start_time = df["timestamp_dt"].iloc[0]
        end_cutoff = start_time + pd.Timedelta(hours=horizon_hours)
        df = df[df["timestamp_dt"] < end_cutoff].reset_index(drop=True)
        
    if df.empty:
        return {"best_window": None, "alternative_windows": [], "worst_window": None}
    
    # Apply scoring row by row
    scores, classes = [], []
    for _, row in df.iterrows():
        s, c = calculate_hourly_score(row, activity)
        scores.append(s)
        classes.append(c)
        
    df["suitability_score"] = scores
    df["classification"] = classes
    
    # Group continuous blocks by classification change
    df["block"] = (df["classification"] != df["classification"].shift(1)).cumsum()
    
    windows = []
    aqi_col = "ensemble" if "ensemble" in df.columns else "predicted_aqi"
    
    for _, group in df.groupby("block"):
        start_time = group["timestamp_dt"].iloc[0]
        end_time = group["timestamp_dt"].iloc[-1] + pd.Timedelta(hours=1)
        
        # Aggregations
        avg_aqi = group[aqi_col].mean() if aqi_col in group.columns else np.nan
        min_aqi = group[aqi_col].min() if aqi_col in group.columns else np.nan
        
        avg_temp = group["temperature"].mean() if "temperature" in group.columns else np.nan
        avg_hum = group["humidity"].mean() if "humidity" in group.columns else np.nan
        avg_wind = group["wind_speed"].mean() if "wind_speed" in group.columns else np.nan
        
        avg_score = group["suitability_score"].mean()
        classification = group["classification"].iloc[0]
        
        explanation = generate_explanation(classification, avg_aqi, avg_temp, activity)
        
        windows.append({
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": len(group),
            "suitability_score": round(avg_score, 1),
            "average_aqi": round(avg_aqi, 1) if not pd.isna(avg_aqi) else None,
            "minimum_aqi": round(min_aqi, 1) if not pd.isna(min_aqi) else None,
            "average_temperature": round(avg_temp, 1) if not pd.isna(avg_temp) else None,
            "average_humidity": round(avg_hum, 1) if not pd.isna(avg_hum) else None,
            "average_wind_speed": round(avg_wind, 1) if not pd.isna(avg_wind) else None,
            "classification": classification,
            "explanation": explanation
        })
        
    # Ranking
    favorable = [w for w in windows if w["classification"] in ["BEST", "GOOD"]]
    favorable = sorted(favorable, key=lambda x: x["suitability_score"], reverse=True)
    
    avoids = [w for w in windows if w["classification"] == "AVOID"]
    avoids = sorted(avoids, key=lambda x: x["suitability_score"])  # Worst first
    
    best_window = favorable[0] if favorable else None
    alt_windows = favorable[1:] if len(favorable) > 1 else []
    
    # Fallback if no favorable windows exist
    if not best_window:
        cautions = [w for w in windows if w["classification"] == "CAUTION"]
        cautions = sorted(cautions, key=lambda x: x["suitability_score"], reverse=True)
        if cautions:
            best_window = cautions[0]
            alt_windows = cautions[1:]
            
    worst_window = avoids[0] if avoids else None
    
    return {
        "best_window": best_window,
        "alternative_windows": alt_windows,
        "avoid_windows": avoids,
        "worst_window": worst_window
    }

def analyze_all_activities(forecast_df, horizon_hours=None):
    """Generates recommendations for all configured activities within the given horizon."""
    results = {}
    for activity in ACTIVITY_PROFILES.keys():
        results[activity] = analyze_forecast_windows(forecast_df, activity, horizon_hours)
    return results
