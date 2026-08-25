from flask import Flask, jsonify
import sys
import os
import traceback
import pandas as pd

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.prediction.predict import get_next_hour_predictions

app = Flask(__name__)

def clean_record(record_dict):
    """Utility to clean up pandas row dictionary for JSON serialization."""
    cleaned = {}
    for k, v in record_dict.items():
        if isinstance(v, (pd.Timestamp, pd.DatetimeIndex)):
            cleaned[k] = v.isoformat()
        elif isinstance(v, (dict, list)):
            cleaned[k] = v
        elif pd.isna(v):
            cleaned[k] = None
        else:
            cleaned[k] = v
    return cleaned

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "AQI Predictor API"
    }), 200

@app.route('/api/prediction/<city>', methods=['GET'])
def get_prediction(city):
    """
    Returns the latest AQI prediction for the requested city.
    Uses the existing prediction pipeline.
    """
    try:
        predictions_df = get_next_hour_predictions()
        
        # Filter for the requested city (case insensitive)
        city_pred = predictions_df[predictions_df['city'].str.lower() == city.lower()]
        
        if city_pred.empty:
            return jsonify({
                "status": "error",
                "message": f"No predictions found for city: {city}"
            }), 404
            
        # Convert to dictionary and clean timestamps/NaNs
        raw_result = city_pred.iloc[0].to_dict()
        result = clean_record(raw_result)
        
        return jsonify({
            "status": "success",
            "data": result
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/aqi/<city>', methods=['GET'])
def get_current_aqi(city):
    """
    Returns the latest available AQI information for the requested city.
    """
    try:
        predictions_df = get_next_hour_predictions()
        city_pred = predictions_df[predictions_df['city'].str.lower() == city.lower()]
        
        if city_pred.empty:
            return jsonify({
                "status": "error",
                "message": f"No data found for city: {city}"
            }), 404
            
        row = city_pred.iloc[0]
        
        timestamp_val = row["current_timestamp"]
        if hasattr(timestamp_val, "isoformat"):
            timestamp_str = timestamp_val.isoformat()
        else:
            timestamp_str = str(timestamp_val)
        
        # Structure actual data using project's existing fields
        response_data = {
            "city": row["city"],
            "timestamp": timestamp_str,
            "aqi": row["current_aqi"],
            "category": row["current_category"],
            "pollutants": row.get("pollutants", {}),
            "weather": row.get("weather", {})
        }
        
        return jsonify({
            "status": "success",
            "data": response_data
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
