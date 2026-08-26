import os
import json
import tempfile
from pathlib import Path
from src.utils.aqi_categories import get_aqi_category
from src.alerts.email_alert import send_aqi_email

# Numerical mappings for category comparisons
CATEGORY_SEVERITY = {
    "Good": 1,
    "Moderate": 2,
    "Unhealthy for Sensitive Groups": 3,
    "Unhealthy": 4,
    "Very Unhealthy": 5
}

RECOMMENDATIONS = {
    "Good": "Air quality is satisfactory, and air pollution poses little or no risk.",
    "Moderate": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups may experience health effects. The general public is less likely to be affected.",
    "Unhealthy": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects. Consider reducing prolonged outdoor activity.",
    "Very Unhealthy": "Health alert: The risk of health effects is increased for everyone. Avoid prolonged outdoor exertion."
}

STATE_FILE = Path("data/alert_state.json")

def load_alert_state(state_file=STATE_FILE):
    """
    Safely loads the alert state from a JSON file.
    If the file is missing or corrupted, returns an empty dictionary.
    """
    if not state_file.exists():
        return {}
    
    try:
        with open(state_file, "r") as f:
            data = f.read().strip()
            if not data:
                return {}
            return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"Warning: Corrupt alert state file detected: {e}. Re-initializing state.")
        return {}
    except Exception as e:
        print(f"Warning: Could not read alert state file: {e}. Re-initializing state.")
        return {}

def save_alert_state(state, state_file=STATE_FILE):
    """
    Atomically saves the alert state to a JSON file by using a temporary file.
    """
    state_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to a temporary file in the same directory to allow atomic replacement
    fd, temp_path_str = tempfile.mkstemp(dir=state_file.parent, prefix="alert_state_", suffix=".tmp")
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        # Atomic replace
        temp_path.replace(state_file)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to atomically write alert state file: {e}")

def check_and_trigger_alerts(predictions_df, state_file=STATE_FILE):
    """
    Evaluates predictions against the alert threshold and triggers notifications.
    """
    # Load alert threshold level
    alert_level_name = os.getenv("AQI_ALERT_LEVEL", "Unhealthy").strip()
    if alert_level_name not in CATEGORY_SEVERITY:
        print(f"Warning: Invalid AQI_ALERT_LEVEL '{alert_level_name}'. Defaulting to 'Unhealthy'.")
        alert_level_name = "Unhealthy"

    threshold_severity = CATEGORY_SEVERITY[alert_level_name]
    state = load_alert_state(state_file)

    for _, row in predictions_df.iterrows():
        city = row["city"]
        predicted_aqi = float(row["next_hour_aqi"])
        category = row["next_hour_category"]
        timestamp = row["prediction_timestamp"]
        
        # Ensure category is mapped to severity
        current_severity = CATEGORY_SEVERITY.get(category, 5) # Default to highest severity if unknown
        recommendation = RECOMMENDATIONS.get(category, "Avoid prolonged outdoor exertion.")

        city_state = state.get(city, {"active": False, "last_alerted_category": None})

        if current_severity >= threshold_severity:
            # Determine if we should send an email
            should_alert = False
            is_escalation = False

            if not city_state["active"]:
                should_alert = True
            else:
                last_cat = city_state["last_alerted_category"]
                last_severity = CATEGORY_SEVERITY.get(last_cat, 0)
                if current_severity > last_severity:
                    should_alert = True
                    is_escalation = True

            if should_alert:
                alert_type = "Escalation Alert" if is_escalation else "Alert"
                print(f"Triggering {alert_type} for {city}. AQI: {predicted_aqi:.2f} ({category})")
                
                # Send SMTP email
                send_aqi_email(
                    city=city,
                    aqi=predicted_aqi,
                    category=category,
                    timestamp=timestamp,
                    recommendation=recommendation,
                    threshold=alert_level_name
                )
                
                # Update state
                state[city] = {
                    "active": True,
                    "last_alerted_category": category
                }
            else:
                print(f"AQI threshold reached for {city} but duplicate alert prevented (Category remains {category}).")
        else:
            # Recovery / Normal level
            if city_state["active"]:
                print(f"AQI recovered for {city}. Resetting alert state (Previous was {city_state['last_alerted_category']}).")
                state[city] = {
                    "active": False,
                    "last_alerted_category": None
                }

    save_alert_state(state, state_file)
