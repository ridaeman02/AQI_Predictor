import os
import sys
import pandas as pd

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.feature_store.hopsworks_connection import get_feature_store

def inspect_feature_groups():
    print("Connecting to Hopsworks Feature Store...")
    fs = get_feature_store()
    print(f"Feature Store Name: {fs.name}")
    
    try:
        fgs = fs.get_feature_groups(name="aqi_features")
    except Exception as e:
        print(f"Error fetching feature groups: {e}")
        return

    print(f"\nFound {len(fgs)} Feature Group(s) matching 'aqi_features':\n")
    for fg in fgs:
        print("=" * 60)
        print(f"Name: {fg.name}")
        print(f"Version: {fg.version}")
        print(f"Description: {fg.description}")
        print(f"Online Enabled: {fg.online_enabled}")
        print(f"Time Travel Format: {fg.time_travel_format}")
        print(f"Primary Key: {fg.primary_key}")
        print(f"Event Time: {fg.event_time}")
        print(f"Features: {[f.name for f in fg.features]}")
        
        # Try reading record count or dataframe
        try:
            df = fg.read()
            print(f"Readable Data: YES - {len(df)} records")
        except Exception as read_err:
            print(f"Readable Data: NO - Error reading: {read_err}")
        print("=" * 60)

if __name__ == "__main__":
    inspect_feature_groups()
