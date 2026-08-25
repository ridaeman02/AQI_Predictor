import os
import sys
import pandas as pd
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.feature_store.hopsworks_connection import get_feature_store

def upload_features():
    """
    Loads historical processed features and uploads them to the Hopsworks feature store.
    """
    print("Starting feature upload process...")
    
    csv_path = "data/processed_features.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed features file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Validation checks
    required_columns = ["city", "timestamp", "target_aqi"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
            
    # Ensure correct data types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Loaded {len(df)} records from {csv_path}.")
    
    # Drop duplicates just in case
    df = df.drop_duplicates(subset=["city", "timestamp"])
    print(f"Total unique records to upload: {len(df)}")
    
    # Get feature store
    try:
        fs = get_feature_store()
    except Exception as e:
        print(f"Failed to connect to Hopsworks: {e}")
        sys.exit(1)
        
    # Create or get feature group
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=3,
        description="AQI predictions dataset with weather and lag features",
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=False,
        time_travel_format="HUDI"
    )
    
    # Upload data
    print("Uploading data to Hopsworks Feature Group 'aqi_features'...")
    aqi_fg.insert(df, write_options={"wait_for_job": True})
    
    print(f"Successfully uploaded and materialized {len(df)} records in Hopsworks Feature Store.")

if __name__ == "__main__":
    upload_features()
