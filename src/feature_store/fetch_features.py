import os
import sys
import pandas as pd

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.feature_store.hopsworks_connection import get_feature_store

def get_training_data(fallback_csv_path="data/processed_features.csv", version=3):
    """
    Attempts to fetch training data from Hopsworks Feature View.
    Falls back to local CSV if Hopsworks or Feature View is unavailable.
    """
    try:
        print("Attempting to connect to Hopsworks Feature Store...")
        fs = get_feature_store()
        
        print("Retrieving 'aqi_features_view' Feature View (Version 1)...")
        try:
            fv = fs.get_feature_view(name="aqi_features_view", version=1)
        except Exception:
            print("Feature View not found. Creating it idempotently from Feature Group...")
            fg = fs.get_feature_group(name="aqi_features", version=version)
            fv = fs.create_feature_view(
                name="aqi_features_view", 
                query=fg.select_all(), 
                version=1
            )
        
        print("Reading Feature View data via batch interface...")
        df = fv.get_batch_data()
        
        print("Connected to Hopsworks")
        print(f"Feature View: {fv.name}")
        print(f"Version: {fv.version}")
        print("Successfully retrieved data")
        print(f"Rows retrieved: {len(df)}")
        print("Training data source: Hopsworks Feature View (aqi_features_view)")
        
        # Sort chronologically to preserve time-series nature
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    except Exception as e:
        print(f"WARNING: Hopsworks Feature View unavailable: {e}")
        print(f"Training data source: Local CSV fallback ({fallback_csv_path})")
        
        if os.path.exists(fallback_csv_path):
            df = pd.read_csv(fallback_csv_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
        else:
            raise FileNotFoundError(f"Local fallback CSV not found at {fallback_csv_path}")

if __name__ == "__main__":
    df = get_training_data()
    print("\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Cities: {df['city'].unique().tolist()}")
    print(f"Target column 'target_aqi' present: {'target_aqi' in df.columns}")

