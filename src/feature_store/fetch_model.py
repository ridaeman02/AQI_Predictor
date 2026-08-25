import os
import sys
import joblib

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.feature_store.hopsworks_connection import get_hopsworks_project

def get_model_from_hopsworks(model_name, version=None):
    """
    Downloads a model from the Hopsworks Model Registry.
    If version is None, fetches the latest version.
    """
    try:
        project = get_hopsworks_project()
        mr = project.get_model_registry()
        
        # Get model
        if version is None:
            # fetch all versions and get the max, or just get_model(name) often returns latest
            model = mr.get_model(model_name)
        else:
            model = mr.get_model(model_name, version=version)
            
        print(f"Downloading model {model_name} (version {model.version}) from Hopsworks...")
        model_dir = model.download()
        
        # Locate the .pkl file inside the downloaded directory
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl file found in downloaded model directory: {model_dir}")
            
        model_path = os.path.join(model_dir, pkl_files[0])
        return joblib.load(model_path)
        
    except Exception as e:
        print(f"Failed to fetch {model_name} from Hopsworks: {e}")
        return None
