import os
import sys
import hopsworks
from dotenv import load_dotenv

# Workaround for Hopsworks SDK bug on Windows where it hardcodes '/tmp' for certificates
if sys.platform == 'win32':
    try:
        os.makedirs('/tmp', exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create /tmp directory required by Hopsworks on Windows: {e}")

def get_hopsworks_project():
    """
    Connects to Hopsworks and returns the project object.
    Requires HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME environment variables.
    """
    load_dotenv()
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
    
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY environment variable not found. Please set it in .env")
        
    if not project_name:
        raise ValueError("HOPSWORKS_PROJECT_NAME environment variable not found. Please set it in .env")
        
    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name
    )
    
    return project

def get_feature_store():
    """
    Returns the feature store for the current Hopsworks project.
    """
    project = get_hopsworks_project()
    return project.get_feature_store()
