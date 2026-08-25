import shap
import pandas as pd
import numpy as np

def generate_global_explanation(model, background_data, feature_names):
    """
    Generates global feature importance using SHAP for tree-based models.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(background_data)
        
        # Calculate mean absolute SHAP values for global importance
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": mean_shap
        }).sort_values(by="importance", ascending=False)
        
        return importance_df
    except Exception as e:
        print(f"Error generating global SHAP explanation: {e}")
        return None

def generate_local_explanation(model, instance, feature_names):
    """
    Generates local SHAP explanation for a single prediction instance.
    """
    try:
        explainer = shap.TreeExplainer(model)
        # TreeExplainer expects 2D array
        instance_2d = instance.reshape(1, -1) if len(instance.shape) == 1 else instance
        shap_values = explainer.shap_values(instance_2d)
        
        # Get base value (expected value)
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]
            
        explanation = {
            "base_value": float(expected_value),
            "features": []
        }
        
        # Map values
        sv = shap_values[0] if len(shap_values.shape) == 2 else shap_values
        
        for i, feature in enumerate(feature_names):
            explanation["features"].append({
                "name": feature,
                "value": float(instance_2d[0][i]),
                "contribution": float(sv[i])
            })
            
        # Sort by absolute contribution
        explanation["features"].sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return explanation
        
    except Exception as e:
        print(f"Error generating local SHAP explanation: {e}")
        return None
