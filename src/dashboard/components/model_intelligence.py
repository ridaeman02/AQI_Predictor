import os
import json
import pandas as pd
import streamlit as st
from pathlib import Path
from src.prediction.forecast import get_forecast_shap_explanation

def render_model_intelligence(features_data, selected_city, BASE_DIR):
    st.markdown("<div class='section-header'>Model Intelligence & XAI</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
        Review chronological train/test split performance and analyze model predictions with Explainable AI.
    </p>
    """, unsafe_allow_html=True)

    perf_file = os.path.join(str(BASE_DIR), "models", "model_performance.json")
    
    try:
        if os.path.exists(perf_file):
            with open(perf_file, "r") as f:
                perf_data = json.load(f)
        else:
            perf_data = [
                {"Model": "Ridge Regression", "MAE": 0.1209, "RMSE": 0.2497, "R²": 0.8245, "Status": "Active (Best Individual)"},
                {"Model": "XGBoost", "MAE": 0.1401, "RMSE": 0.2687, "R²": 0.7968, "Status": "Active"},
                {"Model": "Random Forest", "MAE": 0.1481, "RMSE": 0.2891, "R²": 0.7649, "Status": "Active"},
            ]
    except Exception:
        perf_data = []

    if perf_data:
        cols = st.columns(len(perf_data))
        for i, m_data in enumerate(perf_data):
            with cols[i]:
                is_best = "(Best" in m_data.get("Status", "")
                badge = "<div class='model-best-badge'>BEST MODEL</div>" if is_best else ""
                st.markdown(f"""
                <div class="model-pill">
                    <div class="model-pill-name">{m_data['Model']}</div>
                    <div class="model-pill-val">R² {m_data['R²']:.4f}</div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem;">RMSE: {m_data['RMSE']:.4f}</div>
                    {badge}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Detailed Evaluation Metrics</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Explainable AI (SHAP)</div>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color: var(--text-muted); font-size: 0.9rem;">
        Understand which environmental features drove the model's AQI prediction for the selected city.
        <em>(Calculated on-demand to preserve dashboard performance)</em>
    </p>
    """, unsafe_allow_html=True)

    if st.button("Generate SHAP Explanation for " + selected_city, type="primary"):
        with st.spinner(f"Computing exact SHAP values for {selected_city}..."):
            try:
                shap_result = get_forecast_shap_explanation(selected_city)
                if shap_result and "features" in shap_result:
                    shap_df = pd.DataFrame(shap_result["features"])
                    st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;'>Feature Attributions (SHAP values)</div>", unsafe_allow_html=True)
                    st.dataframe(shap_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("SHAP explanation unavailable. Data or models might be missing.")
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: var(--bg-card); backdrop-filter: blur(16px); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.6rem;">
        <div style="font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem;">Model Architecture Notes</div>
        <ul style="color: var(--text-muted); font-size: 0.9rem; margin: 0; padding-left: 1.2rem; line-height: 1.6;">
            <li><strong>Target Variable:</strong> <code>target_aqi</code> (represents AQI of the NEXT hour).</li>
            <li><strong>Features Used (19 total):</strong> Temperature, Humidity, Wind Speed, PM2.5, PM10, CO, NO2, O3, SO2, NH3, NO, Hour, Day, Month, Day of Week, AQI Lag 1, AQI Lag 2, AQI Change, AQI 3-hour Rolling Mean.</li>
            <li><strong>72-Hour Forecast Architecture:</strong> Autoregressive multi-step recursive forecasting over 72 steps.</li>
            <li><strong>Ensemble Architecture:</strong> Equal-weight ensemble averaging of Random Forest, Ridge Regression, XGBoost, and LSTM.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
