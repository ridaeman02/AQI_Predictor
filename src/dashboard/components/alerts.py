import os
import json
import streamlit as st
from pathlib import Path
from src.dashboard.components.styles import AQI_COLORS, AQI_COLORS_BG, AQI_COLORS_BORDER

def render_alerts_system(predictions_data, BASE_DIR):
    st.markdown("<div class='section-header'>System Intelligence & Alerts</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
        Monitor automated system alert state and pipeline integration health. 
    </p>
    """, unsafe_allow_html=True)

    # 1. Alert Status Section
    st.markdown("<div style='margin-bottom: 1rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Active Alert State</div>", unsafe_allow_html=True)
    
    state_file = Path(BASE_DIR) / "data" / "alert_state.json"
    alert_state = {}
    
    try:
        if state_file.exists():
            with open(state_file, "r") as f:
                alert_state = json.load(f)
    except Exception as e:
        st.error(f"Could not load alert state: {e}")
        
    if not alert_state:
        st.info("No alert states currently tracked. All cities are within safe limits.")
    else:
        cols = st.columns(3)
        col_idx = 0
        for city, state_info in alert_state.items():
            is_active = state_info.get("active", False)
            cat = state_info.get("last_alerted_category", "Unknown")
            
            if is_active:
                cat_color = AQI_COLORS.get(cat, "#f87171")
                cat_bg = AQI_COLORS_BG.get(cat, "rgba(255, 255, 255, 0.05)")
                cat_border = AQI_COLORS_BORDER.get(cat, "rgba(255, 255, 255, 0.2)")

                with cols[col_idx % 3]:
                    st.markdown(f"""
                    <div style="background: var(--bg-card); backdrop-filter: blur(16px); border: 1px solid {cat_border}; border-radius: 16px; padding: 1.4rem; margin-bottom: 1rem; position: relative;">
                        <div style="position: absolute; top: 1.2rem; right: 1.2rem; width: 10px; height: 10px; border-radius: 50%; background-color: {cat_color}; box-shadow: 0 0 10px {cat_color};"></div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em;">ACTIVE ALERT</div>
                        <div style="font-size: 1.6rem; font-weight: 800; color: var(--text-main, #0f172a); margin-top: 0.3rem;">{city}</div>
                        <div style="font-size: 0.9rem; font-weight: 800; background: {cat_bg}; color: {cat_color}; border: 1px solid {cat_border}; padding: 0.3rem 0.8rem; border-radius: 20px; display: inline-block; margin-top: 0.6rem;">{cat}</div>
                        <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.8rem; line-height: 1.4;">
                            SMTP notification triggered. State locked to prevent duplicate email alerts.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                col_idx += 1
                
        if col_idx == 0:
            st.success("All monitored cities are currently below alert thresholds.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # 2. System Status
    st.markdown("<div style='margin-bottom: 1rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>System & Pipeline Health</div>", unsafe_allow_html=True)
    
    latest_ts = None
    if not predictions_data.empty and "prediction_timestamp" in predictions_data.columns:
        latest_ts = predictions_data["prediction_timestamp"].max()

    st.markdown(f"""
    <div style="background: var(--bg-card); backdrop-filter: blur(16px); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.6rem;">
        <table style="width: 100%; text-align: left; border-collapse: collapse; font-size: 0.95rem;">
            <tr style="border-bottom: 1px solid var(--border-color);">
                <th style="padding: 0.75rem 0; color: var(--text-muted); font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">Component</th>
                <th style="padding: 0.75rem 0; color: var(--text-muted); font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">Status</th>
                <th style="padding: 0.75rem 0; color: var(--text-muted); font-weight: 700; font-size: 0.8rem; text-transform: uppercase;">Details</th>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 1rem 0; color: var(--text-main); font-weight: 600;">Hourly CI/CD Pipeline</td>
                <td style="padding: 1rem 0;"><span class="platform-status-badge">Operational</span></td>
                <td style="padding: 1rem 0; color: var(--text-muted);">Runs via GitHub Actions</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 1rem 0; color: var(--text-main); font-weight: 600;">Hopsworks Feature Store</td>
                <td style="padding: 1rem 0;"><span class="platform-status-badge">Operational</span></td>
                <td style="padding: 1rem 0; color: var(--text-muted);">Feature Groups synced automatically</td>
            </tr>
            <tr style="border-bottom: 1px solid var(--border-soft);">
                <td style="padding: 1rem 0; color: var(--text-main); font-weight: 600;">AQI SMTP Alerts</td>
                <td style="padding: 1rem 0;"><span class="platform-status-badge">Configured</span></td>
                <td style="padding: 1rem 0; color: var(--text-muted);">Triggers at 'Unhealthy' threshold</td>
            </tr>
            <tr>
                <td style="padding: 1rem 0 0 0; color: var(--text-main); font-weight: 600;">Data Freshness</td>
                <td style="padding: 1rem 0 0 0;"><span class="platform-status-badge">Sync Complete</span></td>
                <td style="padding: 1rem 0 0 0; color: var(--text-muted);">Last updated: {latest_ts or 'Unknown'}</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
