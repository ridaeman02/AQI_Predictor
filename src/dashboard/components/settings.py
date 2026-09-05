import json
import os
import streamlit as st
from pathlib import Path

SETTINGS_FILE = Path("data/app_settings.json")

def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        'alert_threshold': 'Unhealthy (AQI >= 151)',
        'auto_refresh': '1 Hour (Pipeline Interval)',
        'browser_notifications': True
    }

def save_settings(settings):
    os.makedirs(SETTINGS_FILE.parent, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=4)
    st.toast("Settings saved successfully!", icon="✅")

def render_settings():
    st.markdown("<div class='section-header'>Platform Settings & Preferences</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
        Configure user dashboard defaults, display preferences, and notification triggers.
    </p>
    """, unsafe_allow_html=True)

    current_settings = load_settings()

    st.markdown("""
    <div style="background: var(--bg-card); backdrop-filter: blur(16px); border: 1px solid var(--border-color); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-main); margin-bottom: 1rem;">
            Pipeline & Alert Preferences
        </div>
    """, unsafe_allow_html=True)

    threshold_opts = ["Unhealthy (AQI >= 151)", "Unhealthy for Sensitive Groups (AQI >= 101)", "Very Unhealthy (AQI >= 201)"]
    refresh_opts = ["1 Hour (Pipeline Interval)", "30 Minutes", "Manual Refresh Only"]

    threshold_idx = threshold_opts.index(current_settings['alert_threshold']) if current_settings['alert_threshold'] in threshold_opts else 0
    refresh_idx = refresh_opts.index(current_settings['auto_refresh']) if current_settings['auto_refresh'] in refresh_opts else 0

    new_threshold = st.selectbox("Automated Alert Threshold", threshold_opts, index=threshold_idx)
    new_refresh = st.selectbox("Dashboard Data Auto-Refresh", refresh_opts, index=refresh_idx)
    new_notifications = st.checkbox("Enable Browser Notification Hooks", value=current_settings.get('browser_notifications', True))

    if (new_threshold != current_settings['alert_threshold'] or 
        new_refresh != current_settings['auto_refresh'] or 
        new_notifications != current_settings['browser_notifications']):
        
        current_settings.update({
            'alert_threshold': new_threshold,
            'auto_refresh': new_refresh,
            'browser_notifications': new_notifications
        })
        save_settings(current_settings)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 14px; padding: 1.25rem;">
        <div style="font-weight: 700; font-size: 0.85rem; color: #818cf8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.3rem;">
            Configuration Note
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5;">
            Local settings are maintained in session state. Operational pipeline configurations and SMTP credentials are managed securely via repository environment variables and GitHub Actions secrets.
        </div>
    </div>
    """, unsafe_allow_html=True)
