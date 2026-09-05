import os
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Clear Streamlit module cache for styles to fix ImportError
if 'src.dashboard.components.styles' in sys.modules:
    del sys.modules['src.dashboard.components.styles']
if 'src.dashboard.components.overview' in sys.modules:
    del sys.modules['src.dashboard.components.overview']
if 'src.dashboard.components.alerts' in sys.modules:
    del sys.modules['src.dashboard.components.alerts']

# Import Modular Dashboard Components
from src.dashboard.components.styles import load_css
from src.dashboard.components.overview import render_overview
from src.dashboard.components.city_intelligence import render_city_intelligence
from src.dashboard.components.forecasting import render_forecasting
from src.dashboard.components.model_intelligence import render_model_intelligence
from src.dashboard.components.alerts import render_alerts_system
from src.dashboard.components.settings import load_settings, render_settings

# ============================================================
# PAGE CONFIGURATION
# ============================================================
LOGO_PATH = BASE_DIR / "src" / "dashboard" / "assets" / "logo.png"

try:
    from PIL import Image
    icon = Image.open(str(LOGO_PATH)) if LOGO_PATH.exists() else None
except Exception:
    icon = None

st.set_page_config(
    page_title="AirSight AI",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Settings and CSS
try:
    settings = load_settings()
    theme_mode = settings.get("theme", "Light")
except:
    theme_mode = "Light"

load_css(theme_mode)

LOGO_PATH = BASE_DIR / "src" / "dashboard" / "assets" / "logo.png"

# ============================================================
# SIDEBAR NAVIGATION & CONTROLS
# ============================================================
with st.sidebar:
    # High-End Logo App Icon Rendering
    if LOGO_PATH.exists():
        import base64
        with open(LOGO_PATH, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()
        st.markdown(f'''
        <div style="display: flex; align-items: center; gap: 15px; margin-top: 5px; margin-bottom: 25px; padding-left: 5px;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 55px; height: 55px; border-radius: 12px; box-shadow: 0 4px 10px rgba(14, 165, 233, 0.2); border: 1px solid rgba(14, 165, 233, 0.15); object-fit: cover;">
            <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a; letter-spacing: -0.02em; line-height: 1.1;">
                AirSight<br>AI
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: #0ea5e9;'>AirSight AI</h2>", unsafe_allow_html=True)
        
    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;'>Global Controls</div>", unsafe_allow_html=True)
    
    # Data Loading (needs to happen here so we know available cities for the dropdown)
    DATA_FILE = BASE_DIR / "data" / "processed_features.csv"
    MODEL_DIR = BASE_DIR / "models"

    @st.cache_data(ttl=3600)
    def load_app_data():
        if not DATA_FILE.exists():
            return pd.DataFrame(), pd.DataFrame()
        features_df = pd.read_csv(DATA_FILE)
        features_df["timestamp"] = pd.to_datetime(features_df["timestamp"])
        
        from src.prediction.predict import get_next_hour_predictions
        try:
            predictions_df = get_next_hour_predictions(str(DATA_FILE), str(MODEL_DIR))
        except:
            predictions_df = pd.DataFrame()
            
        return features_df, predictions_df

    features_data, predictions_data = load_app_data()
    
    available_cities = ["Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta"]
    cities_in_preds = list(predictions_data["city"].unique()) if not predictions_data.empty else []
    select_options = [c for c in available_cities if c in cities_in_preds] or cities_in_preds

    selected_city = st.selectbox("Target City", select_options, key="sidebar_city")
    selected_timeframe = st.selectbox("Time Horizon", ["Live / Next-Hour", "24-Hour", "48-Hour", "72-Hour"])
    
    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 1.5rem; margin-bottom: 0.5rem;'>Navigation</div>", unsafe_allow_html=True)
    
    # Navigation Menu
    navigation = st.radio("MAIN MENU", [
        "Home",
        "City Details",
        "Forecast",
        "AI Performance",
        "Alerts",
        "Settings"
    ], label_visibility="collapsed")
    
    st.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; margin-top: 2rem;">
        Developed by Rida Eman<br>
        <a href="https://github.com/ridaeman02/AQI_Predictor" target="_blank" style="color: #0ea5e9; text-decoration: none;">[ Source Code ]</a>
    </div>
    """, unsafe_allow_html=True)

if features_data.empty or predictions_data.empty:
    st.error("Data pipeline offline. Please ensure models and data are available.")
    st.stop()

# ============================================================
# DASHBOARD HEADER & WELCOME
# ============================================================
if navigation == "Home":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(14, 165, 233, 0.08); border: 1px solid rgba(14, 165, 233, 0.15);">
        <h1 style="color: #0f172a !important; margin: 0 0 0.5rem 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">Welcome to <span style="color: #0ea5e9;">AirSight AI</span></h1>
        <p style="font-size: 1.1rem; color: #475569; margin: 0; max-width: 650px; line-height: 1.6;">
            Next-Generation Air Quality Forecasting & Environmental Intelligence. Monitor real-time pollutants, view 72-hour AI predictions, and analyze historical atmospheric trends.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style="padding: 1.5rem; background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
        <div>
            <h2 style="margin: 0; font-size: 1.5rem; color: var(--text-main) !important;">{navigation}</h2>
            <p style="margin: 0; color: var(--text-muted); font-size: 0.9rem;">AirSight AI Intelligence System</p>
        </div>
        <div style="text-align: right;">
            <span style="background-color: #dcfce7; color: #166534; padding: 0.3rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">&check; System Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)



# ============================================================
# ROUTING
# ============================================================
if navigation == "Home":
    render_overview(features_data, predictions_data, selected_city, selected_timeframe)
elif navigation == "City Details":
    render_city_intelligence(features_data, selected_city, predictions_data)
elif navigation == "Forecast":
    render_forecasting(selected_city)
elif navigation == "AI Performance":
    render_model_intelligence(features_data, selected_city, BASE_DIR)
elif navigation == "Alerts":
    render_alerts_system(predictions_data, BASE_DIR)
elif navigation == "Settings":
    render_settings()