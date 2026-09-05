import streamlit as st

AQI_COLORS = {
    "Good": "#10b981",
    "Moderate": "#f59e0b",
    "Unhealthy for Sensitive Groups": "#f97316",
    "Unhealthy": "#ef4444",
    "Very Unhealthy": "#8b5cf6",
    "Hazardous": "#be123c"
}

AQI_COLORS_BG = {
    "Good": "rgba(16, 185, 129, 0.1)",
    "Moderate": "rgba(245, 158, 11, 0.1)",
    "Unhealthy for Sensitive Groups": "rgba(249, 115, 22, 0.1)",
    "Unhealthy": "rgba(239, 68, 68, 0.1)",
    "Very Unhealthy": "rgba(139, 92, 246, 0.1)",
    "Hazardous": "rgba(190, 18, 60, 0.1)"
}

AQI_COLORS_BORDER = {
    "Good": "rgba(16, 185, 129, 0.3)",
    "Moderate": "rgba(245, 158, 11, 0.3)",
    "Unhealthy for Sensitive Groups": "rgba(249, 115, 22, 0.3)",
    "Unhealthy": "rgba(239, 68, 68, 0.3)",
    "Very Unhealthy": "rgba(139, 92, 246, 0.3)",
    "Hazardous": "rgba(190, 18, 60, 0.3)"
}

def load_css(theme_mode="Light"):
    st.markdown("""
    <style>
    /* Light Blueish Theme for AirSight AI */
    :root {
        --primary-color: #0284c7;
        --bg-main: #f0f9ff;
        --card-bg: #ffffff;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --border-color: #e0f2fe;
    }
    
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-main) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-main) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    /* Hero Cards */
    .hero-aqi-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(2, 132, 199, 0.05);
    }
    .hero-aqi-location {
        color: var(--text-muted);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .hero-aqi-value {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 0.5rem 0;
        color: var(--text-main);
    }
    .hero-aqi-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.03);
    }
    .metric-header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .metric-card-title {
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.85rem;
    }
    .metric-card-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-main);
    }
    
    /* Map Container */
    .map-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Sidebar Elevation & Scrollbar Hiding */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        box-shadow: 4px 0 20px rgba(14, 165, 233, 0.08) !important;
        border-right: 1px solid var(--border-color) !important;
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
    }
    
    [data-testid="stSidebar"] > div:first-child {
        -ms-overflow-style: none;
        scrollbar-width: none;
    }
    
    /* Hide scrollbar for Chrome, Safari and Opera */
    [data-testid="stSidebar"] ::-webkit-scrollbar, 
    [data-testid="stSidebar"] > div:first-child::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        background: transparent !important;
    }
    
    /* Dim inactive navigation items to reduce clutter/confusion */
    [data-testid="stSidebar"] .stRadio label {
        opacity: 0.4 !important;
        transition: opacity 0.2s ease-in-out;
    }
    
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        opacity: 1.0 !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        opacity: 1.0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
