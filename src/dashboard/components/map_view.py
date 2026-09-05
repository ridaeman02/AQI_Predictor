import pandas as pd
import streamlit as st
import plotly.express as px

CITY_COORDS = {
    "Lahore": (31.5204, 74.3587),
    "Karachi": (24.8607, 67.0011),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
}

AQI_COLORS = {
    "Good": "#10b981",
    "Moderate": "#fbbf24",
    "Unhealthy for Sensitive Groups": "#fb923c",
    "Unhealthy": "#f87171",
    "Very Unhealthy": "#c084fc",
}

def render_aqi_map(predictions_data, selected_city=None, zoom_city=False, key=None):
    """
    Renders a highly reliable, responsive Plotly Mapbox chart using open-access
    OpenStreetMap tiles that load robustly on all systems without Mapbox API tokens.
    """
    if predictions_data is None or predictions_data.empty:
        return

    map_rows = []
    for _, row in predictions_data.iterrows():
        c_name = row["city"]
        if c_name in CITY_COORDS:
            lat, lon = CITY_COORDS[c_name]
            aqi_val = float(row["current_aqi"])
            cat = row["current_category"]
            color = AQI_COLORS.get(cat, "#9ca3af")
            
            is_selected = (selected_city and c_name.lower() == selected_city.lower())
            
            map_rows.append({
                "City": c_name,
                "lat": lat,
                "lon": lon,
                "AQI": round(aqi_val, 2),
                "Category": cat,
                "Color": color,
                "MarkerSize": 18 if is_selected else 12
            })
            
    map_df = pd.DataFrame(map_rows)
    if map_df.empty:
        return

    # Dynamic camera positioning
    if zoom_city and selected_city and selected_city in CITY_COORDS:
        center_lat, center_lon = CITY_COORDS[selected_city]
        init_zoom = 8.5
    else:
        center_lat, center_lon = 30.3753, 69.3451
        init_zoom = 4.8

    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="Category",
        color_discrete_map=AQI_COLORS,
        size="MarkerSize",
        size_max=18,
        hover_name="City",
        hover_data={"AQI": True, "Category": True, "lat": False, "lon": False, "MarkerSize": False},
        zoom=init_zoom,
        center={"lat": center_lat, "lon": center_lon},
        height=320
    )

    # Use open-street-map (extremely robust raster tiles, renders instantly everywhere)
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=key)
