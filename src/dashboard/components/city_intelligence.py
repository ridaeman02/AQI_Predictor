import streamlit as st
import altair as alt
from src.dashboard.components.map_view import render_aqi_map

def render_city_intelligence(features_data, selected_city, predictions_data=None):
    st.markdown(f"<div class='section-header'>City Intelligence &bull; {selected_city}</div>", unsafe_allow_html=True)
    
    city_trend_feat = features_data[features_data["city"] == selected_city].sort_values("timestamp")
    
    if city_trend_feat.empty:
        st.warning(f"No historical data available for {selected_city}.")
        return

    st.markdown("""
    <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
        Station geographic location, historical air quality trends, pollutant concentrations, and weather metrics.
    </p>
    """, unsafe_allow_html=True)

    # City-focused Map & Historical AQI Trend side-by-side
    map_col, chart_col = st.columns([0.45, 0.55])
    
    with map_col:
        st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Station Focus Map</div>", unsafe_allow_html=True)
        if predictions_data is not None:
            render_aqi_map(predictions_data, selected_city=selected_city, zoom_city=True, key="city_map")

    with chart_col:
        st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>AQI Historical Trend</div>", unsafe_allow_html=True)
        chart_aqi = alt.Chart(city_trend_feat).mark_area(
            line={'color': '#0284c7', 'strokeWidth': 2},
            color=alt.Gradient(
                gradient='linear',
                stops=[alt.GradientStop(color='rgba(2, 132, 199, 0.3)', offset=0),
                       alt.GradientStop(color='rgba(2, 132, 199, 0.0)', offset=1)],
                x1=1, x2=1, y1=1, y2=0
            )
        ).encode(
            x=alt.X("timestamp:T", title="Timeline", axis=alt.Axis(grid=False, labelColor="#475569", titleColor="#475569")),
            y=alt.Y("aqi:Q", title="AQI", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569", titleColor="#475569")),
            tooltip=["timestamp:T", "aqi:Q"]
        ).properties(height=280).interactive()
        st.altair_chart(chart_aqi, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>PM2.5 (µg/m³)</div>", unsafe_allow_html=True)
        chart_pm25 = alt.Chart(city_trend_feat).mark_line(color="#d97706", strokeWidth=2).encode(
            x=alt.X("timestamp:T", title="", axis=alt.Axis(grid=False, labelColor="#475569")),
            y=alt.Y("pm2_5:Q", title="PM2.5", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
            tooltip=["timestamp:T", "pm2_5:Q"]
        ).properties(height=200).interactive()
        st.altair_chart(chart_pm25, use_container_width=True)

    with col2:
        st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>PM10 (µg/m³)</div>", unsafe_allow_html=True)
        chart_pm10 = alt.Chart(city_trend_feat).mark_line(color="#ea580c", strokeWidth=2).encode(
            x=alt.X("timestamp:T", title="", axis=alt.Axis(grid=False, labelColor="#475569")),
            y=alt.Y("pm10:Q", title="PM10", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
            tooltip=["timestamp:T", "pm10:Q"]
        ).properties(height=200).interactive()
        st.altair_chart(chart_pm10, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Weather Conditions</div>", unsafe_allow_html=True)
    wcol1, wcol2, wcol3 = st.columns(3)
    
    with wcol1:
        chart_temp = alt.Chart(city_trend_feat).mark_line(color="#e11d48", strokeWidth=1.8).encode(
            x=alt.X("timestamp:T", title="", axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("temperature:Q", title="Temp (°C)", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
            tooltip=["timestamp:T", "temperature:Q"]
        ).properties(height=160).interactive()
        st.altair_chart(chart_temp, use_container_width=True)

    with wcol2:
        chart_hum = alt.Chart(city_trend_feat).mark_line(color="#0891b2", strokeWidth=1.8).encode(
            x=alt.X("timestamp:T", title="", axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("humidity:Q", title="Humidity (%)", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
            tooltip=["timestamp:T", "humidity:Q"]
        ).properties(height=160).interactive()
        st.altair_chart(chart_hum, use_container_width=True)

    with wcol3:
        chart_wind = alt.Chart(city_trend_feat).mark_line(color="#059669", strokeWidth=1.8).encode(
            x=alt.X("timestamp:T", title="", axis=alt.Axis(labels=False, ticks=False, grid=False)),
            y=alt.Y("wind_speed:Q", title="Wind (m/s)", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
            tooltip=["timestamp:T", "wind_speed:Q"]
        ).properties(height=160).interactive()
        st.altair_chart(chart_wind, use_container_width=True)
