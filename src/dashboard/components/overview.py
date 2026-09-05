import streamlit as st
import altair as alt
import pandas as pd
from pathlib import Path
from src.dashboard.components.styles import AQI_COLORS, AQI_COLORS_BG, AQI_COLORS_BORDER
from src.dashboard.components.map_view import render_aqi_map

BASE_DIR = Path(__file__).resolve().parents[3]
LOGO_PATH = BASE_DIR / "src" / "dashboard" / "assets" / "logo.png"

def render_overview(features_data, predictions_data, selected_city, selected_timeframe="Live / Next-Hour"):
    city_preds = predictions_data[predictions_data["city"] == selected_city]
    if city_preds.empty:
        st.warning(f"No prediction data available for {selected_city}")
        return

    city_pred_row = city_preds.iloc[0]
    current_aqi = city_pred_row["current_aqi"]
    current_cat = city_pred_row["current_category"]
    
    # Default to next-hour
    next_aqi = city_pred_row["next_hour_aqi"]
    next_cat = city_pred_row["next_hour_category"]
    outlook_title = "Next-Hour Outlook"
    
    # Dynamically fetch if 24, 48, or 72 hour is selected
    if selected_timeframe in ["24-Hour", "48-Hour", "72-Hour"]:
        target_hours = int(selected_timeframe.split("-")[0])
        from src.prediction.forecast import forecast_next_72_hours
        with st.spinner(f"Fetching {selected_timeframe} forecast..."):
            try:
                fc_df = forecast_next_72_hours(selected_city, hours=target_hours, live_api=True)
                if not fc_df.empty:
                    target_row = fc_df[fc_df["step"] == target_hours]
                    if not target_row.empty:
                        next_aqi = target_row.iloc[0]["ensemble"]
                        next_cat = target_row.iloc[0]["category"]
                        outlook_title = f"{target_hours}-Hour Outlook"
            except Exception as e:
                st.error("Failed to load extended forecast.")

    # Automatically sync to current live time instead of static CSV time
    update_time = pd.Timestamp.now(tz="UTC").strftime("%B %d, %Y %H:%M UTC")

    cur_color = AQI_COLORS.get(current_cat, "#475569")
    cur_bg = AQI_COLORS_BG.get(current_cat, "rgba(255, 255, 255, 0.8)")
    cur_border = AQI_COLORS_BORDER.get(current_cat, "rgba(255, 255, 255, 0.3)")

    # 1. HERO SECTION & INTERACTIVE GEOGRAPHIC MAP
    hero_col1, hero_col2 = st.columns([0.44, 0.56])

    with hero_col1:
        st.markdown(f"""
        <div class="hero-aqi-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div class="hero-aqi-location">Station Location &bull; <strong>{selected_city.upper()}</strong></div>
                <div class="hero-aqi-value">{current_aqi:.0f}</div>
                <div class="hero-aqi-badge" style="background: {cur_bg}; color: {cur_color}; border: 1px solid {cur_border}; margin-bottom: 1.2rem;">
                    {current_cat}
                </div>
            </div>
            <div>
                <div style="background: rgba(2, 132, 199, 0.03); border: 1px solid var(--border-color); border-radius: 14px; padding: 1rem; margin-bottom: 1rem;">
                    <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em;">{outlook_title}</div>
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 0.3rem;">
                        <span style="font-size: 1.4rem; font-weight: 800; color: var(--text-main);">AQI {next_aqi:.0f}</span>
                        <span style="font-size: 0.8rem; font-weight: 700; color: {AQI_COLORS.get(next_cat, '#fff')};">{next_cat}</span>
                    </div>
                </div>
                <div style="color: var(--text-muted); font-size: 0.75rem; font-weight: 500;">
                    Live System Sync: {update_time}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_col2:
        st.markdown("""
        <div style="margin-bottom: 0.5rem; display: flex; align-items: center; justify-content: space-between;">
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">
                Geographic Station Map
            </div>
            <div style="font-size: 0.75rem; color: var(--accent-blue); font-weight: 700;">
                Live Interactive View
            </div>
        </div>
        """, unsafe_allow_html=True)
        render_aqi_map(predictions_data, selected_city=selected_city, zoom_city=True, key="overview_map")

    # 2. KEY METRICS SECTION WITH SVG ICONS
    st.markdown("<div class='section-header'>Key Environmental Metrics</div>", unsafe_allow_html=True)
    
    city_features = features_data[features_data["city"] == selected_city].sort_values("timestamp", ascending=False)
    if city_features.empty:
        st.warning("No recent environmental data found.")
    else:
        latest = city_features.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header-row">
                    <span class="metric-card-title">AQI Index</span>
                    <div class="metric-card-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0284c7" stroke-width="2"><path d="M12 2v20M2 12h20"/><circle cx="12" cy="12" r="4"/></svg>
                    </div>
                </div>
                <div class="metric-card-val" style="color: {cur_color};">{current_aqi:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header-row">
                    <span class="metric-card-title">PM2.5</span>
                    <div class="metric-card-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
                    </div>
                </div>
                <div class="metric-card-val" style="color: #d97706;">{latest['pm2_5']:.1f}<span class="metric-card-unit">µg/m³</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header-row">
                    <span class="metric-card-title">PM10</span>
                    <div class="metric-card-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>
                    </div>
                </div>
                <div class="metric-card-val" style="color: #ea580c;">{latest['pm10']:.1f}<span class="metric-card-unit">µg/m³</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header-row">
                    <span class="metric-card-title">Temperature</span>
                    <div class="metric-card-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e11d48" stroke-width="2"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>
                    </div>
                </div>
                <div class="metric-card-val" style="color: #e11d48;">{latest['temperature']:.1f}<span class="metric-card-unit">°C</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header-row">
                    <span class="metric-card-title">Humidity</span>
                    <div class="metric-card-icon">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#0891b2" stroke-width="2"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
                    </div>
                </div>
                <div class="metric-card-val" style="color: #0891b2;">{latest['humidity']:.0f}<span class="metric-card-unit">%</span></div>
            </div>
            """, unsafe_allow_html=True)

    # 3. AQI TREND & POLLUTANT ANALYSIS TABS
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Environmental Analysis & Trends</div>", unsafe_allow_html=True)

    tab_trend, tab_pollutants, tab_weather = st.tabs(["AQI Historical Trend", "Pollutant Breakdown", "Weather Relationships"])

    city_trend_feat = features_data[features_data["city"] == selected_city].sort_values("timestamp")

    with tab_trend:
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
            y=alt.Y("aqi:Q", title="AQI Index", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569", titleColor="#475569")),
            tooltip=["timestamp:T", "aqi:Q"]
        ).properties(height=280).interactive()

        st.altair_chart(chart_aqi, use_container_width=True)

    with tab_pollutants:
        pollutant_cols = [c for c in ["pm2_5", "pm10", "co", "no2", "o3", "so2", "nh3", "no"] if c in city_trend_feat.columns]
        if pollutant_cols:
            p_melt = city_trend_feat.melt(id_vars=["timestamp"], value_vars=pollutant_cols, var_name="Pollutant", value_name="Concentration")
            p_melt["Pollutant"] = p_melt["Pollutant"].str.upper()

            chart_p = alt.Chart(p_melt).mark_line(strokeWidth=1.8).encode(
                x=alt.X("timestamp:T", title="", axis=alt.Axis(grid=False, labelColor="#475569")),
                y=alt.Y("Concentration:Q", title="Concentration (µg/m³)", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
                color=alt.Color("Pollutant:N", scale=alt.Scale(scheme="category10")),
                tooltip=["Pollutant:N", alt.Tooltip("timestamp:T", title="Timestamp"), alt.Tooltip("Concentration:Q", format=".2f")]
            ).properties(height=280).interactive()
            st.altair_chart(chart_p, use_container_width=True)

    with tab_weather:
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;'>Temperature vs AQI</div>", unsafe_allow_html=True)
            chart_t_aqi = alt.Chart(city_trend_feat).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X("temperature:Q", title="Temperature (°C)", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
                y=alt.Y("aqi:Q", title="AQI", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
                color=alt.Color("aqi:Q", scale=alt.Scale(scheme="viridis"), legend=None),
                tooltip=["timestamp:T", "temperature:Q", "aqi:Q"]
            ).properties(height=220)
            st.altair_chart(chart_t_aqi, use_container_width=True)

        with w_col2:
            st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.5rem;'>Humidity vs AQI</div>", unsafe_allow_html=True)
            chart_h_aqi = alt.Chart(city_trend_feat).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X("humidity:Q", title="Humidity (%)", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
                y=alt.Y("aqi:Q", title="AQI", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
                color=alt.Color("aqi:Q", scale=alt.Scale(scheme="magma"), legend=None),
                tooltip=["timestamp:T", "humidity:Q", "aqi:Q"]
            ).properties(height=220)
            st.altair_chart(chart_h_aqi, use_container_width=True)

    # 4. NATIONAL STATION COMPARISON
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>National Station Comparison</div>", unsafe_allow_html=True)
    
    comp_melted = predictions_data.melt(
        id_vars=["city"], 
        value_vars=["current_aqi", "next_hour_aqi"],
        var_name="AQI Type", 
        value_name="AQI Value"
    )
    comp_melted["AQI Type"] = comp_melted["AQI Type"].map({
        "current_aqi": "Current AQI", 
        "next_hour_aqi": "Next-Hour Forecast"
    })

    grouped_chart = alt.Chart(comp_melted).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("AQI Type:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("AQI Value:Q", title="AQI Index", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569", titleColor="#475569")),
        color=alt.Color("AQI Type:N", scale=alt.Scale(domain=["Current AQI", "Next-Hour Forecast"], range=["#cbd5e1", "#0284c7"]), legend=alt.Legend(title=None, orient="top", labelColor="#1e293b")),
        column=alt.Column("city:N", title=None, header=alt.Header(labelColor="#0f172a", labelFontSize=12, labelFontWeight="bold")),
        tooltip=["city:N", "AQI Type:N", alt.Tooltip("AQI Value:Q", format=".2f")]
    ).properties(height=260, width=120).configure_view(stroke="transparent")

    st.altair_chart(grouped_chart, use_container_width=True)
