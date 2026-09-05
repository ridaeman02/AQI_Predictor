import pandas as pd
import streamlit as st
import altair as alt
from src.prediction.forecast import forecast_next_72_hours

def render_forecasting(selected_city):
    st.markdown("<div class='section-header'>72-Hour Advanced Forecasting</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
        Autoregressive multi-step recursive forecasting over 72 hours, integrating 5-day weather and 4-day pollutant APIs.
    </p>
    """, unsafe_allow_html=True)

    with st.spinner(f"Generating 72-hour AI forecast for {selected_city}..."):
        try:
            fc_df = forecast_next_72_hours(selected_city, hours=72, live_api=True)
        except Exception as e:
            st.error(f"Forecast generation failed: {e}")
            return
            
    if fc_df is None or fc_df.empty:
        st.warning("Forecast data could not be generated.")
        return

    if "timestamp_dt" not in fc_df.columns:
        if "timestamp" in fc_df.columns:
            fc_df["timestamp_dt"] = pd.to_datetime(fc_df["timestamp"])
        else:
            fc_df["timestamp_dt"] = pd.date_range(start=pd.Timestamp.now(), periods=len(fc_df), freq="h")

    # =========================================================================
    # AIR QUALITY WINDOWS
    # =========================================================================
    st.markdown("<hr style='margin: 2rem 0; opacity: 0.5;'>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 1.5rem; font-weight: 700; font-size: 1.25rem; color: #0f172a; letter-spacing: -0.01em;'>Air Quality Windows</div>", unsafe_allow_html=True)
    
    col_act, col_hor = st.columns([1, 1])
    with col_act:
        selected_activity = st.selectbox("Activity Profile", ["General Outdoor", "Walking", "Running", "Cycling", "Outdoor Work"])
    with col_hor:
        horizon_str = st.radio("Forecast Horizon", ["24 HOURS", "48 HOURS", "72 HOURS"], horizontal=True)
        
    horizon_map = {"24 HOURS": (0, 24), "48 HOURS": (24, 48), "72 HOURS": (48, 72)}
    start_h, end_h = horizon_map[horizon_str]
    
    start_time = fc_df["timestamp_dt"].iloc[0]
    slice_start = start_time + pd.Timedelta(hours=start_h)
    slice_end = start_time + pd.Timedelta(hours=end_h)
    
    sliced_fc_df = fc_df[(fc_df["timestamp_dt"] >= slice_start) & (fc_df["timestamp_dt"] < slice_end)].copy()
    
    from src.recommendation.air_quality_windows import analyze_forecast_windows, calculate_hourly_score
    from src.dashboard.components.styles import AQI_COLORS
    
    windows_data = analyze_forecast_windows(sliced_fc_df, activity=selected_activity, horizon_hours=None)
    best_window = windows_data.get("best_window")
    
    if best_window:
        start_dt = pd.to_datetime(best_window["start_time"])
        end_dt = pd.to_datetime(best_window["end_time"])
        
        # Date logic
        today_date = pd.Timestamp.now(tz="UTC").date()
        if start_dt.date() == today_date:
            day_str = "Today"
        elif start_dt.date() == today_date + pd.Timedelta(days=1):
            day_str = "Tomorrow"
        else:
            day_str = start_dt.strftime("%A")
            
        time_str = f"{start_dt.strftime('%I:%M %p')} — {end_dt.strftime('%I:%M %p')}"
        
        classification = best_window["classification"]
        cls_colors = {"BEST": "#059669", "GOOD": "#2563eb", "CAUTION": "#d97706", "AVOID": "#e11d48"}
        card_color = cls_colors.get(classification, "#475569")
        
        # 1. BEST OUTDOOR WINDOW CARD
        html_card = f"""
<div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.75rem; margin-top: 0.5rem; margin-bottom: 1.5rem; border-left: 5px solid {card_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
        <div style="color: var(--text-muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;">
            {selected_city} &middot; {day_str}
        </div>
    </div>
    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 0.5rem; text-transform: uppercase;">
        {classification} OUTDOOR WINDOW
    </div>
    <div style="font-size: 2rem; font-weight: 800; color: {card_color}; margin-bottom: 1.5rem;">
        {time_str}
    </div>
    <div style="display: flex; flex-wrap: wrap; gap: 2.5rem; margin-bottom: 2rem;">
        <div>
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 0.25rem;">AQI</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{best_window['average_aqi']:.0f}</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 0.25rem;">Score</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: {card_color};">{best_window['suitability_score']:.0f}/100</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 0.25rem;">Temperature</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{best_window['average_temperature']:.1f}&deg;C</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 0.25rem;">Humidity</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{best_window['average_humidity']:.0f}%</div>
        </div>
        <div>
            <div style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600; margin-bottom: 0.25rem;">Wind</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #0f172a;">{best_window['average_wind_speed']:.1f} m/s</div>
        </div>
    </div>
    <div style="background: #f8fafc; padding: 1.25rem; border-radius: 6px; border: 1px solid #f1f5f9; margin-bottom: 1rem;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em;">WHY THIS WINDOW?</div>
        <div style="font-size: 0.95rem; color: #334155; font-weight: 500;">
            {best_window['explanation']}
        </div>
    </div>
</div>
"""
        st.markdown(html_card, unsafe_allow_html=True)
        
        with st.expander("View Important Prediction Factors (Detailed Explainability)"):
            st.markdown("##### 1. Environmental Suitability Factors")
            st.markdown(f"The rule-based recommendation engine selected this window for **{selected_activity}** because:")
            st.markdown(f"- Air quality is classified as **{classification}** (Average AQI: {best_window['average_aqi']:.0f}).")
            st.markdown(f"- Weather conditions (Temp: {best_window['average_temperature']:.1f}&deg;C, Wind: {best_window['average_wind_speed']:.1f} m/s) are favorable for the activity profile.")
            
            st.markdown("---")
            st.markdown("##### 2. Model Prediction Factors")
            st.markdown("The underlying machine learning model predicted the AQI for this specific window based on the following key drivers:")
            
            with st.spinner("Analyzing model drivers via SHAP..."):
                try:
                    import numpy as np
                    from src.explainability.shap_analysis import generate_local_explanation
                    from src.prediction.predict import load_trained_models, FEATURES
                    
                    models = load_trained_models(from_hopsworks=False)
                    if "Random Forest" in models:
                        window_row = fc_df[fc_df["timestamp_dt"] == pd.to_datetime(best_window["start_time"])].iloc[0]
                        f_dict = window_row.get("feature_vector")
                        
                        if isinstance(f_dict, dict) and f_dict:
                            f_arr = np.array([f_dict[f] for f in FEATURES], dtype=np.float32)
                            shap_data = generate_local_explanation(models["Random Forest"], f_arr, FEATURES)
                            
                            if shap_data and "features" in shap_data:
                                top_features = shap_data["features"][:5]
                                for f in top_features:
                                    direction = "increased" if f["contribution"] > 0 else "decreased"
                                    color = "#e11d48" if f["contribution"] > 0 else "#059669"
                                    st.markdown(f"- **{f['name']}** (Value: `{f['value']:.2f}`): <span style='color:{color}; font-weight: 600;'>{direction} predicted AQI</span> by {abs(f['contribution']):.1f} points.", unsafe_allow_html=True)
                            else:
                                st.info("Explanation data is currently unavailable.")
                        else:
                            st.info("Feature vector data not captured for this forecast hour.")
                    else:
                        st.info("Random Forest model is required for SHAP explainability.")
                except Exception as e:
                    st.error(f"Could not load prediction factors: {e}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 2. TIMELINE CHART
        st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Timeline Outlook</div>", unsafe_allow_html=True)
        
        timeline_df = sliced_fc_df.copy()
            
        timeline_df["window_class"] = timeline_df.apply(lambda r: calculate_hourly_score(r, activity=selected_activity)[1], axis=1)
        
        timeline_chart = alt.Chart(timeline_df).mark_bar(size=12).encode(
            x=alt.X("timestamp_dt:T", title=""),
            y=alt.Y("window_class:N", title="", sort=["BEST", "GOOD", "CAUTION", "AVOID"], axis=alt.Axis(grid=True)),
            color=alt.Color("window_class:N", scale=alt.Scale(
                domain=["BEST", "GOOD", "CAUTION", "AVOID"],
                range=["#059669", "#2563eb", "#d97706", "#e11d48"]
            ), legend=None),
            tooltip=[
                alt.Tooltip("timestamp_dt:T", title="Time", format="%b %d, %H:%M"),
                alt.Tooltip("window_class:N", title="Suitability"),
                alt.Tooltip("ensemble:Q", title="Predicted AQI", format=".0f"),
                alt.Tooltip("temperature:Q", title="Temperature (&deg;C)", format=".1f")
            ]
        ).properties(height=180).interactive()
        
        st.altair_chart(timeline_chart, use_container_width=True)
        
        # 3. ALTERNATIVES AND AVOIDS
        col_alt, col_avoid = st.columns(2)
        with col_alt:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #475569; margin-bottom: 0.75rem; text-transform: uppercase;'>Alternative Windows</div>", unsafe_allow_html=True)
            alts = windows_data.get("alternative_windows", [])
            if alts:
                for w in alts[:3]:
                    st_time = pd.to_datetime(w['start_time']).strftime('%b %d, %I:%M %p')
                    end_time = pd.to_datetime(w['end_time']).strftime('%I:%M %p')
                    c_color = cls_colors.get(w['classification'], "#94a3b8")
                    st.markdown(f"""
                    <div style='padding: 0.75rem 1rem; background: white; border: 1px solid #e2e8f0; border-left: 3px solid {c_color}; margin-bottom: 0.5rem; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;'>
                            <div style='font-weight: 700; font-size: 0.9rem; color: #0f172a;'>{st_time} — {end_time}</div>
                            <div style='font-size: 0.75rem; font-weight: 700; color: {c_color};'>{w['classification']}</div>
                        </div>
                        <div style='font-size: 0.8rem; color: #64748b;'>AQI: {w['average_aqi']:.0f} | Score: {w['suitability_score']:.0f}/100</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; font-style: italic;'>No viable alternative windows found.</div>", unsafe_allow_html=True)
                
        with col_avoid:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #475569; margin-bottom: 0.75rem; text-transform: uppercase;'>Periods to Avoid</div>", unsafe_allow_html=True)
            avoids = windows_data.get("avoid_windows", [])
            if avoids:
                for w in avoids[:3]:
                    st_time = pd.to_datetime(w['start_time']).strftime('%b %d, %I:%M %p')
                    end_time = pd.to_datetime(w['end_time']).strftime('%I:%M %p')
                    st.markdown(f"""
                    <div style='padding: 0.75rem 1rem; background: #fdf2f8; border: 1px solid #fce7f3; border-left: 3px solid #e11d48; margin-bottom: 0.5rem; border-radius: 6px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;'>
                            <div style='font-weight: 700; font-size: 0.9rem; color: #9f1239;'>{st_time} — {end_time}</div>
                            <div style='font-size: 0.75rem; font-weight: 700; color: #e11d48;'>AVOID</div>
                        </div>
                        <div style='font-size: 0.8rem; color: #be123c;'>AQI: {w['average_aqi']:.0f} (Dangerous Levels)</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; font-style: italic;'>No dangerous periods detected!</div>", unsafe_allow_html=True)
    else:
        st.info(f"Insufficient data or no valid forecast windows found for {selected_city}.")

    st.markdown("<hr style='margin: 2rem 0; opacity: 0.5;'>", unsafe_allow_html=True)
    # =========================================================================

    # Target Milestone Cards
    st.markdown("<div style='margin-bottom: 1rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Forecast Milestones</div>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3 = st.columns(3)
    
    milestones = [24, 48, 72]
    cols = [m_col1, m_col2, m_col3]
    
    from src.dashboard.components.styles import AQI_COLORS
    
    for step_val, col in zip(milestones, cols):
        step_row = fc_df[fc_df["step"] == step_val]
        if not step_row.empty:
            row = step_row.iloc[0]
            val = float(row.get("ensemble", 0))
            cat = row.get("category", "Unknown")
            cat_color = AQI_COLORS.get(cat, "var(--text-main)")
            time_str = row["timestamp_dt"].strftime("%b %d, %H:%M")
            day_num = step_val // 24
            
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {cat_color};">
                    <div class="metric-header-row">
                        <span class="metric-card-title">DAY {day_num} <span style="font-size: 0.7rem; font-weight: normal;">(+{step_val}h)</span></span>
                        <span style="font-size: 0.75rem; color: var(--text-muted);">{time_str}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-top: 0.5rem;">
                        <div class="metric-card-val" style="color: {cat_color};">{val:.0f}</div>
                        <div style="font-weight: 600; font-size: 0.85rem; color: {cat_color}; background: {cat_color}1A; padding: 0.2rem 0.6rem; border-radius: 4px;">{cat}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Predicted AQI Timeline</div>", unsafe_allow_html=True)
    
    chart_ensemble = alt.Chart(fc_df).mark_line(color="#0284c7", strokeWidth=2.5).encode(
        x=alt.X("timestamp_dt:T", title="Timestamp (UTC)", axis=alt.Axis(grid=False, labelColor="#475569", titleColor="#475569")),
        y=alt.Y("ensemble:Q", title="Predicted AQI (Ensemble)", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569", titleColor="#475569")),
        tooltip=[
            alt.Tooltip("step:Q", title="Hour Step (t+)"),
            alt.Tooltip("timestamp_dt:T", title="Timestamp"),
            alt.Tooltip("ensemble:Q", title="Ensemble AQI", format=".2f"),
            alt.Tooltip("category:N", title="Predicted Category")
        ]
    ).properties(height=300).interactive()
    
    st.altair_chart(chart_ensemble, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);'>Multi-Model Forecast Comparison</div>", unsafe_allow_html=True)
    
    models_to_compare = [col for col in ["random_forest", "ridge", "xgboost", "lstm", "ensemble"] if col in fc_df.columns and not fc_df[col].isna().all()]
    id_columns = [col for col in ["timestamp_dt", "step", "category"] if col in fc_df.columns]
    
    comp_df = fc_df.melt(
        id_vars=id_columns,
        value_vars=models_to_compare,
        var_name="Model",
        value_name="Predicted AQI"
    )
    
    model_labels = {
        "random_forest": "Random Forest",
        "ridge": "Ridge Regression",
        "xgboost": "XGBoost",
        "lstm": "LSTM",
        "ensemble": "Ensemble"
    }
    comp_df["Model"] = comp_df["Model"].map(model_labels)

    chart_models = alt.Chart(comp_df).mark_line(strokeWidth=1.8, opacity=0.85).encode(
        x=alt.X("timestamp_dt:T", title="", axis=alt.Axis(grid=False, labelColor="#475569")),
        y=alt.Y("Predicted AQI:Q", title="AQI", axis=alt.Axis(gridColor="rgba(15,23,42,0.06)", labelColor="#475569")),
        color=alt.Color("Model:N", 
            scale=alt.Scale(
                domain=["Random Forest", "Ridge Regression", "XGBoost", "LSTM", "Ensemble"],
                range=["#059669", "#d97706", "#e11d48", "#7c3aed", "#0284c7"]
            ),
            legend=alt.Legend(title=None, orient="bottom", labelColor="#1e293b")
        ),
        tooltip=["Model:N", alt.Tooltip("timestamp_dt:T", title="Timestamp"), alt.Tooltip("Predicted AQI:Q", format=".2f")]
    ).properties(height=280).interactive()
    
    st.altair_chart(chart_models, use_container_width=True)

    with st.expander("View Complete Forecast Data"):
        disp_cols = ["step", "timestamp"] + models_to_compare + ["category"]
        disp_cols = [c for c in disp_cols if c in fc_df.columns]
        disp_fc = fc_df[disp_cols].copy()
        
        rename_map = {
            "step": "Step (t+h)",
            "timestamp": "Timestamp",
            "random_forest": "Random Forest",
            "ridge": "Ridge Regression",
            "xgboost": "XGBoost",
            "lstm": "LSTM",
            "ensemble": "Ensemble",
            "category": "Risk Category"
        }
        disp_fc = disp_fc.rename(columns=rename_map)
        st.dataframe(disp_fc, use_container_width=True, hide_index=True)
