import pandas as pd

def calculate_epa_aqi(pm25):
    """
    Calculate the US EPA AQI for PM2.5 concentration.
    Uses the standard EPA piecewise linear formula.
    """
    if pd.isna(pm25):
        return None
        
    pm25 = float(pm25)
    
    # Breakpoints: (BP_Low, BP_High, I_Low, I_High)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    
    for bp_low, bp_high, i_low, i_high in breakpoints:
        if bp_low <= pm25 <= bp_high:
            return round(((i_high - i_low) / (bp_high - bp_low)) * (pm25 - bp_low) + i_low)
            
    # If beyond chart, extrapolate based on highest bucket
    return round(((500 - 401) / (500.4 - 350.5)) * (pm25 - 350.5) + 401)

def get_aqi_category(aqi):
    """
    Convert continuous EPA AQI (0-500) into category.
    """
    if pd.isna(aqi):
        return "Unknown"
        
    aqi = round(float(aqi))

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"