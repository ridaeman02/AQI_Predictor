def get_aqi_category(aqi):
    """
    Convert predicted AQI value into the project's AQI category.

    Project AQI scale:
    1 = Good
    2 = Moderate
    3 = Unhealthy for Sensitive Groups
    4 = Unhealthy
    5 = Very Unhealthy
    """

    aqi = round(float(aqi))

    if aqi <= 1:
        return "Good"

    elif aqi == 2:
        return "Moderate"

    elif aqi == 3:
        return "Unhealthy for Sensitive Groups"

    elif aqi == 4:
        return "Unhealthy"

    else:
        return "Very Unhealthy"