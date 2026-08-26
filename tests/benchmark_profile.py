import sys
import os
sys.path.append(os.path.abspath('.'))

import time
import pandas as pd
import numpy as np
from src.prediction.predict import load_trained_models, FEATURES, DATA_FILE, MODEL_DIR
from src.prediction.forecast import fetch_openweather_forecast, forecast_next_72_hours

t0 = time.perf_counter()
models = load_trained_models(MODEL_DIR)
t1 = time.perf_counter()
print(f"load_trained_models: {t1 - t0:.4f}s")

t0 = time.perf_counter()
w, p = fetch_openweather_forecast("Lahore")
t1 = time.perf_counter()
print(f"fetch_openweather_forecast: {t1 - t0:.4f}s")

t0 = time.perf_counter()
df_fc = forecast_next_72_hours("Lahore", hours=72)
t1 = time.perf_counter()
print(f"forecast_next_72_hours: {t1 - t0:.4f}s")
