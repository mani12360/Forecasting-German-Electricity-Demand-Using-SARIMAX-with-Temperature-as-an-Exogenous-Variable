# PART 3 - SARIMA MODEL
# German Electricity Demand Forecasting
import warnings
import os

from statsmodels.tools.sm_exceptions import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf

# Create folder for plots only
plot_dir = "SARIMA_Plots"
os.makedirs(plot_dir, exist_ok=True)

# Load Weekly Data
weekly = pd.read_csv(
    "weekly_load.csv",
    index_col=0,
    parse_dates=True
)
weekly.index = pd.DatetimeIndex(weekly.index)
weekly = weekly.asfreq("W-SUN")
series = weekly["load"]

# Train/Test Split
# Last 2 years = 104 weeks
forecast_horizon = 104
train = series[:-forecast_horizon]
test = series[-forecast_horizon:]
print("Training samples :", len(train))
print("Testing samples  :", len(test))
# SARIMA Grid Search
p = range(0, 7)
d = range(0, 3)
q = range(0, 7)

seasonal_order = (1, 1, 1, 52)
best_aic = np.inf
best_order = None
best_model = None
orders = list(itertools.product(p, d, q))
total_models = len(orders)
print("\nSearching for best SARIMA model...\n")

for i, order in enumerate(orders):
    print(f"Model {i+1}/{total_models}", end="\r")
    try:
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)

        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order
            best_model = results

    except Exception:
        continue
print("\nSearch Complete!\n")

# Best Model
print("Best SARIMA Model")
print("--------------------------")
print("Order           :", best_order)
print("Seasonal Order  :", seasonal_order)
print("AIC             :", round(best_aic, 2))

# Residual Diagnostics
residuals = best_model.resid
# Residual Plot
plt.figure(figsize=(12,5))
plt.plot(residuals)
plt.title("SARIMA Residuals")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.grid(True)
plt.savefig(
    os.path.join(
        plot_dir,
        "SARIMA_Residuals.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Residual Distribution Plot
plt.figure(figsize=(8,5))
plt.hist(
    residuals,
    bins=30
)

plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(
    os.path.join(
        plot_dir,
        "Residual_Distribution.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ACF Plot
plt.figure(figsize=(10,5))
plot_acf(
    residuals,
    lags=60
)

plt.title("Residual Autocorrelation")
plt.savefig(
    os.path.join(
        plot_dir,
        "Residual_ACF.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# Forecast
forecast = best_model.get_forecast(
    steps=forecast_horizon
)

forecast_mean = forecast.predicted_mean

confidence = forecast.conf_int()

# Evaluation
rmse = np.sqrt(
    mean_squared_error(
        test,
        forecast_mean
    )
)
print("\nRMSE =", round(rmse,2))

# Forecast Plot
plt.figure(figsize=(16,7))
plt.plot(
    train.index,
    train,
    label="Training"
)

plt.plot(
    test.index,
    test,
    label="Actual",
    linewidth=2
)

plt.plot(
    test.index,
    forecast_mean,
    color="red",
    linewidth=2,
    label="SARIMA Forecast"
)

plt.fill_between(
    test.index,
    confidence.iloc[:,0],
    confidence.iloc[:,1],
    alpha=0.3,
    label="95% Confidence Interval"
)

plt.title(
    "SARIMA Forecast (Last Two Years)"
)

plt.xlabel("Date")
plt.ylabel("Electricity Demand")

plt.legend()
plt.grid(True)
plt.savefig(
    os.path.join(
        plot_dir,
        "SARIMA_Forecast_Last_Two_Years.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Save Forecast
forecast_df = pd.DataFrame({
    "Actual": test,
    "Forecast": forecast_mean,
    "Lower95": confidence.iloc[:,0],
    "Upper95": confidence.iloc[:,1]
})

forecast_df.to_csv(
    "sarima_forecast.csv"
)
print("\nForecast saved.")
print("Plots saved in folder:", plot_dir)