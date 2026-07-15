# PART 4 - SARIMAX WITH TEMPERATURE
# German Electricity Demand Forecasting
import warnings
warnings.filterwarnings("ignore")

import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Creates folder for plots only
plot_dir = "SARIMAX_Plots"
os.makedirs(plot_dir, exist_ok=True)

# Load Electricity Data
weekly_load = pd.read_csv(
    "weekly_load.csv",
    index_col=0,
    parse_dates=True
)

weekly_load.index = pd.to_datetime(weekly_load.index)
weekly_load.index = weekly_load.index.tz_localize(None)
weekly_load = weekly_load.asfreq("W-SUN")

# Download Berlin Temperature Data
print("Downloading Berlin temperature data...")
url = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=52.52"
    "&longitude=13.41"
    "&start_date=2015-01-01"
    "&end_date=2020-10-31"
    "&daily=temperature_2m_mean"
    "&timezone=GMT"
)
response = requests.get(url)
weather = response.json()

# Convert Temperature Data to DataFrame
temp_df = pd.DataFrame({
    "date": weather["daily"]["time"],
    "temperature": weather["daily"]["temperature_2m_mean"]
})
temp_df["date"] = pd.to_datetime(temp_df["date"])
temp_df["date"] = temp_df["date"].dt.tz_localize(None)
temp_df.set_index("date", inplace=True)

# Weekly Average Temperature
weekly_temp = (
    temp_df
    .resample("W")
    .mean()
)
weekly_temp = weekly_temp.asfreq("W-SUN")

# Merge Load + Temperature
data = weekly_load.join(
    weekly_temp,
    how="inner"
)

data.columns = [
    "load",
    "temperature"
]
print("\nMerged Data")
print(data.head())

# Train/Test Split
forecast_horizon = 104
train = data.iloc[:-forecast_horizon]
test = data.iloc[-forecast_horizon:]

# SARIMAX Model
# Replace orders if Part 3 has different values
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)

# Fit SARIMAX Model
print("\nTraining SARIMAX model...")

model = SARIMAX(
    train["load"],
    exog=train[["temperature"]],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)
print(results.summary())

# Forecast
forecast = results.get_forecast(
    steps=forecast_horizon,
    exog=test[["temperature"]]
)

forecast_mean = forecast.predicted_mean
confidence = forecast.conf_int()

# Metrics
rmse = np.sqrt(
    mean_squared_error(
        test["load"],
        forecast_mean
    )
)

mae = mean_absolute_error(
    test["load"],
    forecast_mean
)

mape = np.mean(
    np.abs(
        (test["load"] - forecast_mean)
        / test["load"]
    )
) * 100

print("\nPerformance")
print("RMSE :", round(rmse, 2))
print("MAE :", round(mae, 2))
print("MAPE :", round(mape, 2))

# Forecast Plot
plt.figure(figsize=(16, 7))
plt.plot(
    train.index,
    train["load"],
    label="Training"
)

plt.plot(
    test.index,
    test["load"],
    linewidth=2,
    label="Actual"
)

plt.plot(
    test.index,
    forecast_mean,
    color="red",
    linewidth=2,
    label="SARIMAX Forecast"
)

plt.fill_between(
    test.index,
    confidence.iloc[:, 0],
    confidence.iloc[:, 1],
    alpha=0.3,
    label="95% CI"
)
plt.title(
    "SARIMAX Forecast with Temperature"
)
plt.xlabel("Date")
plt.ylabel("Electricity Demand")
plt.legend()
plt.grid(True)

# Save forecast plot only
plt.savefig(
    os.path.join(
        plot_dir,
        "SARIMAX_Forecast_with_Temperature.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# Temperature Relationship Plot
plt.figure(figsize=(8, 6))
plt.scatter(
    data["temperature"],
    data["load"]
)

plt.xlabel(
    "Temperature (°C)"
)
plt.ylabel(
    "Electricity Demand"
)
plt.title(
    "Temperature vs Electricity Demand"
)
plt.grid(True)

# Save temperature plot only
plt.savefig(
    os.path.join(
        plot_dir,
        "Temperature_vs_Electricity_Demand.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# Save Forecast Results
results_df = pd.DataFrame({
    "Actual": test["load"],
    "Forecast": forecast_mean,
    "Temperature": test["temperature"]
})

results_df.to_csv(
    "sarimax_temperature_forecast.csv"
)

print("\nForecast saved.")
print("\nPlots saved in folder:", plot_dir)