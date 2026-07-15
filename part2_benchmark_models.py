import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load weekly data created in Part 1
weekly = pd.read_csv(
    "weekly_load.csv",
    index_col=0,
    parse_dates=True
)
series = weekly["load"]

# Train/Test Split
# Last two years (~104 weeks)
forecast_horizon = 104
train = series[:-forecast_horizon]
test = series[-forecast_horizon:]

print("Training observations:", len(train))
print("Testing observations :", len(test))

# Mean Forecast
mean_value = train.mean()
mean_forecast = np.repeat(mean_value, forecast_horizon)

# Naive Forecast
naive_value = train.iloc[-1]
naive_forecast = np.repeat(naive_value, forecast_horizon)

# Drift Forecast
first = train.iloc[0]
last = train.iloc[-1]
slope = (last - first) / (len(train) - 1)
drift_forecast = np.array([
    last + slope * (i + 1)
    for i in range(forecast_horizon)
])

# Seasonal Naive Forecast
# Weekly seasonality = 52 weeks
season = 52
seasonal_forecast = []
for i in range(forecast_horizon):
    seasonal_forecast.append(
        train.iloc[-season + (i % season)]
    )

seasonal_forecast = np.array(seasonal_forecast)

# Evaluation
def rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))
results = pd.DataFrame({
    "Model":[
        "Mean",
        "Naive",
        "Drift",
        "Seasonal Naive"
    ],
    "RMSE":[
        rmse(test, mean_forecast),
        rmse(test, naive_forecast),
        rmse(test, drift_forecast),
        rmse(test, seasonal_forecast)
    ]
})
print("\nModel Comparison")
print(results)

# Plot Forecasts
plt.figure(figsize=(16,7))
plt.plot(train.index, train,
        label="Training",
        color="black")

plt.plot(test.index, test,
        label="Actual",
        linewidth=3)

plt.plot(test.index, mean_forecast,
        label="Mean")

plt.plot(test.index, naive_forecast,
        label="Naive")

plt.plot(test.index, drift_forecast,
        label="Drift")

plt.plot(test.index, seasonal_forecast,
        label="Seasonal Naive")
plt.title("Benchmark Forecast Models")
plt.xlabel("Date")
plt.ylabel("Electricity Demand")
plt.legend()
plt.grid(True)
plt.show()

# Save results
results.to_csv("benchmark_results.csv", index=False)
print("\nBenchmark results saved.")