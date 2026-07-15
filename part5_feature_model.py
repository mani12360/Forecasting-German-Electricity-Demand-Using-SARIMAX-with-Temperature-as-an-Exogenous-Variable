# PART 5
# Feature-Based Forecasting
# Gradient Boosting Regressor
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Create folder for plots only
plot_dir = "Gradient_Boosting_Plots"
os.makedirs(plot_dir, exist_ok=True)

# Load Weekly Electricity Data
weekly_load = pd.read_csv(
    "weekly_load.csv",
    index_col=0,
    parse_dates=True
)

weekly_load.index = pd.to_datetime(
    weekly_load.index
)

# Remove timezone only if it exists
if weekly_load.index.tz is not None:
    weekly_load.index = weekly_load.index.tz_localize(None)

weekly_load = weekly_load.asfreq("W-SUN")

# Load Weekly Temperature
weekly_temp = pd.read_csv(
    "weekly_temperature.csv",
    index_col=0,
    parse_dates=True
)

weekly_temp.index = pd.to_datetime(
    weekly_temp.index
)

# Remove timezone only if it exists
if weekly_temp.index.tz is not None:
    weekly_temp.index = weekly_temp.index.tz_localize(None)
weekly_temp = weekly_temp.asfreq("W-SUN")

# Merge Data
df = weekly_load.join(
    weekly_temp,
    how="inner"
)

df.columns = [
    "load",
    "temperature"
]

# Feature Engineering
# Load lags
for lag in [1, 2, 3, 4, 52]:
    df[f"lag_{lag}"] = df["load"].shift(lag)

# Temperature lags
df["temp_lag1"] = df["temperature"].shift(1)
df["temp_lag2"] = df["temperature"].shift(2)

# Rolling means
df["rolling4"] = (
    df["load"]
    .rolling(4)
    .mean()
)

df["rolling12"] = (
    df["load"]
    .rolling(12)
    .mean()
)

# Calendar features
df["month"] = df.index.month
df["week"] = (
    df.index.isocalendar()
    .week
    .astype(int)
)
df["quarter"] = df.index.quarter

# Remove missing values
df = df.dropna()

# Train/Test Split
forecast_horizon = 104

train = df.iloc[:-forecast_horizon]
test = df.iloc[-forecast_horizon:]

X_train = train.drop(
    columns=["load"]
)
y_train = train["load"]

X_test = test.drop(
    columns=["load"]
)
y_test = test["load"]

# Gradient Boosting Model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

# Metrics
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred
    )
)

mae = mean_absolute_error(
    y_test,
    pred
)

mape = np.mean(
    np.abs(
        (y_test - pred) / y_test
    )
) * 100

r2 = r2_score(
    y_test,
    pred
)

print("\nPerformance")
print("----------------")
print("RMSE :", round(rmse, 2))
print("MAE  :", round(mae, 2))
print("MAPE :", round(mape, 2))
print("R²   :", round(r2, 4))

# Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(
    ascending=False
)
print("\nFeature Importance")
print(importance)

# Save Feature Importance Plot
plt.figure(figsize=(10, 6))
importance.plot.bar()
plt.title(
    "Gradient Boosting Feature Importance"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Importance"
)

plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        plot_dir,
        "Gradient_Boosting_Feature_Importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# Forecast Plot
plt.figure(figsize=(16, 7))

plt.plot(
    train.index,
    train["load"],
    label="Training"
)

plt.plot(
    y_test.index,
    y_test,
    linewidth=2,
    label="Actual"
)

plt.plot(
    y_test.index,
    pred,
    color="red",
    linewidth=2,
    label="Gradient Boosting"
)

plt.title(
    "Gradient Boosting Forecast"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Electricity Demand"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        plot_dir,
        "Gradient_Boosting_Forecast.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# Save Forecast Results
results = pd.DataFrame({
    "Actual": y_test,
    "Prediction": pred
})
results.to_csv(
    "gradient_boosting_forecast.csv"
)
print("\nForecast saved.")
print("\nPlots saved in:", plot_dir)