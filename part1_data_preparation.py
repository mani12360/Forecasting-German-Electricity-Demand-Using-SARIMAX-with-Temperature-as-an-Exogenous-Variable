# PART 1 - German Electricity Demand Analysis
# MSc Data Science Assignment

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create folder to save plots
plot_folder = "part_1 plots"
os.makedirs(plot_folder, exist_ok=True)

# Load dataset
file = "time_series_60min_singleindex.csv"
df = pd.read_csv(file)

# Convert timestamp
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])

# Use timestamp as index
df.set_index('utc_timestamp', inplace=True)

# Keep German electricity demand only
df = df[['DE_load_actual_entsoe_transparency']]
df.columns = ['load']

# Keep required period
df = df.loc['2015-01-01':'2020-10-31']

print(df.head())
print(df.info())

# Missing values
print("\nMissing values:")
print(df.isna().sum())

# Fill missing values
df['load'] = df['load'].interpolate()

# Create Daily and Weekly data
daily = df.resample('D').mean()
weekly = df.resample('W').mean()

print("\nHourly observations:", len(df))
print("Daily observations :", len(daily))
print("Weekly observations:", len(weekly))

# Basic statistics
print("\nSummary Statistics")
print(weekly.describe())

# Plot Hourly Data
plt.figure(figsize=(15,5))
plt.plot(df.index, df['load'])
plt.title("German Electricity Demand (Hourly)")
plt.ylabel("MW")
plt.xlabel("Date")
plt.grid(True)

plt.savefig(
    os.path.join(plot_folder, "hourly_electricity_demand.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# Plot Daily Data
plt.figure(figsize=(15,5))
plt.plot(daily.index, daily['load'])
plt.title("Daily Electricity Demand")
plt.ylabel("MW")
plt.xlabel("Date")
plt.grid(True)

plt.savefig(
    os.path.join(plot_folder, "daily_electricity_demand.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# Plot Weekly Data
plt.figure(figsize=(15,5))
plt.plot(weekly.index, weekly['load'])
plt.title("Weekly Electricity Demand")
plt.ylabel("MW")
plt.xlabel("Date")
plt.grid(True)

plt.savefig(
    os.path.join(plot_folder, "weekly_electricity_demand.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# Seasonal Decomposition
decomposition = seasonal_decompose(
    weekly['load'],
    model='additive',
    period=52
)

fig = decomposition.plot()
fig.set_size_inches(15, 10)

fig.savefig(
    os.path.join(plot_folder, "seasonal_decomposition.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ADF Test
print("\nAugmented Dickey-Fuller Test")

result = adfuller(weekly['load'])

print("ADF Statistic :", result[0])
print("p-value       :", result[1])
print("Critical Values")

for key, value in result[4].items():
    print(key, value)

# First Difference
weekly_diff = weekly.diff().dropna()

print("\nADF after First Difference")

result = adfuller(weekly_diff['load'])

print("ADF Statistic :", result[0])
print("p-value       :", result[1])

# Plot Differenced Series
plt.figure(figsize=(15,5))
plt.plot(weekly_diff)
plt.title("First Differenced Weekly Series")
plt.ylabel("Differenced Load")
plt.xlabel("Date")
plt.grid(True)

plt.savefig(
    os.path.join(plot_folder, "first_difference.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ACF (Original Series)
plot_acf(
    weekly['load'],
    lags=60
)

plt.savefig(
    os.path.join(plot_folder, "acf_original.png"),
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# PACF (Original Series)
plot_pacf(
    weekly['load'],
    lags=60,
    method="ywm"
)
plt.savefig(
    os.path.join(plot_folder, "pacf_original.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ACF (Differenced Series)
plot_acf(
    weekly_diff['load'],
    lags=60
)
plt.savefig(
    os.path.join(plot_folder, "acf_differenced.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# PACF (Differenced Series)
plot_pacf(
    weekly_diff['load'],
    lags=60,
    method="ywm"
)
plt.savefig(
    os.path.join(plot_folder, "pacf_differenced.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# Save processed data
daily.to_csv("daily_load.csv")
weekly.to_csv("weekly_load.csv")

print("\nFinished Part 1")
print("All plots have been saved")