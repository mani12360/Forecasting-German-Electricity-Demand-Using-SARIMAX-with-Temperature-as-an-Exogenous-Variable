# PART 6
# Hourly Electricity Demand Forecasting
# LSTM Deep Learning Model
# German Electricity Demand
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Create folder for plots
plot_dir = "LSTM_Plots"
os.makedirs(
    plot_dir,
    exist_ok=True
)

# Load Hourly German Electricity Data
hourly = pd.read_csv(
    "time_series_60min_singleindex.csv",
    index_col=0,
    parse_dates=True
)

hourly.index = pd.to_datetime(
    hourly.index
)

# Remove timezone if present
if hourly.index.tz is not None:
    hourly.index = hourly.index.tz_localize(None)

# Select Germany electricity demand
series = hourly[
    "DE_load_actual_entsoe_transparency"
]

# Ensure hourly frequency
series = series.asfreq("h")

# Remove missing values
series = series.dropna()
print("\nHourly Dataset")
print("----------------")
print(series.head())
print(
    "\nTotal hourly observations:",
    len(series)
)

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(
    series.values.reshape(-1,1)
)

# Create LSTM Sequences
def create_sequences(
        data,
        window
):
    X = []
    y = []
    for i in range(
        window,
        len(data)
    ):
        X.append(
            data[i-window:i]
        )
        y.append(
            data[i]
        )
    return np.array(X), np.array(y)

# Previous 7 days of hourly data
sequence_length = 168

X, y = create_sequences(
    scaled_data,
    sequence_length
)

print("\nInput shape:")
print(X.shape)

# Train/Test Split
# Last 2 years
forecast_hours = 24 * 365 * 2

X_train = X[:-forecast_hours]
X_test = X[-forecast_hours:]

y_train = y[:-forecast_hours]
y_test = y[-forecast_hours:]

print("\nTraining samples:", X_train.shape)
print("Testing samples :", X_test.shape)

# LSTM Model
# Hyperparameters:
# Layers = 2
# Units = 64 + 32
# Dropout = 0.2
# Learning rate = 0.001
model = Sequential()
model.add(
    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            sequence_length,
            1
        )
    )
)
model.add(
    Dropout(0.2)
)

model.add(
    LSTM(
        32
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(1)
)

model.compile(
    optimizer=Adam(
        learning_rate=0.001
    ),
    loss="mean_squared_error"
)
model.summary()

# Train Model
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Forecast
prediction = model.predict(
    X_test
)

# Reverse scaling
prediction = scaler.inverse_transform(
    prediction
)
actual = scaler.inverse_transform(
    y_test
)

# Evaluation Metrics
rmse = np.sqrt(
    mean_squared_error(
        actual,
        prediction
    )
)

mae = mean_absolute_error(
    actual,
    prediction
)

mape = np.mean(
    np.abs(
        (actual - prediction)
        /
        actual
    )
) * 100

r2 = r2_score(
    actual,
    prediction
)

print("\nPerformance")
print("----------------")
print(
    "RMSE :",
    round(rmse,2)
)
print(
    "MAE  :",
    round(mae,2)
)
print(
    "MAPE :",
    round(mape,2)
)
print(
    "R²   :",
    round(r2,4)
)

# Forecast Plot
plt.figure(
    figsize=(16,7)
)
plt.plot(
    actual,
    label="Actual"
)
plt.plot(
    prediction,
    label="LSTM Forecast"
)

plt.title(
    "LSTM Forecast - German Hourly Electricity Demand"
)

plt.xlabel(
    "Hours"
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
        "LSTM_Hourly_Forecast.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# Training Loss Plot
plt.figure(
    figsize=(10,5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "LSTM Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE Loss"
)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(
        plot_dir,
        "LSTM_Training_Loss.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# Save Forecast Results
results = pd.DataFrame({
    "Actual": actual.flatten(),
    "Forecast": prediction.flatten()
})

results.to_csv(
    "lstm_hourly_forecast.csv",
    index=False
)

print(
    "\nLSTM forecast saved."
)

print(
    "Plots saved in:",
    plot_dir
)