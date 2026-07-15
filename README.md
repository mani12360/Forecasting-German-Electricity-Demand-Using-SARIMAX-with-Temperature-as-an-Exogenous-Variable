# Electricity Demand Forecasting using SARIMAX

## Project Overview

This project focuses on forecasting weekly electricity demand in Germany using the **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous Variables)** model. The aim is to predict future electricity demand by learning historical demand patterns while also considering temperature as an external influencing factor.

Electricity demand changes throughout the year because of seasonal behaviour and weather conditions. During colder periods, electricity usage generally increases due to heating requirements. By including temperature in the forecasting model, the predictions become more accurate compared to using historical demand alone.

This project was developed as part of a time series forecasting assignment using Python.

---

## Objectives

The main objectives of this project are:

- Forecast future electricity demand.
- Analyse seasonal patterns in electricity consumption.
- Investigate the effect of temperature on electricity demand.
- Build a SARIMAX forecasting model.
- Evaluate forecasting performance using standard error metrics.

---

## Dataset

The dataset contains weekly observations of:

- Date
- Electricity Demand
- Temperature

Temperature is used as an **exogenous variable**, meaning it provides additional information that helps improve forecasting accuracy.

---

## Technologies Used

- Python 3
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn

---

## Project Structure

```
Electricity-Demand-Forecasting/
│
├── data/
│   ├── electricity_demand.csv
│   └── temperature.csv
│
├── notebooks/
│   └── Electricity_Demand_Forecasting.ipynb
│
├── report/
│   └── Electricity_Demand_Forecasting_Report.pdf
│
├── images/
│   ├── demand_plot.png
│   ├── forecast_plot.png
│   └── residual_plot.png
│
├── README.md
└── requirements.txt
```

---

## Methodology

The forecasting process follows several stages.

### 1. Data Preparation

The data was first cleaned and prepared by:

- Loading the dataset
- Converting the date column into datetime format
- Setting the date as the time index
- Checking for missing values
- Exploring seasonal patterns

---

### 2. Exploratory Data Analysis

The electricity demand data was visualised to identify:

- Long-term trends
- Seasonal behaviour
- Weekly demand fluctuations
- Relationship between temperature and electricity demand

---

### 3. Model Development

A SARIMAX model was developed because it combines:

- Autoregressive (AR)
- Differencing (I)
- Moving Average (MA)
- Seasonal components
- Exogenous variables

Unlike a standard ARIMA model, SARIMAX allows external variables such as temperature to influence predictions.

---

### 4. Forecasting

The model was trained using historical electricity demand together with temperature observations.

Forecasts were then generated for the test dataset and compared against actual electricity demand.

---

### 5. Model Evaluation

The forecasting model was evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Correlation

These metrics measure how closely the predicted values match the actual electricity demand.

---

## Results

The SARIMAX model successfully captured the main seasonal patterns in electricity demand.

Including temperature improved the forecasting performance because electricity consumption is strongly influenced by weather conditions.

Final evaluation results:

| Metric | Value |
|---------|--------|
| MAE | 3710.02 |
| RMSE | 4567.31 |
| MAPE | 7.05% |
| Correlation | 0.637 |

Overall, the model produced reliable forecasts while maintaining good interpretability.

---

## Installation

Clone this repository.

```bash
git clone https://github.com/yourusername/Electricity-Demand-Forecasting.git
```

Move into the project folder.

```bash
cd Electricity-Demand-Forecasting
```

Install the required libraries.

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
Electricity_Demand_Forecasting.ipynb
```

Run all cells to reproduce the analysis and forecasting results.

---

## Required Libraries

```text
pandas
numpy
matplotlib
statsmodels
scikit-learn
```

or install manually:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn
```

---

## Model Limitations

Although the model performs well, several factors were not included:

- Public holidays
- Industrial electricity usage
- Renewable energy production
- Economic activity
- Population changes
- Unexpected events

Including these variables could further improve forecasting accuracy.

---

## Future Improvements

Possible future enhancements include:

- Adding humidity and wind speed
- Using holiday calendars
- Automatic parameter optimisation
- Comparing SARIMAX with:
  - Random Forest
  - XGBoost
  - LSTM Neural Networks
- Hybrid statistical and machine learning models

---

## Learning Outcomes

Through this project I learned how to:

- Prepare time series datasets
- Explore seasonal patterns
- Build SARIMAX forecasting models
- Use exogenous variables
- Evaluate forecasting accuracy
- Interpret forecasting results

---

## References

Box, G., Jenkins, G., Reinsel, G., & Ljung, G. (2015). *Time Series Analysis: Forecasting and Control*. Wiley.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice.*

Statsmodels Documentation:
https://www.statsmodels.org/

---

## Author

**Manikanta Nagendla**

Student ID: **24098557**

University Assignment – Electricity Demand Forecasting using SARIMAX

---

## License

This project was created for academic purposes only.
