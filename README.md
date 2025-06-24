# ⚡ Electricity Price Forecasting Using Weather and Energy Data

This project predicts **hourly electricity prices** using **time series data** of energy consumption and weather conditions across major cities in Spain. It uses deep learning (LSTM) with carefully engineered features and sequence-based inputs.



## 📁 Project Structure

---

1. Electricity_Price_Forecasting_Project_Using_Weather_and_Energy_Data_With_Time_Series_Analysis.ipynb
2.  energy_dataset.csv
3. weather_features.csv
4. multivariate_lstm.h5
5. README.md





## 📌 Problem Statement

---

Electricity prices are influenced by factors like weather, energy load, and time. This project aims to **forecast the next hour’s electricity price** using historical data and time series modeling. Accurate forecasts can help improve energy trading, planning, and grid management.

---

## 🧠 What This Project Does

- Merges hourly **electricity load** and **weather** data
- Feature engineering including:
  - `hour`, `weekday`, `month`
  - `business hour` encoding
  - `temperature range` and `weighted temperature`
- Time series correlation analysis:
  - **Autocorrelation**
  - **Partial Autocorrelation (PACF)**
  - **Cross-correlation**
- Data reshaping into supervised sequences
- Model training:
  - ✅ Multivariate **LSTM**
  - ✅ **XGBoost**
- Evaluation using **RMSE**

---

## 📊 Data Sources

- `energy_dataset.csv` — Hourly electricity load and price data
- `weather_features.csv` — Hourly temperature data from:
  - Madrid
  - Barcelona
  - Valencia
  - Seville
  - Bilbao

---

## 🔧 Feature Engineering

- Extracted from timestamps:
  - `hour`, `weekday`, `month`
- Categorical conversion:
  - `business hour` encoded as:
    - 0 → off-peak
    - 1 → mid-peak
    - 2 → peak hours
- Calculated:
  - `temp_range_city = temp_max - temp_min`
  - `temp_weighted` using weights for each city
- Applied `MinMaxScaler` to normalize features

---

## 🔍 Correlation Insights

Using **PACF and cross-correlation**:
- Strong lags: `t-1`, `t-2`, `t-24`
- Input window: 24 hours of past data used as input for predicting next hour

---

## 🧠 LSTM Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(24, 17)),
    Flatten(),
    Dense(200, activation='relu'),
    Dropout(0.1),
    Dense(1)  # For single-step prediction
])

```
---

## 🧪 Training
```
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   'multivariate_lstm.h5', monitor=('val_loss'), save_best_only=True)
optimizer = tf.keras.optimizers.Adam(lr=6e-3, amsgrad=True)

multivariate_lstm.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=metric)
history = multivariate_lstm.fit(train, epochs=120,
                                validation_data=validation,
                                callbacks=[early_stopping, 
                                           model_checkpoint])

```

