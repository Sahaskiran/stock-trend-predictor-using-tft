import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime as dt

# =====================
# Date range
# =====================
start = '2010-01-01'
end = dt.date.today()

# =====================
# App Title
# =====================
st.title('📈 Stock Trend Prediction (TFT Model)')

# =====================
# Stock input
# =====================
user_input = st.text_input(
    "Enter stock ticker (Example: AAPL, TSLA, MSFT, RELIANCE.NS)", 
    'AAPL'
)

# Currency symbol
currency = "₹" if user_input.upper().endswith(".NS") else "$"

# =====================
# Download stock data
# =====================
df = yf.download(user_input, start=start, end=end)

if df.empty:
    st.error("❌ Invalid stock ticker or no data available.")
    st.stop()

# Handle MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# =====================
# Data description
# =====================
st.subheader(f'Data from 2010 to {end}')
st.write(df.describe())

# =====================
# Visualizations
# =====================
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.index, df.Close)
plt.xlabel("Year")
plt.ylabel(f"Price ({currency})")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
st.pyplot(fig)

# 100 MA
st.subheader('Closing Price vs Time with 100 MA')
ma100 = df.Close.rolling(100).mean()

fig = plt.figure(figsize=(12, 6))
plt.plot(df.index, df.Close, label='Close')
plt.plot(df.index, ma100, label='100 MA')
plt.xlabel("Year")
plt.ylabel(f"Price ({currency})")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# 100 & 200 MA
st.subheader('Closing Price vs Time with 100 MA & 200 MA')
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12, 6))
plt.plot(df.index, df.Close, label='Close')
plt.plot(df.index, ma100, label='100 MA')
plt.plot(df.index, ma200, label='200 MA')
plt.xlabel("Year")
plt.ylabel(f"Price ({currency})")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

# Volume
st.subheader('Volume vs Time')
fig = plt.figure(figsize=(12, 6))
plt.bar(df.index, df.Volume, alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Volume")
plt.xticks(rotation=45)
st.pyplot(fig)

# =====================
# Data Split
# =====================
data_training = pd.DataFrame(df['Close'][:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data_training)

# =====================
# Load Model
# =====================
model = load_model('tft_model.h5')

# =====================
# Prepare Test Data
# =====================
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.transform(final_df)

x_test, y_test = [], []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test).reshape(-1, 1)

# =====================
# Prediction
# =====================
y_predicted = model.predict(x_test, verbose=0)

# =====================
# Inverse Scaling
# =====================
scale_factor = 1 / scaler.scale_[0]

y_test = y_test * scale_factor
y_predicted = y_predicted * scale_factor

y_test = y_test.flatten()
y_predicted = y_predicted.flatten()

st.subheader('📊 Predicted vs Original Price')

# ✅ ADD THIS LINE (important)
test_dates = df.index[len(df) - len(y_test):]

fig = plt.figure(figsize=(12,6))

# ✅ CHANGE plotting lines
plt.plot(test_dates, y_test, label="Actual")
plt.plot(test_dates, y_predicted, label="Predicted")

plt.xlabel("Date")
plt.ylabel(f"Price ({currency})")

# ✅ Format dates
import matplotlib.dates as mdates
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.legend()

st.pyplot(fig)

# =====================
# Future Forecast (7 Days)
# =====================
forecast_days = 7
last_100 = input_data[-100:].reshape(1, 100, 1)

future_predictions = []

for _ in range(forecast_days):
    next_scaled = model.predict(last_100, verbose=0)[0][0]
    future_predictions.append(next_scaled)
    last_100 = np.append(last_100[:,1:,:], [[[next_scaled]]], axis=1)

future_predictions = np.array(future_predictions).reshape(-1,1) * scale_factor

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq='D'
)

# =====================
# Market Status
# =====================
market_status = [
    "Market Closed (Weekend)" if d.weekday() >= 5 else "Market Open"
    for d in future_dates
]

# =====================
# Forecast Graph
# =====================
st.subheader("📈 Next-Week Price Forecast")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df.index[-150:], df.Close.iloc[-150:], label="Actual")
ax.plot(future_dates, future_predictions, '--o', label="Forecast")

ax.set_xlabel("Year")
ax.set_ylabel(f"Price ({currency})")

ax.legend()
st.pyplot(fig)

# =====================
# Forecast Table
# =====================
st.subheader("📅 Next 7 Days Forecasted Prices")

forecast_table = pd.DataFrame({
    "Date": future_dates.strftime("%d %b %Y"),
    "Predicted Price": [
        f"{currency}{p[0]:.2f}" if status == "Market Open" else "—"
        for p, status in zip(future_predictions, market_status)
    ],
    "Market Status": market_status
})

st.table(forecast_table)

if "Market Closed (Weekend)" in market_status:
    st.info("ℹ️ Stock markets are closed on weekends.")

# =====================
# Weekly Metrics
# =====================
weekly_actual, weekly_pred = [], []

max_len = min(len(y_test), len(y_predicted))

for i in range(0, max_len - 5, 5):
    weekly_actual.append(y_test[i + 5] - y_test[i])
    weekly_pred.append(y_predicted[i + 5] - y_predicted[i])

weekly_actual = np.array(weekly_actual)
weekly_pred = np.array(weekly_pred)

neutral_pct = 0.005
price_mean = np.mean(y_test)

actual_labels = np.where(
    weekly_actual > neutral_pct * price_mean, 1,
    np.where(weekly_actual < -neutral_pct * price_mean, 0, -1)
)

predicted_labels = np.where(
    weekly_pred > neutral_pct * price_mean, 1,
    np.where(weekly_pred < -neutral_pct * price_mean, 0, -1)
)

actual_labels = actual_labels.reshape(-1)
predicted_labels = predicted_labels.reshape(-1)

mask = (actual_labels != -1) & (predicted_labels != -1)

actual_labels = actual_labels[mask]
predicted_labels = predicted_labels[mask]

w_acc = accuracy_score(actual_labels, predicted_labels)
w_prec = precision_score(actual_labels, predicted_labels, zero_division=0)
w_rec = recall_score(actual_labels, predicted_labels, zero_division=0)
w_f1 = f1_score(actual_labels, predicted_labels, zero_division=0)

st.subheader("📅 Weekly Direction Metrics")

st.write(f"Accuracy: {w_acc*100:.2f}%")
st.write(f"Precision: {w_prec*100:.2f}%")
st.write(f"Recall: {w_rec*100:.2f}%")
st.write(f"F1 Score: {w_f1*100:.2f}%")

st.success("✅ Stock forecasting completed!")