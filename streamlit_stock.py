# stock_predictor_full_comparison.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, SimpleRNN, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten

st.title("📊 Stock Price Predictor (Open & Close) - All Models")

model_type = st.selectbox("Choose Model", ["Random Forest", "GRU", "RNN", "CNN"])
uploaded_file = st.file_uploader("Upload CSV with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 60
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i])
        y.append(scaled_data[i, [0, 3]])
    X, y = np.array(X), np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    if model_type == "Random Forest":
        X_rf = X.reshape(X.shape[0], -1)
        X_train_rf, X_val_rf = X_rf[:len(X_train)], X_rf[len(X_train):]
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_train_rf, y_train)
        val_pred = model.predict(X_val_rf)

    elif model_type == "GRU":
        model = Sequential([
            GRU(50, return_sequences=False, input_shape=(PAST_DAYS, 5)),
            Dense(2)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        val_pred = model.predict(X_val)

    elif model_type == "RNN":
        model = Sequential([
            SimpleRNN(50, return_sequences=False, input_shape=(PAST_DAYS, 5)),
            Dense(2)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        val_pred = model.predict(X_val)

    elif model_type == "CNN":
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(PAST_DAYS, 5)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(2)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        val_pred = model.predict(X_val)

    # Reverse scaling for plotting
    padded_pred = np.zeros((val_pred.shape[0], 5))
    padded_pred[:, 0] = val_pred[:, 0]
    padded_pred[:, 3] = val_pred[:, 1]
    pred_val_prices = scaler.inverse_transform(padded_pred)

    actual_val = scaler.inverse_transform(np.concatenate(
        [y_val[:, [0]], np.zeros((len(y_val), 2)), y_val[:, [1]], np.zeros((len(y_val), 1))], axis=1))

    st.subheader("📈 Actual vs Predicted")
    fig_val, ax_val = plt.subplots(figsize=(12, 6))
    ax_val.plot(actual_val[:, 0], label='Actual Open', linestyle='--')
    ax_val.plot(pred_val_prices[:, 0], label='Predicted Open')
    ax_val.plot(actual_val[:, 3], label='Actual Close', linestyle='--')
    ax_val.plot(pred_val_prices[:, 3], label='Predicted Close')
    ax_val.set_title(f"{model_type} Model: Validation Performance")
    ax_val.legend()
    st.pyplot(fig_val)

    # Predict next 20 days
    st.subheader("🔮 Predicting Next 20 Days")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        X_test.append(test_data[i - PAST_DAYS:i])
    X_test = np.array(X_test)

    if model_type == "Random Forest":
        test_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
    else:
        test_pred = model.predict(X_test)

    padded = np.zeros((test_pred.shape[0], 5))
    padded[:, 0] = test_pred[:, 0]
    padded[:, 3] = test_pred[:, 1]
    predicted_prices = scaler.inverse_transform(padded)

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=20, freq='B')
    pred_df = pd.DataFrame({
        'Predicted Open': predicted_prices[:, 0],
        'Predicted Close': predicted_prices[:, 3]
    }, index=future_dates)

    st.dataframe(pred_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pred_df.index, pred_df['Predicted Open'], label='Predicted Open', marker='o')
    ax.plot(pred_df.index, pred_df['Predicted Close'], label='Predicted Close', marker='x')
    ax.set_title("Next 20 Day Forecast")
    ax.legend()
    st.pyplot(fig)

    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Predictions", csv, f"predicted_{model_type.lower()}.csv", "text/csv")
