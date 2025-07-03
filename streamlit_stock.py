# stock_predictor_advanced.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, SimpleRNN, Dense, Conv1D, MaxPooling1D, Flatten, Input, Dropout, LayerNormalization, GlobalAveragePooling1D, LSTM
from tensorflow.keras import Model
import pandas_ta as ta

st.set_page_config(layout="wide")
st.title("📊 Ultra Accurate Stock Price Predictor with Deep Feature Engineering")

model_type = st.selectbox("Choose Model", ["LSTM", "GRU", "RNN", "CNN", "Random Forest", "Transformer"])
uploaded_file = st.file_uploader("Upload your CSV", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # 🌟 Advanced Feature Engineering
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close']).iloc[:, 0]
    df['Volume_Change'] = df['Volume'].pct_change()
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA10', 'MA50', 'EMA10', 'Momentum', 'RSI', 'MACD', 'Volume_Change']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 60
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i])
        y.append(scaled_data[i, [0, 3]])
    X, y = np.array(X), np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    def transformer_encoder(inputs, head_size=128, num_heads=4, ff_dim=256, dropout=0.2):
        x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)
        res = x
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return LayerNormalization(epsilon=1e-6)(x + res)

    def build_transformer_model(input_shape, num_layers=4):
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(num_layers):
            x = transformer_encoder(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(2)(x)
        return Model(inputs, outputs)

    st.info("🧠 Training model... Please wait")

    if model_type == "Random Forest":
        X_rf = X.reshape(X.shape[0], -1)
        model = RandomForestRegressor(n_estimators=200, max_depth=25, random_state=42, n_jobs=-1)
        model.fit(X_rf[:len(X_train)], y_train)
        val_pred = model.predict(X_rf[len(X_train):])
    else:
        if model_type == "Transformer":
            model = build_transformer_model(X.shape[1:], num_layers=4)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            val_pred = model.predict(X_val)
        else:
            model = Sequential()
            if model_type == "LSTM":
                model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
                model.add(Dropout(0.2))
                model.add(LSTM(64))
            elif model_type == "GRU":
                model.add(GRU(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
                model.add(Dropout(0.2))
                model.add(GRU(64))
            elif model_type == "RNN":
                model.add(SimpleRNN(64, return_sequences=False, input_shape=(X.shape[1], X.shape[2])))
            elif model_type == "CNN":
                model.add(Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], X.shape[2])))
                model.add(MaxPooling1D(pool_size=2))
                model.add(Flatten())
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(2))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
            val_pred = model.predict(X_val)

    # 🎯 Add Metrics
    val_mape = mean_absolute_percentage_error(y_val, val_pred) * 100
    val_mse = mean_squared_error(y_val, val_pred)
    st.success(f"✅ Model trained successfully | Validation MAPE: {val_mape:.2f}% | MSE: {val_mse:.4f}")

    st.subheader("📉 Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_val[:, 0], label='Actual Open', linestyle='--')
    ax.plot(val_pred[:, 0], label='Predicted Open')
    ax.plot(y_val[:, 1], label='Actual Close', linestyle='--')
    ax.plot(val_pred[:, 1], label='Predicted Close')
    ax.set_title(f"{model_type} Model - Prediction Accuracy")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🔍 All Prediction Values")
    pred_df = pd.DataFrame({
        "Actual Open": y_val[:, 0],
        "Predicted Open": val_pred[:, 0],
        "Actual Close": y_val[:, 1],
        "Predicted Close": val_pred[:, 1],
    })
    st.dataframe(pred_df)

    st.subheader("📈 Future Prediction (10 Days)")
    future_input = X[-1]
    future_preds = []
    for _ in range(10):
        inp = future_input.reshape(1, PAST_DAYS, X.shape[2])
        if model_type == "Random Forest":
            inp_rf = inp.reshape(1, -1)
            pred = model.predict(inp_rf)
        else:
            pred = model.predict(inp, verbose=0)
        future_preds.append(pred[0])
        next_row = np.append(pred[0], [0] * (X.shape[2] - 2))
        future_input = np.vstack((future_input[1:], next_row))

    future_preds = np.array(future_preds)
    base = np.zeros((future_preds.shape[0], X.shape[2]))
    base[:, 0] = future_preds[:, 0]
    base[:, 3] = future_preds[:, 1]
    inv = scaler.inverse_transform(base)
    future_open = inv[:, 0]
    future_close = inv[:, 3]
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=10, freq='B')
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Open": future_open,
        "Predicted Close": future_close
    })
    st.dataframe(future_df)

    fig2, ax2 = plt.subplots()
    ax2.plot(future_df["Date"], future_df["Predicted Open"], label="Future Open")
    ax2.plot(future_df["Date"], future_df["Predicted Close"], label="Future Close")
    ax2.set_title("🔮 Forecast for Next 10 Days")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    st.success("🎉 Deep model training and forecasting complete!")
