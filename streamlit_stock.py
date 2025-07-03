# stock_predictor_all_models.py (Enhanced with Future Predictions)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, SimpleRNN, Dense, Conv1D, MaxPooling1D, Flatten, Input, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras import Model
import pandas_ta as ta

st.title("📊 Enhanced Stock Price Predictor with Feature Engineering and Future Prediction")

model_type = st.selectbox("Choose Model", ["Random Forest", "GRU", "RNN", "CNN", "Transformer"])
uploaded_file = st.file_uploader("Upload CSV with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Feature Engineering
    df['Return_Open'] = df['Open'].pct_change()
    df['Return_Close'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df.dropna(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return_Open', 'Return_Close', 'MA5', 'MA20', 'Volatility', 'RSI']
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 60
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i])
        y.append(scaled_data[i, [0, 3]])  # Open and Close
    X, y = np.array(X), np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=64, dropout=0.1):
        x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
        x = Dropout(dropout)(x)
        x = LayerNormalization(epsilon=1e-6)(x + inputs)
        res = x
        x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return LayerNormalization(epsilon=1e-6)(x + res)

    def build_transformer_model(input_shape):
        inputs = Input(shape=input_shape)
        x = transformer_encoder(inputs)
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(2)(x)
        return Model(inputs, outputs)

    if model_type == "Random Forest":
        X_rf = X.reshape(X.shape[0], -1)
        X_train_rf, X_val_rf = X_rf[:len(X_train)], X_rf[len(X_train):]
        model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        model.fit(X_train_rf, y_train)
        val_pred = model.predict(X_val_rf)
    else:
        model = Sequential()
        if model_type == "GRU":
            model.add(GRU(64, return_sequences=False, input_shape=(PAST_DAYS, X.shape[2])))
        elif model_type == "RNN":
            model.add(SimpleRNN(64, return_sequences=False, input_shape=(PAST_DAYS, X.shape[2])))
        elif model_type == "CNN":
            model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(PAST_DAYS, X.shape[2])))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
        elif model_type == "Transformer":
            model = build_transformer_model(X.shape[1:])

        if model_type != "Transformer":
            model.add(Dense(2))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            val_pred = model.predict(X_val)
        else:
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            val_pred = model.predict(X_val)

    st.subheader("📈 Validation Results")
    val_mape = mean_absolute_percentage_error(y_val, val_pred) * 100
    st.write(f"Validation MAPE: {val_mape:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_val[:, 0], label='Actual Open', linestyle='--')
    ax.plot(val_pred[:, 0], label='Predicted Open')
    ax.plot(y_val[:, 1], label='Actual Close', linestyle='--')
    ax.plot(val_pred[:, 1], label='Predicted Close')
    ax.set_title(f"{model_type} Model: Actual vs Predicted")
    ax.legend()
    st.pyplot(fig)

    pred_df = pd.DataFrame({
        "Actual Open": y_val[:, 0],
        "Predicted Open": val_pred[:, 0],
        "Actual Close": y_val[:, 1],
        "Predicted Close": val_pred[:, 1],
    })
    st.subheader("📋 All Predicted Values")
    st.dataframe(pred_df)

    # 🔮 Future Predictions
    st.subheader("🔮 Future Predictions (Next 10 Days)")
    future_days = 10
    future_input = X[-1]
    future_preds = []

    for _ in range(future_days):
        inp = future_input.reshape(1, PAST_DAYS, X.shape[2])
        pred = model.predict(inp, verbose=0)
        future_preds.append(pred[0])
        next_input = np.concatenate((future_input[1:], [np.append(pred[0], [0] * (X.shape[2] - 2))]), axis=0)
        future_input = next_input

    future_preds = np.array(future_preds)
    template_row = np.zeros((future_preds.shape[0], X.shape[2]))
    template_row[:, 0] = future_preds[:, 0]
    template_row[:, 3] = future_preds[:, 1]
    scaled_back = scaler.inverse_transform(template_row)
    future_open = scaled_back[:, 0]
    future_close = scaled_back[:, 3]

    dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='B')
    future_df = pd.DataFrame({
        "Date": dates,
        "Predicted Open": future_open,
        "Predicted Close": future_close
    })
    st.dataframe(future_df)

    fig2, ax2 = plt.subplots()
    ax2.plot(future_df["Date"], future_df["Predicted Open"], label="Future Open")
    ax2.plot(future_df["Date"], future_df["Predicted Close"], label="Future Close")
    ax2.set_title("📈 Future Stock Price Predictions (Next 10 Days)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    st.success("✅ Model improved and extended with future prediction!")
