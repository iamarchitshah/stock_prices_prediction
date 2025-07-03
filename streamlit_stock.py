# transformer_stock_prediction.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers

st.title("🔮 Transformer-Based Stock Price Predictor")

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
        y.append(scaled_data[i, [0, 3]])  # Predict Open and Close
    X, y = np.array(X), np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return layers.LayerNormalization(epsilon=1e-6)(x + res)

    def build_model(input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2)(x)
        return tf.keras.Model(inputs, outputs)

    st.write("⏳ Training Transformer model...")
    model = build_model(X_train.shape[1:])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32, verbose=1)

    val_pred = model.predict(X_val)

    st.subheader("📈 Validation Results")
    padded_val = np.zeros((val_pred.shape[0], 5))
    padded_val[:, 0] = val_pred[:, 0]
    padded_val[:, 3] = val_pred[:, 1]
    pred_val_prices = scaler.inverse_transform(padded_val)

    actual_val = scaler.inverse_transform(np.concatenate(
        [y_val[:, [0]], np.zeros((len(y_val), 2)), y_val[:, [1]], np.zeros((len(y_val), 1))], axis=1))

    fig_val, ax_val = plt.subplots(figsize=(12, 6))
    ax_val.plot(actual_val[:, 0], label='Actual Open', linestyle='--')
    ax_val.plot(pred_val_prices[:, 0], label='Predicted Open')
    ax_val.plot(actual_val[:, 3], label='Actual Close', linestyle='--')
    ax_val.plot(pred_val_prices[:, 3], label='Predicted Close')
    ax_val.set_title("Actual vs Predicted (Validation)")
    ax_val.legend()
    st.pyplot(fig_val)

    # Predict next 20 days
    st.subheader("🔮 Predicting Next 20 Days")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        sample = test_data[i - PAST_DAYS:i]
        X_test.append(sample)
    X_test = np.array(X_test)

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
    ax.set_title("Predicted Prices (Next 20 Days)")
    ax.legend()
    st.pyplot(fig)

    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predicted_transformer.csv", "text/csv")
