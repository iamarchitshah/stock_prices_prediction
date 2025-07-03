# advanced_stock_prediction.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from keras.models import Model
from keras.layers import (Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization,
                         Flatten, Activation, RepeatVector, Permute, Multiply, Lambda)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import keras.backend as K

st.title("🔬 Advanced Stock Price Predictor with Attention (Open & Close)")
uploaded_file = st.file_uploader("Upload CSV with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 180
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i])
        y.append(scaled_data[i, [0, 3]])  # Open & Close

    X, y = np.array(X), np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    def attention_block(inputs):
        attention = Dense(1, activation='tanh')(inputs)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(inputs.shape[-1])(attention)
        attention = Permute([2, 1])(attention)
        sent_representation = Multiply()([inputs, attention])
        return Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = attention_block(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(2)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

    st.write("⏳ Training the model with Attention and Stacked Bi-LSTM layers...")

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=64, callbacks=callbacks, verbose=1)

    # Predict next 20 days
    st.subheader("🔮 Predicting Next 20 Days (Open & Close)")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        X_test.append(test_data[i - PAST_DAYS:i])
    X_test = np.array(X_test)

    predicted_scaled = model.predict(X_test)
    padded = np.zeros((predicted_scaled.shape[0], 5))
    padded[:, 0] = predicted_scaled[:, 0]  # Open
    padded[:, 3] = predicted_scaled[:, 1]  # Close
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
    ax.set_title("Predicted Stock Prices (Next 20 Business Days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "advanced_predictions.csv", "text/csv")
