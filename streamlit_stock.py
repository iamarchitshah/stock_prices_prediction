import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

st.title("📈 Deep LSTM Stock Price Prediction - Company Grade")

uploaded_file = st.file_uploader("Upload stock CSV file (include Date and Open columns)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns or 'Open' not in df.columns:
        st.error("CSV must include 'Date' and 'Open' columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Use only Open prices
    dataset = df[['Open']].values

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    PAST_DAYS = 120
    X_train, y_train = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X_train.append(scaled_data[i - PAST_DAYS:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    st.write("Training LSTM model (100 epochs)...")
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Prediction
    st.subheader("📊 Predicting Future Prices")
    test_data = scaled_data[-(PAST_DAYS + 20):]  # predict next 20 days
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        X_test.append(test_data[i - PAST_DAYS:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Actual prices (if future known, else dummy for plot)
    actual_prices = dataset[-20:] if len(dataset) >= 20 else predicted_price

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_prices, color='red', label='Real Stock Price')
    ax.plot(predicted_price, color='blue', label='Predicted Stock Price')
    ax.set_title('Stock Price Prediction (Next 20 days)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Show predictions
    st.subheader("📋 Predicted Prices")
    pred_df = pd.DataFrame(predicted_price, columns=['Predicted Price'])
    st.dataframe(pred_df)

    # CSV download
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predicted_prices.csv", mime="text/csv")
