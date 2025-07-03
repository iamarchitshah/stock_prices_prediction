import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN

st.title("📈 Stock Price Prediction using RNN / LSTM / GRU")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV file (with 'Date' and 'Open' columns)", type="csv")
model_type = st.selectbox("Select Model Type", ["LSTM", "GRU", "RNN"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns or 'Open' not in df.columns:
        st.error("CSV must contain 'Date' and 'Open' columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    data = df[['Open']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 120
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Model definition
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
    elif model_type == 'GRU':
        model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50))
    elif model_type == 'RNN':
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50))

    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    st.write(f"Training {model_type} model for 100 epochs...")
    model.fit(X, y, epochs=100, batch_size=32, verbose=1)

    st.subheader("🔮 Predicting Next 20 Days")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        X_test.append(test_data[i - PAST_DAYS:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(1, 21), predicted_prices, label='Predicted Price', color='blue')
    ax.set_title(f"Predicted Stock Prices for Next 20 Days using {model_type}")
    ax.set_xlabel("Future Days")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Display Table
    pred_df = pd.DataFrame(predicted_prices, columns=["Predicted Price"])
    st.dataframe(pred_df)
    st.download_button("📥 Download Predictions as CSV", pred_df.to_csv(index=False), "predicted_prices.csv", "text/csv")
