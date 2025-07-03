import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from keras.callbacks import EarlyStopping

st.title("📈 Stock Price Prediction using RNN / LSTM / GRU")

# Upload CSV File
uploaded_file = st.file_uploader("Upload CSV file (must include 'Date' and 'Open' columns)", type="csv")
model_type = st.selectbox("Select Model Type", ["LSTM", "GRU", "RNN"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Date' not in df.columns or 'Open' not in df.columns:
        st.error("CSV must contain 'Date' and 'Open' columns.")
        st.stop()

    # Preprocess
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

    # Build model
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

    # Early stopping
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    st.write(f"⏳ Training {model_type} model (max 20 epochs)...")
    model.fit(X, y, epochs=20, batch_size=32, verbose=1, callbacks=[early_stop])

    # Predict next 20 days
    st.subheader("🔮 Predicting Next 20 Business Days")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        X_test.append(test_data[i - PAST_DAYS:i, 0])
    X_test = np.array(X_test).reshape((-1, PAST_DAYS, 1))

    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)

    # Create date range for predictions
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=20, freq='B')

    # Display predictions with dates
    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predicted_prices.flatten()
    }).set_index("Date")

    st.subheader("📅 Predicted Prices")
    st.dataframe(pred_df)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pred_df.index, pred_df["Predicted Price"], marker='o', label='Predicted Price')
    ax.set_title(f"Predicted Stock Prices for Next 20 Days using {model_type}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Download
    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Predictions as CSV", csv, "predicted_prices.csv", "text/csv")
