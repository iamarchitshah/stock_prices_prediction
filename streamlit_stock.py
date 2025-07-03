import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN
from keras.callbacks import EarlyStopping

st.title("📊 Predict Open & Close Prices using RNN / LSTM / GRU")

uploaded_file = st.file_uploader("Upload CSV file with 'Date', 'Open' & 'Close'", type="csv")
model_type = st.selectbox("Select Model Type", ["LSTM", "GRU", "RNN"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Date' not in df.columns or 'Open' not in df.columns or 'Close' not in df.columns:
        st.error("CSV must contain 'Date', 'Open', and 'Close' columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    data = df[['Open', 'Close']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 120
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i])
        y.append(scaled_data[i])  # y will have both [Open, Close]
    X, y = np.array(X), np.array(y)

    # Build model
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 2)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
    elif model_type == 'GRU':
        model.add(GRU(50, return_sequences=True, input_shape=(X.shape[1], 2)))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50))
    elif model_type == 'RNN':
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(X.shape[1], 2)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50))

    model.add(Dropout(0.2))
    model.add(Dense(2))  # Predicts both Open and Close

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    st.write(f"⏳ Training {model_type} model...")
    model.fit(X, y, epochs=20, batch_size=32, verbose=1, callbacks=[early_stop])

    # Predict next 20 days
    st.subheader("🔮 Predicting Next 20 Business Days (Open & Close)")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        X_test.append(test_data[i - PAST_DAYS:i])
    X_test = np.array(X_test)

    predicted = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted)

    # Create date range for predictions
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=20, freq='B')

    # Display predictions
    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Open": predicted_prices[:, 0],
        "Predicted Close": predicted_prices[:, 1]
    }).set_index("Date")

    st.subheader("📅 Predicted Open & Close Prices")
    st.dataframe(pred_df)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pred_df.index, pred_df["Predicted Open"], label="Predicted Open", marker='o')
    ax.plot(pred_df.index, pred_df["Predicted Close"], label="Predicted Close", marker='x')
    ax.set_title(f"Next 20 Day Prediction using {model_type}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Download
    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("📥 Download CSV", csv, "predicted_open_close.csv", "text/csv")
