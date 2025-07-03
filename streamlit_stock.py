import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from keras.callbacks import EarlyStopping

st.title("🚀 Improved Stock Price Predictor - Open & Close (Bidirectional LSTM / GRU / RNN)")

uploaded_file = st.file_uploader("Upload CSV file with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'", type="csv")
model_type = st.selectbox("Select Model", ["LSTM", "GRU", "RNN"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain: {', '.join(required_cols)}")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 180
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i])
        y.append(scaled_data[i, [0, 3]])  # Open and Close
    X, y = np.array(X), np.array(y)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    # Build model
    model = Sequential()
    if model_type == 'LSTM':
        model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(100)))
    elif model_type == 'GRU':
        model.add(Bidirectional(GRU(100, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3))
        model.add(Bidirectional(GRU(100)))
    elif model_type == 'RNN':
        model.add(Bidirectional(SimpleRNN(100, return_sequences=True), input_shape=(X.shape[1], X.shape[2])))
        model.add(Dropout(0.3))
        model.add(Bidirectional(SimpleRNN(100)))

    model.add(Dropout(0.3))
    model.add(Dense(2))  # Output: [Open, Close]

    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    st.write("⏳ Training model with validation & early stopping...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=150, batch_size=32, verbose=1, callbacks=[early_stop])

    # Predict next 20 business days
    st.subheader("🔮 Predicting Next 20 Business Days (Open & Close)")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        X_test.append(test_data[i - PAST_DAYS:i])
    X_test = np.array(X_test)

    predicted_scaled = model.predict(X_test)

    # Prepare dummy rows to inverse transform
    predicted_full = np.zeros((predicted_scaled.shape[0], 5))  # 5 features
    predicted_full[:, 0] = predicted_scaled[:, 0]  # Open
    predicted_full[:, 3] = predicted_scaled[:, 1]  # Close

    predicted_prices = scaler.inverse_transform(predicted_full)
    predicted_open = predicted_prices[:, 0]
    predicted_close = predicted_prices[:, 3]

    # Create date range
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=20, freq='B')

    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Open": predicted_open,
        "Predicted Close": predicted_close
    }).set_index("Date")

    st.subheader("📅 Predicted Prices")
    st.dataframe(pred_df)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pred_df.index, pred_df["Predicted Open"], label="Predicted Open", marker='o')
    ax.plot(pred_df.index, pred_df["Predicted Close"], label="Predicted Close", marker='x')
    ax.set_title(f"Next 20 Day Stock Prediction ({model_type})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Download
    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("📥 Download CSV", csv, "predicted_open_close.csv", "text/csv")
