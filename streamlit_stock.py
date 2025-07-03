import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

st.title("📈 Stock Price Prediction")
st.markdown("Train with **RNN**, predict with **LSTM** and **GRU**")

uploaded_file = st.file_uploader("📁 Upload your stock CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Basic check
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("❌ CSV must have 'Date' and 'Close' columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Close']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_len = 60
    X, y = create_sequences(scaled_data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train RNN model
    st.write("🔁 Training RNN model...")
    rnn_model = Sequential([
        SimpleRNN(50, activation='tanh', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    rnn_model.compile(optimizer='adam', loss='mean_squared_error')
    rnn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    rnn_pred = rnn_model.predict(X_test)
    rnn_pred_rescaled = scaler.inverse_transform(rnn_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    rnn_mse = mean_squared_error(y_test_rescaled, rnn_pred_rescaled)

    st.success(f"RNN MSE: {rnn_mse:.2f}")

    # Train and predict with LSTM and GRU
    def predict_with(cell_type):
        model = Sequential()
        if cell_type == 'LSTM':
            model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
        elif cell_type == 'GRU':
            model.add(GRU(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        pred = model.predict(X_test)
        pred_rescaled = scaler.inverse_transform(pred)
        mse = mean_squared_error(y_test_rescaled, pred_rescaled)
        return pred_rescaled, mse

    lstm_pred, lstm_mse = predict_with('LSTM')
    st.success(f"LSTM MSE: {lstm_mse:.2f}")
    gru_pred, gru_mse = predict_with('GRU')
    st.success(f"GRU MSE: {gru_mse:.2f}")

    # Plot all predictions
    st.subheader("📊 Prediction Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_rescaled, label='Actual', linestyle='--', color='black')
    ax.plot(rnn_pred_rescaled, label='RNN')
    ax.plot(lstm_pred, label='LSTM')
    ax.plot(gru_pred, label='GRU')
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Show and export predictions
    for model_name, pred_data in {
        'RNN': rnn_pred_rescaled,
        'LSTM': lstm_pred,
        'GRU': gru_pred
    }.items():
        st.markdown(f"### 📈 {model_name} Predictions (last 30)")
        pred_df = pd.DataFrame({
            'Actual Price': y_test_rescaled.flatten(),
            'Predicted Price': pred_data.flatten()
        })
        pred_df['Error'] = pred_df['Actual Price'] - pred_df['Predicted Price']
        st.dataframe(pred_df.tail(30))

        # Display last 5 differences as sentences
        st.markdown("📌 Last 5 Prediction Differences:")
        for i, row in pred_df.tail(5).iterrows():
            st.write(f"Actual = {row['Actual Price']:.2f} | Predicted = {row['Predicted Price']:.2f} | Error = {row['Error']:.2f}")

        # CSV download
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {model_name} Predictions",
            data=csv,
            file_name=f"{model_name.lower()}_predictions.csv",
            mime='text/csv'
        )

else:
    st.warning("📁 Please upload a CSV file with 'Date' and 'Close' columns to start.")
