import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, LSTM, Dense

st.title("📈 Smart Stock Price Prediction - RNN vs GRU vs LSTM")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    data = df[['Close']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    seq_len = 60
    X, y = create_sequences(scaled_data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    def build_and_train(model_type):
        model = Sequential()
        if model_type == 'RNN':
            model.add(SimpleRNN(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
        elif model_type == 'GRU':
            model.add(GRU(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
        elif model_type == 'LSTM':
            model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        return y_test_rescaled, y_pred_rescaled, mse

    results = {}
    for m in ['RNN', 'GRU', 'LSTM']:
        st.write(f"Training **{m}** model...")
        actual, predicted, mse = build_and_train(m)
        results[m] = {'actual': actual, 'predicted': predicted, 'mse': mse}
        st.success(f"{m} MSE: {mse:.2f}")

    # Display predicted prices (LSTM)
st.subheader("🔢 Predicted vs Actual Prices with Error")

# Loop through all 3 models and show results
for model_name in ['RNN', 'GRU', 'LSTM']:
    st.markdown(f"### 🧠 {model_name} Predictions")

    # Create DataFrame
    df_pred = pd.DataFrame({
        'Actual Price': results[model_name]['actual'].flatten(),
        'Predicted Price': results[model_name]['predicted'].flatten()
    })
    df_pred['Difference (Error)'] = df_pred['Actual Price'] - df_pred['Predicted Price']

    # Show table
    st.dataframe(df_pred.tail(30))

    # Print last 5 prediction sentences
    st.markdown("📌 Prediction Differences (last 5 shown):")
    for i, row in df_pred.tail(5).iterrows():
        st.write(f"Actual = {row['Actual Price']:.2f} | Predicted = {row['Predicted Price']:.2f} | Error = {row['Difference (Error)']:.2f}")

    # CSV Download button
    csv = df_pred.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download {model_name} Predictions as CSV",
        data=csv,
        file_name=f"{model_name.lower()}_predictions.csv",
        mime='text/csv'
    )
