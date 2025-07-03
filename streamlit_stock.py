import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

st.title("📈 Enhanced Stock Price Prediction (Company-Level)")

uploaded_file = st.file_uploader("📁 Upload stock CSV (must include Date, Open, High, Low, Close, Volume)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate required columns
    required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        st.error(f"CSV must contain columns: {', '.join(required_columns)}")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Use multiple features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i, 3])  # Predicting Close
        return np.array(X), np.array(y)

    seq_len = 90
    X, y = create_sequences(scaled_data, seq_len)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    y_test_actual = scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :3], y_test.reshape(-1, 1), X_test[:, -1, 4:]], axis=1)
    )[:, 3]

    def build_deep_model(model_type='LSTM'):
        model = Sequential()
        if model_type == 'LSTM':
            model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.3))
            model.add(LSTM(64))
        elif model_type == 'GRU':
            model.add(GRU(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.3))
            model.add(GRU(64))
        elif model_type == 'RNN':
            model.add(SimpleRNN(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.3))
            model.add(SimpleRNN(64))

        model.add(Dropout(0.3))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    models = {}
    predictions = {}
    mses = {}

    for model_type in ['RNN', 'LSTM', 'GRU']:
        st.write(f"🔁 Training **{model_type}** model...")
        model = build_deep_model(model_type)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)
        y_pred = model.predict(X_test)

        # Reconstruct full feature for inverse transform
        y_pred_full = np.concatenate([X_test[:, -1, :3], y_pred, X_test[:, -1, 4:]], axis=1)
        y_pred_actual = scaler.inverse_transform(y_pred_full)[:, 3]

        mse = mean_squared_error(y_test_actual, y_pred_actual)

        models[model_type] = model
        predictions[model_type] = y_pred_actual
        mses[model_type] = mse
        st.success(f"{model_type} MSE: {mse:.2f}")

    # Plot
    st.subheader("📊 Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test_actual, label="Actual", color='black', linestyle='--')
    for model_type in predictions:
        ax.plot(predictions[model_type], label=f"{model_type}")
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Show table and download
    for model_type in predictions:
        st.markdown(f"### 📈 {model_type} Predictions (last 30)")
        pred_df = pd.DataFrame({
            'Actual Price': y_test_actual,
            'Predicted Price': predictions[model_type]
        })
        pred_df['Error'] = pred_df['Actual Price'] - pred_df['Predicted Price']
        st.dataframe(pred_df.tail(30))

        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {model_type} Predictions",
            data=csv,
            file_name=f"{model_type.lower()}_predictions.csv",
            mime='text/csv'
        )
else:
    st.info("📁 Please upload a valid CSV to begin training and prediction.")
