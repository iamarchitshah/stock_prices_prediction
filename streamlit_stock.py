import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

# Load and prepare data
df = pd.read_csv("TCS - Sheet1.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
data = df[['Close']].values

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_len = 60
X, y = create_sequences(scaled_data, seq_len)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Train RNN model
rnn_model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(X_train.shape[1], 1)),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Predict using RNN
rnn_pred = rnn_model.predict(X_test)
rnn_pred_rescaled = scaler.inverse_transform(rnn_pred)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
rnn_mse = mean_squared_error(y_test_rescaled, rnn_pred_rescaled)

# Define helper to clone architecture and use RNN-trained weights
def clone_and_predict(X_test, rnn_weights, cell_type):
    model = Sequential()
    if cell_type == 'LSTM':
        model.add(LSTM(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
    elif cell_type == 'GRU':
        model.add(GRU(50, activation='tanh', input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train from scratch (because weights are not transferable)
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
    pred = model.predict(X_test)
    pred_rescaled = scaler.inverse_transform(pred)
    mse = mean_squared_error(y_test_rescaled, pred_rescaled)
    return pred_rescaled, mse

# Train and predict with GRU and LSTM (for comparison only)
lstm_pred, lstm_mse = clone_and_predict(X_test, None, 'LSTM')
gru_pred, gru_mse = clone_and_predict(X_test, None, 'GRU')

# Print Results
print(f"RNN MSE:  {rnn_mse:.2f}")
print(f"LSTM MSE: {lstm_mse:.2f}")
print(f"GRU MSE:  {gru_mse:.2f}")

# Plot Comparison
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual", linestyle='--', color='black')
plt.plot(rnn_pred_rescaled, label="RNN")
plt.plot(lstm_pred, label="LSTM")
plt.plot(gru_pred, label="GRU")
plt.title("Stock Price Prediction: RNN vs LSTM vs GRU")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
