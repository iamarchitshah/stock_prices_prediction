# advanced_stock_prediction_rf.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

st.title("🌲 Random Forest Stock Predictor (Open & Close)")
model_type = st.selectbox("Choose Model Type", ["Random Forest", "GRU", "RNN"])

uploaded_file = st.file_uploader("Upload CSV with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'", type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    PAST_DAYS = 90
    X, y = [], []
    for i in range(PAST_DAYS, len(scaled_data)):
        X.append(scaled_data[i - PAST_DAYS:i].flatten())
        y.append(scaled_data[i, [0, 3]])  # Open & Close

    X, y = np.array(X), np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        st.write("⏳ Training Random Forest...")
        model.fit(X_train, y_train)

    else:
        from keras.models import Model
        from keras.layers import Input, GRU, SimpleRNN, Dense, Dropout, Bidirectional, BatchNormalization, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from keras.optimizers import Adam
        import keras.backend as K

        def attention_block(inputs):
            attention = Dense(1, activation='tanh')(inputs)
            attention = Flatten()(attention)
            attention = Activation('softmax')(attention)
            attention = RepeatVector(inputs.shape[-1])(attention)
            attention = Permute([2, 1])(attention)
            sent_representation = Multiply()([inputs, attention])
            return Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

        X_seq = X.reshape((X.shape[0], PAST_DAYS, 5))
        X_train_seq, X_val_seq = X_seq[:len(X_train)], X_seq[len(X_train):]

        inputs = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
        if model_type == "GRU":
            x = Bidirectional(GRU(64, return_sequences=True))(inputs)
            x = Dropout(0.2)(x)
            x = Bidirectional(GRU(32, return_sequences=True))(x)
        else:  # RNN
            x = Bidirectional(SimpleRNN(64, return_sequences=True))(inputs)
            x = Dropout(0.2)(x)
            x = Bidirectional(SimpleRNN(32, return_sequences=True))(x)

        x = BatchNormalization()(x)
        x = attention_block(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(2)(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

        st.write(f"⏳ Training the {model_type} model with Attention...")
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1),
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
        ]

        history = model.fit(X_train_seq, y_train, validation_data=(X_val_seq, y_val),
                            epochs=30, batch_size=64, callbacks=callbacks, verbose=1)

        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Train Loss')
        ax_loss.plot(history.history['val_loss'], label='Val Loss')
        ax_loss.set_title("Loss Curve")
        ax_loss.legend()
        st.pyplot(fig_loss)

    # Predict on validation
    st.subheader("📈 Validation Results")
    if model_type == "Random Forest":
        val_pred = model.predict(X_val)
    else:
        val_pred = model.predict(X_val_seq)

    padded_val = np.zeros((val_pred.shape[0], 5))
    padded_val[:, 0] = val_pred[:, 0]
    padded_val[:, 3] = val_pred[:, 1]
    pred_val_prices = scaler.inverse_transform(padded_val)

    actual_val = scaler.inverse_transform(np.concatenate(
        [y_val[:, [0]], np.zeros((len(y_val), 2)), y_val[:, [1]], np.zeros((len(y_val), 1))], axis=1))

    fig_val, ax_val = plt.subplots(figsize=(12, 6))
    ax_val.plot(actual_val[:, 0], label='Actual Open', linestyle='--')
    ax_val.plot(pred_val_prices[:, 0], label='Predicted Open')
    ax_val.plot(actual_val[:, 3], label='Actual Close', linestyle='--')
    ax_val.plot(pred_val_prices[:, 3], label='Predicted Close')
    ax_val.set_title("Actual vs Predicted (Validation)")
    ax_val.legend()
    st.pyplot(fig_val)

    # Predict next 20 days
    st.subheader("🔮 Predicting Next 20 Days")
    test_data = scaled_data[-(PAST_DAYS + 20):]
    X_test = []
    for i in range(PAST_DAYS, PAST_DAYS + 20):
        sample = test_data[i - PAST_DAYS:i]
        X_test.append(sample.flatten() if model_type == "Random Forest" else sample)
    X_test = np.array(X_test)
    
    if model_type != "Random Forest":
        X_test = X_test.reshape((X_test.shape[0], PAST_DAYS, 5))

    test_pred = model.predict(X_test)
    padded = np.zeros((test_pred.shape[0], 5))
    padded[:, 0] = test_pred[:, 0]
    padded[:, 3] = test_pred[:, 1]
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
    ax.set_title("Predicted Prices (Next 20 Days)")
    ax.legend()
    st.pyplot(fig)

    csv = pred_df.to_csv().encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "predicted_rf.csv", "text/csv")
