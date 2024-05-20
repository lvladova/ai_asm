import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from meteostat import Daily, Point

# Fetching data
location = Point(40.7128, -74.0060)  # New York City coordinates
start = pd.Timestamp('2019-01-01')
end = pd.Timestamp('2020-12-31')
data = Daily(location, start, end)
data = data.fetch()

# Data Preprocessing
data.fillna(method='ffill', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data['tavg'] = scaler.fit_transform(data[['tavg']])


# Create sequences for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


look_back = 3
X, Y = create_dataset(data['tavg'].values, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# LSTM Model Architecture
def build_lstm_model():
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model


# Train the LSTM model
lstm_model = build_lstm_model()
checkpoint = ModelCheckpoint('lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

lstm_model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val),
               callbacks=[checkpoint, early_stopping, reduce_lr])

# Load the best model
lstm_model = load_model('lstm_model.h5')


# Evaluate the LSTM model
def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluate the model using RMSE and MAE metrics.
    """
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform([y_test]).reshape(-1, 1)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
    # Calculate MAE
    mae = mean_absolute_error(y_test_inv, predictions)
    return rmse, mae, predictions, y_test_inv


# Evaluation of LSTM
lstm_rmse, lstm_mae, lstm_predictions, Y_val_inv = evaluate_model(lstm_model, X_val, Y_val, scaler)
print(f'LSTM RMSE: {lstm_rmse:.2f}, LSTM MAE: {lstm_mae:.2f}')


# Reward-based LSTM Model Architecture
def build_reward_lstm_model():
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Train the Reward-based LSTM model
reward_lstm_model = build_reward_lstm_model()
checkpoint = ModelCheckpoint('reward_lstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

reward_lstm_model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val),
                      callbacks=[checkpoint, early_stopping, reduce_lr])

# Load the best model
reward_lstm_model = load_model('reward_lstm_model.h5')

# Evaluation of Reward-based LSTM
reward_lstm_rmse, reward_lstm_mae, reward_lstm_predictions, _ = evaluate_model(reward_lstm_model, X_val, Y_val, scaler)
print(f'Reward-based LSTM RMSE: {reward_lstm_rmse:.2f}, Reward-based LSTM MAE: {reward_lstm_mae:.2f}')


# Plotting predictions
def plot_predictions(actual, lstm_pred, reward_lstm_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Values')
    plt.plot(lstm_pred, label='LSTM Predictions', alpha=0.7)
    plt.plot(reward_lstm_pred, label='Reward-based LSTM Predictions', alpha=0.7)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend()
    plt.show()


plot_predictions(Y_val_inv, lstm_predictions, reward_lstm_predictions)
