import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from meteostat import Daily, Point

import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # Added ReduceLROnPlateau


# Data Fetching
location = Point(40.7128, -74.0060)
start = pd.Timestamp('2019-01-01')
end = pd.Timestamp('2019-12-31')
data = Daily(location, start, end)
data = data.fetch()

# Data Preprocessing for LSTM
data.fillna(method='ffill', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data['tavg'] = scaler.fit_transform(data[['tavg']])


# Create sequences for LSTM
def create_dataset(dataset, look_back=1):
    """
    Create input sequences and corresponding output values for LSTM model.
    """
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
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.3),
    LSTM(100),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Callbacks
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)  # Adjust learning rate based on validation loss

# Training the LSTM Model
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val), callbacks=[checkpoint, early_stopping, reduce_lr])

# Load the best model
model = load_model('model.h5')


# Evaluate the model
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

trainScore, trainMAE, trainPredictions, Y_train_inv = evaluate_model(model, X_train, Y_train, scaler)
print(f'Train Score: {trainScore:.2f} RMSE, {trainMAE:.2f} MAE')

history = model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val), callbacks=[checkpoint, early_stopping])


# Plotting training progress
def plot_training_history(history):
    """
    Plot the training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


plot_training_history(history)


# Predict on new data
def predict_next_value(model, raw_data, scaler, look_back):
    """
    Predict the next value in a time series using the trained LSTM model.
    """
    raw_data = np.array(raw_data).reshape(-1, 1)
    scaled_data = scaler.transform(raw_data)
    scaled_data = scaled_data.reshape(1, look_back, 1)
    
    predicted = model.predict(scaled_data)
    predicted = scaler.inverse_transform(predicted)
    return predicted


new_sequence = data['tavg'].values[-look_back:]
predicted_temp = predict_next_value(model, new_sequence, scaler, look_back)
print(f"Predicted Temperature: {predicted_temp[0][0]:.2f}")


def plot_predictions(actual, predicted):
    """
    Plot the actual and predicted values of a time series.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual Values')
    plt.plot(predicted, label='Predicted Values', alpha=0.7)
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


plot_predictions(Y_train_inv, trainPredictions)
