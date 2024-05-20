import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from meteostat import Daily, Point

import matplotlib.pyplot as plt


# Fetch weather data for a specific location and time range
location = Point(40.7128, -74.0060)
start = pd.Timestamp('2019-01-01')
end = pd.Timestamp('2019-12-31')
data = Daily(location, start, end)
data = data.fetch()

# Data Preprocessing
# Fill missing values with the previous value
data.fillna(method='ffill', inplace=True)
# Scale the temperature values to a range of 0 to 1
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
# Define the LSTM model architecture
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.3),
    LSTM(100),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM Model
# Define callbacks for model checkpointing and early stopping
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the LSTM model
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val), callbacks=[checkpoint, early_stopping])

# Load the best model
# Load the saved best model
model = load_model('model.h5')

# Prediction and Evaluation for LSTM
# Make predictions using the LSTM model
lstm_predictions = model.predict(X_val)
# Inverse transform the scaled predictions
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Calculate RMSE for LSTM
# Calculate the root mean squared error (RMSE) for LSTM predictions
lstm_rmse = np.sqrt(mean_squared_error(Y_val, lstm_predictions))

# Setup Markov Chain
# Discretize the temperature values into bins
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
data_discrete = discretizer.fit_transform(data[['tavg']])

# Create transition matrix
# Create a transition matrix based on the discretized temperature values
transition_matrix = np.zeros((5, 5))
for i in range(1, len(data_discrete) - 1):
    current_state = int(data_discrete[i])
    next_state = int(data_discrete[i + 1])
    transition_matrix[current_state, next_state] += 1

# Normalize the transition matrix
# Normalize the transition matrix to obtain probabilities
transition_matrix = np.divide(transition_matrix, transition_matrix.sum(axis=1)[:, np.newaxis], out=np.zeros_like(transition_matrix), where=transition_matrix.sum(axis=1)[:, np.newaxis]!=0)

# Prediction using Markov Chain
# Use the Markov Chain to predict the next temperature value
predicted_mc = [transition_matrix[int(state)].argmax() for state in data_discrete[-len(X_val):]]

# Inverse transform predictions
# Inverse transform the discretized predictions
predicted_mc = discretizer.inverse_transform(np.array(predicted_mc).reshape(-1, 1))

# Calculate RMSE for Markov Chain
# Calculate the root mean squared error (RMSE) for Markov Chain predictions
mc_rmse = np.sqrt(mean_squared_error(Y_val, predicted_mc))

# Display results
# Print the RMSE values for LSTM and Markov Chain predictions
print(f"LSTM RMSE: {lstm_rmse:.2f}")
print(f"Markov Chain RMSE: {mc_rmse:.2f}")

# Plotting predictions
# Plot the actual, LSTM, and Markov Chain predictions
plt.figure(figsize=(10, 5))
plt.plot(Y_val, label='Actual', color='black')
plt.plot(lstm_predictions, label='LSTM Predictions', color='blue')
plt.plot(predicted_mc, label='Markov Predictions', color='red')
plt.title('Comparison of LSTM and Markov Chain Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()
