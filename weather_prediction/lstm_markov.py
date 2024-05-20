import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from meteostat import Daily, Point

import matplotlib.pyplot as plt


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
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


look_back = 3
X, Y = create_dataset(data['tavg'].values, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and validation
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# LSTM Model Architecture
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(look_back, 1)),  # First layer with return_sequences
    Dropout(0.3),  # Increased dropout
    LSTM(100),  # Second LSTM layer
    Dropout(0.3),  # Increased dropout
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the LSTM Model
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val))

# Markov Chain model setup
bins = 5  # Increase the number of bins for finer granularity
discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
discrete_data = discretizer.fit_transform(data[['tavg']]).flatten()

# Transition matrix calculation
transition_matrix = np.zeros((bins, bins))
for i in range(len(discrete_data) - 1):
    current_state = int(discrete_data[i])
    next_state = int(discrete_data[i + 1])
    transition_matrix[current_state][next_state] += 1

# Normalize the transition matrix
transition_matrix /= np.sum(transition_matrix, axis=1)[:, np.newaxis]


# Prediction using Markov Chain
def markov_predict(state, transition_matrix):
    return np.argmax(transition_matrix[state])

# Evaluate and compare both models
predicted_markov = [markov_predict(int(state), transition_matrix) for state in discrete_data[-look_back-1:-1]]
predicted_lstm = model.predict(X[-look_back-1:-1]).flatten()
actual_values = discrete_data[-look_back:]

# Inverse transform for visualization
predicted_markov = discretizer.inverse_transform(np.array(predicted_markov).reshape(-1, 1)).flatten()
predicted_lstm = scaler.inverse_transform(predicted_lstm.reshape(-1, 1)).flatten()
actual_values = discretizer.inverse_transform(actual_values.reshape(-1, 1)).flatten()

# Calculate RMSE for both models
rmse_markov = np.sqrt(mean_squared_error(actual_values, predicted_markov))
rmse_lstm = np.sqrt(mean_squared_error(actual_values, predicted_lstm))

# Display results
print(f"Markov Chain RMSE: {rmse_markov:.2f}")
print(f"LSTM RMSE: {rmse_lstm:.2f}")

# Plotting predictions
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='Actual', color='black')
plt.plot(predicted_markov, label='Markov Predictions', linestyle='--', marker='o', color='red')
plt.plot(predicted_lstm, label='LSTM Predictions', linestyle='--', marker='x', color='blue')
plt.title('Comparison of Markov Chain and LSTM Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()
