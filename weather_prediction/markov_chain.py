import numpy as np
import pandas as pd
from meteostat import Daily, Point
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt


# Initialize model parameters
location = Point(40.7128, -74.0060)  # New York City coordinates
start = pd.Timestamp('2019-01-01')
end = pd.Timestamp('2019-12-31')
bins = 5  # Number of bins for discretization

# Fetch data from Meteostat
data = Daily(location, start, end)
data = data.fetch()

# Data Preprocessing
data.fillna(method='ffill', inplace=True)
features = ['tavg', 'tmin', 'tmax']
scaler = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
data[features] = scaler.fit_transform(data[features])

# Split data into train and test
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Create transition matrix for a second-order Markov Chain
num_states = bins ** len(features)
transition_matrix = np.zeros((num_states, num_states, num_states))

# Populate transition matrix from train data
for i in range(2, len(train_data)):
    prev_state = int(sum([train_data.iloc[i-2][f] * (bins ** idx) for idx, f in enumerate(features)]))
    curr_state = int(sum([train_data.iloc[i-1][f] * (bins ** idx) for idx, f in enumerate(features)]))
    next_state = int(sum([train_data.iloc[i][f] * (bins ** idx) for idx, f in enumerate(features)]))
    transition_matrix[prev_state, curr_state, next_state] += 1

# Normalize the transition matrix
transition_matrix /= transition_matrix.sum(axis=2, keepdims=True) + 1e-6  # Avoid division by zero


# Prediction function
def predict_next_state(prev_state, curr_state, matrix):
    """
    Predicts the next state based on the previous state and current state using the transition matrix.
    Args:
        prev_state (int): Previous state index.
        curr_state (int): Current state index.
        matrix (ndarray): Transition matrix.
    Returns:
        int: Index of the predicted next state.
    """
    return np.argmax(matrix[prev_state, curr_state])


# Prediction on test data
actual = []
predicted = []

for i in range(2, len(test_data)):
    prev_state = int(sum([test_data.iloc[i-2][f] * (bins ** idx) for idx, f in enumerate(features)]))
    curr_state = int(sum([test_data.iloc[i-1][f] * (bins ** idx) for idx, f in enumerate(features)]))
    actual_state = int(sum([test_data.iloc[i][f] * (bins ** idx) for idx, f in enumerate(features)]))
    predicted_state = predict_next_state(prev_state, curr_state, transition_matrix)
    
    actual.append(actual_state)
    predicted.append(predicted_state)

# Convert actual and predicted into 2D arrays with 3 features
actual = np.array([[a // bins**2, (a // bins) % bins, a % bins] for a in actual])
predicted = np.array([[p // bins**2, (p // bins) % bins, p % bins] for p in predicted])

# Convert actual and predicted to original scale
actual = scaler.inverse_transform(actual)
predicted = scaler.inverse_transform(predicted)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f"Test RMSE: {rmse:.2f}")

mae = mean_absolute_error(actual, predicted)
print(f"Test MAE: {mae:.2f}")

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted', alpha=0.7)
plt.title('Actual vs Predicted Temperatures')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()


print(f"Predicted next state (temperature values): {predicted[0]}")

