import numpy as np
import pandas as pd
from meteostat import Daily, Point
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

# Initialize model parameters
location = Point(40.7128, -74.0060)  # Latitude and longitude of the location
start = pd.Timestamp('2019-01-01')  # Start date for data fetching
end = pd.Timestamp('2019-12-31')  # End date for data fetching
bins = 5  # Number of bins for discretization

# Fetch data from Meteostat
data = Daily(location, start, end)
data = data.fetch()

# Data Preprocessing
data.fillna(method='ffill', inplace=True)  # Fill missing values with forward fill
features = ['tavg', 'tmin', 'tmax']  # Features to be used for modeling
scaler = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')  # Discretize features into bins
data[features] = scaler.fit_transform(data[features])  # Apply discretization to the features

# Split data into train and test
train_data = data[:int(0.8 * len(data))]  # 80% of data for training
test_data = data[int(0.8 * len(data)):]  # 20% of data for testing

# Create reward-augmented transition matrix for a second-order Markov Chain
num_states = bins ** len(features)  # Total number of states in the Markov Chain
transition_rewards = np.zeros((num_states, num_states, num_states))  # Initialize transition matrix with rewards

# Populate transition matrix with rewards
for i in range(2, len(train_data)):
    prev_state = int(sum([train_data.iloc[i-2][f] * (bins ** idx) for idx, f in enumerate(features)]))  # Calculate previous state index
    curr_state = int(sum([train_data.iloc[i-1][f] * (bins ** idx) for idx, f in enumerate(features)]))  # Calculate current state index
    next_state = int(sum([train_data.iloc[i][f] * (bins ** idx) for idx, f in enumerate(features)]))  # Calculate next state index
    reward = -abs(train_data.iloc[i]['tavg'] - train_data.iloc[i-1]['tavg'])  # Negative error as reward
    transition_rewards[prev_state, curr_state, next_state] += reward  # Update transition matrix with reward

# Normalize rewards
transition_probs = np.sum(transition_rewards, axis=2, keepdims=True)  # Calculate sum of rewards for each (prev_state, curr_state) pair
transition_probs[transition_probs == 0] = 1  # Avoid division by zero
normalized_rewards = transition_rewards / transition_probs  # Normalize rewards


# Prediction function with reward maximization
def predict_next_state_with_rewards(prev_state, curr_state, reward_matrix):
    return np.argmax(reward_matrix[prev_state, curr_state])  # Predict next state with maximum reward


# Prediction on test data using rewards
predicted_rewards = []
for i in range(2, len(test_data)):
    prev_state = int(sum([test_data.iloc[i-2][f] * (bins ** idx) for idx, f in enumerate(features)]))  # Calculate previous state index
    curr_state = int(sum([test_data.iloc[i-1][f] * (bins ** idx) for idx, f in enumerate(features)]))  # Calculate current state index
    predicted_state = predict_next_state_with_rewards(prev_state, curr_state, normalized_rewards)  # Predict next state using rewards
    predicted_rewards.append(predicted_state)  # Append predicted state to the list

# Convert predicted to original scale
predicted_rewards = np.array(predicted_rewards)  # Convert predicted_rewards to numpy array
predicted_rewards = np.array([[p // bins**2, (p // bins) % bins, p % bins] for p in predicted_rewards])  # Convert predicted_rewards to a 2D array with 3 features
predicted_rewards = scaler.inverse_transform(predicted_rewards)  # Convert predicted_rewards to original scale

# Define actual as the actual state indices in the test data
actual = [int(sum([test_data.iloc[i][f] * (bins ** idx) for idx, f in enumerate(features)])) for i in range(2, len(test_data))]  # Calculate actual state indices

# Convert actual into a 2D array with 3 features
actual = np.array([[a // bins**2, (a // bins) % bins, a % bins] for a in actual])  # Convert actual to a 2D array with 3 features

# Convert actual to original scale
actual = scaler.inverse_transform(actual)  # Convert actual to original scale

# Calculate RMSE with rewards
rmse_rewards = np.sqrt(mean_squared_error(actual, predicted_rewards))  # Calculate root mean squared error
print(f"Test RMSE with Rewards: {rmse_rewards:.2f}")  # Print RMSE with rewards

# Calculate MAE with rewards
mae_rewards = mean_absolute_error(actual, predicted_rewards)  # Calculate mean absolute error
print(f"Test MAE with Rewards: {mae_rewards:.2f}")  # Print MAE with rewards

# Plotting with rewards
plt.figure(figsize=(15, 5))
plt.plot(actual, label='Actual')
plt.plot(predicted_rewards, label='Predicted with Rewards', alpha=0.7)
plt.title('Actual vs Predicted Temperatures with Rewards')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()
