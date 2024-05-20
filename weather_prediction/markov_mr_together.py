import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from meteostat import Daily, Point
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Initialize model parameters
location = Point(40.7128, -74.0060)  # New York City coordinates
start = pd.Timestamp('2019-01-01')
end = pd.Timestamp('2020-12-31')
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
def markov_predict(state, transition_matrix):
    return np.argmax(transition_matrix[state])


# Prediction on test data
actual = []
predicted = []

for i in range(2, len(test_data)):
    prev_state = int(sum([test_data.iloc[i-2][f] * (bins ** idx) for idx, f in enumerate(features)]))
    curr_state = int(sum([test_data.iloc[i-1][f] * (bins ** idx) for idx, f in enumerate(features)]))
    actual_state = int(sum([test_data.iloc[i][f] * (bins ** idx) for idx, f in enumerate(features)]))
    predicted_state = markov_predict((prev_state, curr_state), transition_matrix)
    
    actual.append(actual_state)
    predicted.append(predicted_state)

# Convert actual and predicted into 2D arrays with 3 features
actual = np.array([[a // bins**2, (a // bins) % bins, a % bins] for a in actual])
predicted = np.array([[p // bins**2, (p // bins) % bins, p % bins] for p in predicted])

# Convert actual and predicted to original scale
actual = scaler.inverse_transform(actual)
predicted = scaler.inverse_transform(predicted)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(actual, predicted))
mae = mean_absolute_error(actual, predicted)
print(f"Markov Chain - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Plotting Markov Chain results
plt.figure(figsize=(15, 5))
plt.plot(actual, label='Actual')
plt.plot(predicted, label='Predicted', alpha=0.7)
plt.title('Markov Chain: Actual vs Predicted Temperatures')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Create reward-augmented transition matrix for a second-order Markov Chain with rewards
transition_rewards = np.zeros((num_states, num_states, num_states))

# Populate transition matrix with rewards
for i in range(2, len(train_data)):
    prev_state = int(sum([train_data.iloc[i-2][f] * (bins ** idx) for idx, f in enumerate(features)]))
    curr_state = int(sum([train_data.iloc[i-1][f] * (bins ** idx) for idx, f in enumerate(features)]))
    next_state = int(sum([train_data.iloc[i][f] * (bins ** idx) for idx, f in enumerate(features)]))
    reward = -abs(train_data.iloc[i]['tavg'] - train_data.iloc[i-1]['tavg'])  # Negative error as reward
    transition_rewards[prev_state, curr_state, next_state] += reward

# Normalize rewards
transition_probs = np.sum(transition_rewards, axis=2, keepdims=True)
transition_probs[transition_probs == 0] = 1  # Avoid division by zero
normalized_rewards = transition_rewards / transition_probs


# Prediction function with reward maximization
def predict_next_state_with_rewards(prev_state, curr_state, reward_matrix):
    return np.argmax(reward_matrix[prev_state, curr_state])


# Prediction on test data using rewards
predicted_rewards = []

for i in range(2, len(test_data)):
    prev_state = int(sum([test_data.iloc[i-2][f] * (bins ** idx) for idx, f in enumerate(features)]))
    curr_state = int(sum([test_data.iloc[i-1][f] * (bins ** idx) for idx, f in enumerate(features)]))
    predicted_state = predict_next_state_with_rewards(prev_state, curr_state, normalized_rewards)
    predicted_rewards.append(predicted_state)

# Convert predicted to original scale
predicted_rewards = np.array(predicted_rewards)
# Convert predicted_rewards into a 2D array with 3 features
predicted_rewards = np.array([[p // bins**2, (p // bins) % bins, p % bins] for p in predicted_rewards])

# Convert predicted_rewards to original scale
predicted_rewards = scaler.inverse_transform(predicted_rewards)

# Define actual as the actual state indices in the test data
actual = [int(sum([test_data.iloc[i][f] * (bins ** idx) for idx, f in enumerate(features)])) for i in range(2, len(test_data))]

# Convert actual into a 2D array with 3 features
actual = np.array([[a // bins**2, (a // bins) % bins, a % bins] for a in actual])

# Convert actual to original scale
actual = scaler.inverse_transform(actual)

# Calculate RMSE with rewards
rmse_rewards = np.sqrt(mean_squared_error(actual, predicted_rewards))
mae_rewards = mean_absolute_error(actual, predicted_rewards)
print(f"Markov Chain with Rewards - RMSE: {rmse_rewards:.2f}, MAE: {mae_rewards:.2f}")

# Plotting with rewards
plt.figure(figsize=(15, 5))
plt.plot(actual, label='Actual')
plt.plot(predicted_rewards, label='Predicted with Rewards', alpha=0.7)
plt.title('Markov Chain with Rewards: Actual vs Predicted Temperatures')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Plotting both results together for comparison
plt.figure(figsize=(15, 5))
plt.plot(actual, label='Actual', color='black')
plt.plot(predicted, label='Markov Chain Predicted', alpha=0.7, color='blue')
plt.plot(predicted_rewards, label='Markov Chain with Rewards Predicted', alpha=0.7, color='green')
plt.title('Comparison of Markov Chain and Markov Chain with Rewards: Actual vs Predicted Temperatures')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()

# Plotting validation loss
plt.figure(figsize=(15, 5))
plt.plot(np.arange(len(actual)), actual, label='Actual', color='black')
plt.plot(np.arange(len(predicted)), predicted, label='Markov Chain Predicted', alpha=0.7, color='blue')
plt.plot(np.arange(len(predicted_rewards)), predicted_rewards, label='Markov Chain with Rewards Predicted', alpha=0.7, color='green')
plt.title('Validation Loss: Actual vs Predicted Temperatures')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()


