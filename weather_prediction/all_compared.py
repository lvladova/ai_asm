import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from meteostat import Daily, Point


# Data Fetching
location = Point(40.7128, -74.0060)  # New York City coordinates
start = pd.Timestamp('2019-01-01')
end = pd.Timestamp('2020-12-31')
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
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# LSTM Model Architecture
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(look_back, 1)),
    Dropout(0.3),
    LSTM(100),
    Dropout(0.3),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Callbacks
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)  # Adjust learning rate based on validation loss

# Training the LSTM Model
history_lstm = model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val), callbacks=[checkpoint, early_stopping, reduce_lr])

# Load the best model
model = load_model('model.h5')


# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform([y_test]).reshape(-1, 1)
    
    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
    mae = mean_absolute_error(y_test_inv, predictions)
    return rmse, mae, predictions, y_test_inv


rmse_lstm, mae_lstm, trainPredictions_lstm, Y_train_inv_lstm = evaluate_model(model, X_train, Y_train, scaler)
print(f'LSTM Train RMSE: {rmse_lstm:.2f}, MAE: {mae_lstm:.2f}')

# LSTM with Rewards
history = model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2, validation_data=(X_val, Y_val), callbacks=[checkpoint, early_stopping])

# Evaluate the model with rewards
rmse_lstm_reward, mae_lstm_reward, trainPredictions_lstm_reward, Y_train_inv_lstm_reward = evaluate_model(model, X_train, Y_train, scaler)
print(f'LSTM with Rewards Train RMSE: {rmse_lstm_reward:.2f}, MAE: {mae_lstm_reward:.2f}')

# Markov Chain Model
bins = 5
discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
data['tavg'] = discretizer.fit_transform(data[['tavg']])
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

num_states = bins
transition_matrix = np.zeros((num_states, num_states))

for i in range(len(train_data) - 1):
    current_state = int(train_data.iloc[i]['tavg'])
    next_state = int(train_data.iloc[i + 1]['tavg'])
    transition_matrix[current_state][next_state] += 1

transition_matrix /= np.sum(transition_matrix, axis=1)[:, None]


def markov_predict(state, transition_matrix):
    return np.argmax(transition_matrix[state])


actual_markov = []
predicted_markov = []

for i in range(len(test_data) - 1):
    current_state = int(test_data.iloc[i]['tavg'])
    actual_markov.append(int(test_data.iloc[i + 1]['tavg']))
    predicted_markov.append(markov_predict(current_state, transition_matrix))

actual_markov = np.array(actual_markov).reshape(-1, 1)
predicted_markov = np.array(predicted_markov).reshape(-1, 1)
actual_markov = discretizer.inverse_transform(actual_markov)
predicted_markov = discretizer.inverse_transform(predicted_markov)

rmse_markov = np.sqrt(mean_squared_error(actual_markov, predicted_markov))
mae_markov = mean_absolute_error(actual_markov, predicted_markov)
print(f'Markov Chain RMSE: {rmse_markov:.2f}, MAE: {mae_markov:.2f}')

# Markov Chain with Rewards
transition_rewards = np.zeros((num_states, num_states))

for i in range(len(train_data) - 1):
    current_state = int(train_data.iloc[i]['tavg'])
    next_state = int(train_data.iloc[i + 1]['tavg'])
    reward = -abs(train_data.iloc[i + 1]['tavg'] - train_data.iloc[i]['tavg'])
    transition_rewards[current_state][next_state] += reward

transition_probs = np.sum(transition_rewards, axis=1, keepdims=True)
transition_probs[transition_probs == 0] = 1
normalized_rewards = transition_rewards / transition_probs


def predict_next_state_with_rewards(current_state, reward_matrix):
    return np.argmax(reward_matrix[current_state])


actual_reward = []
predicted_reward = []

for i in range(len(test_data) - 1):
    current_state = int(test_data.iloc[i]['tavg'])
    actual_reward.append(int(test_data.iloc[i + 1]['tavg']))
    predicted_reward.append(predict_next_state_with_rewards(current_state, normalized_rewards))

actual_reward = np.array(actual_reward).reshape(-1, 1)
predicted_reward = np.array(predicted_reward).reshape(-1, 1)
actual_reward = discretizer.inverse_transform(actual_reward)
predicted_reward = discretizer.inverse_transform(predicted_reward)

rmse_markov_reward = np.sqrt(mean_squared_error(actual_reward, predicted_reward))
mae_markov_reward = mean_absolute_error(actual_reward, predicted_reward)
print(f'Markov Chain with Rewards RMSE: {rmse_markov_reward:.2f}, MAE: {mae_markov_reward:.2f}')

# Plotting comparison
plt.figure(figsize=(14, 7))

# LSTM comparison
plt.subplot(2, 2, 1)
plt.plot(Y_train_inv_lstm, label='Actual')
plt.plot(trainPredictions_lstm, label='LSTM Predicted')
plt.title('LSTM')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(Y_train_inv_lstm_reward, label='Actual')
plt.plot(trainPredictions_lstm_reward, label='LSTM with Rewards Predicted')
plt.title('LSTM with Rewards')
plt.legend()

# Markov comparison
plt.subplot(2, 2, 3)
plt.plot(actual_markov, label='Actual')
plt.plot(predicted_markov, label='Markov Predicted')
plt.title('Markov Chain')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(actual_reward, label='Actual')
plt.plot(predicted_reward, label='Markov with Rewards Predicted')
plt.title('Markov Chain with Rewards')
plt.legend()

plt.tight_layout()
plt.show()

# Table for RMSE and MAE
results = {
    'Model': ['LSTM', 'LSTM with Rewards', 'Markov Chain', 'Markov Chain with Rewards'],
    'RMSE': [rmse_lstm, rmse_lstm_reward, rmse_markov, rmse_markov_reward],
    'MAE': [mae_lstm, mae_lstm_reward, mae_markov, mae_markov_reward]
}

results_df = pd.DataFrame(results)
print(results_df)
