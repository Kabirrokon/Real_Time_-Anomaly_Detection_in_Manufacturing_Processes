# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate Synthetic Manufacturing Data
np.random.seed(42)
data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
anomalies = np.random.randint(950, 1000, 5)
data[anomalies] += np.random.normal(5, 1, 5)

# Preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1))

# Prepare Sequences
X = []
seq_length = 50
for i in range(len(data) - seq_length):
    X.append(data[i:i + seq_length])
X = np.array(X)

# LSTM Autoencoder Model
model = Sequential()
model.add(LSTM(32, activation='relu', input_shape=(seq_length, 1), return_sequences=False))
model.add(Dense(seq_length))
model.compile(optimizer='adam', loss='mse')

# Train the Model
model.fit(X, X, epochs=10, batch_size=16, verbose=1)

# Predict and Calculate Reconstruction Error
predictions = model.predict(X)
mse = np.mean(np.power(X[:, :, 0] - predictions, 2), axis=1)


# Set a Threshold for Anomaly Detection
threshold = np.percentile(mse, 95)
anomalies = mse > threshold

# Plot Anomalies
plt.figure(figsize=(12, 6))
plt.plot(data, label='Data')
# Create a full-length anomaly array
full_anomalies = np.zeros(len(data), dtype=bool)
full_anomalies[seq_length:len(anomalies) + seq_length] = anomalies

# Plot with fixed anomaly indices
plt.scatter(np.where(full_anomalies)[0], scaler.inverse_transform(data)[full_anomalies], color='red', label='Anomalies')

plt.title('Real-Time Anomaly Detection in Manufacturing Process')
plt.xlabel('Time')
plt.ylabel('Sensor Value')
plt.legend()
plt.show()