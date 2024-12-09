import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf


filename = 'sydneyUniqueSortedLatitudes.csv'  
data = pd.read_csv(filename, header=None, skiprows=2)  

data = data.head(1000000)

data.columns = ['latitude', 'index']


data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
data['index'] = pd.to_numeric(data['index'], errors='coerce')
data = data.dropna()
data = data.sort_values(by='latitude').reset_index(drop=True)

# Verify dataset
print("Dataset Preview:")
print(data.head())
print(f"Dataset shape: {data.shape}")


x = data['latitude'].values.reshape(-1, 1)
y = data['index'].values

# Normalize the input values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Root Model: Linear Regression
root_model = LinearRegression()
num_segments = 10
y_cdf = (y / max(y)) * (num_segments - 1)  
print("Training the root model...")
root_model.fit(x_scaled, y_cdf)


segment_predictions = root_model.predict(x_scaled)
segment_predictions = np.clip(np.round(segment_predictions), 0, num_segments - 1).astype(int)

# Define Neural Network for Second-Layer Models
class IndexPredictionNetwork(tf.keras.Model):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        super(IndexPredictionNetwork, self).__init__()
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_dim, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(output_dim, activation=None)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

# Train a Neural Network for each segment
print("Training second-layer models...")
segment_models = []
for segment in range(num_segments):
    mask = segment_predictions == segment
    x_segment = x_scaled[mask]
    y_segment = y[mask]

    print(f"Training segment {segment} with {len(x_segment)} samples...")
    if len(x_segment) > 0:
        model = IndexPredictionNetwork(input_dim=1, hidden_dims=[64, 32], output_dim=1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        model.fit(x_segment, y_segment, epochs=10, batch_size=32, verbose=1)
        segment_models.append(model)
    else:
        segment_models.append(None) 

def rmi_predict(x_values):
    root_predictions = root_model.predict(x_values)
    root_segments = np.clip(np.round(root_predictions), 0, num_segments - 1).astype(int)

    final_predictions = np.zeros(len(x_values))
    for segment in range(num_segments):
        mask = root_segments == segment
        if np.any(mask) and segment_models[segment]:
            final_predictions[mask] = segment_models[segment](x_values[mask]).numpy().squeeze()

    return final_predictions

print("Generating predictions")
y_pred = rmi_predict(x_scaled)
print(f"Sample predictions: {y_pred[:5]}")


plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='blue', label='True Indices', alpha=0.5)
plt.scatter(x, y_pred, c='red', label='Predicted Indices', alpha=0.5)
plt.title('Recursive Model Index (RMI) Predictions')
plt.xlabel('Latitude')
plt.ylabel('Indices')
plt.legend()
plt.show()


