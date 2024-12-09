#only this cell
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define the model architecture in TensorFlow
class IndexPredictionNetwork(tf.keras.Model):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1, output_activation='linear'):
        super(IndexPredictionNetwork, self).__init__()
        self.hidden_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_dim, activation='leaky_relu'))
            prev_dim = hidden_dim

        # Final output layer
        self.output_layer = tf.keras.layers.Dense(output_dim, activation='exponential')

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def generate_training_data(filename_read, num_samples=100000):
    input_values = pd.read_csv(filename_read, header=None)
    data = input_values[1:num_samples]
    x = data[[0]].values.astype(np.float64)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    indices = np.arange(len(x))

    return x, indices

def train_index_prediction_network(input_values, indices, epochs=150, learning_rate=0.001):
    X = tf.convert_to_tensor(input_values, dtype=tf.float64)
    y = tf.convert_to_tensor(indices, dtype=tf.int32)
    model = IndexPredictionNetwork()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae'])
    history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)

    return model, input_values, indices, history

def visualize_predictions(model, input_values, true_indices):
    predictions = model(input_values).numpy()
    plt.figure(figsize=(10, 6))
    plt.scatter(input_values, true_indices, c='blue', label='True Indices', alpha=0.5)
    plt.scatter(input_values, predictions, c='red', label='Predicted Indices', alpha=0.5)
    plt.title('Index Prediction Neural Network')
    plt.xlabel('Input Values')
    plt.ylabel('Indices')
    plt.legend()
    plt.show()

filename = '/content/sydneyUniqueSortedLatitudes.csv'
for size in [10000, 100000, 1000000]:
  input_values, indices = generate_training_data(filename, num_samples=size)
  model, input_values, indices, history = train_index_prediction_network(input_values, indices, epochs=1)
  visualize_predictions(model, input_values, indices)


filename = '/content/lognormalSortedData.csv'
for size in [10000, 100000, 1000000]:
  input_values, indices = generate_training_data(filename, num_samples=size)
  model, input_values, indices, history = train_index_prediction_network(input_values, indices, epochs=1)
  visualize_predictions(model, input_values, indices)


