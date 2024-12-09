import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

#grid search
def build_model(hp):
    model = tf.keras.Sequential()
    
    for i in range(hp.Int('num_hidden_layers', 1, 3)):
        model.add(tf.keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice(f'activation_{i}', ['relu', 'leaky_relu', 'silu'])
        ))
    
    model.add(tf.keras.layers.Dense(1, activation='exponential'))
    
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def generate_training_data(filename_read, num_samples=100000):
    input_values = pd.read_csv(filename_read, header=None)
    data = input_values[1:num_samples]
    x = data[[0]].values.astype(np.float64)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    indices = np.arange(len(x))
    return x, indices

def perform_grid_search(input_values, indices, max_trials=10, epochs=50):
    X = tf.convert_to_tensor(input_values, dtype=tf.float64)
    y = tf.convert_to_tensor(indices, dtype=tf.int32)
    
    tuner = RandomSearch(
        build_model,
        objective='val_mae',
        max_trials=max_trials,
        executions_per_trial=1,
        directory='grid_search_results',
        project_name='index_prediction'
    )
    
    tuner.search(X, y, epochs=epochs, validation_split=0.2, verbose=1)
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_model, best_hyperparameters

def visualize_predictions(model, input_values, true_indices):
    predictions = model.predict(input_values)
    plt.figure(figsize=(10, 6))
    plt.scatter(input_values, true_indices, c='blue', label='True Indices', alpha=0.5)
    plt.scatter(input_values, predictions, c='red', label='Predicted Indices', alpha=0.5)
    plt.title('Index Prediction Neural Network')
    plt.xlabel('Input Values')
    plt.ylabel('Indices')
    plt.legend()
    plt.show()

# Main execution
filenames = ['/content/sydneyUniqueSortedLatitudes.csv', '/content/lognormalSortedData.csv']
sample_sizes = [10000, 100000, 1000000]
size = sample_sizes[0]

for filename in filenames:
        print(f"Processing {filename} with {size} samples")
        input_values, indices = generate_training_data(filename, num_samples=size)
        best_model, best_hyperparameters = perform_grid_search(input_values, indices)
        
        print("Best Hyperparameters:")
        print(best_hyperparameters.values)
        
        visualize_predictions(best_model, input_values, indices)