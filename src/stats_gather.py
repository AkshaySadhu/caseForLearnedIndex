import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import psutil
from NN import train_index_prediction_network, generate_training_data

# Function to get model size
def get_model_size():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # in bytes

# Function to measure training time
def measure_training_time(X, y):
    start_time = time.time()
    
    train_sort_idx = np.argsort(X)
    X_train = X[train_sort_idx]
    y_train = y[train_sort_idx]
    model, input_values, indices, history = train_index_prediction_network(X_train, y_train, epochs=1)
    
    end_time = time.time()
    return end_time - start_time, model, X_train, y_train


import time
def measure_prediction_time(cs, x_new):
    start_time = time.time()
    y_new = []
    for x in x_new:
        y_new.append(cs(x))
    y_new = np.array(y_new)
    end_time = time.time()
    return end_time - start_time, y_new

# Read the CSV file
df = pd.read_csv('lognormalUniqueSorted.csv', header=None)

# Test different sizes
sizes = [10000, 100000, 1000000]
results = {}

for size in sizes:
    print(f"\nProcessing size: {size}")
    
    # Take subset of data    
    initial_memory = get_model_size()
    X, y = generate_training_data(size)

    train_time, cs, X_train, y_train = measure_training_time(X, y)
    
    final_memory = get_model_size()
    model_size = final_memory - initial_memory
    
    x_new = np.linspace(min(X_train), max(X_train), size)
    
    # Measure prediction time
    pred_time, y_new = measure_prediction_time(cs, x_new)
    
    # Store results
    results[size] = {
        'model_size_bytes': model_size,
        'training_time_seconds': train_time,
        'prediction_time_seconds': pred_time
    }

    print(results)
    
    # # Plot results
    # plt.figure(figsize=(10, 6))
    # visualize_predictions(cs, X_train, y_train)
    # plt.plot(X_train, y_train, 'b-', label='Original Points', linewidth=1)
    # plt.plot(x_new, y_new, 'r-', label='Spline Interpolation', linewidth=1)
    # plt.xlabel('Latitude')
    # plt.ylabel('Index')
    # plt.grid(True)
    # plt.legend()
    # plt.title(f'Spline Interpolation (n={size})')
    # plt.show()

# Print results
print("\nFinal Results:")
print("-" * 50)
for size, metrics in results.items():
    print(f"\nSize: {size} rows")
    print(f"Model Size: {metrics['model_size_bytes']/1024/1024:.2f} MB")
    print(f"Training Time: {metrics['training_time_seconds']:.4f} seconds")
    print(f"Prediction Time: {metrics['prediction_time_seconds']:.4f} seconds")

# Calculate MAD for the largest size
def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))