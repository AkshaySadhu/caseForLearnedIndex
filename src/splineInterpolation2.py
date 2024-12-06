import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import psutil

def get_model_size():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def measure_training_time(X, y):
    start_time = time.time()
    
    train_sort_idx = np.argsort(X)
    X_train = X[train_sort_idx]
    y_train = y[train_sort_idx]
    cs = CubicSpline(X_train, y_train)
    
    end_time = time.time()
    return end_time - start_time, cs, X_train, y_train

def measure_prediction_time(cs, x_new):
    start_time = time.time()
    y_new = []
    for x in x_new:
        y_new.append(cs(x))
    y_new = np.array(y_new)
    end_time = time.time()
    return end_time - start_time, y_new

df = pd.read_csv('sydneyUniqueSortedLatitudes.csv')

sizes = [10000, 100000, 1000000]
results = {}

def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

for size in sizes:
    print(f"\nProcessing size: {size}")
    
    X = df['latitude'].values[:size]
    y = df['index'].values[:size]
    
    initial_memory = get_model_size()
    
    train_time, cs, X_train, y_train = measure_training_time(X, y)
    
    final_memory = get_model_size()
    model_size = final_memory - initial_memory
    
    x_new = np.linspace(min(X_train), max(X_train), size)
    
    pred_time, y_new = measure_prediction_time(cs, x_new)
    
    results[size] = {
        'model_size_bytes': model_size,
        'training_time_seconds': train_time,
        'prediction_time_seconds': pred_time
    }

    x_for_mad = df['latitude'].values[:size]
    y_true = df['index'].values[:size]
    y_pred = cs(x_for_mad)
    print(y_pred[0:10])
    print(y_true[0:10])
    mad = mean_absolute_deviation(y_true, y_pred)
    print(f"\nMean Absolute Deviation for size {size}: {mad:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_train, y_train, 'b-', label='Original Points', linewidth=1)
    plt.plot(x_new, y_new, 'r-', label='Spline Interpolation', linewidth=1)
    plt.xlabel('Latitude')
    plt.ylabel('Index')
    plt.grid(True)
    plt.legend()
    plt.title(f'Spline Interpolation (n={size})')
    plt.show()

print("\nFinal Results:")
print("-" * 50)
for size, metrics in results.items():
    print(f"\nSize: {size} rows")
    print(f"Model Size: {metrics['model_size_bytes']/1024/1024:.2f} MB")
    print(f"Training Time: {metrics['training_time_seconds']:.4f} seconds")
    print(f"Prediction Time: {metrics['prediction_time_seconds']:.4f} seconds")
