import csv
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, sys
from sklearn.metrics import mean_squared_error

def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right)//2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)

def binary_search(arr, target, left, right):
    index = binary_search_recursive(arr, target, left, right)
    return arr[index]

def fit_piecewise_polynomial(X, y, ranges, max_rmse=1000):
    case_polynomials = {}
    total_rmse = float('inf')
    iterations = 0

    while total_rmse > max_rmse and iterations < 100:
        case_polynomials.clear()
        predictions = np.zeros_like(y)

        for i, (low, high) in enumerate(ranges):
            mask = (X >= low) & (X < high)
            X_range = X[mask]
            y_range = y[mask]

            if len(X_range) > 1:
                degree = 1
                coeffs = np.polyfit(X_range, y_range, degree)
                poly = np.poly1d(coeffs)
                case_polynomials[f"Range_{i}"] = {
                    'range': (low, high),
                    'poly': poly
                }
                predictions[mask] = poly(X_range)

        # Calculate RMSE
        total_rmse = np.sqrt(mean_squared_error(y, predictions))
        print(f"Iteration {iterations + 1}: RMSE = {total_rmse:.2f}")

        if total_rmse > max_rmse:
            errors = []
            for i, (low, high) in enumerate(ranges):
                mask = (X >= low) & (X < high)
                if np.sum(mask) > 0:
                    range_rmse = np.sqrt(mean_squared_error(y[mask], predictions[mask]))
                    errors.append((range_rmse, i))

            if errors:
                _, worst_range_idx = max(errors)
                low, high = ranges[worst_range_idx]
                mid = (low + high) / 2
                new_ranges = list(ranges)
                new_ranges[worst_range_idx:worst_range_idx + 1] = [(low, mid), (mid, high)]
                ranges = new_ranges

        iterations += 1

    return case_polynomials, ranges, total_rmse


def calculate_model_size(case_polynomials):
    total_size = 0
    for case, poly_info in case_polynomials.items():
        total_size += sys.getsizeof(case)
        total_size += sys.getsizeof(poly_info['poly'].coef)
        total_size += sys.getsizeof(poly_info['range'])
    return total_size


def predict_latitude(latitude, case_polynomials):
    for case, poly_info in case_polynomials.items():
        low, high = poly_info['range']
        if low <= latitude < high:
            return poly_info['poly'](latitude)
    return None


def load_data(file_path, num_rows=None):
    latitudes = []
    indices = []
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        for i, row in enumerate(reader):
            if num_rows and i >= num_rows:
                break
            latitudes.append(float(row[0]))
            indices.append(float(row[1]))
    return latitudes, indices


if __name__ == "__main__":
    start_train_time = time.time_ns()

    data_path = "../data/lognormalSortedData.csv"
    data = pd.read_csv(data_path, header=None, low_memory=False)

    data[0] = pd.to_numeric(data[0], errors="coerce")  # Latitudes
    data[1] = pd.to_numeric(data[1], errors="coerce")  # Indices
    data = data.dropna()
    X = data[0].values
    y = data[1].values

    initial_ranges = [(1.0, 1200.0), (1200.0, 3000.0), (3000.0, 180825.0)]

    case_polynomials, final_ranges, final_rmse = fit_piecewise_polynomial(
        X, y, initial_ranges, max_rmse=100
    )
    no = len(final_ranges)
    print(f"Number of polynomials: {no} ")
    elapsed_train_time = time.time_ns() - start_train_time
    print("Time required to train the model (ns):", elapsed_train_time)

    model_size = calculate_model_size(case_polynomials)
    print(f"Total model size: {model_size} bytes")

    sizes = [10000, 100000, 1000000]
    test_path = "../data/lognormalSortedData.csv"

    for size in sizes:
        print(f"\nTesting with {size} rows:")
        test_latitudes, ground_truth_indices = load_data(test_path, size)
        predicted_indices = []

        start_test_time = time.time()

        for lat in test_latitudes:
            pred_index = predict_latitude(lat, case_polynomials)
            predicted_indices.append(pred_index if pred_index is not None else np.nan)

        elapsed_test_time = time.time() - start_test_time
        print(f"Time taken for testing {len(test_latitudes)} entries: {elapsed_test_time:.2f} seconds")

        mse = mean_squared_error(ground_truth_indices, predicted_indices) / len(test_latitudes)
        print(f"Mean Squared Error (Training Loss): {mse:.4f}")

        plt.figure(figsize=(10, 6))
        plt.scatter(test_latitudes, ground_truth_indices, color="blue", label="Ground Truth")
        plt.scatter(test_latitudes, predicted_indices, color="red", label="Predicted", alpha=0.6)
        plt.xlabel("Latitude")
        plt.ylabel("Index")
        plt.title(f"Validation: Predicted vs Ground Truth Indices (Size: {size})")
        plt.legend()
        plt.grid()
        plt.show()

        test_path_bs = "../data/lognormalSortedData.csv"
        test_latitudes, ground_truth_indices = load_data(test_path, 1000)
        predicted_indices = []

        start_test_time = time.time()

        for lat in test_latitudes:
            pred_index = predict_latitude(lat, case_polynomials)
            predicted_indices.append(pred_index if pred_index is not None else np.nan)
            temp_lat = binary_search(test_latitudes, lat, max(int(math.floor(pred_index)) - 300, 0), min(int(math.floor(pred_index)) + 300, len(test_latitudes)))
            if temp_lat == lat:
                print("Found value")
            else:
                print("Not Found")

        elapsed_test_time_bs = time.time() - start_test_time

        print(f"Time with Binary Search: {elapsed_test_time_bs}")

