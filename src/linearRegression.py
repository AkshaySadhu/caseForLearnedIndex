import csv
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

start_time = time.time()
# Load the dataset
data_path = "../data/lognormalUniqueSorted.csv"  # Replace with your file path
data = pd.read_csv(data_path, header=None)

# Convert data to numeric, handle any errors
data[0] = pd.to_numeric(data[0], errors='coerce')  # Latitudes
data[1] = pd.to_numeric(data[1], errors='coerce')  # Indices
data = data.dropna()  # Drop invalid rows

# Extract latitude (X) and index (y)
X = data[0].values
y = data[1].values

# Define ranges for case-based equations
ranges = [(0, 700000), (1000000, 100000000)]  # Example ranges, customize as needed
case_polynomials = {}

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data Points")  # Plot raw data points

# Fit polynomial for each range
for i, (low, high) in enumerate(ranges):
    # Filter data within the range
    mask = (X >= low) & (X < high)
    X_range = X[mask]
    y_range = y[mask]

    if len(X_range) > 1:  # Ensure there's enough data for fitting
        degree = 1  # Degree of the polynomial (adjustable)
        coeffs = np.polyfit(X_range, y_range, degree)
        poly = np.poly1d(coeffs)
        case_polynomials[f"Case {i+1}: {low} <= x < {high}"] = poly

        # Plot the fitted polynomial curve
        X_pred = np.linspace(low, high, 500)
        y_pred = poly(X_pred)
        plt.plot(X_pred, y_pred, label=f"Case {i+1}: Degree {degree}")

elapsed = time.time() - start_time
# Function to calculate the model size
def calculate_model_size(case_polynomials):
    total_size = 0
    for case, poly in case_polynomials.items():
        # Size of the polynomial coefficients
        coefficients_size = sys.getsizeof(poly.coefficients)
        # Size of the polynomial intercept (if present)
        intercept_size = sys.getsizeof(poly.c)

        total_size += coefficients_size + intercept_size

    return total_size


# Calculate the model size
model_size = calculate_model_size(case_polynomials)

# Print the model size in bytes
print(f"Total model size: {model_size} bytes")

# Function to query a specific latitude
def predict_latitude(latitude):
    for i, (low, high) in enumerate(ranges):
        if low <= latitude < high:
            poly = case_polynomials.get(f"Case {i + 1}: {low} <= x < {high}")
            if poly:
                return poly(latitude)
    return None


# Function to load a specific number of rows from a CSV file
def load_data(file_path, num_rows=None):
    latitudes = []
    indices = []
    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)
        for i, row in enumerate(reader):
            if num_rows and i >= num_rows:
                break
            latitudes.append(float(row[0]))  # Latitude
            indices.append(float(row[1]))  # Index
    return latitudes, indices



# Test for different dataset sizes
sizes = [10000, 100000, 1000000]  # 10k, 100k, 1M rows

test_path = "../data/lognormalUniqueSorted.csv"  # Replace with your test data path

for size in sizes:
    print(f"\nTesting with {size} rows:")

    # Load a subset of the test data
    test_latitudes, ground_truth_indices = load_data(test_path, size)

    # Make predictions and calculate performance metrics
    predicted_indices = []
    print(f"Testing data of size: {len(test_latitudes)}")

    start_test_time = time.time()

    for lat in test_latitudes:
        pred_index = predict_latitude(lat)  # Using your custom prediction function
        predicted_indices.append(pred_index if pred_index is not None else np.nan)

    elapsed_test_time = time.time() - start_test_time
    print(f"Time taken for testing {len(test_latitudes)} entries: {elapsed_test_time:.2f} seconds")

    # Plot the results for the current size
    plt.figure(figsize=(10, 6))
    plt.scatter(test_latitudes, ground_truth_indices, color="blue", label="Ground Truth")
    plt.scatter(test_latitudes, predicted_indices, color="red", label="Predicted", alpha=0.6)
    plt.xlabel("Latitude")
    plt.ylabel("Index")
    plt.title(f"Validation: Predicted vs Ground Truth Indices (Size: {size})")
    plt.legend()
    plt.grid()
    plt.show()