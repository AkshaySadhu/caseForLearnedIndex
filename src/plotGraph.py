import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data_path = "../data/sydneyUniqueSortedLatitudes.csv"  # Replace with your actual file path
data = pd.read_csv(data_path, header=None)

# Convert data to numeric, handle any errors
data[0] = pd.to_numeric(data[0], errors='coerce')  # Convert latitudes to numeric
data[1] = pd.to_numeric(data[1], errors='coerce')  # Convert indices to numeric

# Drop rows with missing values (if any)
data = data.dropna()

# Extract latitudes (X) and indices (y)
X = data[0].values  # Latitude
y = data[1].values  # Index

# Plot the data points
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Data Points")
plt.xlabel("Latitude")
plt.ylabel("Index")
plt.title("Latitude vs Index")
plt.legend()
plt.grid()
plt.show()
