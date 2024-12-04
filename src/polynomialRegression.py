import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
def read_csv(file_path):
    latitudes = []
    indices = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            latitudes.append(float(row[0]))
            indices.append(int(row[1]))
    return np.array(latitudes), np.array(indices)

file_path = '../data/lognormalUniqueSorted.csv'  # Replace with your file path
latitudes, indices = read_csv(file_path)

# Step 2: Prepare polynomial features
degree = 40  # Adjust the degree of the polynomial as needed
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(latitudes.reshape(-1, 1))

# Step 3: Train the Polynomial Regression model
model = LinearRegression()
model.fit(X_poly, indices)

# Step 4: Make predictions
predictions = model.predict(X_poly)

# Step 5: Evaluate the model
mse = mean_squared_error(indices, predictions)
print(f"Polynomial Regression (degree {degree})")
print(f"Mean Squared Error: {mse:.2f}")

# Step 6: Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(latitudes, indices, color='blue', label='Data Points')
plt.plot(np.sort(latitudes), model.predict(poly.fit_transform(np.sort(latitudes).reshape(-1, 1))),
         color='red', label=f'Polynomial Regression (degree {degree})')
plt.xlabel('Latitude')
plt.ylabel('Index')
plt.title(f'Polynomial Regression Fit (degree {degree})')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Query specific latitudes
query_latitudes = [-28.1221605, -19.4968715, -17.2085636]  # Replace with your latitudes
query_poly = poly.transform(np.array(query_latitudes).reshape(-1, 1))
query_predictions = model.predict(query_poly)

for lat, pred in zip(query_latitudes, query_predictions):
    print(f"Latitude: {lat} -> Predicted Index: {pred:.2f}")
