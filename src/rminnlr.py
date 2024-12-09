import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#Linear Regression
def fit_piecewise_polynomial(X, y, ranges, max_rmse=100):
    case_polynomials = {}
    total_rmse = float('inf')
    iterations = 0

    while total_rmse > max_rmse and iterations < 100:
        case_polynomials.clear()
        predictions = np.zeros_like(y)

        # Fit polynomial for each range
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


#Neural Network for RMI
class IndexPredictionNetwork(tf.keras.Model):
    def __init__(self, input_dim=1, hidden_dims=[64, 32], output_dim=1):
        super(IndexPredictionNetwork, self).__init__()

        self.hidden_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(tf.keras.layers.Dense(hidden_dim, activation='leaky_relu'))
            prev_dim = hidden_dim

        self.output_layer = tf.keras.layers.Dense(output_dim,activation='exponential')

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


#RMI with two layers
def build_rmi(X, y, initial_ranges, max_rmse=1000, epochs=50, batch_size=32):
    case_polynomials, final_ranges, final_rmse = fit_piecewise_polynomial(X, y, initial_ranges, 100)
    print(f"PLR Model Final RMSE: {final_rmse:.2f}")

    segment_models = {}
    for i, (low, high) in enumerate(final_ranges):
        mask = (X >= low) & (X < high)
        X_segment = X[mask].reshape(-1, 1)
        y_segment = y[mask]

        if len(X_segment) > 1:
            scaler = StandardScaler()
            X_segment_normalized = scaler.fit_transform(X_segment)

            model = IndexPredictionNetwork()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                          loss='mse', metrics=['mae'])
            model.fit(X_segment_normalized, y_segment, epochs=epochs, batch_size=batch_size, verbose=1)

            
            segment_models[f"Range_{i}"] = {
                'range': (low, high),
                'model': model,
                'scaler': scaler
            }
        else:
            print(f"Skipping segment {i} due to insufficient data.")

    return case_polynomials, segment_models, final_ranges


#Predict with RMI
def rmi_predict(X, case_polynomials, segment_models, fallback_value):
    predictions = np.zeros_like(X)

    for i, x in enumerate(X):
        segment_poly = None
        for case, poly_info in case_polynomials.items():
            low, high = poly_info['range']
            if low <= x < high:
                segment_poly = poly_info['poly']
                break

        if segment_poly is None:
            predictions[i] = fallback_value
            continue

        predicted = None
        for segment_name, segment_info in segment_models.items():
            low, high = segment_info['range']
            if low <= x < high:
                model = segment_info['model']
                scaler = segment_info['scaler']
                x_normalized = scaler.transform([[x]])
                predicted = model.predict(x_normalized).flatten()[0]
                break

    
        predictions[i] = predicted if predicted is not None else fallback_value

    return predictions


# Main Execution
if __name__ == "__main__":
    # Load and preprocess the dataset
    data_path = 'sydneyUniqueSortedLatitudes.csv'
    data = pd.read_csv(data_path, header=None, low_memory=False)

    # Convert data to numeric, handle any errors
    data[0] = pd.to_numeric(data[0], errors="coerce")  # Latitudes
    data[1] = pd.to_numeric(data[1], errors="coerce")  # Indices
    data = data.dropna()
    X = data[0].values
    y = data[1].values

    # Initial ranges
    initial_ranges = [(-50, -35.2), (-35.2, -34.8), (-34.8, -10)]
    #initial_ranges = [(1.0, 1200.0), (1200.0, 3000.0), (3000.0, 180825.0)]

    # Train RMI
    case_polynomials, segment_models, final_ranges = build_rmi(X, y, initial_ranges, max_rmse=1000)

    # Test RMI Predictions
    fallback_value = np.mean(y)  # Fallback for missing predictions
    predictions = rmi_predict(X, case_polynomials, segment_models, fallback_value)

    # Remove NaN values for evaluation
    valid_mask = ~np.isnan(predictions)
    valid_y = y[valid_mask]
    valid_predictions = predictions[valid_mask]

    # Evaluate performance
    mse = mean_squared_error(valid_y, valid_predictions)
    print(f"RMI Model MSE: {mse:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", label="Ground Truth", alpha=0.5)
    plt.scatter(X[valid_mask], valid_predictions, color="red", label="RMI Predictions", alpha=0.5)
    plt.xlabel("Latitude")
    plt.ylabel("Index")
    plt.title("Recursive Model Index Predictions")
    plt.legend()
    plt.show()
