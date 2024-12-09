import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load and preprocess the dataset
filename = 'lognormalUniqueSorted.csv'
data = pd.read_csv(filename, header=None)
data.columns = ['Integer', 'Index']

# Ensure numeric values and drop invalid rows
data['Integer'] = pd.to_numeric(data['Integer'], errors='coerce')
data['Index'] = pd.to_numeric(data['Index'], errors='coerce')
data = data.dropna()
data = data.sort_values(by='Integer').reset_index(drop=True)

# Normalize the input values (Integer)
scaler = StandardScaler()
x = scaler.fit_transform(data['Integer'].values.reshape(-1, 1))
y = data['Index'].values

# Root Model: Random Forest Regressor
num_segments = 20  
y_cdf = (y / max(y)) * (num_segments - 1) 
root_model = RandomForestRegressor(n_estimators=100, random_state=42)
root_model.fit(x, y_cdf)

# Predict segments using the Root Model
segment_predictions = root_model.predict(x)
segment_predictions = np.clip(np.round(segment_predictions), 0, num_segments - 1).astype(int)

# Train a Neural Network for each segment with increased epochs and regularization
print("Training second-layer models...")
segment_models = []
for segment in range(num_segments):
    mask = segment_predictions == segment
    x_segment = x[mask]
    y_segment = y[mask]

    if len(x_segment) > 0:
        # Define and train the neural network model with regularization
        model = Sequential()
        model.add(Dense(256, activation='leaky_relu', input_dim=1, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(128, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(64, activation='leaky_relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Dense(1, activation='exponential'))

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        model.fit(x_segment, y_segment, epochs=150, batch_size=64, verbose=1)  # Increase epochs
        segment_models.append(model)
    else:
        segment_models.append(None)

# Modify the inference function to smooth predictions
def rmi_predict_smooth(x_values, smooth_window=5):
    root_predictions = root_model.predict(x_values)
    root_segments = np.clip(np.round(root_predictions), 0, num_segments - 1).astype(int)

    final_predictions = np.zeros(len(x_values))
    for segment in range(num_segments):
        mask = root_segments == segment
        if np.any(mask) and segment_models[segment]:
            predictions = segment_models[segment].predict(x_values[mask]).flatten()
            # Apply a moving average to smooth predictions
            predictions = np.convolve(predictions, np.ones(smooth_window) / smooth_window, mode='same')
            final_predictions[mask] = predictions

    return final_predictions

# Predictions using the RMI with smoothing
print("Generating predictions...")
y_pred = rmi_predict_smooth(x)
print(f"Sample predictions: {y_pred[:5]}")

# Visualization: True vs Predicted Indices (Smoothed Plot)
plt.figure(figsize=(10, 6))
plt.scatter(data['Integer'], y, c='blue', label='True Indices', alpha=0.5)
plt.scatter(data['Integer'], y_pred, c='red', label='Predicted Indices', alpha=0.5)
plt.title('Recursive Model Index (RMI) Predictions')
plt.xlabel('Integer')
plt.ylabel('Index')
plt.legend()
plt.show()

