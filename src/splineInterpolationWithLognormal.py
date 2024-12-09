import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('lognormal.csv', header=None) 
X = df[0].drop_duplicates().sort_values().values
y = np.arange(len(X))

X_train = X
y_train = y

cs = CubicSpline(X_train, y_train)

def predict_index(value):
    return cs(value)

x_new = np.linspace(min(X_train), max(X_train), 100)
y_new = cs(x_new)

plt.figure(figsize=(8, 4))
plt.scatter(X_train, y_train, color='blue', label='Original Points')
plt.plot(x_new, y_new, 'r-', label='Spline Interpolation')
plt.xlabel('Values')
plt.ylabel('Indices')
plt.grid(True)
plt.legend()
plt.title('Index Prediction using Cubic Spline')
plt.show()

def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# print('First 10 actual: ', y_test[0:10])
# print('First 10 predicted: ', y_pred[0:10])
# mad = mean_absolute_deviation(y_test, y_pred)
# print(f"Mean Absolute Deviation: {mad}")
