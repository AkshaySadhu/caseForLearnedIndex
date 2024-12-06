import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

df = pd.read_csv('sydneyUniqueSortedLatitudes.csv')

X_train = df['latitude'].values.reshape(-1, 1)
y_train = df['index'].values

train_sort_idx = np.argsort(X_train.ravel())
X_train = X_train[train_sort_idx]
y_train = y_train[train_sort_idx]

spline_model = make_pipeline(
    SplineTransformer(n_knots=25, degree=3),
    LinearRegression()
)

spline_model.fit(X_train, y_train)

def predict_index(value):
    return spline_model.predict([[value]])[0]

x_new = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
y_new = spline_model.predict(x_new)

def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

plt.figure(figsize=(8, 4))
plt.scatter(X_train, y_train, color='blue', label='Original Points')
plt.plot(x_new, y_new, 'r-', label='Spline Interpolation')
plt.xlabel('Values')
plt.ylabel('Indices')
plt.grid(True)
plt.legend()
plt.title('Index Prediction using Spline Transformer')
plt.show()