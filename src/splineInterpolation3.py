import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import time
import os
import psutil
import pickle

df = pd.read_csv('dummyLatitudes.csv')

X_train = df['latitude'].values.reshape(-1, 1)
y_train = df['index'].values

train_sort_idx = np.argsort(X_train.ravel())
X_train = X_train[train_sort_idx]

# mask = (X_train >= -35) & (X_train <= -30)
# X_train = (X_train[mask]).reshape(-1,1)
y_train = np.arange(0, len(X_train))

#y_train = y_train[train_sort_idx]
def get_model_size():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


spline_model = make_pipeline(
    SplineTransformer(n_knots=16, degree=3),
    LinearRegression()
)

spline_model.fit(X_train, y_train)

with open("spline_model.pkl", "wb") as f:
    pickle.dump(spline_model, f, protocol=5)


def predict_index(value):
    return spline_model.predict([[value]])[0]

x_new = np.linspace(min(X_train), max(X_train), 1000).reshape(-1, 1)
y_new = spline_model.predict(x_new)

def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

print(y_train)
plt.figure(figsize=(8, 4))
plt.scatter(X_train, y_train, color='blue', label='Original Points')
plt.plot(x_new, y_new, 'r-', label='Spline Interpolation')
plt.xlabel('Values')
plt.ylabel('Indices')
plt.grid(True)
plt.legend()
plt.title('Index Prediction using Spline Transformer')
plt.show()