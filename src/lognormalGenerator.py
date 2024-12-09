import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mean = 0
sigma = 1
data = np.random.lognormal(mean, sigma, 4000000)

# data = np.round(data).astype(int)
data = np.unique(data)
# data = np.round(data)
sorted_data = np.sort(data)

data_with_indices = pd.DataFrame({
    'Integer': 1000*sorted_data,
    'Index': range(len(sorted_data))
})

data_with_indices.to_csv('lognormalSortedData.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(sorted_data, np.arange(1, len(sorted_data) + 1) / len(sorted_data), label='Log-Normal CDF')

plt.title('Log-Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)

plt.show()