import pandas as pd
import numpy as np

df = pd.read_csv('../../data/lognormal_g.csv', header=None)  # Specify header=None for files without headers
X = df[0].drop_duplicates().sort_values().values  # Now column 0 will be recognized
y = np.arange(len(X))

X_train=X
y_train=y

df_output = pd.DataFrame({"Latitude": X_train, "Index": y_train})

# Save the data to a CSV file
output_path = "../../data/lognormalUniqueSorted.csv"  # Adjust path as needed
df_output.to_csv(output_path, index=False, header=True)
