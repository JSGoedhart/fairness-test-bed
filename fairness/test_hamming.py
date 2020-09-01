import numpy as np
import pandas as pd
import os

data_path = os.path.join('fairness', 'data', 'preprocessed', 'ricci_numerical-binsensitive.csv')
df = pd.read_csv(data_path, sep = ',')

print(df)


bool_cols = [col for col in df if np.isin(df[col].dropna().unique(), [0, 1]).all()]
print(bool_cols)
print(df[bool_cols])