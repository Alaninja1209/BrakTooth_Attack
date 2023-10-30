import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Reading the data
data = pd.read_csv(r"C:\Users\alana\Documents\TecMTY\Esiee\Machine_Learning\insurance.csv")
print(data.shape)

# Estimate parameters
regresion = LinearRegression()
x = data[['age', 'bmi']]
y = data.charges
regresion.fit(x, y)

print(regresion.coef_)
print(regresion.intercept_)

# Prepare prediction
x_new = np.array([35, 24.9])
regresion.predict(x_new)