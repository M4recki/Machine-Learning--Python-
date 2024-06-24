# Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Loading the Iris dataset

dataset = pd.read_csv('iris.csv')

# Creating the matrix of features (X) and the dependent variable vector (y)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Printing the matrix of features and the dependent variable vector

print(X)
print(y)