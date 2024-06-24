# Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset

dataset = pd.read_csv('pima-indians-diabetes.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Print the number of missing entries in each column

print("Missing values :  \n", dataset.isnull().sum())

# Configure an instance of the SimpleImputer class

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the DataFrame

imputer.fit(X[:, 1:8])

# Apply the transform to the DataFrame

X[:, 1:8] = imputer.transform(X[:, 1:8])

# Print your updated matrix of features

print("\nUpdated features matrix : \n", X)