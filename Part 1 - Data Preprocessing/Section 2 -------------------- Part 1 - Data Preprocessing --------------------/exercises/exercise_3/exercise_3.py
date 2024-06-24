# Importing the necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset

dataset = pd.read_csv("titanic.csv")

# Identify the categorical data

categorical_features = ['Sex', 'Embarked', 'Pclass']

# Implement an instance of the ColumnTransformer class

ct = ColumnTransformer(
    [("encoder", OneHotEncoder(), categorical_features)], remainder="passthrough"
)

# Apply the fit_transform method on the instance of ColumnTransformer

X = ct.fit_transform(dataset)

le = LabelEncoder()
y = le.fit_transform(dataset["Survived"])

# Print the updated matrix of features and the dependent variable vector
print("Updated matrix of features: \n", X)
print("Updated dependent variable vector: \n", y)
