# Import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset

dataset = pd.read_csv("iris.csv")

# Separate features and target

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Split the dataset into an 80-20 training-test set

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply feature scaling on the training and test sets

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print the scaled training and test sets

print("Scaled training set:\n", X_train)
print("Scaled test set:\n", X_test)
