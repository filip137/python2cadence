from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_moons_data(n_samples=1000, noise=0.1, random_state=42):
    # Generate the moons dataset
    X, Y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Split the data: 60% for training, 40% for validation and test
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, random_state=random_state)
    
    # Split the remaining 40%: 20% for validation, 20% for test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=random_state)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# Prepare the moons dataset
X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_moons_data()

# Print shapes of the datasets
print("Shapes of datasets:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_val:", Y_val.shape)
print("Y_test:", Y_test.shape)

# Print example input vectors
print("\nExample input vectors (X_train):")
print(X_train[:5])  # Print the first 5 input vectors from training set

print("\nCorresponding labels (Y_train):")
print(Y_train[:5])  # Print the corresponding labels from training set

print("\nExample input vectors (X_val):")
print(X_val[:5])  # Print the first 5 input vectors from validation set

print("\nCorresponding labels (Y_val):")
print(Y_val[:5])  # Print the corresponding labels from validation set

print("\nExample input vectors (X_test):")
print(X_test[:5])  # Print the first 5 input vectors from test set

print("\nCorresponding labels (Y_test):")
print(Y_test[:5])  # Print the corresponding labels from test set