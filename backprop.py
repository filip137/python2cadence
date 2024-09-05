import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ensure plots are shown in Spyder

# Generate the moons dataset
n_samples = 4000
noise = 0.1
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
def create_model(hidden_units):
    model = Sequential([
        Dense(hidden_units, input_shape=(2,), activation='relu'),  # Input layer with 2 nodes, first hidden layer
        #Dense(hidden_units, activation='relu'),
        #Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer with 1 node (binary classification)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

# Function to visualize the true labels, predictions, and decision boundary
def plot_predictions(X, y_true, y_pred, model, hidden_units):
    plt.figure(figsize=(18, 6))

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict the class for each point in the grid
    grid_predictions = model.predict(grid)
    grid_predictions = (grid_predictions > 0.5).astype(int).reshape(xx.shape)

    # Plot true labels
    plt.subplot(1, 3, 1)
    plt.scatter(X[y_true == 0][:, 0], X[y_true == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y_true == 1][:, 0], X[y_true == 1][:, 1], color='blue', label='Class 1')
    plt.title('True Labels')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # Plot predictions
    plt.subplot(1, 3, 2)
    plt.scatter(X[y_pred == 0][:, 0], X[y_pred == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y_pred == 1][:, 0], X[y_pred == 1][:, 1], color='blue', label='Class 1')
    plt.title(f'Predicted Labels with {hidden_units} Hidden Units')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    # Plot decision boundary
    plt.subplot(1, 3, 3)
    plt.contourf(xx, yy, grid_predictions, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[y_true == 0][:, 0], X[y_true == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y_true == 1][:, 0], X[y_true == 1][:, 1], color='blue', label='Class 1')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Experiment with different numbers of hidden units
hidden_units_list = [4]

for hidden_units in hidden_units_list:
    print(f'Training with {hidden_units} hidden units per layer')
    
    model = create_model(hidden_units)
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    
    # Train the model and store the history
    history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Hidden Units: {hidden_units}, Test Accuracy: {accuracy}')
    
    # Plot the training and validation accuracy and loss
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy with {hidden_units} Hidden Units')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss with {hidden_units} Hidden Units')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    # Visualize the results
    plot_predictions(X_test, y_test, y_pred, model, hidden_units)