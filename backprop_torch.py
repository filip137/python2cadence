import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# Generate the moons dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Moons Dataset')
plt.show()

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create a TensorDataset
dataset = TensorDataset(X_tensor, y_tensor)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the neural network
class ComplexNetwork(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(ComplexNetwork, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.output_layer = nn.Linear(n_hidden2, n_output)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout during training only
        x = self.output_layer(x)  # No activation here for CrossEntropyLoss
        return x

# Instantiate the network
mynetwork = ComplexNetwork(n_input=2, n_hidden1=4, n_hidden2=4, n_output=2)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mynetwork.parameters(), lr=0.1)

# Training loop
num_of_epochs = 10
for epoch in range(num_of_epochs):
    running_loss = 0.0
    mynetwork.train()  # Ensure the model is in training mode
    for inputs, labels in dataloader:
        # Forward pass
        prediction = mynetwork(inputs)

        # Calculate loss
        loss = criterion(prediction, labels)

        # Zero gradients, backward pass, and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_of_epochs}], Loss: {running_loss / len(dataloader):.4f}')

def plot_decision_boundary(model, X, y, grid_step=0.01, cmap=plt.cm.Paired):
    # Set the model to evaluation mode
    model.eval()

    # Generate a grid of points covering the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # Convert the grid to a tensor and pass it through the model
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model(grid)
        _, Z = torch.max(Z, 1)
        Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)

    # Plot the original data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()
