import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
data = pd.read_csv('data/Real_estate.csv')

# Select features and target
X = data.iloc[:, 1:7].values  # Features
y = data.iloc[:, 7].values    # Target

# Normalize both features AND target
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_normalized = X_scaler.fit_transform(X)
y_normalized = y_scaler.fit_transform(y.reshape(-1, 1))

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y_normalized, dtype=torch.float32)

# Create a DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(6, 28)
        self.hidden2 = nn.Linear(28, 29)
        self.hidden3 = nn.Linear(29, 30)
        self.output = nn.Linear(30, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        x = self.output(x)
        return x

model = MLP()

import torch.optim as optim

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    epoch_losses = []
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    avg_loss = sum(epoch_losses) / len(epoch_losses)
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

# After training loop
model.eval()
with torch.no_grad():
    # Get predictions
    y_pred_normalized = model(X_tensor)
    
    # Denormalize predictions
    y_pred = y_scaler.inverse_transform(y_pred_normalized.numpy())
    
    # Compute MSE on original scale
    mse = np.mean((y - y_pred.ravel())**2)
    print("Mean Squared Error:", mse)