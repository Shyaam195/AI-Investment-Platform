import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- LOAD DATA ----------------
df = pd.read_csv("app/data/companies.csv")

X = df[["Growth", "Risk", "RSI", "Volatility"]].values
y = df["MarketCap"].values

# convert to float32
X = X.astype(np.float32)
y = y.astype(np.float32)

# reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], 1, X.shape[1]))

# ---------------- CONVERT TO TENSORS ----------------
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).view(-1, 1)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ---------------- MODEL ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel()

# ---------------- LOSS & OPTIMIZER ----------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------------- TRAINING ----------------
epochs = 10

for epoch in range(epochs):
    for batch_X, batch_y in loader:

        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ---------------- SAVE MODEL ----------------
torch.save(model.state_dict(), "app/ml/deep_model.pt")

print("✅ PyTorch model trained successfully!")