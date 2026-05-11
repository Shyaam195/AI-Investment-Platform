import torch
import numpy as np
import torch.nn as nn

# ---------------- MODEL ----------------
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---------------- LOAD MODEL ----------------
model = LSTMModel()
model.load_state_dict(torch.load("app/ml/deep_model.pt"))
model.eval()

# ---------------- SAMPLE DATA ----------------
sample = np.array([[0.7, 0.2, 60, 0.3]], dtype=np.float32)
sample = sample.reshape((1, 1, 4))
input_tensor = torch.tensor(sample)

# ---------------- PREDICTION ----------------
with torch.no_grad():
    prediction = model(input_tensor).item()

# ---------------- DECISION LOGIC ----------------
if prediction > 2000000000:
    decision = "🟢 BUY"
elif prediction > 1000000000:
    decision = "🟡 HOLD"
else:
    decision = "🔴 SELL"

print("Predicted MarketCap:", prediction)
print("Investment Decision:", decision)

def get_recommendation(score):
    if score >= 70:
        return "BUY"

    elif score >= 50:
        return "HOLD"

    else:
        return "SELL"