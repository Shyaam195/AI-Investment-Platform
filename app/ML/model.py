import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def predict_signal(features):
    prediction = model.predict([features])[0]

    if prediction == 2:
        return "BUY"
    elif prediction == 1:
        return "HOLD"
    else:
        return "SELL"

# Load dataset
df = pd.read_csv("app/data/companies.csv")

# Convert recommendation to numbers
label_map = {
    "BUY": 2,
    "HOLD": 1,
    "SELL": 0
}

df["Target"] = df["Recommendation"].map(label_map)

# Features
X = df[["Growth", "Risk", "RSI", "Volatility", "MarketCap"]]
y = df["Target"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "app/ml/trading_model.pkl")

print("Model trained and saved!")