import pandas as pd
import joblib

from sklearn.neural_network import MLPRegressor

# LOAD DATA
df = pd.read_csv("app/data/companies.csv")

# FEATURES
X = df[["Growth", "Risk", "RSI", "Volatility"]]

# TARGET
y = df["MarketCap"]

# BUILD MODEL
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    max_iter=500,
    random_state=42
)

# TRAIN MODEL
model.fit(X, y)

# SAVE MODEL
joblib.dump(model, "app/ML/deep_market_model.pkl")

print("Deep Learning Model Trained Successfully!")