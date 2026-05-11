import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor

# LOAD DATA
df = pd.read_csv("app/data/companies.csv")

# FEATURES
X = df[["Growth", "Risk", "RSI", "Volatility"]]

# TARGET
y = df["MarketCap"]

# TRAIN MODEL
model = RandomForestRegressor()
model.fit(X, y)

# SAVE MODEL
joblib.dump(model, "app/ML/market_model.pkl")

print("Model trained successfully!")