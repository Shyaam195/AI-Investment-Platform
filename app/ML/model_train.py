import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# load real data
df = pd.read_csv("app/data/companies.csv")

# input features (what AI learns from)
X = df[["Growth", "Risk", "RSI", "Volatility"]]

# output (what AI predicts)
y = df["MarketCap"]

# train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# save model
joblib.dump(model, "app/ml/model.pkl")

print("Model trained successfully")