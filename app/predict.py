import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "ml", "model.pkl")

model = joblib.load(model_path)

sample_data = pd.DataFrame([{
    "Growth": 90,
    "Risk": 20,
    "RSI": 65,
    "Volatility": 15
}])

prediction = model.predict(sample_data)

print("Predicted MarketCap:", prediction[0])