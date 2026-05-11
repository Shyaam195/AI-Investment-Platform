import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "ml", "model.pkl")

model = joblib.load(model_path)

def predict(growth, risk, rsi, volatility):
    data = np.array([[growth, risk, rsi, volatility]])
    return model.predict(data)[0]