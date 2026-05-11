import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierEngine:

    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)

    def detect(self, df):
        df = df.copy()

        features = ["AI_Score", "Risk", "Volatility", "Growth"]
        df[features] = df[features].fillna(df[features].mean())

        preds = self.model.fit_predict(df[features])

        df["Is_Outlier"] = preds == -1
        df["Outlier_Score"] = np.where(preds == -1, 1, 0)

        return df