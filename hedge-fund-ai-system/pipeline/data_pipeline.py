import pandas as pd

class DataPipeline:

    def __init__(self, path):
        self.path = path

    def load_data(self):
        return pd.read_csv(self.path)

    def validate_data(self, df):
        required = ["Company","MarketCap","Growth","Risk","RSI","Volatility"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df

    def clean(self, df):
        df = df.drop_duplicates()
        df = df.fillna(0)
        return df

    def feature_engineering(self, df):
        df["Momentum"] = df["RSI"] - 50
        df["Risk_Intensity"] = df["Risk"] * df["Volatility"]
        return df

    def scale_features(self, df, cols):
        df = df.copy()
        for c in cols:
            df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min() + 1e-9)
        return df