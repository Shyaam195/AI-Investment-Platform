class FeatureEngine:

    def add_indicators(self, df):

        # RSI (improved)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()

        # Volatility
        df["Volatility"] = df["Close"].pct_change().rolling(10).std()

        # Momentum
        df["Momentum"] = df["Close"] - df["Close"].shift(5)

        # Bollinger Bands
        df["BB_Mid"] = df["Close"].rolling(20).mean()
        df["BB_Upper"] = df["BB_Mid"] + 2 * df["Close"].rolling(20).std()
        df["BB_Lower"] = df["BB_Mid"] - 2 * df["Close"].rolling(20).std()

        return df