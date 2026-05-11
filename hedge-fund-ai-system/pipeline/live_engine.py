import yfinance as yf

class LiveEngine:

    def get_price(self, symbol: str):
        try:
            df = yf.Ticker(symbol).history(period="1d", interval="1m")
            return float(df["Close"].iloc[-1]) if not df.empty else 0.0
        except Exception:
            return 0.0

    def get_multi_prices(self, symbols):
        return {s: self.get_price(s) for s in symbols}