import yfinance as yf

def get_stock_data(ticker):

    stock = yf.Ticker(ticker)

    info = stock.info

    history = stock.history(period="6mo")

    historical_prices = {
        "dates": history.index.strftime("%Y-%m-%d").tolist(),
        "prices": history["Close"].fillna(0).tolist()
    }

    return {
        "company": info.get("longName"),
        "current_price": info.get("currentPrice"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "sector": info.get("sector"),
        "historical_data": historical_prices
    }
    