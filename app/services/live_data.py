import yfinance as yf

def get_stock_data(symbol):

    stock = yf.Ticker(symbol)

    info = stock.info

    return {
        "Company": info.get("shortName"),
        "CurrentPrice": info.get("currentPrice"),
        "MarketCap": info.get("marketCap")
    }