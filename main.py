from fastapi import FastAPI
from app.services.market_data import get_stock_data
from app.services.scoring import calculate_score

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "AI Investment Platform Running Successfully"
    }

@app.get("/stock/{ticker}")
def stock_analysis(ticker: str):

    stock_data = get_stock_data(ticker)

    ai_result = calculate_score(stock_data)

    return {
        "company": stock_data.get("company"),
        "current_price": stock_data.get("current_price"),
        "market_cap": stock_data.get("market_cap"),
        "pe_ratio": stock_data.get("pe_ratio"),
        "sector": stock_data.get("sector"),
        "historical_data": stock_data.get("historical_data"),
        "ai_analysis": ai_result
    }
    
    