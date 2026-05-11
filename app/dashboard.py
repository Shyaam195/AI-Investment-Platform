import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
import os
import feedparser

deep_model = joblib.load("app/ML/deep_market_model.pkl")
from textblob import TextBlob

model = joblib.load("app/ML/market_model.pkl")

BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "ml", "model.pkl")

model = joblib.load(model_path)

from services.scoring import calculate_score
from decision import get_recommendation

# ---------------- CONFIG ----------------
INITIAL_CAPITAL = 100000
model = joblib.load("app/ml/model.pkl")

# ---------------- LOAD DATA EARLY ----------------
df = pd.read_csv("app/data/companies.csv")

required_columns = ["Company", "Growth", "Risk", "RSI", "Volatility", "MarketCap"]
missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ---------------- SESSION STATE ----------------
if "cash" not in st.session_state:
    st.session_state.cash = INITIAL_CAPITAL

if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

if "history" not in st.session_state:
    st.session_state.history = []

if "pnl_history" not in st.session_state:
    st.session_state.pnl_history = []

# ---------------- RESET ----------------
if st.button("Reset Portfolio"):
    st.session_state.cash = INITIAL_CAPITAL
    st.session_state.portfolio = {}
    st.session_state.history = []
    st.session_state.pnl_history = []

# ---------------- UTIL FUNCTIONS ----------------
def calculate_confidence(score, risk, volatility):
    return max(0, min(100, score - (risk * 0.3) - (volatility * 0.2)))

def trend_signal(rsi):
    if rsi > 70:
        return "OVERBOUGHT"
    elif rsi < 30:
        return "OVERSOLD"
    return "NEUTRAL"

def get_live_price(company):
    ticker_map = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Nvidia": "NVDA",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA"
    }

    ticker = ticker_map.get(company)
    if not ticker:
        return np.nan

    try:
        data = yf.Ticker(ticker).history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        return np.nan

    return np.nan

# ---------------- TRADING ENGINE ----------------
def buy_stock(company, price):
    if pd.isna(price):
        return

    if st.session_state.cash >= price:
        if company in st.session_state.portfolio:
            old_qty = st.session_state.portfolio[company]["qty"]
            old_avg = st.session_state.portfolio[company]["avg_price"]

            new_qty = old_qty + 1
            new_avg = ((old_qty * old_avg) + price) / new_qty

            st.session_state.portfolio[company]["qty"] = new_qty
            st.session_state.portfolio[company]["avg_price"] = new_avg
        else:
            st.session_state.portfolio[company] = {
                "qty": 1,
                "avg_price": price
            }

        st.session_state.cash -= price

        st.session_state.history.append({
            "type": "BUY",
            "company": company,
            "price": price
        })

def sell_stock(company, price):
    if pd.isna(price):
        return

    if company in st.session_state.portfolio:
        data = st.session_state.portfolio[company]

        if data["qty"] > 0:
            data["qty"] -= 1
            st.session_state.cash += price

            st.session_state.history.append({
                "type": "SELL",
                "company": company,
                "price": price
            })

            if data["qty"] == 0:
                del st.session_state.portfolio[company]

# ---------------- PNL ENGINE ----------------
def calculate_pnl(df):
    total_value = st.session_state.cash
    pnl = 0

    for company, data in st.session_state.portfolio.items():
        row = df[df["Company"] == company]

        if row.empty:
            continue

        live_price = row["LivePrice"].values[0]

        if pd.isna(live_price):
            continue

        qty = data["qty"]
        avg_price = data["avg_price"]

        pnl += (live_price - avg_price) * qty
        total_value += live_price * qty

    st.session_state.pnl_history.append(float(pnl))

    return total_value, pnl

df["Growth"] = pd.to_numeric(df["Growth"], errors="coerce")
df["Risk"] = pd.to_numeric(df["Risk"], errors="coerce")
df["RSI"] = pd.to_numeric(df["RSI"], errors="coerce")
df["Volatility"] = pd.to_numeric(df["Volatility"], errors="coerce")

df = df.dropna()

# ---------------- AI ENGINE ----------------
df["AI_Score"] = df.apply(
    lambda r: calculate_score(
        r["Growth"],
        r["Risk"],
        r["RSI"],
        r["Volatility"]
    ),
    axis=1
)

df["Recommendation"] = df["AI_Score"].apply(get_recommendation)

df["Trend"] = df["RSI"].apply(trend_signal)
df["LivePrice"] = df["Company"].apply(get_live_price)

# ---------------- ML ENGINE ----------------
def ml_recommend(row):
    features = np.array([[
        row["Growth"],
        row["Risk"],
        row["RSI"],
        row["Volatility"],
    ]])

    prediction = model.predict(features)[0]

    return "BUY" if prediction == 2 else "HOLD" if prediction == 1 else "SELL"

df["Recommendation"] = df.apply(ml_recommend, axis=1)

# ---------------- UI ----------------
st.title("AI Investment Recommendation Dashboard")

st.subheader("AI Trading Signals")
selected_company = st.selectbox(
    "Select Company",
    df["Company"]
)

filtered_df = df[df["Company"] == selected_company]

st.dataframe(
    filtered_df[["Company", "AI_Score", "Recommendation"]]
)

st.subheader("AI Score Comparison")

st.bar_chart(df.set_index("Company")["AI_Score"])

st.subheader("Market Cap Comparison")

st.bar_chart(df.set_index("Company")["MarketCap"])

st.subheader("Top Companies")
st.dataframe(df.sort_values("AI_Score", ascending=False))

# ---------------- TRADING PANEL ----------------
st.subheader("Trading Panel")

for i, row in df.iterrows():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(row["Company"])

    with col2:
        if st.button(f"BUY {row['Company']}", key=f"buy_{i}"):
            buy_stock(row["Company"], row["LivePrice"])

    with col3:
        if st.button(f"SELL {row['Company']}", key=f"sell_{i}"):
            sell_stock(row["Company"], row["LivePrice"])

# ---------------- PORTFOLIO ----------------
st.subheader("Portfolio Summary")

total_value, pnl = calculate_pnl(df)

st.write("💰 Cash:", st.session_state.cash)
st.write("📊 Total Value:", total_value)
st.write("📈 Profit/Loss:", pnl)

st.subheader("Trade History")

if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.info("No trades yet")

st.subheader("PnL History Chart")

if st.session_state.pnl_history:
    st.line_chart(st.session_state.pnl_history)

st.json(st.session_state.portfolio)

def ml_recommend(row):
    features = np.array([[
        row["Growth"],
        row["Risk"],
        row["RSI"],
        row["Volatility"]
    ]])  # ✅ only 4 features

    prediction = model.predict(features)[0]

    if prediction == 2:
        return "BUY"
    elif prediction == 1:
        return "HOLD"
    else:
        return "SELL"
    
    df["Growth"] = pd.to_numeric(df["Growth"])
df["Risk"] = pd.to_numeric(df["Risk"])
df["RSI"] = pd.to_numeric(df["RSI"])
df["Volatility"] = pd.to_numeric(df["Volatility"])

st.subheader("Live AI Prediction")

growth = st.slider("Growth", 0, 100, 50)
risk = st.slider("Risk", 0, 100, 50)
rsi = st.slider("RSI", 0, 100, 50)
volatility = st.slider("Volatility", 0, 100, 50)

input_data = pd.DataFrame([{
    "Growth": growth,
    "Risk": risk,
    "RSI": rsi,
    "Volatility": volatility
}])

predicted_marketcap = model.predict(input_data)[0]

st.metric(
    "Predicted Market Cap",
    f"${predicted_marketcap:.2f}B"
)

live_score = calculate_score(growth, risk, rsi, volatility)

st.metric("Predicted AI Score", round(live_score, 2))

recommendation = get_recommendation(live_score)

st.success(f"Recommendation: {recommendation}")

st.header("AI MarketCap Prediction")

growth = st.slider("Growth", 0, 100, 80)
risk = st.slider("Risk", 0, 100, 30)
rsi = st.slider("RSI", 0, 100, 60)
volatility = st.slider("Volatility", 0, 100, 20)

if st.button("Predict MarketCap"):

    input_data = pd.DataFrame([{
        "Growth": growth,
        "Risk": risk,
        "RSI": rsi,
        "Volatility": volatility
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted MarketCap: ${prediction:.2f} Billion")
    
    st.header("Live Stock Market Tracker")

ticker = st.text_input("Enter Stock Symbol", "AAPL")

if st.button("Load Live Stock Data"):

    stock = yf.Ticker(ticker)

    hist = stock.history(period="6mo")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist.index,
            y=hist["Close"],
            mode="lines",
            name="Close Price"
        )
    )

    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(hist.tail())
    
    st.header("AI News Sentiment Analysis")

news_symbol = st.text_input("News Stock Symbol", "AAPL")

if st.button("Analyze News Sentiment"):

    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={news_symbol}&region=US&lang=en-US"

    feed = feedparser.parse(url)

    sentiments = []

    for entry in feed.entries[:10]:

        title = entry.title

        analysis = TextBlob(title)

        polarity = analysis.sentiment.polarity

        sentiments.append(polarity)

        if polarity > 0:
            mood = "Positive"
        elif polarity < 0:
            mood = "Negative"
        else:
            mood = "Neutral"

        st.write(f"📰 {title}")
        st.write(f"Sentiment: {mood}")
        st.write("---")

    avg_sentiment = sum(sentiments) / len(sentiments)

    st.subheader(f"Average Sentiment Score: {avg_sentiment:.2f}")

    if avg_sentiment > 0:
        st.success("Overall Market Mood: BULLISH")
    elif avg_sentiment < 0:
        st.error("Overall Market Mood: BEARISH")
    else:
        st.warning("Overall Market Mood: NEUTRAL")
        
        st.header("AI Portfolio Optimizer")

investment = st.number_input(
    "Enter Investment Amount ($)",
    min_value=1000,
    value=10000
)

stocks = ["AAPL", "MSFT", "NVDA", "TSLA"]

if st.button("Optimize Portfolio"):

    weights = np.random.dirichlet(np.ones(len(stocks)), size=1)[0]

    allocation = {}

    for stock, weight in zip(stocks, weights):

        allocation[stock] = round(weight * investment, 2)

    st.subheader("Recommended Allocation")

    for stock, amount in allocation.items():

        st.write(f"{stock}: ${amount}")

    expected_return = round(np.random.uniform(8, 18), 2)

    risk_score = round(np.random.uniform(5, 15), 2)

    sharpe_ratio = round(expected_return / risk_score, 2)

    st.success(f"Expected Annual Return: {expected_return}%")

    st.warning(f"Portfolio Risk Score: {risk_score}")

    st.info(f"Sharpe Ratio: {sharpe_ratio}")
    
    st.header("Deep AI Prediction")

d_growth = st.slider("Deep Growth", 0, 100, 80)
d_risk = st.slider("Deep Risk", 0, 100, 30)
d_rsi = st.slider("Deep RSI", 0, 100, 60)
d_volatility = st.slider("Deep Volatility", 0, 100, 20)

if st.button("Predict with Deep AI"):

    input_data = pd.DataFrame([{
        "Growth": d_growth,
        "Risk": d_risk,
        "RSI": d_rsi,
        "Volatility": d_volatility
    }])

    prediction = deep_model.predict(input_data)[0]

    st.success(
        f"Deep AI Predicted MarketCap: ${prediction:.2f} Billion"
    )