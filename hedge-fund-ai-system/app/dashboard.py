import streamlit as st
import streamlit as st

st.write("🚀 DASHBOARD LOADED SUCCESSFULLY")
st.write("DEBUG: Step 1 OK - imports loaded")
st.write("DEBUG: Step 2 OK - data pipeline starting")
st.write("DEBUG: Step 3 OK - AI engine running")

from pipeline.data_pipeline import DataPipeline
from pipeline.live_engine import LiveEngine
from pipeline.outlier_engine import OutlierEngine

from agents.risk_agent import analyze_risk
from agents.market_agent import market_trend
from agents.health_agent import company_health
from agents.recommendation_agent import get_recommendation

st.set_page_config(page_title="Hedge Fund v3", layout="wide")

# ---------------- LOAD DATA ----------------
pipeline = DataPipeline("app/data/companies.csv")
df = pipeline.load_data()
df = pipeline.validate_data(df)
df = pipeline.clean(df)

df = pipeline.feature_engineering(df)
df = pipeline.scale_features(df, ["Growth","Risk","RSI","Volatility"])

# ---------------- LIVE ENGINE ----------------
engine = LiveEngine()
stocks = ["AAPL","MSFT","GOOGL","TSLA"]

prices = engine.get_multi_prices(stocks)

# ---------------- AI FEATURES ----------------
df["AI_Score"] = df["Growth"] * 0.4 + (1 - df["Risk"]) * 0.3 + df["RSI"] * 0.3

df["Risk_Level"] = df.apply(lambda r: analyze_risk(r["Risk"], r["Volatility"]), axis=1)
df["Market_Trend"] = df["RSI"].apply(market_trend)
df["Health"] = df.apply(lambda r: company_health(r["Growth"], r["Risk"]), axis=1)
df["Recommendation"] = df["AI_Score"].apply(get_recommendation)

# ---------------- OUTLIERS ----------------
outlier_engine = OutlierEngine()
df = outlier_engine.detect(df)

# ---------------- UI ----------------
st.title("🏦 Hedge Fund AI System v3")

st.subheader("📊 Live Prices")
st.write(prices)

st.subheader("📊 Dataset")
st.dataframe(df)

st.subheader("⚠️ Outliers")
st.dataframe(df[df["Is_Outlier"]])

st.subheader("📈 AI Score")
st.bar_chart(df.set_index("Company")["AI_Score"])