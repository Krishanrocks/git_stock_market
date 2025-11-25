# app.py — Full Streamlit app with local company_info JSON (Clean Card Style overview)
import os
import re
import json
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import requests

# ---------------- MODULE IMPORTS ----------------
from modules.stock_lists import nifty50, other50, all_stocks
from modules.data_manager import load_local_stock, download_and_save_all_stocks
from modules.indicator import add_indicators
from modules.ai_insights import generate_ai_insights
from modules.predictor import run_stock_prediction_parallel, PREDICTIONS_DIR

# NEW MODULES
from modules.trend_analysis import trend_strength_score, trend_vs_prediction_summary
from modules.sentiment_engine import compute_sentiment
from modules.explanation_engine import explanation_from_indicators

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Indian Stock AI Dashboard", layout="wide")
st.title("Indian Stock Market AI Dashboard")
st.markdown("Local Data Source: **Parquet Cache (Yahoo Finance)**")

running_cloud = "STREAMLIT_RUNTIME" in os.environ

# ---------------- SIDEBAR ----------------
st.sidebar.header("Dashboard Options")
view_option = st.sidebar.radio(
    "Choose View:",
    ["Top Gainers/Losers", "Stock Details", "Predict Top 10 Stocks"]
)

group_option = st.sidebar.selectbox(
    "Select Stock Group",
    ["Nifty 50", "Other NSE 50", "All NSE Stocks"],
    index=2
)

stock_list = (
    nifty50 if group_option == "Nifty 50"
    else other50 if group_option == "Other NSE 50"
    else all_stocks
)

# ---------------- REFRESH LOGIC ----------------
running_cloud = "STREAMLIT_RUNTIME" in os.environ

if running_cloud:
    # On Streamlit Cloud: do not show refresh button
    st.sidebar.info("🔒 Data refresh is not available on this deployment.")
else:
    # Local machine: show refresh with confirmation
    if st.sidebar.button("🔄 Refresh All Data"):
        confirm = st.sidebar.radio(
            "⚠️ Are you sure you want to refresh all stock data? This may take several minutes.",
            ["No, cancel", "Yes, refresh now"],
            index=0
        )
        if confirm == "Yes, refresh now":
            with st.spinner("🔄 Downloading latest data..."):
                download_and_save_all_stocks(all_stocks)
            st.sidebar.success("✔ Data refreshed successfully!")
        else:
            st.sidebar.warning("❌ Refresh cancelled.")


# ---------------- LOCAL COMPANY INFO LOADER ----------------
# Look for possible filenames (preferred order)
POSSIBLE_INFO_PATHS = [
    "data/company_info.json",
    "data/company_info_full.json",
    "company_info.json",
    "/mnt/data/company_info_full.json",
    "/mnt/data/company_info.json"
]

INFO_FILE = None
for p in POSSIBLE_INFO_PATHS:
    if os.path.exists(p):
        INFO_FILE = p
        break

def load_company_info(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

if INFO_FILE:
    company_cache = load_company_info(INFO_FILE)
else:
    # no local file found — keep empty cache
    company_cache = {}

def save_company_info_local(data, path="data/company_info.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------- UTIL FUNCTIONS ----------------
@st.cache_data
def compute_returns(stock_list, period_days):
    data = []
    for ticker in stock_list:
        df = load_local_stock(ticker)
        if df.empty:
            continue
        df = df.sort_values("Date")
        if len(df) < period_days:
            continue
        first, last = df["Close"].iloc[-period_days], df["Close"].iloc[-1]
        data.append({"Ticker": ticker, "Return (%)": (last - first) / first * 100})
    return pd.DataFrame(data)


def safe_info(ticker):
    """
    Primary behavior: use local company_cache (company_info.json).
    If not present, attempt NSE API fetch (best-effort) and then fallback to yfinance.info.
    The returned dict will contain keys similar to yf `.info` for compatibility:
      - 'sector', 'industry', 'longBusinessSummary'
    """
    # 1) Local cache
    if ticker in company_cache:
        entry = company_cache[ticker]
        # normalize keys used by your app:
        return {
            "sector": entry.get("sector", "-"),
            "industry": entry.get("industry", "-"),
            "longBusinessSummary": entry.get("summary", entry.get("longBusinessSummary", ""))
        }

    # 2) Try NSE API (only if not running in strict cloud without access)
    try:
        symbol = ticker.replace(".NS", "")
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/"
        }
        session = requests.Session()
        resp = session.get(url, headers=headers, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            info = {
                "sector": data.get("info", {}).get("sector", "-"),
                "industry": data.get("metadata", {}).get("industry", "-"),
                "longBusinessSummary": data.get("securityInfo", {}).get("industryInfo", "") or ""
            }
            # add to local cache for future runs (attempt to save)
            try:
                company_cache[ticker] = {
                    "sector": info["sector"],
                    "industry": info["industry"],
                    "summary": info["longBusinessSummary"]
                }
                # try save to default path
                save_company_info_local(company_cache, path="data/company_info.json")
            except Exception:
                pass
            return info
    except Exception:
        # NSE likely blocked — ignore silently and try next fallback
        pass

    # 3) Fallback to yfinance.info (may be empty on cloud but useful locally)
    try:
        yinfo = yf.Ticker(ticker).info or {}
        return {
            "sector": yinfo.get("sector", "-"),
            "industry": yinfo.get("industry", "-"),
            "longBusinessSummary": yinfo.get("longBusinessSummary", "")
        }
    except Exception:
        return {"sector": "-", "industry": "-", "longBusinessSummary": ""}


# ---------------- UI: Top Gainers / Losers ----------------
if view_option == "Top Gainers/Losers":
    st.subheader("Top Gainers & Losers")

    period_option = st.sidebar.selectbox(
        "Select Time Period",
        ["1 Month", "6 Months", "1 Year", "5 Years"], index=2
    )
    days_map = {"1 Month": 21, "6 Months": 126, "1 Year": 252, "5 Years": 1260}

    df = compute_returns(stock_list, days_map[period_option])

    if df.empty:
        st.warning("⚠ No data available.")
    else:
        top_g = df.sort_values("Return (%)", ascending=False).head(10)
        top_l = df.sort_values("Return (%)").head(10)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###  Top 10 Gainers")
            st.dataframe(top_g)
            st.plotly_chart(px.bar(top_g, x="Ticker", y="Return (%)"), use_container_width=True)

        with col2:
            st.markdown("###  Top 10 Losers")
            st.dataframe(top_l)
            st.plotly_chart(px.bar(top_l, x="Ticker", y="Return (%)"), use_container_width=True)


# ---------------- UI: Stock Details ----------------
elif view_option == "Stock Details":
    st.subheader("Stock Details")

    selected_stock = st.sidebar.selectbox("Select Stock", stock_list)
    default_start = pd.Timestamp.today() - pd.DateOffset(years=5)
    start_date = st.sidebar.date_input("Start Date", default_start)
    end_date = st.sidebar.date_input("End Date", pd.Timestamp.today())

    df = load_local_stock(selected_stock, start=start_date, end=end_date)

    if df.empty:
        st.warning(f"No data found for {selected_stock}.")
    else:
        df = df.sort_values("Date")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📘 Overview", "📈 Price Chart", "🕯 Candlestick", "📊 Technical Indicators"]
        )

        # ----- OVERVIEW (Clean Card Style) -----
        with tab1:
            st.write(f"###  {selected_stock} - Company Overview")

            info = safe_info(selected_stock)
            summary_text = info.get("longBusinessSummary", "") or info.get("summary", "")

            # Card HTML (clean boxed style)
            card_html = """
            <div style="
                border-radius:12px;
                border:1px solid #e6e6e6;
                padding:18px;
                background:linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            ">
            <div style="font-size:16px;font-weight:600;margin-bottom:8px;">📘 Company Summary</div>
            <div style="font-size:14px;color:#111;padding-left:6px;">
            {bullets}
            </div>
            <hr style="margin:12px 0;">
            <div style="display:flex;gap:18px;align-items:center;">
                <div style="font-size:14px;"><b>🏭 Sector:</b> {sector}</div>
                <div style="font-size:14px;"><b>🏷 Industry:</b> {industry}</div>
            </div>
            </div>
            """

            # format bullets: split by sentence endings and show as <ul>
            if summary_text and len(summary_text.strip()) > 5:
                # split into sentences more robustly
                sentences = re.split(r'(?<=[.!?])\s+', summary_text.strip())
                # filter short fragments
                sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
                bullets_html = "<ul style='margin:6px 0 6px 18px;padding:0;'>"
                for s in sentences:
                    # remove newlines and trailing spaces
                    clean = s.replace("\n", " ").strip()
                    bullets_html += f"<li style='margin:6px 0;'>{clean}</li>"
                bullets_html += "</ul>"
            else:
                bullets_html = "<div style='color:#666'>No detailed summary available.</div>"

            rendered = card_html.format(
                bullets=bullets_html,
                sector=info.get("sector", "-"),
                industry=info.get("industry", "-")
            )

            st.markdown(rendered, unsafe_allow_html=True)

        # ----- PRICE CHART -----
        with tab2:
            st.markdown("### 📈 Historical Price Chart")
            st.plotly_chart(px.line(df, x="Date", y="Close"), use_container_width=True)
            st.info(
                """
### 📘 What this chart shows:
- Daily closing price movement  
- Helps identify long-term uptrend/downtrend  
- Good for spotting momentum shifts  
"""
            )

        # ----- CANDLESTICK -----
        with tab3:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                    )
                ]
            )
            fig.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            st.info(
                """
### 📙 Candlestick Interpretation:
- Green candle = buyers stronger  
- Red candle = sellers stronger  
- Long wicks = price rejection  
- Useful for daily sentiment detection  
"""
            )

        # ----- TECHNICAL INDICATORS -----
        with tab4:
            df = add_indicators(df)
            st.markdown("###  AI Technical Insights")

            insights = generate_ai_insights(df)

            if insights:
                cols = st.columns(3)
                for i, ins in enumerate(insights):
                    color = (
                        "green"
                        if any(w in ins.lower() for w in ["bullish", "upward", "positive", "rebound"])
                        else "red"
                        if any(w in ins.lower() for w in ["bearish", "downward", "weakness", "pullback"])
                        else "blue"
                    )
                    with cols[i % 3]:
                        st.markdown(
                            f"""
                            <div style="
                                background:{color};
                                padding:12px;border-radius:10px;color:white;margin:5px;">
                                {ins}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # MA CHART
            if {"MA5", "MA10"}.issubset(df.columns):
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
                fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA5"], name="MA5"))
                fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA10"], name="MA10"))
                st.plotly_chart(fig_ma, use_container_width=True)
                st.info(
                    """
### 📘 Moving Averages (MA5 / MA10)
- MA5 > MA10 → short-term bullish  
- MA5 < MA10 → short-term bearish  
"""
                )

            # RSI CHART
            if "RSI" in df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
                fig_rsi.add_hline(y=70, line_dash="dash")
                fig_rsi.add_hline(y=30, line_dash="dash")
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.info(
                    """
### 📘 RSI Interpretation:
- RSI > 70 → Overbought  
- RSI < 30 → Oversold  
"""
                )


# ---------------- UI: Predict Top 10 Stocks ----------------
elif view_option == "Predict Top 10 Stocks":
    st.subheader("Predict Top 10 Stocks to Buy / Avoid")

    model_choice = st.sidebar.selectbox(
        "Prediction Model", ["1 Month Model", "6 Month Model", "1 Year Model"], index=2
    )

    stock_key = {"Nifty 50": "Nifty50", "Other NSE 50": "Other50", "All NSE Stocks": "NSE"}[group_option]
    time_key = {"1 Month Model": "1m", "6 Month Model": "6m", "1 Year Model": "1y"}[model_choice]

    csv_path = f"{PREDICTIONS_DIR}/{stock_key}_{time_key}.csv"

    if os.path.exists(csv_path):
        dfp = pd.read_csv(csv_path)
        st.success("Loaded  predictions.")
    else:
        model_map = {
            "1 Month Model": "models/model_1m.pkl",
            "6 Month Model": "models/model_6m.pkl",
            "1 Year Model": "models/model_6m.pkl",
        }
        model = load(model_map[model_choice])

        dfp_dict, out = run_stock_prediction_parallel(stock_list, model, stock_key, time_key)
        # run_stock_prediction_parallel returns dict with keys 'buys','holds','sells' (updated earlier)
        # but older code expected 'not_buys' — handle both gracefully
        if isinstance(dfp_dict, dict):
            if "buys" in dfp_dict and "not_buys" in dfp_dict:
                dfp = pd.concat([dfp_dict["buys"], dfp_dict["not_buys"]])
            elif "buys" in dfp_dict and "sells" in dfp_dict:
                dfp = pd.concat([dfp_dict.get("buys", pd.DataFrame()), dfp_dict.get("sells", pd.DataFrame())])
            else:
                # fallback: try to read returned path
                dfp = pd.DataFrame()
        else:
            dfp = pd.DataFrame()

        st.success(f"Prediction complete → {out}")

    # ensure dfp exists and has expected columns
    if "dfp" not in locals() or dfp is None or dfp.empty:
        st.warning("No prediction results to display.")
    else:
        buys = dfp[dfp["Decision"] == "Buy"].head(10)
        avoids = dfp[dfp["Decision"] != "Buy"].head(10)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Top 10 Buys")
            st.dataframe(buys)

        with col2:
            st.markdown("### ❌ Top 10 Avoids")
            st.dataframe(avoids)

        # -------- Detailed Cards --------
        st.markdown("## 📋 Detailed Prediction Analysis")
        cols = st.columns(2)

        for idx, row in buys.reset_index(drop=True).iterrows():
            ticker = row["Ticker"]
            df_local = add_indicators(load_local_stock(ticker))
            score = trend_strength_score(df_local)
            info = safe_info(ticker)
            sentiment = compute_sentiment(info.get("longBusinessSummary", ""))
            reasons = explanation_from_indicators(df_local)
            summary = trend_vs_prediction_summary(df_local, row["Confidence (%)"])

            card = f"""
            <div style='border:1px solid #ccc;padding:12px;border-radius:10px;
            margin:6px;background:#e8f5e9;'>
            <b>{idx+1}. {ticker}</b><br>
            Confidence: {row['Confidence (%)']}%<br>
            Trend Score: {score}/100<br>
            Sentiment: {sentiment}<br><br>
            <b>Trend vs ML Prediction:</b><br>{summary}<br><br>
            <b>Reasons:</b>
            <ul>{"".join(f"<li>{r}</li>" for r in reasons)}</ul>
            </div>
            """

            cols[idx % 2].markdown(card, unsafe_allow_html=True)