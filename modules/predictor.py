# modules/predictor.py
import os
import pandas as pd
import numpy as np
import concurrent.futures
from joblib import load
from modules.data_manager import load_local_stock

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # modules/
PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "predictions")
PREDICTIONS_DIR = os.path.abspath(PREDICTIONS_DIR)

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ⚠️ FEATURES MATCH TRAINING EXACTLY
FEATURES = [
    "RSI", "MACD", "Signal",
    "MA5", "MA10", "MA20", "MA50", "MA100", "MA200",
    "EMA20", "EMA50", "EMA200",
    "Volatility", "Momentum",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Return", "Return_5D", "Return_10D", "Return_20D",
    "Volume_Ratio", "Position_6M",
]


# ---------------- COMPUTE FEATURES ----------------
def compute_features(df, window=14):
    if df is None or df.empty:
        return df

    df = df.copy()

    # === RETURNS ===
    df["Return"] = df["Close"].pct_change()
    df["Return_5D"] = df["Close"].pct_change(5)
    df["Return_10D"] = df["Close"].pct_change(10)
    df["Return_20D"] = df["Close"].pct_change(20)

    # === VOLATILITY ===
    df["Volatility"] = df["Return"].rolling(10).std()

    # === MOMENTUM ===
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    df["Momentum_5"] = df["Close"].diff(5)
    df["Momentum_10"] = df["Close"].diff(10)
    df["Momentum_20"] = df["Close"].diff(20)

    # === MOVING AVERAGES ===
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # === EMA ===
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # === RSI ===
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # === MACD / SIGNAL ===
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # === VOLUME RATIO ===
    df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

    # === POSITION ===
    df["Position_6M"] = df["Close"] / df["Close"].rolling(120).mean()

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# ---------------- PREDICT SINGLE STOCK ----------------
def predict_one_stock(ticker, model):
    try:
        df = load_local_stock(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None

        df = compute_features(df)
        df = df.dropna(subset=FEATURES)

        last = df.iloc[-1]
        X = last[FEATURES].values.reshape(1, -1)

        prob = model.predict_proba(X)[0, 1]

        # BETTER THRESHOLD
        pred = int(prob > 0.42)

        return {
            "Ticker": ticker,
            "Decision": "Buy" if pred == 1 else "Not Buy",
            "Confidence (%)": round(prob * 100, 2),
            "Close": round(last["Close"], 2),
        }
    except Exception as e:
        print(f"⚠️ {ticker} failed: {e}")
        return None


# ---------------- LOAD LOCAL PREDICTION ----------------
def load_local_prediction(stock_list_name, timeframe):
    file_name = f"{stock_list_name}_{timeframe}.csv"
    path = os.path.join(PREDICTIONS_DIR, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        buys = df[df["Decision"] == "Buy"].head(10)
        not_buys = df[df["Decision"] != "Buy"].head(10)
        return {"buys": buys, "not_buys": not_buys}, path
    return None, None


# ---------------- RUN PARALLEL PREDICTIONS ----------------
def run_stock_prediction_parallel(stocks, model, stock_list_name="NSE", timeframe="1y"):

    # 1. Try loading existing predictions
    local_res, local_path = load_local_prediction(stock_list_name, timeframe)
    if local_res:
        return local_res, local_path

    # 2. Run new predictions
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for res in executor.map(lambda s: predict_one_stock(s, model), stocks):
            if res:
                results.append(res)

    if not results:
        return {}, None

    df = pd.DataFrame(results)

    sorted_df = df.sort_values("Confidence (%)", ascending=False)

    buys = sorted_df.head(10).copy()
    buys["Decision"] = "Buy"

    not_buys = sorted_df.tail(10).copy()
    not_buys["Decision"] = "Not Buy"

    # Save file
    final_df = pd.concat([buys, not_buys], ignore_index=True)
    save_path = os.path.join(PREDICTIONS_DIR, f"{stock_list_name}_{timeframe}.csv")
    final_df.to_csv(save_path, index=False)

    return {"buys": buys, "not_buys": not_buys}, save_path
