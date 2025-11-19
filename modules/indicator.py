import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame):
    if df.empty:
        return df

    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)

    # ============================
    # Moving Averages
    # ============================
    df["MA5"]   = df["Close"].rolling(5).mean()
    df["MA10"]  = df["Close"].rolling(10).mean()
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    # EMA
    df["EMA20"]  = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"]  = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # ============================
    # RSI (14)
    # ============================
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))

    # ============================
    # MACD + Signal
    # ============================
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"]   = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df
