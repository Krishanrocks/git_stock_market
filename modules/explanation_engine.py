# modules/explanation_engine.py
import numpy as np

def explanation_from_indicators(df):
    if df is None or df.empty:
        return ["Not enough data"]

    last = df.iloc[-1]
    prev5 = df.iloc[-5:] if len(df) > 5 else df

    reasons = []

    # RSI
    if "RSI" in df.columns:
        r = last["RSI"]
        if r > 70: reasons.append("RSI overbought → possible pullback")
        elif r < 30: reasons.append("RSI oversold → rebound possible")
        else: reasons.append("RSI neutral")

    # MA
    if "MA5" in df.columns and "MA10" in df.columns:
        if last["MA5"] > last["MA10"]:
            reasons.append("MA5 above MA10 → bullish crossover")
        else:
            reasons.append("MA5 below MA10 → bearish crossover")

    # MACD
    if "MACD" in df.columns and "Signal" in df.columns:
        if last["MACD"] > last["Signal"]:
            reasons.append("MACD positive → bullish momentum")
        else:
            reasons.append("MACD negative → bearish")

    # Volatility
    vol = np.std(prev5["Close"]) / np.mean(prev5["Close"])
    if vol > 0.05:
        reasons.append("High volatility")
    else:
        reasons.append("Low volatility")

    return reasons[:6]
