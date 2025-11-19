import numpy as np

def generate_ai_insights(df):
    """Generate AI-style insights from indicators and recent price action."""
    if df.empty or "Close" not in df.columns:
        return ["⚠️ Not enough data for analysis."]

    insights = []
    recent = df.iloc[-1]
    prev = df.iloc[-5:] if len(df) > 5 else df

    # RSI
    if "RSI" in df.columns:
        rsi = recent["RSI"]
        if rsi > 70:
            insights.append("RSI shows overbought condition — possible short-term pullback (bearish).")
        elif rsi < 30:
            insights.append("RSI indicates oversold zone — possible rebound (bullish).")
        else:
            insights.append("RSI remains neutral — steady market momentum (neutral).")

    # MA crossover
    if "MA5" in df.columns and "MA10" in df.columns:
        if recent["MA5"] > recent["MA10"]:
            insights.append("Short-term trend above long-term — bullish crossover detected.")
        else:
            insights.append("Short-term below long-term — bearish signal forming.")

    # Volatility
    vol = np.std(prev["Close"]) / np.mean(prev["Close"])
    if vol > 0.05:
        insights.append("High short-term volatility — unpredictable movements.")
    else:
        insights.append("Low volatility — stable trend continuation possible.")

    # Momentum
    if len(df) > 10:
        momentum = (recent["Close"] - df["Close"].iloc[-10]) / df["Close"].iloc[-10]
        if momentum > 0.05:
            insights.append("Upward momentum in recent sessions — buyers in control.")
        elif momentum < -0.05:
            insights.append("Downward momentum — selling pressure visible.")
        else:
            insights.append("Flat momentum — market indecisive.")

    # Long trend
    if len(df) > 30:
        trend = df["Close"].rolling(30).mean().iloc[-1] - df["Close"].rolling(30).mean().iloc[-5]
        if trend > 0:
            insights.append("📈 Long-term uptrend remains intact (bullish bias).")
        else:
            insights.append("📉 Long-term weakness detected (bearish bias).")

    # Volume
    if "Volume" in df.columns:
        avg_vol = df["Volume"].tail(20).mean()
        if recent["Volume"] > 1.5 * avg_vol:
            insights.append("Volume spike — strong trader interest.")
        elif recent["Volume"] < 0.5 * avg_vol:
            insights.append("Low volume — weak market participation.")

    return insights[:8]
