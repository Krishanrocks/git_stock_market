# modules/trend_analysis.py

def trend_strength_score(df):
    if df is None or df.empty:
        return 0
    score = 50

    try:
        last = df.iloc[-1]

        # MA Trend
        if "MA5" in df.columns and "MA10" in df.columns and "MA20" in df.columns:
            if last["MA5"] > last["MA10"] > last["MA20"]:
                score += 20
            elif last["MA5"] < last["MA10"] < last["MA20"]:
                score -= 20

        # RSI
        if "RSI" in df.columns:
            r = last["RSI"]
            if r > 60: score += 10
            elif r < 40: score -= 10

        # MACD
        if "MACD" in df.columns and "Signal" in df.columns:
            if last["MACD"] > last["Signal"]:
                score += 10
            else:
                score -= 10

        # Momentum
        if len(df) > 10:
            mom = (df["Close"].iloc[-1] - df["Close"].iloc[-10]) / df["Close"].iloc[-10]
            if mom > 0.03: score += 10
            elif mom < -0.03: score -= 10

    except:
        pass

    return max(0, min(100, score))


def trend_vs_prediction_summary(df, confidence):
    if df is None or df.empty:
        return "No data available."

    trend_signals = []

    # RSI
    if "RSI" in df.columns:
        r = df["RSI"].iloc[-1]
        if r > 60: trend_signals.append("bullish")
        elif r < 40: trend_signals.append("bearish")
        else: trend_signals.append("neutral")

    # MACD
    if "MACD" in df.columns and "Signal" in df.columns:
        if df["MACD"].iloc[-1] > df["Signal"].iloc[-1]:
            trend_signals.append("bullish")
        else:
            trend_signals.append("bearish")

    # MA
    if "MA5" in df.columns and "MA10" in df.columns:
        if df["MA5"].iloc[-1] > df["MA10"].iloc[-1]:
            trend_signals.append("bullish")
        else:
            trend_signals.append("bearish")

    bull = trend_signals.count("bullish")
    bear = trend_signals.count("bearish")

    if bull > bear:
        current = "Bullish"
    elif bear > bull:
        current = "Bearish"
    else:
        current = "Neutral"

    # ML forecast
    if confidence > 80:
        forecast = "Strong Bullish Reversal Expected"
    elif confidence > 60:
        forecast = "Moderately Bullish Outlook"
    elif confidence > 50:
        forecast = "Mild Bullish Probability"
    else:
        forecast = "No Strong Bullish Signal"

    return f"Current Trend: {current}<br>ML Forecast: {forecast}"
