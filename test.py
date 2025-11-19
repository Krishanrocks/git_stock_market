import yfinance as yf
data = yf.download("TITAN.NS", period="1mo")
print(data.tail())
