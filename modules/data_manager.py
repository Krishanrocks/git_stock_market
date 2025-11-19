import os
import time
import random
import yfinance as yf
import pandas as pd
import numpy as np
from modules.indicator import add_indicators
from modules.stock_lists import nifty50, other50, all_stocks

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def fix_yf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize YFinance DataFrame columns."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col if c not in [None, '']]).lower() for col in df.columns]
    else:
        df.columns = [c.lower().strip() for c in df.columns]
    df.columns = [c.split("_")[0] if "_" in c else c for c in df.columns]
    rename_map = {
        "adjclose": "Adj Close", "adj_close": "Adj Close",
        "closeprice": "Close", "closingprice": "Close",
        "close": "Close", "open": "Open", "high": "High",
        "low": "Low", "volume": "Volume"
    }
    df.rename(columns=rename_map, inplace=True)
    df.columns = [c.capitalize() for c in df.columns]
    return df


def download_and_save_all_stocks(stock_list, period="10y", max_retries=3):
    """Download all stocks and save as parquet with indicators."""
    for ticker in stock_list:
        success = False
        for attempt in range(max_retries):
            try:
                print(f"📥 Downloading {ticker} (Attempt {attempt + 1}/{max_retries})...")
                df = yf.download(
                    ticker, period=period,
                    progress=False, auto_adjust=False,
                    repair=True, actions=False, threads=False
                )

                if df.empty:
                    print(f"⚠️ Skipping {ticker}: No valid data.")
                    break

                df = fix_yf_dataframe(df)
                df.reset_index(inplace=True)

                # Ensure mandatory columns exist
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan

                # ✅ ADD ALL INDICATORS
                df = add_indicators(df)

                file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
                df.to_parquet(file_path, index=False)
                print(f"✅ Saved {ticker} → {file_path}")
                success = True
                break

            except Exception as e:
                print(f"❌ Failed for {ticker} (Attempt {attempt + 1}): {e}")
                wait = random.uniform(3, 6) * (attempt + 1)
                print(f"⏳ Waiting {wait:.1f}s before retry...")
                time.sleep(wait)

        if not success:
            print(f"🚫 Giving up on {ticker} after {max_retries} attempts.")

        # Short delay to avoid hitting Yahoo API limits
        time.sleep(random.uniform(1.5, 3.5))

    print("🎉 All stock data saved successfully.")


def load_local_stock(ticker, start=None, end=None):
    """Load local parquet data for a stock."""
    file_path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(file_path)
    except Exception:
        return pd.DataFrame()

    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not expected_cols.issubset(set(df.columns)):
        df = fix_yf_dataframe(df)
        df.reset_index(inplace=True)
        df.to_parquet(file_path, index=False)

    if start:
        df = df[df["Date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["Date"] <= pd.to_datetime(end)]
    return df


# 🚀 Run bulk download for all groups manually
if __name__ == "__main__":
    print("🚀 Starting full NSE download (Nifty50 + Other50 + All NSE)...")
    combined_list = list(set(all_stocks + nifty50 + other50))
    download_and_save_all_stocks(combined_list, period="10y")
