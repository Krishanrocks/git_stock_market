# modules/data_refresher.py
import os
import time
import random
import yfinance as yf
from datetime import datetime, timedelta
from modules.stock_lists import all_stocks
from modules.indicator import add_indicators  # <- added import
import pandas as pd
import numpy as np

DATA_DIR = "data"


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


def refresh_all_data(days_back=2000, max_retries=3):
    """Download or refresh all NSE stock data with indicators."""
    os.makedirs(DATA_DIR, exist_ok=True)
    start = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    print(f"📥 Downloading {len(all_stocks)} stocks from Yahoo Finance ({start} → {end})...")

    for i, ticker in enumerate(all_stocks, 1):
        path = f"{DATA_DIR}/{ticker}.parquet"

        # Skip existing files
        if os.path.exists(path):
            print(f"⏩ [{i}] Skipped (already exists): {ticker}")
            continue

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                print(f"📊 [{i}] Downloading {ticker} (Attempt {attempt}/{max_retries})...")
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    repair=True,
                    actions=False,
                    threads=False
                )

                if df.empty:
                    print(f"⚠️ [{i}] No data for {ticker}")
                    break

                df = fix_yf_dataframe(df)
                df.reset_index(inplace=True)

                # Ensure mandatory columns exist
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col not in df.columns:
                        df[col] = np.nan

                # ✅ Add all technical indicators
                df = add_indicators(df)

                df.to_parquet(path, index=False)
                print(f"✅ [{i}] Saved: {ticker}")
                success = True
                break

            except Exception as e:
                print(f"❌ [{i}] Failed {ticker} (Attempt {attempt}): {e}")
                wait = random.uniform(3, 6) * attempt
                print(f"⏳ Waiting {wait:.1f}s before retry...")
                time.sleep(wait)

        if not success:
            print(f"🚫 [{i}] Giving up on {ticker} after {max_retries} attempts.")

        delay = random.uniform(1.5, 3.5)
        print(f"🕒 Cooling down for {delay:.2f}s...")
        time.sleep(delay)

    print("\n🎯 Data refresh complete.")


if __name__ == "__main__":
    refresh_all_data()
