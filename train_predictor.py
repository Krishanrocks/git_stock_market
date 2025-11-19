# train_predictor.py
import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from modules.stock_lists import all_stocks
from modules.data_manager import load_local_stock
from modules.indicator import add_indicators

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# FEATURES (MATCHED WITH predictor.py)
FEATURES = [
    "RSI", "MACD", "Signal",
    "MA5", "MA10", "MA20", "MA50", "MA100", "MA200",
    "EMA20", "EMA50", "EMA200",
    "Volatility", "Momentum",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Return", "Return_5D", "Return_10D", "Return_20D",
    "Volume_Ratio", "Position_6M"
]


# ---------------- BUILD DATASET ----------------
def build_dataset_from_local():
    all_df = []

    for symbol in all_stocks:
        df = load_local_stock(symbol)
        if df.empty:
            print(f"⚠️ No data for {symbol}")
            continue

        df = df.sort_values("Date")
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(10).std()
        df["Momentum"] = df["Close"] - df["Close"].shift(10)
        df["Momentum_5"] = df["Close"].diff(5)
        df["Momentum_10"] = df["Close"].diff(10)
        df["Momentum_20"] = df["Close"].diff(20)
        df["Return_5D"] = df["Close"].pct_change(5)
        df["Return_10D"] = df["Close"].pct_change(10)
        df["Return_20D"] = df["Close"].pct_change(20)
        df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["Position_6M"] = df["Close"] / df["Close"].rolling(120).mean()

        df = add_indicators(df)
        df["Stock"] = symbol
        df.dropna(inplace=True)

        all_df.append(df)
        print(f"Loaded {symbol} ({len(df)} rows)")

    if not all_df:
        raise ValueError("No valid local data found!")

    final_df = pd.concat(all_df, ignore_index=True)
    final_df.to_csv(f"{DATA_DIR}/all_stock_data.csv", index=False)
    print(f"Saved dataset → {DATA_DIR}/all_stock_data.csv ({len(final_df)} rows)")

    return final_df


# ---------------- TRAIN MODEL ----------------
def train_period_model(df, target_col, model_name):
    print(f"\nTraining model → {model_name} (target = {target_col})")

    needed_cols = FEATURES + [target_col]
    df2 = df.dropna(subset=[c for c in needed_cols if c in df.columns])

    X = df2[FEATURES].astype(float)
    y = df2[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(classification_report(y_test, preds, digits=3))

    path = f"{MODEL_DIR}/{model_name}.pkl"
    joblib.dump(model, path)
    print(f"Saved → {path}")


# ---------------- TRAIN ALL MODELS ----------------
def train_all_models():
    dataset_path = f"{DATA_DIR}/all_stock_data.csv"

    if os.path.exists(dataset_path):
        print("Using cached dataset...")
        df = pd.read_csv(dataset_path)
    else:
        print("Building dataset...")
        df = build_dataset_from_local()

    df = df.sort_values("Date")

    # ----- IMPROVED TARGET THRESHOLDS -----
    df["Future_1m"] = df["Close"].shift(-20) / df["Close"] - 1
    df["Future_3m"] = df["Close"].shift(-60) / df["Close"] - 1
    df["Future_6m"] = df["Close"].shift(-120) / df["Close"] - 1

    # More balanced thresholds
    df["Target_1m"] = (df["Future_1m"] > 0.01).astype(int)
    df["Target_3m"] = (df["Future_3m"] > 0.04).astype(int)
    df["Target_6m"] = (df["Future_6m"] > 0.08).astype(int)

    df.dropna(inplace=True)

    train_period_model(df, "Target_1m", "model_1m")
    train_period_model(df, "Target_3m", "model_3m")
    train_period_model(df, "Target_6m", "model_6m")

    print("ALL MODELS TRAINED SUCCESSFULLY")


if __name__ == "__main__":
    train_all_models()
