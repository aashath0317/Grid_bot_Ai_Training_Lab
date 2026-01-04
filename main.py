import argparse
import os

# Changed imports for standalone use
from download_data import get_historical_data
from model_trainer import ModelTrainer
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Train AI Model")
    parser.add_argument("--pair", type=str, default="SOL/USDT", help="Trading Pair")
    parser.add_argument("--interval", type=str, default="15m", help="Candle Interval (e.g., 1m, 5m, 15m, 1h)")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--start_date", type=str, help="Start Date (YYYY-MM-DD) for full history")
    parser.add_argument(
        "--label_threshold", type=float, default=0.02, help="Price rise % to qualify as Pump (default 0.02 = 2%)"
    )
    parser.add_argument(
        "--data_file", type=str, default=None, help="Path to data (default: data/{pair}_{interval}.csv)"
    )

    args = parser.parse_args()

    # Generate default filename if not provided
    if args.data_file is None:
        safe_pair = args.pair.replace("/", "_")
        args.data_file = f"data/{safe_pair}_{args.interval}.csv"

    # 1. Get Data
    if os.path.exists(args.data_file):
        print(f"Loading data from {args.data_file}...")
        df = pd.read_csv(args.data_file, index_col="timestamp", parse_dates=True)
    else:
        print("Data file not found. Downloading...")
        try:
            if args.start_date:
                start_str = args.start_date
                print(f"Downloading from {start_str}...")
            else:
                from datetime import datetime, timedelta

                start_date = datetime.now() - timedelta(days=args.days)
                start_str = start_date.strftime("%Y-%m-%d")
                print(f"Downloading last {args.days} days...")

            df = get_historical_data(args.pair, args.interval, start_str)

            # Save for future use
            os.makedirs(os.path.dirname(args.data_file), exist_ok=True)
            df.to_csv(args.data_file)
        except Exception as e:
            print(f"Failed to download data: {e}")
            return

    if df.empty:
        print("No data available to train.")
        return

    # 2. Train
    print("Initializing Trainer...")

    # Calculate lookahead for ~1 hour
    lookahead_map = {"1m": 60, "5m": 12, "15m": 4, "30m": 2, "1h": 1}
    lookahead = lookahead_map.get(args.interval, 4)  # Default to 4 if unknown
    print(f"Lookahead set to {lookahead} candles ({args.interval} interval)")

    trainer = ModelTrainer()
    trainer.train(df, label_threshold=args.label_threshold, lookahead=lookahead)


if __name__ == "__main__":
    main()
