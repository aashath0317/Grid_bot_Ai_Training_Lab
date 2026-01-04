import argparse
from datetime import datetime, timedelta
import os
import time

import pandas as pd
import requests

# Constants
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"


def get_historical_data(symbol, interval, start_str, end_str=None):
    """
    Downloads historical klines from Binance.
    """
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000) if end_str else int(time.time() * 1000)

    data = []
    limit = 1000

    print(f"Downloading {symbol} ({interval}) from {start_str} to {end_str or 'now'}...")

    while start_ts < end_ts:
        params = {
            "symbol": symbol.replace("/", ""),
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit,
        }

        try:
            response = requests.get(BINANCE_API_URL, params=params)
            response.raise_for_status()
            klines = response.json()

            if not klines:
                break

            for k in klines:
                # Binance columns: Open Time, Open, High, Low, Close, Volume, ...
                data.append([k[0], float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])

            # Update start_ts to the last timestamp + 1ms to avoid duplicates
            last_ts = klines[-1][0]
            start_ts = last_ts + 1

            # Rate limit respect
            time.sleep(0.1)

            # Progress update
            current_date = pd.to_datetime(last_ts, unit="ms")
            print(f"Downloaded up to {current_date}...", end="\r")

        except Exception as e:
            print(f"Error downloading data: {e}")
            time.sleep(1)

    print(f"\nDownload complete! Total records: {len(data)}")

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Historical Data from Binance")
    parser.add_argument("--pair", type=str, default="SOL/USDT", help="Trading Pair (e.g. SOL/USDT)")
    parser.add_argument("--interval", type=str, default="15m", help="Candle Interval")
    parser.add_argument("--days", type=int, default=365, help="Days of history to download")
    parser.add_argument("--start_date", type=str, help="Start Date (YYYY-MM-DD). Overrides --days.")
    parser.add_argument("--output", type=str, default="data/historical_data.csv", help="Output file path")

    args = parser.parse_args()

    if args.start_date:
        start_str = args.start_date
    else:
        start_date = datetime.now() - timedelta(days=args.days)
        start_str = start_date.strftime("%Y-%m-%d")

    # Create data dir if not exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = get_historical_data(args.pair, args.interval, start_str)

    if not df.empty:
        df.to_csv(args.output)
        print(f"Data saved to {args.output}")
    else:
        print("No data found.")
