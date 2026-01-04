# AI Model Prediction Lab

This repository contains a standalone environment for training XGBoost models to predict crypto price spikes ("Pumps").

## Features
- **Data Acquisition**: Automatically downloads historical data from Binance.
- **Feature Engineering**: Calculates RSI, ATR, Bollinger Bands, PPO, ADX, and specialized Multi-Timeframe / Interaction features (e.g., Volatility * Momentum).
- **Model Training**: Uses XGBoost (GPU-accelerated) to classify high-volatility hikes.
- **Precision Optimization**: Includes tools for Threshold Tuning and Class Balancing.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have CUDA drivers installed if using GPU acceleration for XGBoost.*

## Usage

Run the training script with your desired pair and interval.

### Example: Training on SOL/USDT (5m interval)
```bash
python main.py --pair "SOL/USDT" --interval "5m" --start_date "2017-08-01" --label_threshold 0.015
```

- `--pair`: Trading pair (e.g., SOL/USDT, BTC/USDT)
- `--interval`: Candle timeframe (1m, 5m, 15m, 1h)
- `--start_date`: Start date for historical data (YYYY-MM-DD)
- `--label_threshold`: Percentage move required to be labeled a "Pump" (e.g., 0.015 = 1.5%)

## Output
The script outputs:
- **Class Balance**: Percentage of "Pump" candles vs Total.
- **Threshold Table**: Precision/Recall matrix for thresholds 0.50 - 0.95.
- **Feature Importance**: Which indicators are driving decisions.
- **Model File**: Saves `ai_model.joblib` if performance meets criteria.

## Files
- `main.py`: Entry point for downloading and training.
- `model_trainer.py`: Core logic for Feature Engineering and XGBoost training.
- `indicators.py`: Library of technical indicator calculations.
- `download_data.py`: Utility to fetch Binance klines.
