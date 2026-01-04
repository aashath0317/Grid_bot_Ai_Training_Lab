import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # First value is simple average
    # Subsequent values use Wilder's smoothing (alpha=1/period)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return pd.DataFrame({"upper": upper_band, "middle": sma, "lower": lower_band})


def calculate_volume_moving_average(volume_series: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Volume Moving Average.
    """
    return volume_series.rolling(window=period).mean()


def detect_volume_spike(volume_series: pd.Series, period: int = 20, threshold: float = 2.0) -> pd.Series:
    """
    Detects if current volume is 'threshold' times larger than the average.
    """
    vma = calculate_volume_moving_average(volume_series, period)
    return volume_series > (vma * threshold)


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    """
    return series.rolling(window=period).mean()


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns DataFrame with 'macd', 'signal', 'hist'.
    """
    exp1 = calculate_ema(series, period=fast)
    exp2 = calculate_ema(series, period=slow)
    macd = exp1 - exp2
    signal_line = calculate_ema(macd, period=signal)
    hist = macd - signal_line
    return pd.DataFrame({"macd": macd, "signal": signal_line, "hist": hist})


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate TR (True Range) - already done in ATR but needed for DM
    tr = calculate_atr(df, period) * period  # ATR is smoothed TR, so maybe recalculate raw TR?
    # Better to stick to standard formula

    # True Range
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    pos_dm = pd.Series(pos_dm, index=df.index)
    neg_dm = pd.Series(neg_dm, index=df.index)

    # Smooth
    tr_smooth = tr.ewm(alpha=1 / period, adjust=False).mean()
    pos_dm_smooth = pos_dm.ewm(alpha=1 / period, adjust=False).mean()
    neg_dm_smooth = neg_dm.ewm(alpha=1 / period, adjust=False).mean()

    # DI (+DI and -DI)
    pos_di = 100 * (pos_dm_smooth / tr_smooth)
    neg_di = 100 * (neg_dm_smooth / tr_smooth)

    # DX
    dx = 100 * (abs(pos_di - neg_di) / (pos_di + neg_di))

    # ADX
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    # ADX
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def calculate_ppo(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate Percentage Price Oscillator (PPO).
    Normalized version of MACD: (FastEMA - SlowEMA) / SlowEMA * 100
    Better for comparing assets with different prices or long histories.
    """
    exp1 = calculate_ema(series, period=fast)
    exp2 = calculate_ema(series, period=slow)

    # Avoid division by zero
    try:
        ppo = ((exp1 - exp2) / exp2) * 100
    except Exception:
        ppo = pd.Series(0, index=series.index)

    signal_line = calculate_ema(ppo, period=signal)
    hist = ppo - signal_line
    return pd.DataFrame({"ppo": ppo, "ppo_signal": signal_line, "ppo_hist": hist})
