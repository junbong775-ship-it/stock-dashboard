import pandas as pd


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta    = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float('nan'))
    rsi_series = 100 - 100 / (1 + rs)
    val = rsi_series.dropna()
    return float(val.iloc[-1]) if not val.empty else 50.0


def calc_macd(close: pd.Series,
              fast: int = 12,
              slow: int = 26,
              signal: int = 9) -> dict[str, pd.Series]:
    ema_fast    = close.ewm(span=fast, adjust=False).mean()
    ema_slow    = close.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return {'macd': macd_line, 'signal': signal_line, 'hist': histogram}


def calc_mfi(hist: pd.DataFrame, period: int = 14) -> float:
    tp      = (hist['High'] + hist['Low'] + hist['Close']) / 3
    mf      = tp * hist['Volume']
    pos_mf  = mf.where(tp > tp.shift(1), 0)
    neg_mf  = mf.where(tp < tp.shift(1), 0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfi     = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, float('nan'))))
    val = mfi.dropna()
    return float(val.iloc[-1]) if not val.empty else 50.0


def add_indicators(data: dict) -> dict:
    hist = data['hist']
    data = dict(data)
    data['rsi']  = calc_rsi(hist['Close'])
    data['mfi']  = (calc_mfi(hist)
                    if 'Volume' in hist.columns and hist['Volume'].sum() > 0
                    else 50.0)
    data['macd'] = calc_macd(hist['Close'])
    return data
