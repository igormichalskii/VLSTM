import yfinance as yf
import numpy as np
import pandas as pd

def fetch_and_clean_data(symbol="SPY", vix_symbol="^VIX"):
    print(f"1. Fetching {symbol}, and {vix_symbol} data...")

    # We now pull Volume alongside the Close price
    spy = yf.download(symbol, start="2014-01-01")[['Close', 'Volume']]
    spy.rename(columns={'Close': 'close', 'Volume': 'volume'}, inplace=True)

    vix = yf.download(vix_symbol, start="2014-01-01")[['Close']]
    vix.rename(columns={'Close': 'vix_close'}, inplace=True)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)

    df = spy.join(vix, how='inner')

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)
    df['vol_21d_sma'] = df['volume'].rolling(window=21).mean()
    df['volume_surge'] = df['volume'] / df['vol_21d_sma']
    df['price_21d_sma'] = df['close'].rolling(window=21).mean()
    df['bb_width'] = (4 * df['close'].rolling(window=21).std()) / df['price_21d_sma']

    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - macd_signal

    clean_df = df.dropna()
    features_to_keep = [
        'close',
        'log_return',
        'realized_vol',
        'vix_close',
        'volume_surge',
        'bb_width',
        'macd_hist'
    ]
    clean_df = clean_df[features_to_keep]

    output_file = f"{symbol}_VIX_daily_clean.parquet"
    clean_df.to_parquet(output_file)
    print(f"Dataset secured | Shape: {clean_df.shape}. Saved to {output_file}")
    return clean_df

if __name__ == "__main__":
    fetch_and_clean_data()