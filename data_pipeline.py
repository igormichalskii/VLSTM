import yfinance as yf
import numpy as np
import pandas as pd

def fetch_and_clean_data(symbol="SPY", vix_symbol="^VIX"):
    print(f"1. Fetching {symbol}, and {vix_symbol} data...")

    # We now pull Volume alongside the Close price
    spy = yf.download(symbol, start="2010-01-01")[['Close']]
    spy.rename(columns={'Close': 'close'}, inplace=True)

    vix = yf.download(vix_symbol, start="2010-01-01")[['Close']]
    vix.rename(columns={'Close': 'vix_close'}, inplace=True)

    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.droplevel(1)

    df = spy.join(vix, how='inner')

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)

    clean_df = df.dropna()

    output_file = f"{symbol}_VIX_daily_clean.parquet"
    clean_df.to_parquet(output_file)
    print(f"Dataset secured | Shape: {clean_df.shape}. Saved to {output_file}")
    return clean_df

if __name__ == "__main__":
    fetch_and_clean_data()