import yfinance as yf
import numpy as np
import pandas as pd

def fetch_and_clean_data(symbol: str="SPY", vix_symbol: str="^VIX", start_date="2010-01-01"):
    print(f"1. Fetching data for {symbol} and {vix_symbol}...")
    spy = yf.Ticker(symbol).history(start=start_date)[['Close']]
    spy.rename(columns={'Close': 'close'}, inplace=True)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()

    vix = yf.Ticker(symbol).history(start=start_date)[['Close']]
    vix.rename(columns={'Close': 'vix_close'}, inplace=True)
    vix.index = pd.to_datetime(spy.index).tz_localize(None).normalize()

    print("2. Merging datasets and calculating targets...")

    df = spy.join(vix, how='inner')

    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)

    clean_df = df.dropna()
    output_file = f"{symbol}_VIX_daily_clean.parquet"
    clean_df.to_parquet(output_file)

    print(f"3. Data scrubbed and saved to {output_file}. | Shape: {clean_df.shape}.")
    return clean_df

if __name__ == "__main__":
    fetch_and_clean_data()