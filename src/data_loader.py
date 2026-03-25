import yfinance as yf
import pandas as pd

def load_data(tickers: list, start_date: str, end_date: str = None) -> pd.DataFrame:
    """Downloads and cleans daily prices for a cross-asset universe."""
    print(f"Downloading data for {tickers} from {start_date}...")
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        prices = raw_data['Close']
        prices = prices.ffill().dropna()
        print(f"Data successfully loaded. Final shape: {prices.shape}")
        return prices
    except Exception as e:
        print(f"Critical Error loading data from yfinance: {e}")
        raise