import yfinance as yf
import pandas as pd

def load_data(tickers: list, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Downloads and cleans daily adjusted close prices for a list of tickers.
    """
    print(f"Downloading data for {tickers} from {start_date}...")
    
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # ==========================================
        # THE FIX: yfinance API Update
        # ==========================================
        # yfinance >= 0.2.51 automatically adjusts prices and removed 'Adj Close'. 
        # The 'Close' column now acts as our adjusted price.
        prices = raw_data['Close']
        
        # Forward fill missing values to prevent lookahead bias, then drop initial NaNs
        prices = prices.ffill().dropna()
        
        print(f"Data successfully loaded and cleaned. Final shape: {prices.shape}")
        return prices
        
    except Exception as e:
        print(f"Critical Error loading data from yfinance: {e}")
        raise