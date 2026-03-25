import pandas as pd
import numpy as np
import os

def calculate_momentum_features(prices: pd.DataFrame, windows: list = [21, 63, 126, 252]) -> pd.DataFrame:
    """
    Calculates rolling returns for roughly 1m, 3m, 6m, and 12m periods.
    Using trading days (252 days/year).
    """
    features = pd.DataFrame(index=prices.index)
    
    for w in windows:
        # Calculate percentage change over the rolling window
        rolling_ret = prices.pct_change(periods=w)
        rolling_ret.columns = [f"{col}_mom_{w}d" for col in prices.columns]
        features = pd.concat([features, rolling_ret], axis=1)
        
    return features

def calculate_risk_features(prices: pd.DataFrame, windows: list = [21, 63]) -> pd.DataFrame:
    """
    Calculates annualized rolling volatility, rolling 1-year drawdown, and rolling Sharpe.
    """
    daily_returns = prices.pct_change()
    features = pd.DataFrame(index=prices.index)
    
    # 1. Rolling Volatility (Annualized)
    for w in windows:
        # Standard deviation * square root of trading days
        vol = daily_returns.rolling(window=w).std() * np.sqrt(252)
        vol.columns = [f"{col}_vol_{w}d" for col in prices.columns]
        features = pd.concat([features, vol], axis=1)
        
    # 2. Rolling 1-Year Drawdown
    # Calculates how far the current price is from the 252-day high
    rolling_max = prices.rolling(window=252, min_periods=1).max()
    drawdown = (prices / rolling_max) - 1
    drawdown.columns = [f"{col}_dd_252d" for col in prices.columns]
    features = pd.concat([features, drawdown], axis=1)
    
    # 3. Rolling Sharpe Ratio (252-day annualized)
    rolling_mean_ret = daily_returns.rolling(window=252).mean() * 252
    rolling_vol_252 = daily_returns.rolling(window=252).std() * np.sqrt(252)
    sharpe = rolling_mean_ret / (rolling_vol_252 + 1e-6) # 1e-6 prevents division by zero
    sharpe.columns = [f"{col}_sharpe_252d" for col in prices.columns]
    features = pd.concat([features, sharpe], axis=1)
    
    return features

def calculate_cross_asset_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates market regime indicators, such as cross-asset dispersion.
    High dispersion means assets are moving independently; low means they are highly correlated.
    """
    daily_returns = prices.pct_change()
    
    # Calculate the cross-sectional standard deviation of returns across all assets for each day,
    # then smooth it with a 21-day moving average to reduce noise.
    dispersion = daily_returns.std(axis=1).rolling(window=21).mean()
    
    return pd.DataFrame({"market_dispersion_21d": dispersion})

def build_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to compile all features into a single DataFrame.
    """
    print("Building momentum features...")
    mom_features = calculate_momentum_features(prices)
    
    print("Building risk features...")
    risk_features = calculate_risk_features(prices)
    
    print("Building cross-asset features...")
    cross_features = calculate_cross_asset_features(prices)
    
    # Concatenate all feature DataFrames along the columns
    all_features = pd.concat([mom_features, risk_features, cross_features], axis=1)
    
    # CRITICAL: Shift all features by 1 period to prevent lookahead bias!
    # This ensures yesterday's closing data is used to generate today's signal.
    shifted_features = all_features.shift(1)
    
    # Drop rows with NaN values. 
    cleaned_features = shifted_features.dropna()
    print(f"Feature engineering complete. Usable days: {len(cleaned_features)}")
    
    return cleaned_features

if __name__ == "__main__":
    # Test block to execute when running this file directly
    prices_path = 'data/raw_prices.csv'
    
    if os.path.exists(prices_path):
        prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        feature_df = build_all_features(prices_df)
        feature_df.to_csv('data/features.csv')
        print("Features saved to data/features.csv")
    else:
        print("Error: raw_prices.csv not found. Run data_loader.py first.")