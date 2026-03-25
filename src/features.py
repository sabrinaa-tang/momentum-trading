import pandas as pd
import numpy as np

def calculate_momentum_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates 3m, 6m, 12m rolling returns and Moving Average distances."""
    features = pd.DataFrame(index=prices.index)
    windows = [63, 126, 252] # 3m, 6m, 12m in trading days
    
    # 1. Rolling Returns
    for w in windows:
        rolling_ret = prices.pct_change(periods=w)
        rolling_ret.columns = [f"{col}_mom_{w}d" for col in prices.columns]
        features = pd.concat([features, rolling_ret], axis=1)
        
    # 2. Moving Averages (50d / 200d distance)
    ma_50 = prices.rolling(50).mean()
    ma_200 = prices.rolling(200).mean()
    ma_dist = (ma_50 / ma_200) - 1
    ma_dist.columns = [f"{col}_ma_dist" for col in prices.columns]
    features = pd.concat([features, ma_dist], axis=1)
    
    return features

def calculate_risk_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates annualized rolling volatility and 1-year drawdown."""
    daily_returns = prices.pct_change()
    features = pd.DataFrame(index=prices.index)
    
    for w in [21, 63]:
        vol = daily_returns.rolling(window=w).std() * np.sqrt(252)
        vol.columns = [f"{col}_vol_{w}d" for col in prices.columns]
        features = pd.concat([features, vol], axis=1)
        
    rolling_max = prices.rolling(window=252, min_periods=1).max()
    drawdown = (prices / rolling_max) - 1
    drawdown.columns = [f"{col}_dd_252d" for col in prices.columns]
    features = pd.concat([features, drawdown], axis=1)
    
    return features

def calculate_cross_asset_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates cross-asset dispersion (market regime indicator)."""
    daily_returns = prices.pct_change()
    dispersion = daily_returns.std(axis=1).rolling(window=21).mean()
    return pd.DataFrame({"market_dispersion_21d": dispersion})

def build_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Master function to compile all features."""
    print("Building features (Momentum, MAs, Risk, Dispersion)...")
    features = pd.concat([
        calculate_momentum_features(prices),
        calculate_risk_features(prices),
        calculate_cross_asset_features(prices)
    ], axis=1)
    
    # CRITICAL: Shift by 1 to prevent lookahead bias
    cleaned_features = features.shift(1).dropna()
    print(f"Feature engineering complete. Usable days: {len(cleaned_features)}")
    return cleaned_features