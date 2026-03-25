import pandas as pd
import numpy as np
import os

def generate_trend_signals(features: pd.DataFrame, assets: list) -> pd.DataFrame:
    """
    Generates binary directional signals based on 12-month (252-day) momentum.
    +1 for Long (positive trend), -1 for Short (negative trend).
    """
    signals = pd.DataFrame(index=features.index)
    
    for asset in assets:
        mom_12m_col = f"{asset}_mom_252d"
        # If momentum > 0, long (1). Else, short (-1).
        signals[asset] = np.where(features[mom_12m_col] > 0, 1, -1)
        
    return signals

def calculate_equal_weights(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Allocates capital equally across all assets.
    Gross exposure is maintained at 100% (1.0).
    """
    n_assets = len(signals.columns)
    # Divide the signal (+1 or -1) by the number of assets
    weights = signals / n_assets
    return weights

def calculate_inverse_vol_weights(signals: pd.DataFrame, features: pd.DataFrame, assets: list) -> pd.DataFrame:
    """
    Allocates capital inversely proportional to the asset's 63-day volatility.
    Normalizes weights so absolute gross exposure equals 100%.
    """
    raw_weights = pd.DataFrame(index=signals.index)
    
    for asset in assets:
        vol_col = f"{asset}_vol_63d"
        # Inverse of volatility (adding 1e-6 to prevent division by zero)
        inv_vol = 1.0 / (features[vol_col] + 1e-6)
        
        # Raw weight = Direction * Magnitude
        raw_weights[asset] = signals[asset] * inv_vol

    # Normalize across the row so the sum of absolute weights equals 1.0
    row_abs_sums = raw_weights.abs().sum(axis=1)
    normalized_weights = raw_weights.div(row_abs_sums, axis=0)
    
    return normalized_weights

if __name__ == "__main__":
    features_path = 'data/features.csv'
    
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
        # FIX: Only extract tickers from columns we explicitly know are momentum features
        assets = list(set([col.split('_')[0] for col in features_df.columns if '_mom_252d' in col]))
        assets.sort()
        print(f"Detected universe: {assets}")
        
        # 1. Generate Signals
        signals = generate_trend_signals(features_df, assets)
        
        # 2. Construct Portfolios
        eq_weights = calculate_equal_weights(signals)
        vol_weights = calculate_inverse_vol_weights(signals, features_df, assets)
        
        # 3. Save targets
        eq_weights.to_csv('data/weights_equal.csv')
        vol_weights.to_csv('data/weights_vol_scaled.csv')
        
        print("Portfolio weights generated successfully.")
        print(f"Equal Weight snapshot (first row):\n{eq_weights.iloc[0].to_dict()}")
        print(f"Vol-Scaled snapshot (first row):\n{vol_weights.iloc[0].to_dict()}")
        
    else:
        print("Error: features.csv not found. Run features.py first.")