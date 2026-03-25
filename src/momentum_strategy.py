import pandas as pd
import numpy as np

def generate_base_signals(features: pd.DataFrame, assets: list, mom_window: int = 252) -> pd.DataFrame:
    """
    Generates directional signals based on the 12-month (252 trading days) momentum.
    Rule: Long (1) if momentum > 0, Short (-1) if momentum < 0.
    """
    # Extract the correct momentum columns
    mom_cols = [f"{asset}_mom_{mom_window}d" for asset in assets]
    mom_data = features[mom_cols].copy()
    
    # np.sign returns 1 for positive, -1 for negative, 0 for exactly zero.
    signals = np.sign(mom_data)
    
    # Handle the extremely rare case where return is exactly 0.000%
    signals = signals.replace(0, 1) 
    
    # Rename columns to clean asset names for easier alignment later
    signals.columns = assets
    return signals

def calculate_volatility_weights(features: pd.DataFrame, assets: list, vol_window: int = 63) -> pd.DataFrame:
    """
    Calculates inverse-volatility weights based on the 3-month (63 trading days) rolling volatility.
    """
    vol_cols = [f"{asset}_vol_{vol_window}d" for asset in assets]
    vol_data = features[vol_cols].copy()
    
    # Calculate inverse volatility (1 / Volatility)
    inv_vol = 1.0 / vol_data
    inv_vol.columns = assets
    
    # Normalize the weights so that the gross exposure sums to 100% (1.0) each day
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    
    return weights

def build_target_positions(features: pd.DataFrame, assets: list) -> pd.DataFrame:
    """
    Combines signals and weights, and applies the crucial lookahead shift.
    """
    print("Generating base momentum signals...")
    signals = generate_base_signals(features, assets, mom_window=252)
    
    print("Calculating inverse-volatility weights...")
    weights = calculate_volatility_weights(features, assets, vol_window=63)
    
    # Target Position = Direction * Size
    target_positions = signals * weights
    
    # =====================================================================
    # THE MOST IMPORTANT LINE IN THE PROJECT: THE LOOKAHEAD SHIFT
    # =====================================================================
    # The signals and weights were calculated using the closing price of Day T.
    # We cannot trade on Day T's close using Day T's data. 
    # We must shift the target positions forward by 1 day so they apply to Day T+1.
    executable_positions = target_positions.shift(1)
    
    # Drop the first row which is now NaN due to the shift
    cleaned_positions = executable_positions.dropna()
    print("Target positions calculated and shifted successfully.")
    
    return cleaned_positions