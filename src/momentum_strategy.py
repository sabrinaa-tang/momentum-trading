import pandas as pd
import numpy as np

def generate_trend_signals(features: pd.DataFrame, assets: list) -> pd.DataFrame:
    """
    Generates binary signals: Long (1) if 12m momentum > 0, Flat/Cash (0) otherwise.
    Avoiding Short (-1) prevents massive drag during equity bull markets.
    """
    signals = pd.DataFrame(index=features.index)
    for asset in assets:
        mom_12m_col = f"{asset}_mom_252d"
        signals[asset] = np.where(features[mom_12m_col] > 0, 1, 0)
    return signals

def calculate_inverse_vol_weights(signals: pd.DataFrame, features: pd.DataFrame, assets: list) -> pd.DataFrame:
    """Allocates capital inversely proportional to 63-day volatility."""
    raw_weights = pd.DataFrame(index=signals.index)
    for asset in assets:
        vol_col = f"{asset}_vol_63d"
        inv_vol = 1.0 / (features[vol_col] + 1e-6)
        raw_weights[asset] = signals[asset] * inv_vol

    row_sums = raw_weights.sum(axis=1)
    # Normalize, but if row_sum is 0 (all signals are 0), keep weights at 0 (100% Cash)
    weights = raw_weights.div(row_sums.replace(0, 1e-6), axis=0)
    return weights.fillna(0)

def generate_random_strategy(index, columns) -> pd.DataFrame:
    """Random coin-flip strategy for benchmark ablation."""
    random_signals = pd.DataFrame(np.random.choice([0, 1], size=(len(index), len(columns))), 
                                  index=index, columns=columns)
    row_sums = random_signals.sum(axis=1)
    return random_signals.div(row_sums.replace(0, 1e-6), axis=0).fillna(0)