import pandas as pd
import numpy as np

def run_backtest(returns: pd.DataFrame, target_weights: pd.DataFrame, tc: float = 0.001) -> pd.Series:
    """
    Vectorized backtest with MONTHLY rebalancing logic.
    Weights drift daily with market moves, but transaction costs are only incurred 
    at month-end when portfolios are officially rebalanced.
    """
    common_idx = returns.index.intersection(target_weights.index)
    returns = returns.loc[common_idx]
    target_weights = target_weights.loc[common_idx]
    
    # Identify end of month dates for rebalancing
    eom_dates = returns.resample('BME').last().index
    rebalance_mask = returns.index.isin(eom_dates)
    
    actual_weights = np.zeros_like(target_weights.values)
    turnover = np.zeros(len(returns))
    current_w = np.zeros(target_weights.shape[1])
    
    ret_vals = returns.values
    target_vals = target_weights.values
    
    for i in range(len(returns)):
        if i > 0:
            # Drift weights based on previous day's return
            drifted_w = current_w * (1 + ret_vals[i-1])
            sum_drifted = np.nansum(np.abs(drifted_w))
            current_w = drifted_w / sum_drifted if sum_drifted > 0 else np.zeros_like(current_w)
                
        # Rebalance if it's the first day or an end-of-month date
        if i == 0 or rebalance_mask[i]:
            target = target_vals[i]
            turnover[i] = np.nansum(np.abs(target - current_w))
            current_w = target
            
        actual_weights[i] = current_w
        
    # Shift actual weights to represent holding them into the next day
    actual_weights_df = pd.DataFrame(actual_weights, index=returns.index, columns=returns.columns)
    gross_returns = (actual_weights_df.shift(1) * returns).sum(axis=1)
    
    net_returns = gross_returns - (pd.Series(turnover, index=returns.index) * tc)
    return net_returns