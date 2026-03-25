import pandas as pd
import numpy as np
import os

def run_backtest(returns: pd.DataFrame, weights: pd.DataFrame, tc: float = 0.001) -> pd.Series:
    """
    Runs a vectorized backtest with transaction costs.
    returns: Daily asset returns.
    weights: Target portfolio weights.
    tc: Transaction cost per unit of turnover (default 0.1% or 10 bps).
    """
    # Forward fill weights to handle any missing days, then fillna with 0
    weights = weights.ffill().fillna(0)
    
    # Ensure returns and weights align perfectly
    common_idx = returns.index.intersection(weights.index)
    returns = returns.loc[common_idx]
    weights = weights.loc[common_idx]
    
    # Calculate daily gross returns
    # Since our features were shifted by 1 in features.py, the weights for day T 
    # were calculated using data up to T-1. Thus, it is safe to multiply weights at T by returns at T.
    gross_returns = (weights * returns).sum(axis=1)
    
    # Calculate turnover (absolute change in weights day-over-day)
    # .shift(1) represents the portfolio we held yesterday.
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum() # Initial allocation turnover
    
    # Calculate net returns
    net_returns = gross_returns - (turnover * tc)
    
    return net_returns

if __name__ == "__main__":
    print("Loading data for backtest...")
    
    paths = {
        'prices': 'data/raw_prices.csv',
        'eq_weights': 'data/weights_equal.csv',
        'vol_weights': 'data/weights_vol_scaled.csv',
        'ml_preds': 'data/ml_predictions.csv'
    }
    
    if all(os.path.exists(p) for p in paths.values()):
        # Load all components
        prices = pd.read_csv(paths['prices'], index_col=0, parse_dates=True)
        eq_weights = pd.read_csv(paths['eq_weights'], index_col=0, parse_dates=True)
        vol_weights = pd.read_csv(paths['vol_weights'], index_col=0, parse_dates=True)
        ml_preds = pd.read_csv(paths['ml_preds'], index_col=0, parse_dates=True)
        
        # Calculate daily asset returns
        daily_returns = prices.pct_change().dropna()
        
        # Align all dataframes to the same index (the ML predictions are the shortest)
        common_idx = ml_preds.index.intersection(daily_returns.index)
        daily_returns = daily_returns.loc[common_idx]
        eq_weights = eq_weights.loc[common_idx]
        vol_weights = vol_weights.loc[common_idx]
        ml_preds = ml_preds.loc[common_idx]
        
        print(f"Running backtests from {common_idx[0].date()} to {common_idx[-1].date()}...")
        
        # 1. Base Strategy: Equal Weight
        eq_net_ret = run_backtest(daily_returns, eq_weights)
        
        # 2. Base Strategy: Vol-Scaled
        vol_net_ret = run_backtest(daily_returns, vol_weights)
        
        # 3. ML Strategy: Random Forest
        # We scale the Vol-Weighted portfolio by the probability of a good regime.
        # If the model is 90% sure it's a good regime, we take 90% of our target position.
        rf_weights = vol_weights.multiply(ml_preds['rf_regime_prob'], axis=0)
        rf_net_ret = run_backtest(daily_returns, rf_weights)
        
        # 4. ML Strategy: Logistic Regression
        lr_weights = vol_weights.multiply(ml_preds['lr_regime_prob'], axis=0)
        lr_net_ret = run_backtest(daily_returns, lr_weights)
        
        # Combine all strategy returns into a single DataFrame
        strategy_returns = pd.DataFrame({
            'Equal_Weight_Mom': eq_net_ret,
            'Vol_Scaled_Mom': vol_net_ret,
            'RF_Enhanced_Mom': rf_net_ret,
            'LR_Enhanced_Mom': lr_net_ret
        })
        
        # Add SPY as our Buy & Hold Benchmark
        if 'SPY' in daily_returns.columns:
            strategy_returns['Benchmark_SPY'] = daily_returns['SPY']
            
        strategy_returns.to_csv('data/backtest_returns.csv')
        print("Backtest complete! Daily returns saved to data/backtest_returns.csv")
        
    else:
        print("Error: Missing required files. Ensure Day 1-4 scripts ran successfully.")