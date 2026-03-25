import pandas as pd
import numpy as np

def run_backtest(prices: pd.DataFrame, positions: pd.DataFrame, trading_fee_bps: int = 10) -> pd.DataFrame:
    """
    Simulates portfolio returns given historical prices and target positions.
    Includes transaction costs based on daily turnover.
    
    Args:
        prices: Daily adjusted close prices.
        positions: Executable target weights for each asset (already shifted to avoid lookahead).
        trading_fee_bps: Transaction cost in basis points (10 bps = 0.1%).
    """
    # 1. Calculate actual daily percentage returns of the underlying assets
    asset_returns = prices.pct_change()
    
    # Align the dates exactly (positions might be slightly shorter due to dropped NaNs)
    asset_returns = asset_returns.reindex(positions.index)
    
    # 2. Calculate Gross Daily Portfolio Returns
    # Since our positions are already shifted (from Day 3), the position we hold on Day T 
    # earns the asset return generated on Day T.
    portfolio_gross_returns = (positions * asset_returns).sum(axis=1)
    
    # 3. Calculate Daily Turnover
    # How much did our weights change from yesterday to today?
    # Fill the first day's NaN with the absolute positions (since we bought the initial portfolio)
    weight_changes = positions.diff().fillna(positions.abs())
    
    # Sum the absolute changes across all assets to get total daily turnover
    daily_turnover = weight_changes.abs().sum(axis=1)
    
    # 4. Apply Transaction Costs
    tc_decimal = trading_fee_bps / 10000.0
    daily_tc_drag = daily_turnover * tc_decimal
    
    # 5. Calculate Net Returns
    portfolio_net_returns = portfolio_gross_returns - daily_tc_drag
    
    # 6. Compile the results
    results = pd.DataFrame({
        'gross_return': portfolio_gross_returns,
        'net_return': portfolio_net_returns,
        'turnover': daily_turnover
    })
    
    return results

def calculate_cumulative_performance(returns: pd.Series) -> pd.Series:
    """
    Converts daily returns into a cumulative equity curve starting at 1.0 ($1).
    """
    return (1 + returns).cumprod()