import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_metrics(returns: pd.Series, name: str) -> dict:
    """
    Calculates standard institutional risk-adjusted metrics for a return stream.
    """
    # 252 trading days in a year
    annualized_return = returns.mean() * 252
    annualized_vol = returns.std() * np.sqrt(252)
    
    # Sharpe Ratio (Assuming 0% risk-free rate for simplicity in this project)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
    # Maximum Drawdown
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        "Strategy": name,
        "Ann. Return": f"{annualized_return * 100:.2f}%",
        "Ann. Volatility": f"{annualized_vol * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown * 100:.2f}%"
    }

def plot_equity_curves(results_dict: dict):
    """
    Plots the cumulative returns of multiple strategies on a log scale.
    """
    plt.figure(figsize=(12, 6))
    
    for name, returns in results_dict.items():
        cumulative_returns = (1 + returns).cumprod()
        plt.plot(cumulative_returns, label=name, linewidth=2)
        
    plt.title("Strategy Comparison: Cumulative Returns (Log Scale)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (Multiple of Initial)")
    # Using a log scale is standard practice to accurately show percentage growth over time
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    # Save the plot for your README
    plt.savefig("equity_curve.png")
    print("Plot saved as 'equity_curve.png'.")
    plt.show()