import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    returns = returns.dropna()
    n_days = len(returns)
    if n_days == 0: return {}
        
    cum_returns = (1 + returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
    max_dd = ((cum_returns / cum_returns.cummax()) - 1).min()
    
    return {
        'Total Return': f"{total_return * 100:.2f}%",
        'Ann. Return': f"{ann_return * 100:.2f}%",
        'Ann. Volatility': f"{ann_vol * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd * 100:.2f}%"
    }

def plot_performance(returns_df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    cum_returns = (1 + returns_df).cumprod() * 100 
    
    plt.figure(figsize=(12, 6))
    for col in cum_returns.columns:
        plt.plot(cum_returns.index, cum_returns[col], label=col, linewidth=1.5)
    plt.title('Strategy Ablation Study: Cumulative Equity Curve (Log Scale)')
    plt.yscale('log')
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'equity_curve.png'), dpi=300); plt.close()
    
    plt.figure(figsize=(12, 4))
    for col in cum_returns.columns:
        drawdown = (cum_returns[col] / cum_returns[col].cummax()) - 1
        plt.plot(drawdown.index, drawdown * 100, label=col, linewidth=1)
    plt.title('Strategy Drawdowns')
    plt.ylabel('Drawdown (%)')
    plt.legend(); plt.grid(True, ls="--", alpha=0.5)
    plt.savefig(os.path.join(save_dir, 'drawdowns.png'), dpi=300); plt.close()

def plot_rolling_sharpe(returns_df: pd.DataFrame, save_dir: str, window: int = 252):
    os.makedirs(save_dir, exist_ok=True)
    rolling_sharpe = (returns_df.rolling(window).mean() / returns_df.rolling(window).std()) * np.sqrt(252)
    plt.figure(figsize=(12, 6))
    for col in rolling_sharpe.columns:
        plt.plot(rolling_sharpe[col], label=col)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title(f'Rolling {window}-Day Sharpe Ratio')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'rolling_sharpe.png')); plt.close()

def plot_regime_visualization(spy_prices: pd.Series, regime_signal: pd.Series, save_dir: str):
    """Visualizes when the ML model was 'ON' vs the SPY benchmark."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    common_idx = spy_prices.index.intersection(regime_signal.index)
    
    plt.plot(common_idx, spy_prices.loc[common_idx], color='black', label='SPY Price')
    plt.fill_between(common_idx, spy_prices.loc[common_idx].min(), spy_prices.loc[common_idx].max(), 
                     where=(regime_signal.loc[common_idx] == 1), color='green', alpha=0.2, label='ML Favorable Regime')
    plt.yscale('log')
    plt.title('ML Regime Signal vs SPY Market Action')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'regime_visualization.png')); plt.close()

def plot_feature_importance(importance_series: pd.Series, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    importance_series.head(15).plot(kind='barh', color='teal')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png')); plt.close()