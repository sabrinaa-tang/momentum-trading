import pandas as pd
import numpy as np
import os
from data_loader import load_data
from features import build_all_features
from momentum_strategy import generate_trend_signals, calculate_inverse_vol_weights, generate_random_strategy
from ml_model import create_labels, train_and_predict_walk_forward, get_feature_importance
from backtest import run_backtest
from evaluation import calculate_metrics, plot_performance, plot_rolling_sharpe, plot_regime_visualization, plot_feature_importance

def main():
    # Configuration matches prompt universe strictly
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'USO']
    start_date = '2010-01-01'
    results_dir = 'results/figures/'
    
    # 1. Pipeline
    prices = load_data(tickers, start_date)
    features_df = build_all_features(prices)
    
    # 2. Base Strategies
    signals = generate_trend_signals(features_df, tickers)
    base_weights = calculate_inverse_vol_weights(signals, features_df, tickers)
    random_weights = generate_random_strategy(base_weights.index, tickers)
    
    # 3. ML Labeling & Training
    labels = create_labels(prices, base_weights, horizon=21, threshold=0.000)
    common_idx = features_df.index.intersection(labels.dropna().index)
    X_clean, y_clean = features_df.loc[common_idx], labels.loc[common_idx]

    print(f"Alignment complete. Training on {len(X_clean)} days.")
    rf_probs = train_and_predict_walk_forward(X_clean, y_clean, model_type='rf')
    lr_probs = train_and_predict_walk_forward(X_clean, y_clean, model_type='lr')
    
    # Smooth probabilities to prevent transaction cost whipsawing
    rf_smooth = rf_probs.reindex(base_weights.index).ffill().fillna(0).rolling(window=5).mean()
    lr_smooth = lr_probs.reindex(base_weights.index).ffill().fillna(0).rolling(window=5).mean()

    rf_regime = pd.Series(
        np.where(rf_smooth > 0.50, 1.0, np.where(rf_smooth > 0.35, 0.5, 0.0)), 
        index=rf_smooth.index
    )
    lr_regime = pd.Series(
        np.where(lr_smooth > 0.50, 1.0, np.where(lr_smooth > 0.35, 0.5, 0.0)), 
        index=lr_smooth.index
    )
    
    ml_rf_weights = base_weights.multiply(rf_regime, axis=0)
    ml_lr_weights = base_weights.multiply(lr_regime, axis=0)

    # 4. Monthly Backtesting (0.1% transaction cost)
    daily_rets = prices.pct_change().dropna()
    print("\nRunning strategy backtests with Monthly Rebalancing...")
    
    results = pd.DataFrame({
        '1_Buy_Hold_SPY': daily_rets['SPY'],
        '2_Random_Benchmark': run_backtest(daily_rets, random_weights),
        '3_Base_Momentum': run_backtest(daily_rets, base_weights),
        '4_ML_Logistic_Mom': run_backtest(daily_rets, ml_lr_weights),
        '5_ML_RandomForest_Mom': run_backtest(daily_rets, ml_rf_weights)
    }).dropna()

    # 5. Visualization & Metrics
    plot_performance(results, results_dir)
    plot_rolling_sharpe(results, results_dir) 
    plot_regime_visualization(prices['SPY'], rf_regime, results_dir)
    plot_feature_importance(get_feature_importance(X_clean, y_clean), results_dir)

    print("\n" + "="*85)
    print("PERFORMANCE SUMMARY")
    print("="*85)

    summary_df = pd.DataFrame([
        {**calculate_metrics(results[col]), 'Strategy': col} 
        for col in results.columns
    ])
    
    cols = ['Strategy', 'Ann. Return', 'Ann. Volatility', 'Sharpe Ratio', 'Max Drawdown']
    try:
        print(summary_df[cols].to_markdown(index=False))
    except ImportError:
        print(summary_df[cols].to_string(index=False))

if __name__ == "__main__":
    main()