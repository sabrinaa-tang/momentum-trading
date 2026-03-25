# main.py
from data_loader import load_data
from features import build_all_features
from momentum_strategy import build_target_positions
from ml_model import create_labels, train_and_predict_regime
from backtest import run_backtest, calculate_cumulative_performance
import pandas as pd
from evaluation import calculate_metrics, plot_equity_curves

def main():
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'USO', 'ICLN']
    start_date = '2010-01-01'
    
    # Day 1 & 2
    prices = load_data(tickers, start_date)
    features_df = build_all_features(prices)
    
    # Day 3: Base Positions
    base_positions = build_target_positions(features_df, tickers)
    
    # Day 4: ML Positions
    labels = create_labels(prices, base_positions)
    regime_signals, rf_model = train_and_predict_regime(features_df, labels)
    ml_enhanced_positions = base_positions.multiply(regime_signals, axis=0).dropna()
    
    # --- DAY 5: BACKTESTING ---
    print("\n--- Starting Day 5: Realistic Backtesting ---")
    
    # 1. Backtest Base Momentum (10 bps transaction cost)
    print("Running backtest for Base Momentum...")
    base_results = run_backtest(prices, base_positions, trading_fee_bps=10)
    
    # 2. Backtest ML Enhanced Momentum
    print("Running backtest for ML Enhanced Momentum...")
    ml_results = run_backtest(prices, ml_enhanced_positions, trading_fee_bps=10)
    
    # 3. Create a Benchmark (100% SPY)
    # We create a dummy positions dataframe that is 1.0 for SPY and 0.0 for everything else
    print("Generating SPY Benchmark...")
    benchmark_positions = pd.DataFrame(0, index=base_positions.index, columns=tickers)
    benchmark_positions['SPY'] = 1.0
    benchmark_results = run_backtest(prices, benchmark_positions, trading_fee_bps=0) # No trading fees for Buy & Hold
    
    # Calculate cumulative returns for final output
    base_equity = calculate_cumulative_performance(base_results['net_return'])
    ml_equity = calculate_cumulative_performance(ml_results['net_return'])
    bench_equity = calculate_cumulative_performance(benchmark_results['net_return'])
    
    # --- DAY 6: EVALUATION & VISUALIZATION ---
    print("\n--- Starting Day 6: Risk-Adjusted Metrics ---")
    
    # Compile the daily net returns into a dictionary
    returns_dict = {
        "Buy & Hold SPY": benchmark_results['net_return'],
        "Base Momentum": base_results['net_return'],
        "ML Enhanced Momentum": ml_results['net_return']
    }
    
    # Calculate and print metrics
    metrics_list = []
    for name, returns in returns_dict.items():
        metrics = calculate_metrics(returns, name)
        metrics_list.append(metrics)
    
    # Display as a clean DataFrame
    metrics_df = pd.DataFrame(metrics_list).set_index("Strategy")
    print("\n")
    print(metrics_df.to_string())
    
    # Plot the curves
    print("\nGenerating equity curve plot...")
    plot_equity_curves(returns_dict)

if __name__ == "__main__":
    main()