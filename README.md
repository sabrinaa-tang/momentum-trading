# Cross-Asset Momentum Strategy with ML Regime Filter

## Overview
This project implements a complete, professional-grade quantitative trading pipeline. It combines a traditional cross-asset time-series momentum strategy with a Machine Learning (Random Forest) regime filter to dynamically manage risk and reduce portfolio drawdowns.

The system trades a diversified universe of ETFs (Equities, Treasuries, Gold, Oil) and features a custom-built, vectorized backtesting engine that strictly prevents lookahead bias and accounts for real-world transaction costs.

## Key Features & Methodology
* **Cross-Asset Universe:** SPY (Large Cap Equities), QQQ (Tech Equities), TLT (Long-Term Treasuries), GLD (Gold), and USO (Crude Oil).
* **Inverse-Volatility Sizing:** Instead of equal-weighting (which allows highly volatile assets like Oil to dominate the portfolio's variance), capital is allocated dynamically based on 63-day rolling inverse volatility. This ensures a roughly equal risk contribution from each asset.
* **Base Signal (Time-Series Momentum):** Generates long/short signals based on 252-day (12-month) rolling returns. 
* **Machine Learning Regime Filter:** A Random Forest classifier trained to predict if the base momentum strategy will be profitable over the forward 21-day window. If the model predicts a hostile regime (e.g., high cross-asset dispersion and volatility), the portfolio overrides the momentum signals and dynamically allocates to 100% cash.
* **Strict Lookahead Bias Prevention:** Target weights are explicitly shifted to $T+1$ to ensure signals calculated at the market close are executed using the following day's data. Missing data is strictly forward-filled (`.ffill()`), never interpolated.

## Project Structure
The codebase is fully modularized, separating data ingestion, feature engineering, strategy logic, and backtesting into distinct components.

* `main.py` - The central orchestrator that runs the end-to-end pipeline.
* `data_loader.py` - Handles ingestion and cleaning of daily adjusted close prices via the `yfinance` API.
* `features.py` - Constructs time-series momentum, rolling risk, and cross-asset dispersion features.
* `momentum_strategy.py` - Generates base directional signals and calculates risk-parity target weights.
* `ml_model.py` - Handles time-series train/test splitting, target labeling, and trains the Random Forest classifier (constrained to `max_depth=3` to prevent overfitting noisy financial data).
* `backtest.py` - A vectorized backtester that simulates portfolio returns and deducts daily turnover transaction costs (10 bps).
* `evaluation.py` - Calculates institutional risk-adjusted metrics (Sharpe, Max Drawdown) and plots the equity curve.

## Performance & Results
*Backtest Period: 2011 to 2026* | *Transaction Costs: 10 bps*

The addition of the ML regime filter acting as an "emergency brake" significantly improved the risk-adjusted profile of the base momentum strategy, successfully side-stepping major market crashes in 2020 and 2022.

| Strategy | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
| Buy & Hold SPY (Benchmark) | 14.03% | 17.11% | 0.82 | -33.72% |
| Base Momentum | 1.90% | 8.93% | 0.21 | -28.57% |
| **ML Enhanced Momentum** | **4.35%** | **8.12%** | **0.54** | **-21.57%** |