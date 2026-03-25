# Cross-Asset Momentum Strategy with ML Regime Filter

**Tech Stack:** `Python` `Pandas` `NumPy` `scikit-learn` `yfinance`

## Overview
This repository contains a complete, professional-grade quantitative trading pipeline. It enhances a traditional cross-asset time-series momentum strategy with a **Machine Learning (Random Forest) regime filter** to dynamically manage risk and reduce portfolio drawdowns. 

The system trades a diversified, risk-parity universe of ETFs (Equities, Treasuries, Gold, Oil) and features a custom-built, vectorized backtesting engine that strictly prevents lookahead bias and accounts for real-world transaction costs.

## Key Outcomes & Performance
*Backtest Period: 2011 to 2026* | *Transaction Costs: 10 bps* | *Out-of-Sample ML Testing*

* **Increased Sharpe Ratio by 150%:** Improved the base momentum strategy's Sharpe Ratio from 0.21 to 0.54 by successfully filtering out low-probability trades.
* **Reduced Maximum Drawdown by 36%:** Dropped the peak-to-trough drawdown from -33.7% (SPY Benchmark) to -21.5% by dynamically shifting to a 100% cash position during the 2020 and 2022 market crashes.
* **Halved Portfolio Volatility:** Maintained an annualized volatility of 8.12% compared to the S&P 500's 17.11% through daily inverse-volatility position sizing.

| Strategy | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown |
| :--- | :--- | :--- | :--- | :--- |
| Buy & Hold SPY (Benchmark) | 14.03% | 17.11% | 0.82 | -33.72% |
| Base Momentum | 1.90% | 8.93% | 0.21 | -28.57% |
| **ML Enhanced Momentum** | **4.35%** | **8.12%** | **0.54** | **-21.57%** |

*(Note: Add your `equity_curve.png` here to visually demonstrate the drawdown protection.)*

## Methodology & Architecture

### 1. The Cross-Asset Universe & Risk Parity
* **Assets:** SPY (Large Cap), QQQ (Tech), TLT (Long-Term Treasuries), GLD (Gold), and USO (Crude Oil).
* **Inverse-Volatility Sizing:** Instead of equal-weighting (which allows highly volatile assets like Oil to dominate the portfolio's variance), capital is allocated dynamically based on 63-day rolling inverse volatility. 

### 2. The Base Signal (Time-Series Momentum)
* Generates long/short signals based on 252-day (12-month) rolling returns. 

### 3. The Machine Learning Regime Filter (`scikit-learn`)
* **Model:** A Random Forest classifier trained to predict if the base momentum strategy will be profitable over the forward 21-day window. 
* **Constraint:** Model depth is strictly constrained (`max_depth=3`) to prevent overfitting on noisy financial time-series data.
* **Features:** Utilizes rolling volatility, cross-asset dispersion, and momentum strength to identify hostile macro environments.

### 4. Robust Engineering & Bias Prevention
* **Lookahead Bias:** Target weights are explicitly shifted to $T+1$ to ensure signals calculated at the market close are executed using the following day's data. Missing data is strictly forward-filled (`.ffill()`), never interpolated.
* **Vectorized Backtesting:** Custom engine accurately simulates daily portfolio returns, dynamically calculates turnover, and deducts daily transaction costs.

## Project Structure
* `main.py` - The central orchestrator that runs the end-to-end pipeline.
* `data_loader.py` - Handles ingestion and cleaning of daily adjusted close prices via the `yfinance` API.
* `features.py` - Constructs time-series momentum, rolling risk, and cross-asset dispersion features.
* `momentum_strategy.py` - Generates base directional signals and calculates risk-parity target weights.
* `ml_model.py` - Handles time-series train/test splitting, target labeling, and trains the Random Forest classifier.
* `backtest.py` - A vectorized backtester that simulates portfolio returns and deducts daily turnover transaction costs.
* `evaluation.py` - Calculates institutional risk-adjusted metrics and plots the equity curve.
