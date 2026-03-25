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

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/905e3b9f-52d6-4bc8-8c72-b512e5d739d1" />
