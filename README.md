# Cross-Asset Momentum Trading with ML Regime Filter

A quantitative trading framework applying cross-sectional momentum across five
broad-market ETFs, with a machine learning regime overlay trained on two feature
universes. Built with walk-forward validation, explicit signal lagging,
transaction cost modelling, and a clean ablation study.

---

## Results (January 2011 – September 2026)

### Universe: SPY / QQQ / TLT / GLD / USO

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar |
|:---|---:|---:|---:|---:|---:|---:|
| SPY Buy & Hold | 14.3% | 17.0% | 0.75 | 0.89 | -33.7% | 0.42 |
| **Cross-Sectional Momentum** | **13.6%** | **14.5%** | **0.82** | **1.01** | **-24.8%** | **0.55** |
| Momentum + LR (narrow — 5-ticker features) | 7.6% | 11.2% | 0.53 | 0.55 | -22.0% | 0.34 |
| Momentum + RF (narrow — 5-ticker features) | 10.2% | 13.6% | 0.64 | 0.72 | -24.8% | 0.41 |
| Momentum + LR (wide — 12-ticker features) | 10.1% | 12.8% | 0.66 | 0.71 | -24.8% | 0.41 |
| Momentum + RF (wide — 12-ticker features) | 11.5% | 13.8% | 0.72 | 0.84 | -23.0% | 0.50 |

> **Key finding:** Cross-sectional momentum (Sharpe 0.82) beats SPY (0.75) on a
> risk-adjusted basis while sustaining a smaller max drawdown (−24.8% vs −33.7%).
> Expanding the ML feature set from 5 to 12 tickers — adding uncorrelated assets
> (DBA, SLV, FXY, XBI, FXI, UNG, VNQ) as regime signals — consistently improves
> both ML variants: RF_Wide reaches Sharpe 0.72 vs RF_Narrow at 0.64. The ML
> overlay still reduces returns relative to raw momentum in this predominantly
> bull-market sample, but cuts max drawdown to −23.0% and reduces volatility.

---

## Strategy Architecture

Two strategies form the core of the ablation study:

1. **Cross-Sectional Momentum** — ranks assets relative to each other using composite z-scores across four lookback windows (1m/3m/6m/12m). Long the relatively strongest assets, flat on the weakest.
2. **ML Regime Overlay** — applied on top of momentum; scales position sizes to 100%, 50%, or 0% when a classifier predicts favorable, uncertain, or unfavorable forward return regimes.

---

### Universe

**Traded universe** — five liquid broad-market ETFs:

| Ticker | Asset Class | Role |
|:---|:---|:---|
| SPY | US Large-Cap Equity | Risk-on core |
| QQQ | US Technology Equity | High-momentum growth |
| TLT | Long-Duration Treasuries | Flight-to-quality / diversifier |
| GLD | Gold | Inflation hedge / tail risk |
| USO | Crude Oil | Commodity momentum |

**Wide feature universe** — 12 tickers used to train the ML regime filter,
adding 7 uncorrelated assets that carry macro regime information:

| Ticker | Asset Class | Regime signal |
|:---|:---|:---|
| DBA | Agriculture | Commodity inflation regime |
| SLV | Silver | Risk / inflation hybrid |
| FXY | Japanese Yen | Risk-off / rate differential |
| XBI | Biotech | Risk appetite / growth sentiment |
| FXI | China Large-Cap | Global growth / EM risk |
| UNG | Natural Gas | Commodity volatility |
| VNQ | REITs | Rate sensitivity / real assets |

### Signal Generation

**Cross-sectional momentum:** Composite score averaging cross-sectional z-scores
across four lookback windows (1m / 3m / 6m / 12m). Cross-sectional z-scoring at
each date ensures the signal captures *relative* momentum strength across assets,
not just absolute direction. Binary long/flat signal: long when composite z-score > 0,
flat (cash) otherwise.

### Position Sizing
Inverse-volatility weighting using 63-day realised volatility, capped at 40%
per asset to prevent excessive concentration in low-vol assets (typically TLT).
Weights are re-normalised after capping; residual allocation sits in cash.

### ML Regime Overlay
A binary classifier predicts whether the momentum strategy's forward 21-day
return will exceed the expanding-window historical median. When the model
signals an unfavorable regime, position sizes are scaled to 50% or 0% of their
momentum-only values.

Two models trained in parallel on two feature sets:
- **Narrow (51 features)** — features from the 5 traded tickers only
- **Wide (121 features)** — features from all 12 tickers; uncorrelated assets provide macro regime context unavailable from the traded universe alone

Two model types:
- **Logistic Regression** — interpretable baseline with L2 regularisation
- **Random Forest** — nonlinear benchmark (200 trees, max depth 3, balanced class weights)

Both trained with strict **walk-forward (time-series) cross-validation** —
no shuffling, no future data in any training fold.

### Rebalancing & Execution
- Monthly rebalancing at business month-end close
- Daily weight drift between rebalances (weights evolve with market moves)
- 10 bps one-way transaction cost applied at each rebalance
- 1-day execution lag: signals computed at close of day *t*, positions entered at close of day *t+1*

---

## Methodology

### Lookahead Bias Prevention
Every layer of the pipeline enforces strict temporal separation:

```
prices[t]           → features[t]       (data through close of day t only)
features[t]         → signals[t]        (momentum score using features[t])
signals[t]          → weights[t]        (inv-vol sizing at close of day t)
weights[t]          → label[t]          (fwd return t+1..t+21 — no overlap)
features[t]+label[t]→ ml_probs[t]       (walk-forward OOS only)
weights[t]×regime[t]→ portfolio[t]      
portfolio[t]        → return[t+1]       (shift(1) execution lag in backtest)
```

No row in any training set contains information from its own test window.

### ML Validation: Walk-Forward with Gap Purging
`TimeSeriesSplit(n_splits=10, gap=21, test_size=252)` — the `gap=21` parameter
is critical. Without it, the last 21 training labels overlap with the test
period (they include returns from days *inside* the test window), creating
direct data leakage. The gap purges this overlap exactly.

Each test fold covers approximately one calendar year, giving 10 independent
out-of-sample evaluation periods across the full history.

### Label Construction: Expanding-Window Median Split
Labels use an **expanding-window median** rather than a fixed return threshold.
A fixed threshold (e.g. −2%) produces severe class imbalance in trending
markets (>95% positive labels in a post-2010 bull market), causing the model to
learn a degenerate always-positive predictor. The expanding median adapts to
each market regime and consistently produces ~50/50 class balance
(observed: 53% positive), enabling genuine discriminative learning.

### Feature Engineering
Two feature sets are built — narrow (5 traded tickers) and wide (12 tickers):

| Category | Narrow (5 tickers) | Wide (12 tickers) |
|:---|---:|---:|
| Momentum (21d/63d/126d/252d) | 20 | 48 |
| Moving Average distance (50d/200d) | 5 | 12 |
| Volatility (21d/63d annualised) | 10 | 24 |
| Drawdown (252d rolling) | 5 | 12 |
| Mean-reversion z-score (21d/63d) | 10 | 24 |
| Cross-asset dispersion (21d) | 1 | 1 |
| **Total** | **51** | **121** |

All features computed with `min_periods` equal to the full window length —
no partial-window values during warmup.

---

## Ablation Study: Isolating the ML Contribution

```
SPY buy & hold                    →  Sharpe 0.75
Momentum + LR Narrow (5 tickers)  →  Sharpe 0.53
Momentum + RF Narrow (5 tickers)  →  Sharpe 0.64
Momentum + LR Wide  (12 tickers)  →  Sharpe 0.66
Momentum + RF Wide  (12 tickers)  →  Sharpe 0.72
Cross-Sectional Momentum          →  Sharpe 0.82  (best risk-adjusted)
```

**Interpretation:** Cross-sectional momentum beats SPY on Sharpe (0.82 vs 0.75)
with a substantially better drawdown profile (−24.8% vs −33.7%). The wide ML
feature set consistently outperforms the narrow one, confirming that uncorrelated
assets carry regime information the traded universe alone cannot capture.

The ML overlay reduces returns in this sample for three structural reasons:

1. **Bull market bias.** 2011–2026 is predominantly a US equity bull market where
   momentum pays nearly continuously. Reducing exposure mechanically costs returns
   without a commensurate improvement in drawdown protection.

2. **Difficult prediction target.** Predicting whether a 21-day forward return will
   be above-median is a low signal-to-noise classification problem in near-efficient
   markets. The model's OOS accuracy is modest, and a noisy filter hurts more than it helps.

3. **Transaction cost asymmetry.** Regime-scaled weights incur ~7–14× more annualised
   turnover than the momentum-only strategy (~0.7x), adding meaningful TC drag at each
   monthly rebalance.

The momentum signal is the primary alpha source. ML regime filtering is most likely to add
value over a longer history spanning full cycles, or with a better-specified prediction
target such as tail-risk drawdown events rather than median forward return.

---

## Project Structure

```
momentum-trading/
├── src/
│   ├── data_loader.py          # yfinance download, adjusted prices, validation
│   ├── features.py             # momentum, MA, vol, drawdown, z-score, dispersion features
│   ├── momentum_strategy.py    # cross-sectional z-score signals, inv-vol weighting
│   ├── ml_model.py             # label creation, walk-forward training, importances
│   ├── backtest.py             # vectorised backtest, daily drift, TC modelling
│   ├── evaluation.py           # metrics, ablation table, all visualisations
│   └── main.py                 # pipeline orchestration
├── eda.py                      # correlation analysis for 12-ticker feature universe
└── results/
    └── figures/                # equity curves, drawdowns, rolling Sharpe, regime viz
```

---

## Key Design Decisions

**Cross-sectional vs. time-series momentum — what's the difference?**  
Cross-sectional momentum asks "which asset is strongest *relative to the others*?" and always holds the relatively best assets, rotating into safe-havens (TLT, GLD) during risk-off periods. Time-series momentum asks "is this asset trending *in absolute terms*?" — all assets can be simultaneously flat if none are trending up. The stop-loss in the time-series variant adds a third override: exit even a trending asset if it has drawn down 10% from its recent high. Time-series momentum is implemented in `ts_momentum_strategy.py` and reserved for future comparison work.

**Why long/flat, not long/short?**  
Shorting individual ETFs introduces borrow costs and diverges from how most
institutional cross-asset momentum strategies are implemented. Long/flat with
cash better represents the strategy's risk profile.

**Why inverse-volatility weighting?**  
Inverse-vol normalises each position's risk contribution without requiring a
full covariance matrix estimate (which is noisy in a 5-asset universe).

**Why composite z-score across lookbacks?**  
Single-lookback momentum is sensitive to the chosen window. Averaging
cross-sectional z-scores across 1m/3m/6m/12m is more robust and reflects
the standard multi-horizon approach in the academic literature.

**Why expanding-window median labels?**  
A fixed return threshold produces severely imbalanced labels in trending
markets, causing degenerate classifiers. The expanding median self-calibrates
to the current market regime while maintaining strict temporal separation.

---

## Limitations and Future Work

- **Sample period:** 2011–2026 is predominantly a US equity bull market.
  Performance across a full cycle (including 2000–2002, 2007–2009) would
  require extending the universe to indices with longer histories.

- **Time-series momentum comparison:** `ts_momentum_strategy.py` implements
  time-series momentum with a 10% trailing stop-loss. A head-to-head comparison
  against cross-sectional momentum would clarify the relative-vs-absolute
  strength tradeoff, particularly during periods where all assets trend down
  simultaneously (2022 rate shock, 2020 COVID crash).

- **High-volatility universe stress test:** The framework was previously tested
  on a high-vol universe (XBI, GDX, EWZ, FXI, XOP). Momentum signals failed to
  add value (Sharpe ~0.33 vs random ~0.38), and the time-series stop-loss
  triggered on 96.6% of days due to the fixed 10% threshold being miscalibrated
  for 25–30% vol assets. Future work: a vol-scaled stop (e.g. 1× annualised vol
  per asset) and a mean-reversion signal (more appropriate for commodity/sector
  ETFs) could recover edge in that universe.

- **ML signal quality:** The regime classifier's predictive accuracy is not
  reported. Adding precision/recall curves and calibration plots per fold would
  better characterise where the model adds and destroys value.

- **Alternative ML targets:** Predicting tail-risk events (e.g. rolling max
  drawdown > 5% in the next 21 days) rather than the median return split may
  produce a filter that adds value even in trending markets by protecting
  specifically against sharp drawdowns.

- **Transaction cost sensitivity:** A TC sweep (0–30 bps) would show the
  breakeven cost at which each strategy becomes unviable — relevant for
  comparing futures vs. ETF implementation.
