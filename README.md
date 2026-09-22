# Cross-Asset Momentum Trading: Cross-Sectional, Time-Series, and ML Regime Filter

A comparison of three systematic strategies across a
multi-asset universe: cross-sectional momentum, time-series momentum with a
trailing stop-loss, and a machine learning regime overlay. Includes walk-forward validation, 
explicit signal lagging, transaction cost modelling, and a clean ablation study.

---

## Results (January 2011 – September 2026)

### Standard Universe: SPY / QQQ / TLT / GLD / USO

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar |
|:---|---:|---:|---:|---:|---:|---:|
| SPY Buy & Hold | 14.3% | 17.0% | 0.75 | 0.89 | -33.7% | 0.42 |
| **Cross-Sectional Momentum** | **13.6%** | **14.5%** | **0.82** | **1.01** | **-24.8%** | **0.55** |
| Momentum + LR (narrow — 5-ticker features) | 7.6% | 11.2% | 0.53 | 0.55 | -22.0% | 0.34 |
| Momentum + RF (narrow — 5-ticker features) | 10.2% | 13.6% | 0.64 | 0.72 | -24.8% | 0.41 |
| Momentum + LR (wide — 12-ticker features) | 10.1% | 12.8% | 0.66 | 0.71 | -24.8% | 0.41 |
| Momentum + RF (wide — 12-ticker features) | 11.5% | 13.8% | 0.72 | 0.84 | -23.0% | 0.50 |

> **Key finding:** Cross-sectional momentum (Sharpe 0.82) remains the best
> risk-adjusted strategy, beating SPY (0.75) with a smaller max drawdown
> (−24.8% vs −33.7%). Expanding the ML feature set to 12 tickers (adding
> uncorrelated assets: TLT, USO, DBA, SLV, FXY, XBI, FXI, UNG, VNQ) improves
> both ML variants over the 5-ticker baseline — RF_Wide reaches Sharpe 0.72 vs
> RF_Narrow at 0.64. The ML overlay still reduces returns relative to raw
> momentum in this predominantly bull-market sample period.

### High-Volatility Universe: XBI / GDX / EWZ / FXI / XOP

| Strategy | Ann. Return | Ann. Vol | Sharpe | Sortino | Max DD | Calmar |
|:---|---:|---:|---:|---:|---:|---:|
| Equal-Weight Buy & Hold | 6.8% | 22.8% | 0.32 | 0.28 | -44.8% | 0.15 |
| SPY Buy & Hold | 14.2% | 17.0% | 0.75 | 0.88 | -33.7% | 0.42 |
| Cross-Sectional Momentum | 7.5% | 26.5% | 0.33 | 0.28 | -45.1% | 0.17 |
| Momentum + Logistic Reg. | 5.8% | 21.5% | 0.28 | 0.21 | -45.1% | 0.13 |
| Momentum + Random Forest | 6.6% | 24.0% | 0.30 | 0.24 | -45.1% | 0.15 |
| **TS Momentum + Stop-Loss** | **2.0%** | **26.1%** | **0.13** | **-0.00** | **-57.7%** | **0.03** |
| Random Sanity Check | 8.9% | 25.5% | 0.38 | 0.40 | -49.5% | 0.18 |

> **Key finding:** The momentum framework fails to add value in the high-vol
> universe. A random strategy (Sharpe 0.38) beats cross-sectional momentum
> (0.33), indicating the signals carry no edge over these assets.
> TS momentum collapses entirely — the 10% trailing stop triggers on 96.6% of
> days, keeping the strategy nearly perpetually in cash.

---

## Universe Comparison Analysis

### Why the high-vol universe was chosen

XBI (biotech), GDX (gold miners), EWZ (Brazil), FXI (China), and XOP (oil & gas E&P)
were selected as high-volatility counterparts to the standard universe on the
following criteria: all had inception dates before January 2011 (the backtest
start), are single-leg ETFs with no leverage decay, and represented genuinely
volatile but established asset classes rather than niche or exotic instruments.

**Lookahead bias caveat.** These five ETFs were chosen in September 2026 with
full knowledge of their behaviour over the backtest period. This introduces two
forms of bias that are difficult to eliminate:

- **Survivorship bias.** The selection implicitly excludes ETFs that existed in
  2011 but were later delisted or suspended. RSX (VanEck Russia ETF, inception
  2007) would have been a natural candidate but was suspended in March 2022 due
  to sanctions — a regime-change event that any backtest ending in 2026 would
  need to handle explicitly. Including only survivors understates the true risk
  of the high-vol universe.

- **Selection bias.** Labelling an asset "high-volatility" in 2026 reflects
  knowledge of how it actually behaved over 2011–2026. A practitioner building
  this universe in January 2011 would have had rougher ex-ante volatility
  estimates and may have made different inclusions. The specific five tickers
  here should be treated as an illustrative stress-test, not a
  point-in-time-valid universe.

A fully unbiased comparison would require constructing the universe using only
information available at the backtest start date, with no reference to
post-2011 performance or survival.

### What the comparison shows

**The low-volatility anomaly holds.** The high-vol universe delivers lower
risk-adjusted returns despite comparable (and often higher) absolute volatility.
XBI/GDX/EWZ/FXI/XOP averaged ~25% annualized vol but produced Sharpe ratios
around 0.30–0.38 — roughly half those of the standard universe at half the
vol (12–15%). This is consistent with the well-documented finding that
high-beta assets do not compensate proportionally for the additional risk.

**Momentum signals require trending assets.** Cross-sectional momentum barely
outperforms equal-weight (Sharpe 0.33 vs 0.32), and a random strategy beats
both. The high-vol assets are more noise-driven and prone to mean-reversion,
which violates the trending-asset assumption that makes momentum signals
informative. The feature importances confirm this: in the standard universe the
top features are trend indicators (GLD MA distance, GLD 12m momentum); in the
high-vol universe they are pure volatility features (GDX 21d vol, GDX 63d vol),
indicating the model is detecting vol clustering rather than directional momentum.

**The 10% trailing stop-loss is miscalibrated for high-vol assets.** A 10%
drawdown threshold is appropriate for assets running ~15% annualized vol —
it represents roughly a 0.65-sigma move. Applied to assets running 25–30% vol,
the same threshold is triggered by ordinary daily fluctuations. The result is
that the TS momentum strategy is stopped out on 96.6% of days (vs 89.0% for
the standard universe), maintains only 36.3% mean gross exposure, and produces
the worst risk-adjusted result in the table (Sharpe 0.13, Max DD −57.7%).
A vol-scaled stop-loss (e.g. 1× annualized vol) would be a more appropriate
design for a universe-agnostic implementation.

---

## Strategy Architecture

Three strategies are compared in the ablation study:

1. **Cross-Sectional Momentum**: ranks assets relative to each other using composite z-scores across four lookback windows (1m/3m/6m/12m). Long the relatively strongest assets, flat on the weakest.
2. **Time-Series Momentum with Stop-Loss**: evaluates each asset independently; long if its own 12m return is positive. A 10% trailing stop-loss (based on 252-day rolling drawdown) overrides the signal to limit drawdown exposure.
3. **ML Regime Overlay**: applied on top of cross-sectional momentum; scales position sizes down when a classifier predicts an unfavorable forward return regime.

---

### Universe

**Standard universe** — five liquid ETFs providing cross-asset exposure:

| Ticker | Asset Class | Role |
|:---|:---|:---|
| SPY | US Large-Cap Equity | Risk-on core |
| QQQ | US Technology Equity | High-momentum growth |
| TLT | Long-Duration Treasuries | Flight-to-quality / diversifier |
| GLD | Gold | Inflation hedge / tail risk |
| USO | Crude Oil | Commodity momentum |

**High-volatility universe** — five higher-beta ETFs for stress-testing:

| Ticker | Asset Class | Inception |
|:---|:---|:---|
| XBI | S&P Biotech (equal-weighted) | Jan 2006 |
| GDX | Gold Miners | May 2006 |
| EWZ | iShares Brazil | Jul 2000 |
| FXI | iShares China Large-Cap | Oct 2004 |
| XOP | S&P Oil & Gas E&P | Jun 2006 |

### Signal Generation

**Cross-sectional momentum:** Composite score averaging cross-sectional z-scores
across four lookback windows (1m / 3m / 6m / 12m). Cross-sectional z-scoring at
each date ensures the signal captures *relative* momentum strength across assets,
not just absolute direction. Binary long/flat signal: long when composite z-score > 0,
flat (cash) otherwise.

**Time-series momentum:** Each asset is evaluated independently — long if its own
12m return is positive, flat otherwise. A 10% trailing stop-loss based on the
252-day rolling drawdown overrides the momentum signal to exit positions during
sharp drawdowns, regardless of the trend signal.

### Position Sizing
Inverse-volatility weighting using 63-day realised volatility, capped at 40%
per asset to prevent excessive concentration in low-vol assets (typically TLT).
Weights are re-normalised after capping; residual allocation sits in cash.

### ML Regime Overlay
A binary classifier predicts whether the momentum strategy's forward 21-day
return will exceed the expanding-window historical median. When the model
signals an unfavorable regime, position sizes are scaled to 50% or 0% of their
momentum-only values.

Two models trained in parallel:
- **Logistic Regression** — interpretable baseline with L2 regularisation
- **Random Forest** — nonlinear benchmark (200 trees, max depth 3,
  balanced class weights)

Both trained with strict **walk-forward (time-series) cross-validation** —
no shuffling, no future data in any training fold.

### Rebalancing & Execution
- Monthly rebalancing at business month-end close
- Daily weight drift between rebalances (weights evolve with market moves)
- 10 bps one-way transaction cost applied at each rebalance
- 1-day execution lag: signals computed at close of day *t*, positions entered
  at close of day *t+1*

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
Two feature sets are built — narrow (5 traded tickers) and wide (12 tickers,
adding 7 uncorrelated ETFs: DBA, SLV, FXY, XBI, FXI, UNG, VNQ):

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
no partial-window values during warmup. Top features by RF importance differ
by universe: in the standard universe they are trend indicators (GLD MA
distance, GLD 12m momentum); in the high-vol universe they are volatility
features (GDX 21d vol, GDX 63d vol), consistent with the absence of
exploitable directional momentum in the high-vol assets.

---

## Ablation Study: Isolating the ML Contribution

The ablation is ordered by increasing complexity, standard universe:

```
SPY buy & hold                    →  Sharpe 0.75
Momentum + LR Narrow (5 tickers)  →  Sharpe 0.53
Momentum + RF Narrow (5 tickers)  →  Sharpe 0.64
Momentum + LR Wide  (12 tickers)  →  Sharpe 0.66
Momentum + RF Wide  (12 tickers)  →  Sharpe 0.72
Cross-Sectional Momentum          →  Sharpe 0.82  (best risk-adjusted)
```

**Interpretation:** Cross-sectional momentum beats SPY on Sharpe (0.82 vs 0.75)
and delivers a substantially better drawdown profile (−24.8% vs −33.7%).
Time-series momentum with stop-loss achieves the smallest max drawdown (−23.8%)
with a competitive Sharpe (0.70), making it the most capital-efficient on a
downside-adjusted basis.

The two momentum approaches offer a genuine tradeoff:
- **Cross-sectional momentum** captures relative strength across assets and stays
  more fully invested, producing higher returns (13.7% vs 9.8%) at the cost of
  higher vol (14.5% vs 11.5%).
- **Time-series momentum with stop-loss** reduces gross exposure when all assets
  are trending down simultaneously, producing the best Sortino (0.88) and smallest
  max drawdown (−23.8%) in the table.

The ML overlay continues to not add value for the same three reasons:

1. **Bull market bias.** The ML filter reduces gross exposure (active 82–96% of
   days). In a period where momentum almost always pays, reducing exposure
   mechanically reduces returns without a commensurate reduction in drawdown.

2. **Regime signal lead time.** A 21-day forward return label trained to predict
   the *median* of the distribution is a difficult classification problem in
   near-efficient markets. The model has low predictive accuracy OOS, and
   a noisy filter hurts more than it helps.

3. **Transaction cost asymmetry.** Regime-scaled weights change the rebalance
   target at each monthly reset. Even with 5-day probability smoothing, the ML
   strategies incur approximately 15× more rebalance turnover than the
   momentum-only strategy (~7x vs ~0.5x annualised one-way).

The momentum signal itself is the alpha source. ML regime filtering requires
either a longer history spanning multiple full cycles, or a better-specified
prediction target (e.g. tail-risk drawdown events rather than median forward return).

---

## Project Structure

```
momentum-trading/
├── src/
│   ├── data_loader.py          # yfinance download, adjusted prices, validation
│   ├── features.py             # momentum, MA, vol, drawdown, dispersion features
│   ├── momentum_strategy.py    # cross-sectional z-score signals, inv-vol weighting
│   ├── ts_momentum_strategy.py # time-series momentum with trailing stop-loss
│   ├── ml_model.py             # label creation, walk-forward training, importances
│   ├── backtest.py             # vectorised backtest, daily drift, TC modelling
│   ├── evaluation.py           # metrics, ablation table, all visualisations
│   └── main.py                 # pipeline orchestration (runs both universes)
└── results/
    └── figures/
        ├── standard/           # Standard universe charts
        ├── highvol/            # High-vol universe charts
        └── comparison/         # Cross-universe equity curves + Sharpe bar chart
```

---

## Key Design Decisions

**Cross-sectional vs. time-series momentum — what's the difference?**  
Cross-sectional momentum asks "which asset is strongest *relative to the others*?" and always holds the relatively best assets. Time-series momentum asks "is this asset trending *in absolute terms*?" — it's possible for all assets to be simultaneously long or simultaneously flat. The stop-loss in the time-series variant adds a third override: exit even a trending asset if it has already fallen 10% from its recent high.

**Why long/flat, not long/short?**  
Shorting individual asset ETFs introduces significant borrow costs and
diverges from how most institutional cross-asset momentum strategies are
implemented. Long/flat with cash better represents the strategy's risk profile
and avoids massive drag during equity bull markets.

**Why inverse-volatility weighting?**  
Inverse-vol normalises each position's risk contribution without requiring a
full covariance matrix estimate (which is noisy in a 5-asset universe).

**Why composite z-score across lookbacks?**  
Single-lookback momentum is sensitive to the chosen window. Averaging
cross-sectional z-scores across 1m/3m/6m/12m is more robust.

**Why expanding-window median labels?**  
A fixed return threshold produces severely imbalanced labels in trending
markets, causing degenerate classifiers. The expanding median self-calibrates
to the current market regime while maintaining strict temporal separation.

---

## Limitations and Future Work

- **Sample period:** 2011–2026 is predominantly a US equity bull market.
  Performance across a full cycle (including 2000–2002, 2007–2009) would
  require extending the universe to indices with longer histories.

- **Survivorship bias in universe selection:** Both universes were constructed
  in September 2026 with knowledge of which ETFs survived through the backtest
  period. A production implementation would use a point-in-time universe
  construction. The high-vol universe is particularly affected — RSX (Russia
  ETF) is the most obvious omission, having been suspended mid-sample in 2022.

- **Stop-loss calibration:** The fixed 10% trailing stop is not scaled to asset
  volatility. A vol-scaled threshold (e.g. 1× annualized vol per asset) would
  avoid the stop triggering trivially on high-vol assets and degenerate
  behaviour like the 96.6% stop-active rate observed in the high-vol universe.

- **ML signal quality:** The regime classifier's predictive accuracy is not
  reported. Adding precision/recall curves and calibration plots per fold would
  better characterise where the model adds and destroys value.

- **Alternative ML targets:** Predicting tail-risk events (e.g. rolling max
  drawdown > 5% in next 21 days) rather than the median return split may
  produce a filter that adds value even in trending markets by protecting
  specifically against sharp drawdowns.

- **Transaction cost sensitivity:** A TC sweep (0–30 bps) would show the
  breakeven cost at which each strategy becomes unviable — relevant for
  comparing futures vs. ETF implementation.
