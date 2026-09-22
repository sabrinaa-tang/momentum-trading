import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from features import build_all_features
from momentum_strategy import (
    generate_trend_signals,
    calculate_inverse_vol_weights,
    generate_equal_weight_baseline,
    generate_random_strategy,
)
from ml_model import create_labels, train_and_predict_walk_forward, get_feature_importance
from backtest import run_backtest
from evaluation import (
    compare_strategies,
    plot_performance,
    plot_rolling_sharpe,
    plot_regime_visualization,
    plot_feature_importance,
)

# ── Universes ─────────────────────────────────────────────────────────────────
# Traded assets — momentum signal and weights built from these
MOMENTUM_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "USO"]

# Exogenous regime signals for the wide ML variant — uncorrelated assets only,
# intentionally excluding the 5 traded tickers so the two inputs are orthogonal:
# momentum signal from traded universe, regime signal from uncorrelated universe
WIDE_FEATURE_TICKERS = ["DBA", "SLV", "FXY", "XBI", "FXI", "UNG", "VNQ"]

START_DATE          = "2010-01-01"
BASE_RESULTS        = "results/figures/"
RISK_FREE_RATE      = 0.02
TC                  = 0.001
MAX_ASSET_WEIGHT    = 0.40
LABEL_HORIZON       = 21
LABEL_TYPE          = "median"
LABEL_THRESHOLD     = -0.02
ML_PROB_THRESHOLD   = 0.50
ML_PROB_PARTIAL     = 0.35
ML_SMOOTHING_WINDOW = 5


def build_ml_regime(
    probs: pd.Series,
    target_index: pd.Index,
    high: float = ML_PROB_THRESHOLD,
    low: float  = ML_PROB_PARTIAL,
    smooth: int = ML_SMOOTHING_WINDOW,
) -> pd.Series:
    filled   = probs.reindex(target_index).ffill().fillna(1.0)
    smoothed = filled.rolling(window=smooth, min_periods=1).mean()
    regime   = pd.Series(
        np.where(smoothed > high, 1.0, np.where(smoothed > low, 0.5, 0.0)),
        index=smoothed.index,
    )
    on_pct = (regime > 0).mean()
    print(f"  Regime filter active {on_pct:.1%} of days "
          f"({(regime == 1.0).mean():.1%} full / "
          f"{(regime == 0.5).mean():.1%} half / "
          f"{(regime == 0.0).mean():.1%} off)")
    return regime


def train_ml_pair(X: pd.DataFrame, y: pd.Series, label: str) -> tuple:
    """Trains RF + logistic walk-forward on X/y, returns (rf_regime, lr_regime)."""
    print(f"\n--- ML [{label}]: {X.shape[1]} features ---")
    rf_probs = train_and_predict_walk_forward(X, y, model_type="rf",       horizon=LABEL_HORIZON)
    lr_probs = train_and_predict_walk_forward(X, y, model_type="logistic", horizon=LABEL_HORIZON)
    print(f"  RF regime:  ", end="")
    rf_regime = build_ml_regime(rf_probs, y.index)
    print(f"  LR regime:  ", end="")
    lr_regime = build_ml_regime(lr_probs, y.index)
    return rf_regime, lr_regime


def main() -> None:
    spy_prices = load_data(["SPY"], START_DATE)

    # ── Momentum weights (traded universe) ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Building momentum weights: {MOMENTUM_TICKERS}")
    print(f"{'='*60}")
    mom_prices        = load_data(MOMENTUM_TICKERS, START_DATE)
    features_narrow   = build_all_features(mom_prices)
    signals           = generate_trend_signals(features_narrow, MOMENTUM_TICKERS)
    weights           = calculate_inverse_vol_weights(
        signals, features_narrow, MOMENTUM_TICKERS, max_weight=MAX_ASSET_WEIGHT
    )

    labels = create_labels(
        mom_prices, weights,
        horizon=LABEL_HORIZON,
        threshold=LABEL_THRESHOLD,
        label_type=LABEL_TYPE,
    )
    common_narrow = features_narrow.index.intersection(labels.dropna().index)
    X_narrow = features_narrow.loc[common_narrow]
    y        = labels.loc[common_narrow]
    print(f"  ML training set: {len(X_narrow)} days "
          f"({X_narrow.index[0].date()} → {X_narrow.index[-1].date()})")

    # ── Wide feature set (EDA tickers + 3 non-EDA momentum tickers) ──────────
    print(f"\n{'='*60}")
    print(f"  Building wide features: {WIDE_FEATURE_TICKERS}")
    print(f"{'='*60}")
    wide_prices    = load_data(WIDE_FEATURE_TICKERS, START_DATE)
    features_wide  = build_all_features(wide_prices)
    common_wide    = features_wide.index.intersection(labels.dropna().index)
    X_wide         = features_wide.loc[common_wide]
    y_wide         = labels.loc[common_wide]

    # ── Train both ML variants ────────────────────────────────────────────────
    rf_narrow, lr_narrow = train_ml_pair(X_narrow, y,      label="narrow — 5 tickers")
    rf_wide,   lr_wide   = train_ml_pair(X_wide,   y_wide, label="wide — 12 tickers")

    # ── Apply regime filters to momentum weights ──────────────────────────────
    def scale_weights(regime: pd.Series) -> pd.DataFrame:
        return weights.multiply(regime.reindex(weights.index).ffill().fillna(1.0), axis=0)

    daily_rets = mom_prices.pct_change().dropna().loc[weights.index[0]:]
    spy_rets   = spy_prices.pct_change().dropna()["SPY"].reindex(daily_rets.index)

    print("\nRunning backtests...")
    results = pd.DataFrame({
        "1_EqWeight":    run_backtest(daily_rets, generate_equal_weight_baseline(daily_rets.index, MOMENTUM_TICKERS), tc=TC),
        "2_Random":      run_backtest(daily_rets, generate_random_strategy(daily_rets.index, MOMENTUM_TICKERS),       tc=TC),
        "3_SPY_BuyHold": spy_rets,
        "4_Momentum":    run_backtest(daily_rets, weights,                    tc=TC),
        "5_LR_Narrow":   run_backtest(daily_rets, scale_weights(lr_narrow),   tc=TC),
        "6_RF_Narrow":   run_backtest(daily_rets, scale_weights(rf_narrow),   tc=TC),
        "7_LR_Wide":     run_backtest(daily_rets, scale_weights(lr_wide),     tc=TC),
        "8_RF_Wide":     run_backtest(daily_rets, scale_weights(rf_wide),     tc=TC),
    })

    os.makedirs(BASE_RESULTS, exist_ok=True)
    plot_performance(results, BASE_RESULTS)
    plot_rolling_sharpe(results, BASE_RESULTS, risk_free_rate=RISK_FREE_RATE)
    plot_regime_visualization(spy_prices["SPY"], (rf_wide > 0).astype(int), BASE_RESULTS)
    plot_feature_importance(get_feature_importance(X_wide, y_wide), BASE_RESULTS)

    print(f"\n{'='*90}")
    print("PERFORMANCE SUMMARY")
    print("="*90)
    summary = compare_strategies(
        {col: results[col].dropna() for col in results.columns},
        risk_free_rate=RISK_FREE_RATE,
    )
    try:
        print(summary.to_markdown())
    except ImportError:
        print(summary.to_string())

    print(f"\nAll figures saved under {BASE_RESULTS}")


if __name__ == "__main__":
    main()
