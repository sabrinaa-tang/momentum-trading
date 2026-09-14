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
from ts_momentum_strategy import generate_timeseries_momentum_signals
from ml_model import create_labels, train_and_predict_walk_forward, get_feature_importance
from backtest import run_backtest
from evaluation import (
    compare_strategies,
    plot_performance,
    plot_rolling_sharpe,
    plot_regime_visualization,
    plot_feature_importance,
    plot_universe_comparison,
)

# ── Universe definitions ──────────────────────────────────────────────────────
STANDARD_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "USO"]
HIGHVOL_TICKERS  = ["XBI", "GDX", "EWZ", "FXI", "XOP"]

START_DATE     = "2010-01-01"
BASE_RESULTS   = "results/figures/"
RISK_FREE_RATE = 0.02

LABEL_HORIZON       = 21
LABEL_THRESHOLD     = -0.02
LABEL_TYPE          = "median"
TC                  = 0.001
ML_PROB_THRESHOLD   = 0.50
ML_PROB_PARTIAL     = 0.35
ML_SMOOTHING_WINDOW = 5
MAX_ASSET_WEIGHT    = 0.40


def build_ml_regime(
    probs: pd.Series,
    target_index: pd.Index,
    high: float = ML_PROB_THRESHOLD,
    low: float  = ML_PROB_PARTIAL,
    smooth: int = ML_SMOOTHING_WINDOW,
) -> pd.Series:
    """
    Converts raw walk-forward probabilities into a {0, 0.5, 1.0} regime scalar.

    Warmup rows (NaN predictions) default to 1.0 — full momentum exposure.
    Smoothing window reduces transaction cost drag from daily regime flips.
    """
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


def run_universe_pipeline(
    tickers: list,
    spy_prices: pd.DataFrame,
    universe_label: str,
    save_dir: str,
) -> tuple:
    """
    Runs the full momentum strategy pipeline for a given ticker universe.

    spy_prices is loaded externally so SPY buy-and-hold is always the same
    benchmark regardless of whether SPY is in the trading universe.

    Returns (results_aligned, results_df, rf_regime, X_ml, y_ml, ml_start).
    """
    print(f"\n{'='*70}")
    print(f"  UNIVERSE: {universe_label}")
    print(f"  Tickers:  {tickers}")
    print(f"{'='*70}")
    os.makedirs(save_dir, exist_ok=True)

    prices         = load_data(tickers, START_DATE)
    features_daily = build_all_features(prices)

    signals      = generate_trend_signals(features_daily, tickers)
    base_weights = calculate_inverse_vol_weights(
        signals, features_daily, tickers, max_weight=MAX_ASSET_WEIGHT
    )

    print("\nBuilding time-series momentum signals...")
    ts_signals = generate_timeseries_momentum_signals(features_daily, tickers)
    ts_weights = calculate_inverse_vol_weights(
        ts_signals, features_daily, tickers, max_weight=MAX_ASSET_WEIGHT
    )

    labels = create_labels(
        prices, base_weights,
        horizon=LABEL_HORIZON,
        threshold=LABEL_THRESHOLD,
        label_type=LABEL_TYPE,
    )

    common_idx = features_daily.index.intersection(labels.dropna().index)
    X_ml = features_daily.loc[common_idx]
    y_ml = labels.loc[common_idx]
    print(f"\nML training set: {len(X_ml)} days "
          f"({X_ml.index[0].date()} → {X_ml.index[-1].date()})")

    rf_probs = train_and_predict_walk_forward(
        X_ml, y_ml, model_type="rf",       horizon=LABEL_HORIZON
    )
    lr_probs = train_and_predict_walk_forward(
        X_ml, y_ml, model_type="logistic", horizon=LABEL_HORIZON
    )

    print("\nBuilding ML regime signals...")
    rf_regime = build_ml_regime(rf_probs, base_weights.index)
    lr_regime = build_ml_regime(lr_probs, base_weights.index)

    ml_rf_weights = base_weights.multiply(rf_regime, axis=0)
    ml_lr_weights = base_weights.multiply(lr_regime, axis=0)

    weights_start = base_weights.index[0]
    daily_rets    = prices.pct_change().dropna().loc[weights_start:]
    spy_rets      = spy_prices.pct_change().dropna()["SPY"].reindex(daily_rets.index)

    print("\nRunning backtests...")
    results_dict = {
        "1_EqWeight":    run_backtest(
            daily_rets, generate_equal_weight_baseline(daily_rets.index, tickers), tc=TC
        ),
        "2_SPY_BuyHold": spy_rets,
        "3_Momentum":    run_backtest(daily_rets, base_weights, tc=TC),
        "4_Mom_LogReg":  run_backtest(daily_rets, ml_lr_weights, tc=TC),
        "5_Mom_RF":      run_backtest(daily_rets, ml_rf_weights, tc=TC),
        "6_Random":      run_backtest(
            daily_rets, generate_random_strategy(daily_rets.index, tickers), tc=TC
        ),
        "7_TS_Momentum": run_backtest(daily_rets, ts_weights, tc=TC),
    }
    results_df = pd.DataFrame(results_dict)

    ml_start = results_df.apply(lambda c: c.first_valid_index()).max()
    print(f"\n[INFO] Full history:     {results_df.index[0].date()} → "
          f"{results_df.index[-1].date()}")
    print(f"[INFO] ML window starts: {ml_start.date()} "
          f"({(results_df.index >= ml_start).sum()} trading days)")

    results_aligned = results_df.loc[ml_start:]

    print(f"\nGenerating plots → {save_dir}")
    plot_performance(results_aligned, save_dir)
    plot_rolling_sharpe(results_aligned, save_dir, risk_free_rate=RISK_FREE_RATE)
    plot_regime_visualization(spy_prices["SPY"], (rf_regime > 0).astype(int), save_dir)
    plot_feature_importance(get_feature_importance(X_ml, y_ml), save_dir)

    return results_aligned, results_df, rf_regime, X_ml, y_ml, ml_start


def print_summary(
    label: str,
    results_aligned: pd.DataFrame,
    results_df: pd.DataFrame,
    ml_start: pd.Timestamp,
) -> None:
    print(f"\n{'='*90}")
    print(f"PERFORMANCE SUMMARY — {label}")
    print(f"ML window: {ml_start.date()} → {results_aligned.index[-1].date()}")
    print("="*90)
    summary = compare_strategies(
        {col: results_aligned[col] for col in results_aligned.columns},
        risk_free_rate=RISK_FREE_RATE,
    )
    try:
        print(summary.to_markdown())
    except ImportError:
        print(summary.to_string())

    non_ml = ["1_EqWeight", "2_SPY_BuyHold", "3_Momentum"]
    print(f"\n{'='*90}")
    print(f"FULL HISTORY — {label}")
    print(f"{results_df.index[0].date()} → {results_df.index[-1].date()} "
          f"— non-ML strategies only")
    print("="*90)
    summary_full = compare_strategies(
        {col: results_df[col].dropna() for col in non_ml},
        risk_free_rate=RISK_FREE_RATE,
    )
    try:
        print(summary_full.to_markdown())
    except ImportError:
        print(summary_full.to_string())


def main() -> None:
    # SPY is loaded once and shared as the buy-and-hold benchmark for both universes
    spy_prices = load_data(["SPY"], START_DATE)

    std_aligned, std_full, _, _, _, std_ml_start = run_universe_pipeline(
        STANDARD_TICKERS,
        spy_prices,
        "Standard ETFs (SPY / QQQ / TLT / GLD / USO)",
        os.path.join(BASE_RESULTS, "standard"),
    )

    hv_aligned, hv_full, _, _, _, hv_ml_start = run_universe_pipeline(
        HIGHVOL_TICKERS,
        spy_prices,
        "High-Vol ETFs (XBI / GDX / EWZ / FXI / XOP)",
        os.path.join(BASE_RESULTS, "highvol"),
    )

    print_summary("Standard ETFs (SPY/QQQ/TLT/GLD/USO)", std_aligned, std_full, std_ml_start)
    print_summary("High-Vol ETFs (XBI/GDX/EWZ/FXI/XOP)", hv_aligned,  hv_full,  hv_ml_start)

    comparison_dir = os.path.join(BASE_RESULTS, "comparison")
    plot_universe_comparison(
        std_aligned, hv_aligned,
        comparison_dir,
        risk_free_rate=RISK_FREE_RATE,
    )

    print(f"\nAll figures saved under {BASE_RESULTS}")
    print("  standard/   — Standard ETF universe")
    print("  highvol/    — High-Vol ETF universe")
    print("  comparison/ — Cross-universe comparison charts")


if __name__ == "__main__":
    main()
