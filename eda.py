import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_loader import load_data

TICKERS    = ["TLT", "USO", "DBA", "SLV", "FXY", "XBI", "FXI", "UNG", "VNQ",
              "SPY", "QQQ", "GLD"]
START_DATE = "2010-01-01"

TICKER_LABELS = {
    "TLT": "TLT (20yr Treasury)",
    "USO": "USO (Oil)",
    "DBA": "DBA (Agriculture)",
    "SLV": "SLV (Silver)",
    "XBI": "XBI (Biotech)",
    "FXI": "FXI (China)",
    "UNG": "UNG (Natural Gas)",
    "FXY": "FXY (Japanese Yen)",
    "VNQ": "VNQ (REITs)",
    "SPY": "SPY (S&P 500)",
    "QQQ": "QQQ (Nasdaq 100)",
    "GLD": "GLD (Gold)",
}

os.makedirs("results/eda", exist_ok=True)


def plot_correlation_heatmap(corr: pd.DataFrame, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(title, fontsize=13, pad=12)
    tick_labels = [TICKER_LABELS.get(t, t) for t in corr.columns]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_rolling_correlation(returns: pd.DataFrame, window: int = 63) -> None:
    pairs = [
        ("TLT", "VNQ"),
        ("TLT", "FXY"),
        ("FXY", "SLV"),
        ("USO", "UNG"),
        ("SLV", "DBA"),
    ]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 10), sharex=True)
    for ax, (a, b) in zip(axes, pairs):
        roll_corr = returns[a].rolling(window).corr(returns[b])
        ax.plot(roll_corr.index, roll_corr, linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{a}/{b}", fontsize=9)
        ax.set_ylim(-1, 1)
    axes[0].set_title(f"{window}-day Rolling Pairwise Correlations", fontsize=13)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    path = "results/eda/rolling_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path}")


def print_correlation_summary(corr: pd.DataFrame) -> None:
    corr_unnamed = corr.copy()
    corr_unnamed.index.name = None
    corr_unnamed.columns.name = None
    upper = corr_unnamed.where(np.tril(np.ones(corr_unnamed.shape), k=-1).astype(bool))
    pairs = (
        upper.stack().dropna()
        .reset_index()
        .rename(columns={"level_0": "Asset A", "level_1": "Asset B", 0: "Correlation"})
        .sort_values("Correlation", ascending=False)
    )
    print("\nTop 5 most correlated pairs:")
    print(pairs.head(5).to_string(index=False))
    print("\nTop 5 least correlated (most diversifying) pairs:")
    print(pairs.tail(5).to_string(index=False))


def main() -> None:
    prices = load_data(TICKERS, START_DATE)
    returns = prices.pct_change().dropna()

    # Full-period return correlation
    corr_full = returns.corr()
    plot_correlation_heatmap(
        corr_full,
        "Return Correlation — Full Period (2010–present)",
        "results/eda/correlation_full.png",
    )

    # Last 3-year correlation
    cutoff = returns.index[-1] - pd.DateOffset(years=3)
    corr_recent = returns.loc[returns.index >= cutoff].corr()
    plot_correlation_heatmap(
        corr_recent,
        "Return Correlation — Last 3 Years",
        "results/eda/correlation_recent.png",
    )

    # Rolling correlations for key pairs
    plot_rolling_correlation(returns, window=63)

    # Summary stats
    print("\n=== Full-period correlation matrix ===")
    print(corr_full.round(2).to_string())
    print_correlation_summary(corr_full)


if __name__ == "__main__":
    main()
