import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def create_labels(prices: pd.DataFrame, positions: pd.DataFrame, forward_window: int = 21) -> pd.Series:
    """
    Creates the target variable (y) for our ML model.
    Label = 1 if the momentum strategy makes a positive return over the next 21 days, else 0.
    """
    # 1. Align the signals
    # 'positions' from Day 3 are already shifted to be executed on Day T+1.
    # To evaluate the signal generated at the close of Day T, we temporarily "unshift" it.
    signals_at_t = positions.shift(-1)
    
    # 2. Calculate the actual forward return of the assets from Close T to Close T+21
    # .shift(-forward_window) pulls the future price back to today's row to compute the return
    forward_asset_returns = prices.pct_change(forward_window).shift(-forward_window)
    
    # 3. Calculate how our portfolio would perform over those 21 days
    strategy_fwd_returns = (signals_at_t * forward_asset_returns).sum(axis=1)
    
    # 4. Create binary labels: 1 if profitable, 0 if flat or negative
    labels = (strategy_fwd_returns > 0).astype(int)
    
    # 5. Erase the last 21 days because we don't have future data to label them!
    labels.iloc[-(forward_window+1):] = np.nan
    
    return labels.rename('target')

def train_and_predict_regime(features: pd.DataFrame, labels: pd.Series, train_size: float = 0.7):
    """
    Trains a Random Forest to predict the regime and generates trading signals.
    """
    # Combine features and labels, dropping any rows with NaNs to ensure perfect alignment
    data = pd.concat([features, labels], axis=1).dropna()
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    # =====================================================================
    # PARANOID QUANT CHECK: CHRONOLOGICAL SPLIT
    # =====================================================================
    # NEVER use sklearn's default train_test_split on financial data!
    # It shuffles the data randomly, meaning you will train the model on data from 2020 
    # to predict prices in 2015. We must split strictly by time.
    split_idx = int(len(data) * train_size)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training ML on {len(X_train)} days (Past), Testing on {len(X_test)} days (Future)...")
    
    # Initialize Random Forest. 
    # max_depth=3 is CRITICAL. Financial data is extremely noisy. 
    # Deep trees will memorize the noise (overfitting). Shallow trees find robust, general rules.
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate out-of-sample
    test_preds = rf.predict(X_test)
    print("\n--- ML Model Out-Of-Sample Performance ---")
    print(classification_report(y_test, test_preds))
    
    # Generate predictions for the entire timeline to use in our backtest
    all_preds = rf.predict(X)
    regime_signals = pd.Series(all_preds, index=X.index, name="regime_signal")
    
    return regime_signals, rf