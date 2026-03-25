import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

def create_labels(prices: pd.DataFrame, strategy_weights: pd.DataFrame, horizon: int = 21, threshold: float = -0.02) -> pd.Series:
    """Creates forward-looking labels based on 1-month strategy performance."""
    
    # --- THESE 3 LINES WERE MISSING ---
    daily_returns = prices.pct_change()
    common_idx = daily_returns.index.intersection(strategy_weights.index)
    strat_daily_returns = (strategy_weights.loc[common_idx] * daily_returns.loc[common_idx]).sum(axis=1)
    # ----------------------------------
    
    forward_returns = strat_daily_returns.rolling(window=horizon).sum().shift(-horizon)    
    
    # 1 means Safe (Returns are better than -2%). 0 means CRASH (Returns are worse than -2%)
    labels = (forward_returns > threshold).astype(float)
    labels[forward_returns.isna()] = np.nan
    return labels

def train_and_predict_walk_forward(X: pd.DataFrame, y: pd.Series, model_type: str = 'rf'):
    """Walk-Forward TimeSeries validation to prevent data leakage."""
    valid_mask = y.notna()
    X_clean, y_clean = X[valid_mask], y[valid_mask]
    predictions = pd.Series(index=X_clean.index, dtype=float)
    
    tscv = TimeSeriesSplit(n_splits=10) 
    scaler = StandardScaler()

    for train_index, test_index in tscv.split(X_clean):
        X_train, X_test = X_clean.iloc[train_index], X_clean.iloc[test_index]
        y_train = y_clean.iloc[train_index]
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced', random_state=42, n_jobs=-1)
        else:
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        
        model.fit(X_train_scaled, y_train)
        predictions.iloc[test_index] = model.predict_proba(X_test_scaled)[:, 1]
        
    return predictions

def get_feature_importance(X: pd.DataFrame, y: pd.Series):
    """Retrieves global feature importance."""
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)