import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DATA_PROC, RESULTS_MODELS,
    TARGET_COL, TRAIN_RATIO, TIMESTEPS, LOG_TRANSFORM,
)


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROC, parse_dates=["date"])
    return df


def split_features_target(df: pd.DataFrame):
    # Include TARGET_COL (past values) in X — strong autocorrelation signal.
    # The sequence window [t-23 … t] feeds into predicting t+24.
    feature_cols = [c for c in df.columns if c != "date"]
    X = df[feature_cols].values
    y = df[[TARGET_COL]].values   # keep 2-D for scaler
    return X, y, feature_cols


def train_test_split_timeseries(X: np.ndarray, y: np.ndarray, ratio: float = TRAIN_RATIO):
    split = int(len(X) * ratio)
    return X[:split], X[split:], y[:split], y[split:]


def fit_scalers(X_train: np.ndarray, y_train_transformed: np.ndarray):
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()
    feature_scaler.fit(X_train)
    target_scaler.fit(y_train_transformed)
    return feature_scaler, target_scaler


def scale(X_train, X_test, y_train, y_test, feature_scaler, target_scaler):
    return (
        feature_scaler.transform(X_train),
        feature_scaler.transform(X_test),
        target_scaler.transform(y_train),
        target_scaler.transform(y_test),
    )


def make_sequences(X: np.ndarray, y: np.ndarray, timesteps: int = TIMESTEPS):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i : i + timesteps])
        ys.append(y[i + timesteps, 0])
    return np.array(Xs), np.array(ys)


def save_scalers(feature_scaler: MinMaxScaler, target_scaler: MinMaxScaler) -> None:
    RESULTS_MODELS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_MODELS / "feature_scaler.pkl", "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(RESULTS_MODELS / "target_scaler.pkl", "wb") as f:
        pickle.dump(target_scaler, f)


def load_scalers():
    with open(RESULTS_MODELS / "feature_scaler.pkl", "rb") as f:
        feature_scaler = pickle.load(f)
    with open(RESULTS_MODELS / "target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)
    return feature_scaler, target_scaler


def run(timesteps: int = TIMESTEPS, save: bool = True):
    df = load_processed()
    X, y, feature_cols = split_features_target(df)

    X_train, X_test, y_train_raw, y_test_raw_all = train_test_split_timeseries(X, y)

    # Optional log1p transform on target before scaling
    if LOG_TRANSFORM:
        y_train_t = np.log1p(y_train_raw)
        y_test_t  = np.log1p(y_test_raw_all)
        # Also log1p the Appliances column inside X (same index as TARGET_COL in feature_cols)
        target_idx = feature_cols.index(TARGET_COL)
        X_train_t = X_train.copy()
        X_test_t  = X_test.copy()
        X_train_t[:, target_idx] = np.log1p(X_train[:, target_idx])
        X_test_t[:, target_idx]  = np.log1p(X_test[:, target_idx])
    else:
        y_train_t = y_train_raw
        y_test_t  = y_test_raw_all
        X_train_t = X_train
        X_test_t  = X_test

    feature_scaler, target_scaler = fit_scalers(X_train_t, y_train_t)
    X_train_sc, X_test_sc, y_train_sc, y_test_sc = scale(
        X_train_t, X_test_t, y_train_t, y_test_t, feature_scaler, target_scaler
    )

    X_train_seq, y_train_seq = make_sequences(X_train_sc, y_train_sc, timesteps)
    X_test_seq,  y_test_seq  = make_sequences(X_test_sc,  y_test_sc,  timesteps)

    # y_test_raw: original Wh values (un-transformed), aligned to sequences
    _, y_test_raw = make_sequences(X_test_sc, y_test_raw_all, timesteps)

    if save:
        save_scalers(feature_scaler, target_scaler)

    print(f"Train sequences : {X_train_seq.shape}  y={y_train_seq.shape}")
    print(f"Test  sequences : {X_test_seq.shape}   y={y_test_seq.shape}")
    print(f"Log transform   : {LOG_TRANSFORM}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    return (
        X_train_seq, y_train_seq,
        X_test_seq,  y_test_seq,
        y_test_raw,
        feature_scaler, target_scaler,
        feature_cols,
    )


if __name__ == "__main__":
    run()
