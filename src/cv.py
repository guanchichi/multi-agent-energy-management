"""
5-fold TimeSeriesSplit cross-validation for ML baseline models.
Replicates the Durrani et al. (2025) methodology validation.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from src.config import LOG_TRANSFORM, TARGET_COL, TIMESTEPS, RESULTS_METRICS
from src.sequences import load_processed, split_features_target, make_sequences
from src.models.ml_models import (
    train_linear_regression, train_random_forest, train_svr,
    TFkNN, K_NEIGHBORS, flatten,
)

N_SPLITS = 5


def _inverse_transform(tgt_sc: MinMaxScaler, y_scaled: np.ndarray) -> np.ndarray:
    y = tgt_sc.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    if LOG_TRANSFORM:
        y = np.expm1(y)
    return y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def _fold_data(X_tr_raw, X_te_raw, y_tr_raw, y_te_raw_all, feature_cols):
    """Fit scalers on train fold, build sequences; return raw Wh labels for test."""
    target_idx = feature_cols.index(TARGET_COL)

    if LOG_TRANSFORM:
        X_tr = X_tr_raw.copy(); X_tr[:, target_idx] = np.log1p(X_tr_raw[:, target_idx])
        X_te = X_te_raw.copy(); X_te[:, target_idx] = np.log1p(X_te_raw[:, target_idx])
        y_tr = np.log1p(y_tr_raw)
    else:
        X_tr, X_te = X_tr_raw, X_te_raw
        y_tr = y_tr_raw

    feat_sc = MinMaxScaler().fit(X_tr)
    tgt_sc  = MinMaxScaler().fit(y_tr.reshape(-1, 1))

    X_tr_sc = feat_sc.transform(X_tr)
    X_te_sc = feat_sc.transform(X_te)
    y_tr_sc = tgt_sc.transform(y_tr.reshape(-1, 1))    # shape (N, 1)

    X_tr_seq, y_tr_seq = make_sequences(X_tr_sc, y_tr_sc)
    # Placeholder y for test — only X_te_seq is used
    X_te_seq, _        = make_sequences(X_te_sc, np.zeros((len(X_te_sc), 1)))

    # Raw Wh values aligned to test sequences:
    # sequence i targets row i+TIMESTEPS → indices [TIMESTEPS, TIMESTEPS+len(X_te_seq))
    y_te_raw = y_te_raw_all[TIMESTEPS : TIMESTEPS + len(X_te_seq)]

    return X_tr_seq, y_tr_seq, X_te_seq, y_te_raw, tgt_sc


def run_cv(save: bool = True) -> pd.DataFrame:
    df = load_processed()
    X, y, feature_cols = split_features_target(df)
    y = y.flatten()

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    records = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        if len(te_idx) <= TIMESTEPS:
            print(f"Fold {fold}: test too small ({len(te_idx)} rows), skipping.")
            continue

        print(f"\n--- Fold {fold}/{N_SPLITS}  (train={len(tr_idx):,}  test={len(te_idx):,}) ---")

        X_tr_seq, y_tr_seq, X_te_seq, y_te_raw, tgt_sc = _fold_data(
            X[tr_idx], X[te_idx], y[tr_idx], y[te_idx], feature_cols,
        )
        X_tr_flat = flatten(X_tr_seq)
        X_te_flat = flatten(X_te_seq)

        # LinearRegression
        lr = train_linear_regression(X_tr_flat, y_tr_seq)
        y_pred = _inverse_transform(tgt_sc, lr.predict(X_te_flat, verbose=0).flatten())
        m = _metrics(y_te_raw, y_pred)
        records.append({"fold": fold, "model": "LinearRegression", **m})
        print(f"  LinearRegression  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

        # RandomForest (XGBoost GPU)
        rf = train_random_forest(X_tr_flat, y_tr_seq)
        y_pred = _inverse_transform(tgt_sc, rf.predict(xgb.DMatrix(X_te_flat)))
        m = _metrics(y_te_raw, y_pred)
        records.append({"fold": fold, "model": "RandomForest", **m})
        print(f"  RandomForest      MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

        # SVR (TF Huber)
        svr = train_svr(X_tr_flat, y_tr_seq)
        y_pred = _inverse_transform(tgt_sc, svr.predict(X_te_flat, verbose=0).flatten())
        m = _metrics(y_te_raw, y_pred)
        records.append({"fold": fold, "model": "SVR", **m})
        print(f"  SVR               MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

        # kNN (last-timestep 32-dim, k=15)
        knn = TFkNN(k=K_NEIGHBORS)
        knn.fit(X_tr_seq[:, -1, :], y_tr_seq)
        y_pred = _inverse_transform(tgt_sc, knn.predict(X_te_seq[:, -1, :]))
        m = _metrics(y_te_raw, y_pred)
        records.append({"fold": fold, "model": "kNN", **m})
        print(f"  kNN               MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

    df_cv = pd.DataFrame(records)

    summary = (
        df_cv.groupby("model")[["MAE", "RMSE", "R2"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]

    print("\n=== 5-fold CV Summary (mean ± std) ===")
    for model, row in summary.iterrows():
        print(
            f"  {model:20s}  "
            f"MAE={row['MAE_mean']:.2f}±{row['MAE_std']:.2f}  "
            f"RMSE={row['RMSE_mean']:.2f}±{row['RMSE_std']:.2f}  "
            f"R2={row['R2_mean']:.4f}±{row['R2_std']:.4f}"
        )

    if save:
        RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
        df_cv.to_csv(RESULTS_METRICS / "cv_results.csv", index=False)
        summary.to_csv(RESULTS_METRICS / "cv_summary.csv")
        print(f"\nSaved: {RESULTS_METRICS / 'cv_results.csv'}")
        print(f"Saved: {RESULTS_METRICS / 'cv_summary.csv'}")

    return df_cv


if __name__ == "__main__":
    run_cv()
