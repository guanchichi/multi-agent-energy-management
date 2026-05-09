import numpy as np
import pandas as pd
from scipy import stats

from src.config import DATA_RAW, DATA_PROC, TARGET_COL, OUTLIER_Z_THRESH, OUTLIER_WINDOW


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"]        = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = df["date"].dt.month
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    T   = df["T_out"]
    RH  = df["RH_out"]
    WS  = df["Windspeed"]

    df["discomfort_index"] = T - 0.55 * (1 - 0.01 * RH) * (T - 14.5)

    e = (RH / 100) * 6.105 * np.exp(17.27 * T / (237.7 + T))
    df["apparent_temp"] = T + 0.33 * e - 0.7 * WS - 4.0

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.ffill().bfill()


def smooth_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace |z| > threshold in TARGET_COL with local median of ±OUTLIER_WINDOW rows."""
    df = df.copy()
    col = df[TARGET_COL].astype(float).copy()
    z   = np.abs(stats.zscore(col))
    outlier_idx = np.where(z > OUTLIER_Z_THRESH)[0]

    for i in outlier_idx:
        lo = max(0, i - OUTLIER_WINDOW)
        hi = min(len(col), i + OUTLIER_WINDOW + 1)
        neighbors = np.concatenate([col.iloc[lo:i].values, col.iloc[i+1:hi].values])
        if len(neighbors) > 0:
            col.iloc[i] = np.median(neighbors)

    df[TARGET_COL] = col
    return df


def drop_noise_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["rv1", "rv2"], errors="ignore")


def run(save: bool = True) -> pd.DataFrame:
    df = load_raw()
    df = add_time_features(df)
    df = add_weather_features(df)
    df = handle_missing(df)
    df = smooth_outliers(df)
    df = drop_noise_cols(df)

    if save:
        DATA_PROC.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROC, index=False)
        print(f"Saved → {DATA_PROC}  shape={df.shape}")

    return df


if __name__ == "__main__":
    run()
