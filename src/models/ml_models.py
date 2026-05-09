"""
GPU-accelerated baseline models:
  LinearRegression — TF Dense(1) + MSE (exact equivalent, runs on GPU)
  RandomForest     — XGBoost device='cuda' (tree ensemble on GPU)
  SVR              — TF Dense(1) + Huber loss (approximates SVR, runs on GPU)
  kNN              — TF brute-force kNN via GPU matrix ops
"""
import numpy as np
import pickle
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

from src.config import RESULTS_MODELS, RESULTS_METRICS, LOG_TRANSFORM
from src.sequences import run as build_sequences


def _inverse(target_scaler, y_scaled: np.ndarray) -> np.ndarray:
    """Inverse-scale and optionally undo log1p transform."""
    y = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    if LOG_TRANSFORM:
        y = np.expm1(y)
    return y

K_NEIGHBORS = 15
XGB_ROUNDS  = 300


# ── helpers ───────────────────────────────────────────────────────────────────

def flatten(X_seq: np.ndarray) -> np.ndarray:
    return X_seq.reshape(len(X_seq), -1)


def _save_pkl(name: str, obj) -> None:
    RESULTS_MODELS.mkdir(parents=True, exist_ok=True)
    path = RESULTS_MODELS / f"{name.lower()}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved {name} → {path}")


def _save_pred(name: str, y_pred: np.ndarray) -> None:
    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    path = RESULTS_METRICS / f"{name.lower()}_predictions.npy"
    np.save(path, y_pred)
    print(f"Predictions saved → {path}")


# ── LinearRegression via TF (GPU) ─────────────────────────────────────────────

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
    n_feat = X_train.shape[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(n_feat,)),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=0,
              validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    return model


# ── RandomForest via XGBoost GPU ─────────────────────────────────────────────

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # True RF mode: 1 boosting round, 100 parallel trees per round
    # (equivalent to sklearn RandomForestRegressor(n_estimators=100))
    params = {
        "tree_method":       "hist",
        "device":            "cuda",
        "objective":         "reg:squarederror",
        "max_depth":         6,
        "learning_rate":     1.0,
        "subsample":         0.8,
        "colsample_bynode":  0.8,
        "num_parallel_tree": 100,
        "verbosity":         0,
    }
    model = xgb.train(params, dtrain, num_boost_round=1)
    return model


# ── SVR via TF (GPU) ──────────────────────────────────────────────────────────

def train_svr(X_train: np.ndarray, y_train: np.ndarray) -> keras.Model:
    n_feat = X_train.shape[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(n_feat,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.Huber(delta=0.1))
    model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=0,
              validation_split=0.1,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    return model


# ── kNN via TF GPU (brute-force) ──────────────────────────────────────────────

class TFkNN:
    """GPU brute-force kNN using TensorFlow distance computation."""

    def __init__(self, k: int = K_NEIGHBORS):
        self.k = k
        self._X_train: tf.Tensor | None = None
        self._y_train: tf.Tensor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TFkNN":
        self._X_train = tf.constant(X, dtype=tf.float32)
        self._y_train = tf.constant(y.flatten(), dtype=tf.float32)
        return self

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Uses ||a-b||² = ||a||² + ||b||² - 2·aᵀb to avoid materialising a
        3-D diff tensor (which would be ~21 GB for this dataset).
        """
        X_t = tf.constant(X, dtype=tf.float32)
        train_sq = tf.reduce_sum(tf.square(self._X_train), axis=1, keepdims=True)  # (N,1)
        preds = []
        for start in range(0, len(X_t), batch_size):
            batch = X_t[start : start + batch_size]                     # (B, F)
            batch_sq = tf.reduce_sum(tf.square(batch), axis=1, keepdims=True)  # (B,1)
            # (B, N)  — no 3-D tensor
            dists = batch_sq + tf.transpose(train_sq) - 2.0 * tf.matmul(batch, self._X_train, transpose_b=True)
            _, idx = tf.math.top_k(-dists, k=self.k)                    # (B, k)
            nn_vals = tf.gather(self._y_train, idx)                      # (B, k)
            preds.append(tf.reduce_mean(nn_vals, axis=-1).numpy())
        return np.concatenate(preds)


def train_knn(X_train: np.ndarray, y_train: np.ndarray) -> TFkNN:
    model = TFkNN(k=K_NEIGHBORS)
    model.fit(X_train, y_train)
    return model


# ── unified runner ────────────────────────────────────────────────────────────

def run():
    (
        X_train_seq, y_train_seq,
        X_test_seq,  _,
        y_test_raw,
        feature_scaler, target_scaler,
        _,
    ) = build_sequences(save=False)

    X_train_flat = flatten(X_train_seq)
    X_test_flat  = flatten(X_test_seq)

    predictions = {}

    # --- LinearRegression (TF GPU) ---
    print("\nTraining LinearRegression (TF GPU)...")
    lr = train_linear_regression(X_train_flat, y_train_seq)
    _save_pkl("LinearRegression", lr)
    y_pred = _inverse(target_scaler, lr.predict(X_test_flat, verbose=0))
    predictions["LinearRegression"] = y_pred
    _save_pred("LinearRegression", y_pred)

    # --- RandomForest (XGBoost GPU) ---
    print("\nTraining RandomForest (XGBoost GPU)...")
    rf = train_random_forest(X_train_flat, y_train_seq)
    _save_pkl("RandomForest", rf)
    y_pred = _inverse(target_scaler, rf.predict(xgb.DMatrix(X_test_flat)))
    predictions["RandomForest"] = y_pred
    _save_pred("RandomForest", y_pred)

    # --- SVR (TF GPU) ---
    print("\nTraining SVR (TF GPU)...")
    svr = train_svr(X_train_flat, y_train_seq)
    _save_pkl("SVR", svr)
    y_pred = _inverse(target_scaler, svr.predict(X_test_flat, verbose=0))
    predictions["SVR"] = y_pred
    _save_pred("SVR", y_pred)

    # --- kNN (TF GPU, last-timestep 32-dim features) ---
    # kNN is a tabular method — using the last timestep (32 features) avoids
    # the curse of dimensionality that kills brute-force kNN on 768-dim sequences.
    print("\nTraining kNN (TF GPU, last-timestep 32-dim)...")
    X_train_knn = X_train_seq[:, -1, :]   # (N, 32)
    X_test_knn  = X_test_seq[:, -1, :]    # (N, 32)
    knn = TFkNN(k=K_NEIGHBORS)
    knn.fit(X_train_knn, y_train_seq)
    _save_pkl("kNN", knn)
    y_pred = _inverse(target_scaler, knn.predict(X_test_knn))
    predictions["kNN"] = y_pred
    _save_pred("kNN", y_pred)

    np.save(RESULTS_METRICS / "y_test_raw.npy", y_test_raw)

    return predictions, y_test_raw


if __name__ == "__main__":
    run()
