import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from src.config import (
    RESULTS_MODELS, RESULTS_METRICS,
    TIMESTEPS,
    LSTM_UNITS_1, LSTM_UNITS_2, DENSE_UNITS, DROPOUT_RATE,
    L2_REG, LEARNING_RATE, BATCH_SIZE, EPOCHS, PATIENCE, VAL_SPLIT,
    LOG_TRANSFORM,
)
from src.sequences import run as build_sequences


def build_model(n_features: int, timesteps: int = TIMESTEPS) -> keras.Model:
    reg = keras.regularizers.l2(L2_REG)
    model = keras.Sequential([
        keras.layers.Input(shape=(timesteps, n_features)),
        keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True,
                          kernel_regularizer=reg, recurrent_regularizer=reg),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.LSTM(LSTM_UNITS_2,
                          kernel_regularizer=reg, recurrent_regularizer=reg),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(DENSE_UNITS, activation="relu",
                           kernel_regularizer=reg),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


def train(model: keras.Model, X_train, y_train):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_model(model: keras.Model) -> None:
    RESULTS_MODELS.mkdir(parents=True, exist_ok=True)
    path = RESULTS_MODELS / "lstm.h5"
    model.save(path)
    print(f"Model saved → {path}")


def save_history(history) -> None:
    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history.history)
    df.index.name = "epoch"
    path = RESULTS_METRICS / "lstm_history.csv"
    df.to_csv(path)
    print(f"History saved → {path}")


def load_model() -> keras.Model:
    return keras.models.load_model(RESULTS_MODELS / "lstm.h5")


def predict_inverse(model: keras.Model, X_test, target_scaler) -> np.ndarray:
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_transformed = target_scaler.inverse_transform(y_pred_scaled).flatten()
    if LOG_TRANSFORM:
        return np.expm1(y_pred_transformed)
    return y_pred_transformed


def run():
    (
        X_train_seq, y_train_seq,
        X_test_seq,  _y_test_seq,
        y_test_raw,
        feature_scaler, target_scaler,
        feature_cols,
    ) = build_sequences()

    n_features = X_train_seq.shape[2]
    model = build_model(n_features)
    model.summary()

    print(f"\nTraining LSTM  (log_transform={LOG_TRANSFORM})...")
    history = train(model, X_train_seq, y_train_seq)

    save_model(model)
    save_history(history)

    y_pred = predict_inverse(model, X_test_seq, target_scaler)

    RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_METRICS / "lstm_predictions.npy", y_pred)
    np.save(RESULTS_METRICS / "y_test_raw.npy", y_test_raw)
    print(f"Predictions saved → {RESULTS_METRICS / 'lstm_predictions.npy'}")

    return model, y_pred, y_test_raw, target_scaler


if __name__ == "__main__":
    run()
