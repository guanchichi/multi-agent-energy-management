"""
Reproduce key figures from Durrani et al. (2025):
  Fig 12 — Model comparison (MAE / RMSE / R²) bar charts
  Fig 14 — LSTM training/validation loss curve
  Fig 15 — Actual vs Predicted (LSTM, no DR)
  Fig 16 — DR strategy comparison bar chart (LSTM + 7 strategies)
  Fig 17 — Box-plot of prediction errors per model
  Fig 18 — Heatmap: R² for all model × strategy combos
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.config import RESULTS_FIGURES, RESULTS_METRICS

FIG_DIR = RESULTS_FIGURES / "paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS    = ["LSTM", "LinearRegression", "RandomForest", "SVR", "kNN"]
STRATEGIES = [
    "Peak Clipping", "Valley Filling", "Load Shifting",
    "Load Leveling", "ToU Optimization", "Price Based", "Behavioral DR",
]

COLORS = plt.cm.tab10.colors


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def _load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_METRICS / "all_results.csv")


def _load_history() -> pd.DataFrame:
    return pd.read_csv(RESULTS_METRICS / "lstm_history.csv", index_col="epoch")


def _load_pred(name: str) -> np.ndarray:
    fname = "lstm" if name == "LSTM" else name.lower()
    return np.load(RESULTS_METRICS / f"{fname}_predictions.npy")


def _load_true() -> np.ndarray:
    return np.load(RESULTS_METRICS / "y_test_raw.npy")


# ── Fig 12 — model comparison ─────────────────────────────────────────────────

def fig12_model_comparison(df: pd.DataFrame):
    """Bar chart per metric, one bar per model (averaged over all DR strategies)."""
    grouped = df.groupby("Model")[["MAE", "RMSE", "R2"]].mean().reindex(MODELS).dropna()
    metrics = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, metrics):
        values = grouped[metric]
        bars = ax.bar(values.index, values.values, color=COLORS[:len(values)])
        ax.set_title(metric)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        for bar, v in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Fig 12 — Model Comparison (avg over DR strategies)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig12_model_comparison")


# ── Fig 14 — LSTM loss curve ──────────────────────────────────────────────────

def fig14_loss_curve(history: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.index + 1, history["loss"],     label="Train loss",      linewidth=2)
    ax.plot(history.index + 1, history["val_loss"], label="Val loss", linestyle="--", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Fig 14 — LSTM Training / Validation Loss")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig14_lstm_loss_curve")


# ── Fig 15 — Actual vs Predicted ──────────────────────────────────────────────

def fig15_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, n_plot: int = 1000):
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_plot)
    ax.plot(x, y_true[:n_plot], label="Actual",    linewidth=1.2, alpha=0.85)
    ax.plot(x, y_pred[:n_plot], label="Predicted", linewidth=1.2, alpha=0.85, linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Appliances Energy (Wh)")
    ax.set_title("Fig 15 — LSTM: Actual vs Predicted (first 1000 test steps)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig15_actual_vs_predicted")


# ── Fig 16 — DR strategy comparison (LSTM) ────────────────────────────────────

def fig16_dr_strategies(df: pd.DataFrame):
    lstm_df = df[df["Model"] == "LSTM"].set_index("Strategy").reindex(STRATEGIES)
    metrics = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metrics):
        values = lstm_df[metric]
        bars = ax.bar(values.index, values.values, color=COLORS[:len(values)])
        ax.set_title(metric)
        ax.set_xlabel("DR Strategy")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        for bar, v in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Fig 16 — LSTM: DR Strategy Comparison", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig16_dr_strategy_comparison")


# ── Fig 17 — Error box-plot per model ─────────────────────────────────────────

def fig17_error_boxplot():
    y_true = _load_true()
    errors = {}
    for name in MODELS:
        try:
            y_pred = _load_pred(name)
            n = min(len(y_true), len(y_pred))
            errors[name] = np.abs(y_true[:n] - y_pred[:n])
        except FileNotFoundError:
            pass
    if not errors:
        print("No prediction files found for box-plot — skipping.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(errors.values(), labels=errors.keys(), showfliers=False)
    ax.set_xlabel("Model")
    ax.set_ylabel("Absolute Error (Wh)")
    ax.set_title("Fig 17 — Absolute Error Distribution per Model")
    fig.tight_layout()
    _save(fig, "fig17_error_boxplot")


# ── Fig 18 — R² heatmap ───────────────────────────────────────────────────────

def fig18_r2_heatmap(df: pd.DataFrame):
    pivot = df.pivot(index="Model", columns="Strategy", values="R2")
    pivot = pivot.reindex(index=MODELS, columns=STRATEGIES)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="R²")
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES, rotation=35, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(MODELS)):
        for j in range(len(STRATEGIES)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Fig 18 — R² Heatmap: Model × DR Strategy")
    fig.tight_layout()
    _save(fig, "fig18_r2_heatmap")


# ── Fig 14b — LSTM DR time-series (4×2 subplot) ──────────────────────────────

def fig14_lstm_dr_timeseries():
    from src.dr_strategies import STRATEGIES as DR_FUNS

    y_true = _load_true()
    y_pred = _load_pred("LSTM")

    strategy_order = [
        "Peak Clipping", "Valley Filling", "Load Shifting",
        "Load Leveling", "ToU Optimization", "Price Based", "Behavioral DR",
    ]
    N = 300

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    for idx, name in enumerate(strategy_order):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        y_pred_dr = DR_FUNS[name](y_pred)

        ax.plot(y_true[:N],    color="steelblue", linestyle="--",
                linewidth=1,   alpha=0.7, label="Actual")
        ax.plot(y_pred_dr[:N], color="orange",    linestyle="-",
                linewidth=1.2, label=f"LSTM ({name})")

        ax.set_title(f"LSTM | DR Strategy: {name}", fontsize=10)
        ax.set_xlabel("Time Index", fontsize=9)
        ax.set_ylabel("Energy (Wh)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(labelsize=8)

    # 第 8 格（右下）留空
    axes[3, 1].axis("off")

    fig.suptitle("LSTM Model Evaluation with DR Strategies",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig14_lstm_dr_timeseries")


# ── entry point ───────────────────────────────────────────────────────────────

def run():
    df      = _load_results()
    history = _load_history()
    y_true  = _load_true()
    y_lstm  = _load_pred("LSTM")

    fig12_model_comparison(df)
    fig14_loss_curve(history)
    fig15_actual_vs_predicted(y_true, y_lstm)
    fig16_dr_strategies(df)
    fig17_error_boxplot()
    fig18_r2_heatmap(df)
    print("All figures saved to", FIG_DIR)


if __name__ == "__main__":
    run()
    fig14_lstm_dr_timeseries()
