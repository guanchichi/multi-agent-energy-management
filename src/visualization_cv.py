"""
5-fold CV visualizations for Durrani et al. (2025) replication.
Produces 3 figures saved to results/figures/cv/.
"""
import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_METRICS, RESULTS_FIGURES

CV_FIG_DIR = RESULTS_FIGURES / "cv"

MODEL_ORDER  = ["LSTM", "RandomForest", "LinearRegression", "SVR", "kNN"]
MODEL_LABELS = ["LSTM", "RF", "LR", "SVR", "kNN"]

# LSTM=blue, RF=orange for contrast; others neutral
COLORS = ["#2196F3", "#FF9800", "#9E9E9E", "#9E9E9E", "#9E9E9E"]


def _load_summary() -> pd.DataFrame:
    path = RESULTS_METRICS / "cv_summary.csv"
    df = pd.read_csv(path, index_col=0)
    return df.reindex(MODEL_ORDER)


def _load_results() -> pd.DataFrame:
    path = RESULTS_METRICS / "cv_results.csv"
    return pd.read_csv(path)


def plot_cv_r2_barchart(save: bool = True) -> None:
    CV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()

    means = summary["R2_mean"].values
    stds  = summary["R2_std"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        MODEL_LABELS, means,
        yerr=stds, capsize=6,
        color=COLORS, edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#444444"},
    )

    # Baseline at R²=0
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")

    # Value labels above each bar
    for bar, mean, std in zip(bars, means, stds):
        y_pos = max(mean + std + 0.01, mean + 0.02)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{mean:.3f} ± {std:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("5-fold TimeSeriesSplit CV: R² (mean ± std)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save:
        out = CV_FIG_DIR / "cv_r2_barchart.png"
        fig.savefig(out, dpi=150)
        print(f"Saved → {out}")
    plt.close(fig)


def plot_cv_mae_barchart(save: bool = True) -> None:
    CV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summary()

    means = summary["MAE_mean"].values
    stds  = summary["MAE_std"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        MODEL_LABELS, means,
        yerr=stds, capsize=6,
        color=COLORS, edgecolor="white", linewidth=0.8,
        error_kw={"elinewidth": 1.5, "ecolor": "#444444"},
    )

    # Value labels above each bar
    for bar, mean, std in zip(bars, means, stds):
        y_pos = mean + std + 0.5
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{mean:.2f} ± {std:.2f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("MAE (Wh)", fontsize=12)
    ax.set_title("5-fold TimeSeriesSplit CV: MAE (mean ± std)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    if save:
        out = CV_FIG_DIR / "cv_mae_barchart.png"
        fig.savefig(out, dpi=150)
        print(f"Saved → {out}")
    plt.close(fig)


def plot_cv_perfold_lineplot(save: bool = True) -> None:
    CV_FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()

    # Map to display labels
    label_map = dict(zip(MODEL_ORDER, MODEL_LABELS))
    results["label"] = results["model"].map(label_map)

    palette = {
        "LSTM":  "#2196F3",
        "RF":    "#FF9800",
        "LR":    "#4CAF50",
        "SVR":   "#9C27B0",
        "kNN":   "#F44336",
    }
    markers = {"LSTM": "o", "RF": "s", "LR": "^", "SVR": "D", "kNN": "v"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for label in MODEL_LABELS:
        sub = results[results["label"] == label].sort_values("fold")
        ax.plot(
            sub["fold"], sub["R2"],
            marker=markers[label],
            color=palette[label],
            linewidth=2, markersize=7,
            label=label,
        )

    # Annotation pointing up from below fold 1's lowest R²
    fold1_min = results[results["fold"] == 1]["R2"].min()
    ax.annotate(
        "Fold 1: only 3,290\ntrain samples",
        xy=(1, fold1_min),
        xytext=(1.4, fold1_min - 0.15),
        fontsize=9, color="#555555",
        arrowprops={"arrowstyle": "->", "color": "#888888", "lw": 1.2},
    )

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Per-fold R² across 5 models", fontsize=13)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.tick_params(labelsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    ax.set_ylim(bottom=fold1_min - 0.25)

    fig.tight_layout()
    if save:
        out = CV_FIG_DIR / "cv_perfold_lineplot.png"
        fig.savefig(out, dpi=150)
        print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot_cv_r2_barchart()
    plot_cv_mae_barchart()
    plot_cv_perfold_lineplot()
    print("All CV figures saved to results/figures/cv/")
