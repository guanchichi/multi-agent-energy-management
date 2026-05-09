import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

from src.config import DATA_PROC, RESULTS_FIGURES, TARGET_COL

OUT_DIR = RESULTS_FIGURES / "eda"


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def plot_timeseries(df: pd.DataFrame) -> None:
    """Appliances energy consumption over time (Fig. 2)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(pd.to_datetime(df["date"]), df[TARGET_COL], linewidth=0.5, color="steelblue")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.set_xlabel("Date")
    ax.set_ylabel("Appliances Energy (Wh)")
    ax.set_title("Appliances Energy Consumption Over Time")
    _save(fig, "timeseries.png")


def plot_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of Appliances (Fig. 3)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET_COL], bins=60, kde=True, color="steelblue", ax=ax)
    ax.set_xlabel("Appliances Energy (Wh)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Appliances Energy Consumption")
    _save(fig, "distribution.png")


def plot_hourly_profile(df: pd.DataFrame) -> None:
    """Mean consumption by hour of day."""
    hourly = df.groupby("hour")[TARGET_COL].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hourly.index, hourly.values, color="steelblue", edgecolor="white")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Appliances Energy (Wh)")
    ax.set_title("Average Hourly Energy Profile")
    ax.set_xticks(range(0, 24))
    _save(fig, "hourly_profile.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap (Fig. 4)."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix")
    _save(fig, "correlation_heatmap.png")


def print_descriptive_stats(df: pd.DataFrame) -> None:
    """Descriptive statistics table (Table 2)."""
    key_cols = [TARGET_COL, "lights", "T_out", "RH_out", "Windspeed",
                "discomfort_index", "apparent_temp"]
    stats = df[key_cols].describe().T[["mean", "std", "min", "50%", "max"]]
    stats.columns = ["Mean", "Std", "Min", "Median", "Max"]
    print("\n=== Descriptive Statistics ===")
    print(stats.round(2).to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats.round(2).to_csv(OUT_DIR / "descriptive_stats.csv")
    print(f"Saved → {OUT_DIR / 'descriptive_stats.csv'}")


def run() -> None:
    df = pd.read_csv(DATA_PROC)
    plot_timeseries(df)
    plot_distribution(df)
    plot_hourly_profile(df)
    plot_correlation_heatmap(df)
    print_descriptive_stats(df)


if __name__ == "__main__":
    run()
