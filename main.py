"""
End-to-end pipeline for replicating Durrani et al. (2025).

Usage:
    python main.py [--skip-lstm] [--skip-ml] [--skip-eval] [--skip-viz]
"""
import argparse
import numpy as np


def main(skip_lstm: bool, skip_ml: bool, skip_eval: bool, skip_viz: bool):
    # ── 1. LSTM ──────────────────────────────────────────────────────────────
    if not skip_lstm:
        print("\n=== 1/4  LSTM Training ===")
        from src.models.lstm_model import run as run_lstm
        _, lstm_pred, y_true, _ = run_lstm()
    else:
        print("Skipping LSTM (--skip-lstm)")
        from src.config import RESULTS_METRICS
        lstm_pred = np.load(RESULTS_METRICS / "lstm_predictions.npy")
        y_true    = np.load(RESULTS_METRICS / "y_test_raw.npy")

    # ── 2. Baseline ML ───────────────────────────────────────────────────────
    if not skip_ml:
        print("\n=== 2/4  Baseline ML Models ===")
        from src.models.ml_models import run as run_ml
        ml_preds, _ = run_ml()
    else:
        print("Skipping ML models (--skip-ml)")
        from src.config import RESULTS_METRICS
        ml_preds = {}
        for name in ("LinearRegression", "RandomForest", "SVR", "kNN"):
            path = RESULTS_METRICS / f"{name.lower()}_predictions.npy"
            if path.exists():
                ml_preds[name] = np.load(path)

    # ── 3. Evaluation ────────────────────────────────────────────────────────
    if not skip_eval:
        print("\n=== 3/4  Evaluation (35 combinations) ===")
        from src.evaluation import run as run_eval
        all_preds = {"LSTM": lstm_pred, **ml_preds}
        df = run_eval(all_preds, y_true)
    else:
        print("Skipping evaluation (--skip-eval)")

    # ── 4. Visualisation ─────────────────────────────────────────────────────
    if not skip_viz:
        print("\n=== 4/4  Figures ===")
        from src.visualization import run as run_viz
        run_viz()

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-ml",   action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-viz",  action="store_true")
    args = parser.parse_args()
    main(args.skip_lstm, args.skip_ml, args.skip_eval, args.skip_viz)
