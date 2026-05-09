from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW  = ROOT / "data" / "raw"  / "energydata_complete.csv"
DATA_PROC = ROOT / "data" / "processed" / "processed.csv"

RESULTS_FIGURES = ROOT / "results" / "figures"
RESULTS_MODELS  = ROOT / "results" / "models"
RESULTS_METRICS = ROOT / "results" / "metrics"

TARGET_COL = "Appliances"

TIMESTEPS   = 144
TRAIN_RATIO = 0.70
VAL_SPLIT   = 0.15

LSTM_UNITS_1  = 128
LSTM_UNITS_2  = 64
DENSE_UNITS   = 32
DROPOUT_RATE  = 0.2
RECURRENT_DROPOUT = 0.0   # keep 0 to preserve cuDNN GPU acceleration
L2_REG        = 1e-4
LEARNING_RATE = 0.001
BATCH_SIZE    = 32
EPOCHS        = 200
PATIENCE      = 25
LOG_TRANSFORM = True      # log1p target to normalise right-skewed distribution

PEAK_HOURS    = list(range(18, 22))
OFFPEAK_HOURS = list(range(0, 6))

OUTLIER_Z_THRESH  = 3.0
OUTLIER_WINDOW    = 3
