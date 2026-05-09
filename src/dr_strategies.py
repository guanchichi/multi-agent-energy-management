"""
7 Demand Response strategies applied to a predictions array (Wh).
Each function returns a modified predictions array of the same shape.
Parameters are defined in src/config.py.
"""
import numpy as np
from src.config import PEAK_HOURS, OFFPEAK_HOURS

# ---------- helpers ----------

def _hour_mask(n: int, hours: list[int]) -> np.ndarray:
    """Boolean mask aligned to a length-n array assuming 10-min intervals."""
    hour_of_day = (np.arange(n) % 144) // 6   # 144 steps per day, 6 per hour
    return np.isin(hour_of_day, hours)


# ---------- strategies ----------

def peak_clipping(y: np.ndarray, clip_pct: float = 0.85) -> np.ndarray:
    """Cap peak-hour consumption at clip_pct of the 95th-percentile value.
    Using np.percentile(y, 95) instead of y.max() prevents a single outlier
    from inflating the threshold to the point where the strategy has no effect."""
    out = y.copy()
    threshold = clip_pct * np.percentile(y, 95)
    mask = _hour_mask(len(y), PEAK_HOURS)
    out[mask] = np.minimum(out[mask], threshold)
    return out


def valley_filling(y: np.ndarray, fill_pct: float = 0.30) -> np.ndarray:
    """Raise off-peak consumption toward fill_pct of global max."""
    out = y.copy()
    target = fill_pct * y.max()
    mask = _hour_mask(len(y), OFFPEAK_HOURS)
    out[mask] = np.maximum(out[mask], target)
    return out


def load_shifting(
    y: np.ndarray,
    shift_pct: float = 0.20,
    shift_steps: int = 6,
) -> np.ndarray:
    """Shift shift_pct of peak load forward by shift_steps time-steps."""
    out = y.copy()
    mask = _hour_mask(len(y), PEAK_HOURS)
    peak_idx = np.where(mask)[0]
    for i in peak_idx:
        delta = out[i] * shift_pct
        out[i] -= delta
        j = i + shift_steps
        if j < len(out):
            out[j] += delta
    return out


def load_leveling(y: np.ndarray, window: int = 144) -> np.ndarray:
    """Flatten consumption to a 24-hour rolling mean (144 × 10-min steps).
    Removes all within-day peaks/valleys — maximum distortion strategy."""
    out = np.convolve(y, np.ones(window) / window, mode="same")
    return out


def tou_optimization(
    y: np.ndarray,
    peak_reduce: float = 0.15,
    offpeak_boost: float = 0.10,
) -> np.ndarray:
    """Reduce peak, boost off-peak based on Time-of-Use pricing."""
    out = y.copy()
    peak_mask    = _hour_mask(len(y), PEAK_HOURS)
    offpeak_mask = _hour_mask(len(y), OFFPEAK_HOURS)
    out[peak_mask]    *= (1 - peak_reduce)
    out[offpeak_mask] *= (1 + offpeak_boost)
    return out


def price_based(
    y: np.ndarray,
    price_peak: float = 1.5,
    price_offpeak: float = 0.6,
    price_std: float = 1.0,
    price_avg: float = 1.0,
    elasticity: float = 0.30,
) -> np.ndarray:
    """Constant-elasticity demand model: y_new = y × (price / price_avg)^(−ε).

    Three-tier pricing relative to price_avg = 1.0:
      peak (18–22 h)  : price = 1.5  → demand drops  ≈ 11.5 %
      off-peak (0–6 h): price = 0.6  → demand rises   ≈ 16.6 %
      standard        : price = 1.0  → no change

    Implements the constant elasticity of demand (CED) framework from
    Singh et al. (2025). Elasticity ε = 0.30 is typical for residential
    electricity short-run response.
    """
    out = y.copy().astype(float)
    price = np.full(len(y), price_std, dtype=float)
    price[_hour_mask(len(y), PEAK_HOURS)]    = price_peak
    price[_hour_mask(len(y), OFFPEAK_HOURS)] = price_offpeak
    out *= (price / price_avg) ** (-elasticity)
    return out


def behavioral_dr(
    y: np.ndarray,
    awareness_pct: float = 0.08,
) -> np.ndarray:
    """Uniform reduction reflecting occupant awareness / habit change."""
    return y * (1 - awareness_pct)


# ---------- registry ----------

STRATEGIES: dict[str, callable] = {
    "Peak Clipping":   peak_clipping,
    "Valley Filling":  valley_filling,
    "Load Shifting":   load_shifting,
    "Load Leveling":   load_leveling,
    "ToU Optimization": tou_optimization,
    "Price Based":     price_based,
    "Behavioral DR":   behavioral_dr,
}


def apply_all(y_pred: np.ndarray) -> dict[str, np.ndarray]:
    """Apply every strategy and return {name: modified_array}."""
    return {name: fn(y_pred) for name, fn in STRATEGIES.items()}
