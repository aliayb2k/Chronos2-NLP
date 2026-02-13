from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class WindowConfig:
    context: int
    horizon: int
    stride: int = 1

def make_windows(values: np.ndarray, cfg: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    values: shape [T]
    returns:
      X: [N, context]
      Y: [N, horizon]
    """
    values = np.asarray(values, dtype=np.float32)
    L, H, S = cfg.context, cfg.horizon, cfg.stride
    T = len(values)

    if T < L + H:
        raise ValueError(f"Series too short: T={T} < L+H={L+H}")

    starts = np.arange(0, T - (L + H) + 1, S)
    X = np.stack([values[s:s+L] for s in starts], axis=0)
    Y = np.stack([values[s+L:s+L+H] for s in starts], axis=0)
    return X, Y

def time_split_indices(T: int, train_frac=0.7, val_frac=0.15) -> tuple[slice, slice, slice]:
    """
    Splits by time to avoid leakage.
    """
    train_end = int(T * train_frac)
    val_end = int(T * (train_frac + val_frac))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, T)