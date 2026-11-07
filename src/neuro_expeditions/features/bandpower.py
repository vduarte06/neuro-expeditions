"""Log band power per channel: the classical baseline every foundation model should beat."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}


def log_bandpower(windows: np.ndarray, sfreq: float, bands: dict = BANDS) -> np.ndarray:
    """(N, C, T) -> (N, C * n_bands) log10 power."""
    freqs, psd = welch(windows, fs=sfreq, nperseg=min(windows.shape[-1], int(sfreq)), axis=-1)
    feats = [psd[..., (freqs >= lo) & (freqs < hi)].mean(-1) for lo, hi in bands.values()]
    return np.log10(np.stack(feats, -1) + 1e-30).reshape(len(windows), -1)
