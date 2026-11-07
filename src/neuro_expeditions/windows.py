"""Cut continuous EEG into fixed-length windows and label them from interval annotations."""

from __future__ import annotations

import numpy as np
import pandas as pd

UNLABELED = "unlabeled"
AMBIGUOUS = "ambiguous"


def make_windows(data: np.ndarray, sfreq: float, window_s: float, stride_s: float) -> tuple[np.ndarray, np.ndarray]:
    """(channels, samples) -> (n_windows, channels, window_samples) and window start times in seconds."""
    win, step = int(round(window_s * sfreq)), int(round(stride_s * sfreq))
    starts = np.arange(0, data.shape[1] - win + 1, step)
    windows = np.stack([data[:, s : s + win] for s in starts])
    return windows, starts / sfreq


def label_windows(
    starts: np.ndarray,
    window_s: float,
    events: pd.DataFrame,
    min_cover: float = 0.5,
    max_other: float = 0.1,
) -> np.ndarray:
    """Assign each window the event group covering >= min_cover of it.

    Windows where a second group also covers > max_other are "ambiguous"; windows with no
    group reaching min_cover are "unlabeled".
    """
    groups = events["group"].unique()
    cover = np.zeros((len(starts), len(groups)))
    ends = starts + window_s
    for gi, g in enumerate(groups):
        for start, stop in events.loc[events["group"] == g, ["start", "stop"]].itertuples(index=False):
            cover[:, gi] += np.clip(np.minimum(ends, stop) - np.maximum(starts, start), 0, None)
    cover = np.clip(cover / window_s, 0, 1)

    labels = np.full(len(starts), UNLABELED, dtype=object)
    if not len(groups):
        return labels
    best = cover.argmax(1)
    best_cover = cover[np.arange(len(starts)), best]
    cover[np.arange(len(starts)), best] = 0
    labels[best_cover >= min_cover] = groups[best][best_cover >= min_cover]
    labels[(best_cover >= min_cover) & (cover.max(1) > max_other)] = AMBIGUOUS
    return labels
