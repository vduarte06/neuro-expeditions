"""eventNet seizure detector (Seeuws et al. 2022; SzCORE reproduction, Dan et al. 2024): 1D U-Net, trained on TUSZ.

Thin wrapper around the official SzCORE package `eventnet` (github.com/esl-epfl/eventnet_2024, pinned commit
d138668, installed via pyproject). Everything model-related is the official code:
    - preprocessing: `neuro_expeditions.szcore.to_szcore` = `epilepsy2bids` `Eeg.standardize` (19 10-20
      electrodes, 256 Hz via resampy, common average), i.e. the SzCORE format the model is run on;
    - model + weights: `eventnet.utils.load_model`;
    - windowing: `eventnet.utils.SeizureDataset` (120 s windows, shorter for short recordings, as in `main`);
    - events: `eventnet.utils.events_to_mask` (threshold 0.44, as in `main`).
Only the batching loop is ours: the official `get_dataloader`/`predict` cannot run as published (`n_workers` is not
a DataLoader argument, and batches are stacked into a 3D array that `events_to_mask` cannot iterate). Windows are
passed through the model in order and concatenated, which is what `events_to_mask` expects.

Input is an mne.Raw in volts (any montage-free referential recording), so perturbations can be applied upstream.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import torch
from eventnet import utils

from neuro_expeditions.szcore import to_szcore

THRESHOLD = 0.44
WINDOW_SEC = 120
COLUMNS = ["onset_sec", "offset_sec", "duration_sec", "confidence", "label"]


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str | None = None) -> torch.nn.Module:
    return utils.load_model().to(device or default_device()).eval()


def mask_to_events(mask: np.ndarray, fs: float) -> pd.DataFrame:
    edges = np.diff(np.concatenate([[0], (mask > 0).astype(int), [0]]))
    on, off = np.flatnonzero(edges == 1) / fs, np.flatnonzero(edges == -1) / fs
    return pd.DataFrame({"onset_sec": on, "offset_sec": off, "duration_sec": off - on,
                         "confidence": np.nan, "label": "seizure"}, columns=COLUMNS)  # fmt: skip


@torch.no_grad()
def detect(raw: mne.io.Raw, model: torch.nn.Module | None = None, device: str | None = None,
           thresh: float = THRESHOLD, batch_size: int = 32) -> pd.DataFrame:  # fmt: skip
    device = device or default_device()
    model = model or load_model(device)
    device = next(model.parameters()).device
    eeg = to_szcore(raw)
    duration = eeg.data.shape[1] / eeg.fs

    # Window size exactly as in the official eventnet.main
    win = WINDOW_SEC if duration - 5 >= WINDOW_SEC * 2 else (duration - 5) // 20 * 10
    if win <= 25:
        return pd.DataFrame(columns=COLUMNS)

    dataset = utils.SeizureDataset(eeg.data, window_size_sec=win, fs=eeg.fs)
    centers, durations = [], []
    for i in range(0, len(dataset), batch_size):
        x = torch.stack([dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]).to(device)
        c, d = model(x)
        centers.append(c.squeeze(1).cpu().numpy())
        durations.append(d.squeeze(1).cpu().numpy())

    mask = utils.events_to_mask(np.concatenate(centers), np.concatenate(durations), win, fs=eeg.fs, thresh=thresh)
    y = np.zeros(int(duration * eeg.fs))
    flat = np.concatenate(mask)[: len(y)]
    y[: len(flat)] = flat  # same placement as the official predict(): windows laid end to end from sample 0
    return mask_to_events(y, eeg.fs)
