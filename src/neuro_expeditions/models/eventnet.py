"""eventNet Seizure Detector (Seeuws et al. 2022; Dan et al. 2024).

1D Temporal U-Net encoder-decoder with skip connections trained on the Temple University Hospital
Seizure Corpus (TUSZ). Official SzCORE baseline for event-based seizure detection.

Input: 19-channel standard 10-20 unipolar EEG at 256 Hz.
Output: Center onset and event duration probabilities.

References:
    - Seeuws N., De Vos M., Bertrand A. "EventNet: Detecting Events in EEG", KU Leuven, 2022.
    - Dan J., Shahbazinia A., Kechris C., Atienza D. "Quantifying the Generalization Gap in
      Seizure Detection: A Large-Scale Empirical Benchmark via the SzCORE Challenge", ICML 2024.
    - Checkpoint: https://github.com/esl-epfl/eventnet_2024
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn

from neuro_expeditions import DATA_DIR

MODEL_WEIGHTS_URL = "https://raw.githubusercontent.com/esl-epfl/eventnet_2024/main/eventnet/src/eventnet/model.pth"
CACHE_DIR = DATA_DIR / "cache" / "models" / "eventnet"
WEIGHTS_PATH = CACHE_DIR / "model.pth"

# Official 19 standard 10-20 electrodes expected by eventNet (epilepsy2bids canonical order)
ELECTRODES_19 = (
    "Fp1", "F3", "C3", "P3", "O1",
    "F7", "T3", "T5",
    "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "O2",
    "F8", "T4", "T6",
)
SFREQ = 256.0


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class UNet(nn.Module):
    """1D Temporal U-Net for continuous event detection."""

    def __init__(self, input_channels: int = 19):
        super().__init__()
        bias = False
        channels = 16
        stride = 4
        padding = "same"
        kernel_size = 9
        dilation = 1

        self.up = nn.Upsample(scale_factor=stride)
        self.sigm = nn.Sigmoid()

        # Encoder stages
        self.enc0 = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels, out_channels=channels, dilation=dilation,
                stride=1, kernel_size=kernel_size, padding=padding, padding_mode="reflect",
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True),
        )
        self.down = nn.MaxPool1d(kernel_size=stride, dilation=dilation)

        self.enc1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels, out_channels=2 * channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode="reflect",
            ),
            nn.BatchNorm1d(num_features=2 * channels),
            nn.ELU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv1d(
                in_channels=2 * channels, out_channels=4 * channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode="reflect",
            ),
            nn.BatchNorm1d(num_features=4 * channels),
            nn.ELU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv1d(
                in_channels=4 * channels, out_channels=8 * channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode="reflect",
            ),
            nn.BatchNorm1d(num_features=8 * channels),
            nn.ELU(),
        )

        self.enc4 = nn.Sequential(
            nn.Conv1d(
                in_channels=8 * channels, out_channels=16 * channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding, padding_mode="reflect",
            ),
            nn.BatchNorm1d(num_features=16 * channels),
            nn.ELU(),
        )

        # Decoder stages with skip connections
        self.dec3 = nn.Sequential(
            nn.Conv1d(
                in_channels=(16 + 8) * channels, out_channels=8 * channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding,
            ),
            nn.BatchNorm1d(num_features=8 * channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=8 * channels, out_channels=8 * channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding, padding_mode="reflect",
            ),
            nn.BatchNorm1d(num_features=8 * channels),
            nn.ELU(),
        )

        self.dec2 = nn.Sequential(
            nn.Conv1d(
                in_channels=(8 + 4) * channels, out_channels=4 * channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding,
            ),
            nn.BatchNorm1d(num_features=4 * channels),
            nn.ELU(),
            nn.Conv1d(
                in_channels=4 * channels, out_channels=4 * channels, dilation=dilation,
                kernel_size=kernel_size, stride=1, bias=bias, padding=padding,
            ),
            nn.BatchNorm1d(num_features=4 * channels),
            nn.ELU(),
        )

        self.dec1 = nn.Sequential(
            nn.Conv1d(
                in_channels=(4 + 2) * channels, out_channels=2 * channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding,
            ),
            nn.BatchNorm1d(num_features=2 * channels),
            nn.ELU(inplace=True),
        )

        self.dec0 = nn.Sequential(
            nn.Conv1d(
                in_channels=(2 + 1) * channels, out_channels=channels, dilation=dilation,
                kernel_size=15, stride=1, bias=bias, padding=padding,
            ),
            nn.BatchNorm1d(num_features=channels),
            nn.ELU(inplace=True),
        )

        # Heads
        self.center_logit = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=21, stride=1, padding=padding),
            nn.MaxPool1d(kernel_size=21, stride=1),
        )
        self.duration_logit = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=21, stride=1, padding=padding),
            nn.MaxPool1d(kernel_size=21, stride=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lvl0 = self.enc0(x)
        lvl1 = self.enc1(self.down(lvl0))
        lvl2 = self.enc2(self.down(lvl1))
        lvl3 = self.enc3(self.down(lvl2))
        lvl4 = self.enc4(self.down(lvl3))

        x = self.dec3(torch.cat((self.up(lvl4), lvl3), dim=1))
        x = self.dec2(torch.cat((self.up(x), lvl2), dim=1))
        x = self.dec1(torch.cat((self.up(x), lvl1), dim=1))
        out0 = self.dec0(torch.cat((self.up(x), lvl0), dim=1))

        center = self.sigm(self.center_logit(out0))
        duration = self.sigm(self.duration_logit(out0))

        # Trim boundary convolution padding
        pad = 256 - 21 + 1
        center = center[:, :, pad // 2 : -pad // 2]
        duration = duration[:, :, pad // 2 : -pad // 2]

        return center, duration


def ensure_weights(weights_path: Path = WEIGHTS_PATH) -> Path:
    """Download official eventNet weights if not already present."""
    if weights_path.exists() and weights_path.stat().st_size > 100_000:
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = weights_path.with_suffix(".part")
    import shutil
    req = urllib.request.Request(MODEL_WEIGHTS_URL, headers={"User-Agent": "NeuroExpeditions/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.rename(weights_path)
    return weights_path


def load_model(device: str | None = None, weights_path: Path = WEIGHTS_PATH) -> UNet:
    """Instantiate UNet and load pre-trained TUSZ weights."""
    device = device or default_device()
    path = ensure_weights(weights_path)
    model = UNet(input_channels=len(ELECTRODES_19))
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


SYNONYMS = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}


def preprocess(raw: mne.io.Raw) -> mne.io.Raw:
    """Prepare raw recording to match eventNet input specifications:
    - 19 standard 10-20 channels in canonical unipolar order
    - Resampled to 256 Hz
    """
    raw_copy = raw.copy()

    # Normalize channel names (e.g. "EEG FP1" -> "Fp1", "T7" -> "T3")
    ch_map = {}
    for ch in raw_copy.ch_names:
        clean = ch.replace("EEG ", "").replace("-REF", "").replace("-Ref", "").strip()
        # Map 10-10 synonyms if needed
        clean = SYNONYMS.get(clean, clean)
        # Case match against standard 10-20
        for std in ELECTRODES_19:
            if clean.upper() == std.upper() or SYNONYMS.get(clean.upper()) == std.upper():
                ch_map[ch] = std
                break
    raw_copy.rename_channels(ch_map)

    # Pick the 19 standard channels (or raise if missing)
    available = set(raw_copy.ch_names)
    missing = [c for c in ELECTRODES_19 if c not in available]
    if missing:
        raise ValueError(f"Recording is missing required 10-20 channels for eventNet: {missing}")

    raw_copy.reorder_channels(list(ELECTRODES_19))

    # Resample to 256 Hz if needed
    if abs(raw_copy.info["sfreq"] - SFREQ) > 0.1:
        raw_copy.resample(SFREQ, verbose="error")

    return raw_copy


def detect(
    raw: mne.io.Raw,
    model: UNet | None = None,
    device: str | None = None,
    thresh: float = 0.44,
    window_sec: int = 120,
    min_duration_sec: float = 3.0,
) -> pd.DataFrame:
    """Run eventNet detector on an mne.Raw recording and return detected seizure events.

    Args:
        raw: mne.io.Raw object.
        model: optional pre-loaded UNet model.
        device: 'mps', 'cuda', or 'cpu'.
        thresh: detection threshold for peak center probability (default: 0.44).
        window_sec: sliding window length in seconds (default: 120s).
        min_duration_sec: minimum event duration in seconds.

    Returns:
        DataFrame: [onset_sec, offset_sec, duration_sec, confidence, label]
    """
    device = device or default_device()
    if model is None:
        model = load_model(device=device)

    preprocessed = preprocess(raw)
    data = preprocessed.get_data(units="uV")  # shape (19, n_samples); trained on microvolts
    n_samples = data.shape[1]
    fs = int(preprocessed.info["sfreq"])

    window_size = int(fs * window_sec)
    pad = 256

    if n_samples < window_size + pad:
        return pd.DataFrame(columns=["onset_sec", "offset_sec", "duration_sec", "confidence", "label"])

    window_indices = np.arange(pad, n_samples - window_size, window_size, dtype=int)
    if len(window_indices) == 0:
        return pd.DataFrame(columns=["onset_sec", "offset_sec", "duration_sec", "confidence", "label"])

    # Batch windows
    batch_clips = []
    for idx in window_indices:
        clip = data[:, idx - pad // 2 : idx + window_size + pad // 2]
        batch_clips.append(clip)

    clips_tensor = torch.tensor(np.array(batch_clips), dtype=torch.float32, device=device)

    pred_centers = []
    pred_durations = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(clips_tensor), batch_size):
            batch = clips_tensor[i : i + batch_size]
            centers, durations = model(batch)
            pred_centers.append(centers.squeeze(1).cpu().numpy())
            pred_durations.append(durations.squeeze(1).cpu().numpy())

    pred_centers = np.concatenate(pred_centers, axis=0)
    pred_durations = np.concatenate(pred_durations, axis=0)

    # Event extraction via peak finding
    events = []
    max_duration_sec = 300.0  # eventNet max duration scale (5 minutes)

    for w_i, start_sample in enumerate(window_indices):
        centers_smooth = gaussian_filter1d(pred_centers[w_i], 100)
        durations_smooth = pred_durations[w_i]

        peaks, props = signal.find_peaks(centers_smooth, height=thresh, distance=fs * 30)

        for peak in peaks:
            conf = float(centers_smooth[peak])
            dur_sec = float(durations_smooth[peak] * max_duration_sec)
            dur_sec = max(dur_sec, min_duration_sec)

            onset_sec = (start_sample + peak) / fs - (dur_sec / 2.0)
            offset_sec = onset_sec + dur_sec

            if onset_sec < 0:
                onset_sec = 0.0

            events.append({
                "onset_sec": float(onset_sec),
                "offset_sec": float(offset_sec),
                "duration_sec": float(dur_sec),
                "confidence": conf,
                "label": "seizure",
            })

    df = pd.DataFrame(events)
    if df.empty:
        return pd.DataFrame(columns=["onset_sec", "offset_sec", "duration_sec", "confidence", "label"])

    # Merge overlapping detections
    df = df.sort_values("onset_sec").reset_index(drop=True)
    merged = []
    curr = df.iloc[0].to_dict()

    for i in range(1, len(df)):
        nxt = df.iloc[i].to_dict()
        if nxt["onset_sec"] <= curr["offset_sec"] + 5.0:  # merge within 5s gap
            curr["offset_sec"] = max(curr["offset_sec"], nxt["offset_sec"])
            curr["duration_sec"] = curr["offset_sec"] - curr["onset_sec"]
            curr["confidence"] = max(curr["confidence"], nxt["confidence"])
        else:
            merged.append(curr)
            curr = nxt
    merged.append(curr)

    return pd.DataFrame(merged)
