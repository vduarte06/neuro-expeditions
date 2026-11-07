"""CBraMod (Wang et al., ICLR 2025): criss-cross transformer EEG foundation model, ~4M params.

Pretrained on TUEG with masked patch reconstruction. Loaded through braindecode. Input: 200 Hz, 1 s patches
(200 samples), any number of channels, signal in uV / 100 (same convention as the original repo).
"""

from __future__ import annotations

import mne
import numpy as np
import torch
from braindecode.models import CBraMod

from neuro_expeditions.models.reve import default_device

SFREQ = 200.0
SCALE_UV = 100.0
WEIGHTS = "braindecode/cbramod-pretrained"


def preprocess(raw: mne.io.Raw, line_freq: float = 50.0) -> mne.io.Raw:
    """Average reference, 0.3-75 Hz band-pass, notch, 200 Hz (CBraMod's downstream preprocessing)."""
    raw = raw.copy().pick("eeg")
    raw.set_eeg_reference("average", verbose="error")
    raw.filter(l_freq=0.3, h_freq=75.0, verbose="error")
    raw.notch_filter(line_freq, verbose="error")
    raw.resample(SFREQ, verbose="error")
    return raw


def to_model_units(data_volts: np.ndarray) -> np.ndarray:
    return (data_volts * 1e6 / SCALE_UV).astype(np.float32)


class CBraModEncoder:
    """Frozen CBraMod encoder: windows (N, C, T) at 200 Hz, T a multiple of 200 -> embeddings."""

    def __init__(self, n_chans: int, n_times: int, device: str | None = None):
        self.device = device or default_device()
        self.model = CBraMod.from_pretrained(
            WEIGHTS, n_chans=n_chans, n_times=n_times, sfreq=SFREQ, return_encoder_output=True
        )
        self.model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, windows: np.ndarray, batch_size: int = 64) -> dict[str, np.ndarray]:
        """Returns {"mean": (N, 200)}: encoder output averaged over channels and patches."""
        out = []
        for i in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[i : i + batch_size]).to(self.device)
            feats = self.model(x)
            feats = feats if isinstance(feats, torch.Tensor) else feats["features"]
            out.append(feats.reshape(len(x), -1, feats.shape[-1]).mean(1).cpu())
        return {"mean": torch.cat(out).numpy()}
