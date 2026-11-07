"""REVE (brain-bzh/reve-base): 69M-param EEG foundation model with 4D positional encoding.

Input conventions follow the official repo (github.com/elouayas/reve_eeg):
200 Hz, average reference, 0.3 Hz high-pass, notch at line frequency, signal in uV / 100.
Patches are 1 s (200 samples) with 20-sample overlap, so a 2 s window -> 2 patches per channel.
"""

from __future__ import annotations

import mne
import numpy as np
import torch
from transformers import AutoModel

SFREQ = 200.0
SCALE_UV = 100.0
CLIP = 15.0

# Pinned because the model runs remote code (trust_remote_code=True).
MODEL_ID, MODEL_REVISION = "brain-bzh/reve-base", "dc2a075c287bb2f6c04ee5875bd79535a0f7dba6"
POS_ID, POS_REVISION = "brain-bzh/reve-positions", "befa5b57a455b77cf302daf610c2e9ed8140bace"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def preprocess(raw: mne.io.Raw, line_freq: float = 50.0) -> mne.io.Raw:
    """Return a copy of `raw` prepared the way REVE saw data in pretraining."""
    raw = raw.copy().pick("eeg")
    raw.set_eeg_reference("average", verbose="error")
    raw.filter(l_freq=0.3, h_freq=None, verbose="error")
    raw.notch_filter(line_freq, verbose="error")
    raw.resample(SFREQ, verbose="error")
    return raw


def to_model_units(data_volts: np.ndarray) -> np.ndarray:
    return np.clip(data_volts * 1e6 / SCALE_UV, -CLIP, CLIP).astype(np.float32)


class ReveEncoder:
    """Frozen REVE encoder: windows (N, C, T) at 200 Hz -> embeddings."""

    def __init__(self, device: str | None = None):
        self.device = device or default_device()
        self.model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
        self.bank = AutoModel.from_pretrained(POS_ID, revision=POS_REVISION, trust_remote_code=True)
        self.model.eval().to(self.device)
        self.bank.eval().to(self.device)

    def supported(self, ch_names: list[str]) -> list[str]:
        """Channels REVE has a 3D position for (the bank silently drops unknown names)."""
        return [c for c in ch_names if c in self.bank.mapping]

    @torch.no_grad()
    def embed(self, windows: np.ndarray, ch_names: list[str], batch_size: int = 64) -> dict[str, np.ndarray]:
        """Returns per-window embeddings:
        - "mean": mean over all channel x patch tokens (E,)
        - "attn": REVE's own learned attention pooling (E,)
        - "channel": mean over patches, kept per channel (C, E)
        """
        if self.supported(ch_names) != list(ch_names):
            raise ValueError(f"No REVE position for: {sorted(set(ch_names) - set(self.supported(ch_names)))}")
        pos = self.bank(list(ch_names))
        out = {"mean": [], "attn": [], "channel": []}
        for i in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[i : i + batch_size]).to(self.device)
            tokens = self.model(x, pos.expand(len(x), -1, -1))  # (B, C, patches, E)
            out["mean"].append(tokens.mean((1, 2)).cpu())
            out["attn"].append(self.model.attention_pooling(tokens).cpu())
            out["channel"].append(tokens.mean(2).cpu())
        return {k: torch.cat(v).numpy() for k, v in out.items()}
