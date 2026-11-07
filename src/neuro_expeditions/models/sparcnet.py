"""SPaRCNet Seizure & Rhythmic Continuum Detector (Jing et al. 2023 - Harvard / MGH).

Multi-scale 1D DenseNet trained on extensive ICU and EMU scalp EEG records for detecting
seizures and periodic patterns across the Ictal-Interictal-Injury Continuum (IIIC).

Input: 16-channel standard bipolar double-banana montage at 200 Hz.
Output: 6-class continuum probabilities (Seizure, LPD, GPD, LRDA, GRDA, Other).

Reference:
    Jing J., Ge W., Hong S., et al. "Development of Expert-Level Classification of Seizures
    and Rhythmic and Periodic Patterns During EEG Interpretation", Neurology (2023).
    Repository: https://github.com/bdsp-core/IIIC-SPaRCNet
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from collections import OrderedDict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuro_expeditions import DATA_DIR

MODEL_WEIGHTS_URL = "https://raw.githubusercontent.com/bdsp-core/IIIC-SPaRCNet/main/SPaRCNet/model_1130.pt"
CACHE_DIR = DATA_DIR / "cache" / "models" / "sparcnet"
WEIGHTS_PATH = CACHE_DIR / "model_1130.pt"

SFREQ = 200.0

# 16 bipolar channels formed by 8 pairs of electrodes in double-banana montage
BIPOLAR_PAIRS = [
    # Left temporal chain
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    # Right temporal chain
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    # Left parasagittal chain
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    # Right parasagittal chain
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
]

CLASSES = ["Other", "Seizure", "LPD", "GPD", "LRDA", "GRDA"]


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, conv_bias: bool, batch_norm: bool):
        super().__init__()
        if batch_norm:
            self.add_module("norm1", nn.BatchNorm1d(num_input_features))
        self.add_module("elu1", nn.ELU())
        self.add_module("conv1", nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=conv_bias))
        if batch_norm:
            self.add_module("norm2", nn.BatchNorm1d(bn_size * growth_rate))
        self.add_module("elu2", nn.ELU())
        self.add_module("conv2", nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=conv_bias))
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, growth_rate: int, drop_rate: float, conv_bias: bool, batch_norm: bool):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, conv_bias, batch_norm)
            self.add_module(f"denselayer{i + 1}", layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int, conv_bias: bool, batch_norm: bool):
        super().__init__()
        if batch_norm:
            self.add_module("norm", nn.BatchNorm1d(num_input_features))
        self.add_module("elu", nn.ELU())
        self.add_module("conv", nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=conv_bias))
        self.add_module("pool", nn.AvgPool1d(kernel_size=2, stride=2))


class DenseNetEnconder(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4),
        in_channels: int = 16,
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        conv_bias: bool = True,
        batch_norm: bool = False,
    ):
        super().__init__()
        first_conv = OrderedDict([
            ("conv0", nn.Conv1d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=conv_bias))
        ])
        if batch_norm:
            first_conv["norm0"] = nn.BatchNorm1d(num_init_features)
        first_conv["elu0"] = nn.ELU()
        first_conv["pool0"] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.densenet = nn.Sequential(first_conv)

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                conv_bias=conv_bias,
                batch_norm=batch_norm,
            )
            self.densenet.add_module(f"denseblock{i + 1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    conv_bias=conv_bias,
                    batch_norm=batch_norm,
                )
                self.densenet.add_module(f"transition{i + 1}", trans)
                num_features = num_features // 2

        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.densenet(x)
        out = F.adaptive_avg_pool1d(out, 1)
        return torch.flatten(out, 1)


class DenseNetClassifier(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4),
        in_channels: int = 16,
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        conv_bias: bool = True,
        batch_norm: bool = False,
        drop_fc: float = 0.5,
        num_classes: int = 6,
    ):
        super().__init__()
        self.features = DenseNetEnconder(
            growth_rate=growth_rate,
            block_config=block_config,
            in_channels=in_channels,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            conv_bias=conv_bias,
            batch_norm=batch_norm,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_fc),
            nn.Linear(self.features.num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        out = self.classifier(features)
        return out, features


def ensure_weights(weights_path: Path = WEIGHTS_PATH) -> Path:
    """Download SPaRCNet checkpoint if not present."""
    if weights_path.exists() and weights_path.stat().st_size > 1_000_000:
        return weights_path

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = weights_path.with_suffix(".part")
    req = urllib.request.Request(MODEL_WEIGHTS_URL, headers={"User-Agent": "NeuroExpeditions/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp_path.rename(weights_path)
    return weights_path


def load_model(device: str | None = None, weights_path: Path = WEIGHTS_PATH) -> DenseNetClassifier:
    """Load SPaRCNet pre-trained PyTorch model."""
    import sys
    device = device or default_device()
    path = ensure_weights(weights_path)

    # Register classes in __main__ to satisfy PyTorch legacy unpickler
    main_mod = sys.modules.get("__main__")
    if main_mod:
        setattr(main_mod, "DenseNetClassifier", DenseNetClassifier)
        setattr(main_mod, "DenseNetEnconder", DenseNetEnconder)
        setattr(main_mod, "_DenseBlock", _DenseBlock)
        setattr(main_mod, "_DenseLayer", _DenseLayer)
        setattr(main_mod, "_Transition", _Transition)

    model = torch.load(path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return model


SYNONYMS = {
    "T7": "T3",
    "T8": "T4",
    "P7": "T5",
    "P8": "T6",
}


def derive_bipolar_montage(raw: mne.io.Raw) -> np.ndarray:
    """Extract the 16 double-banana bipolar channels from mne.Raw at 200 Hz.

    Returns:
        np.ndarray of shape (16, n_samples)
    """
    raw_copy = raw.copy()
    if abs(raw_copy.info["sfreq"] - SFREQ) > 0.1:
        raw_copy.resample(SFREQ, verbose="error")

    # Normalize channel names
    name_map = {}
    for ch in raw_copy.ch_names:
        clean = ch.replace("EEG ", "").replace("-REF", "").replace("-Ref", "").strip()
        clean = SYNONYMS.get(clean, clean)
        name_map[ch] = clean
    raw_copy.rename_channels(name_map)

    data_bipolar = []
    for an, ca in BIPOLAR_PAIRS:
        # Case insensitive match
        an_match = next((c for c in raw_copy.ch_names if c.upper() == an.upper()), None)
        ca_match = next((c for c in raw_copy.ch_names if c.upper() == ca.upper()), None)

        if an_match is None or ca_match is None:
            raise ValueError(f"Could not derive bipolar pair ({an}, {ca}). Available: {raw_copy.ch_names}")

        # SPaRCNet was trained on microvolts; in volts its output is constant (blind)
        an_data = raw_copy.get_data(picks=[an_match], units="uV")[0]
        ca_data = raw_copy.get_data(picks=[ca_match], units="uV")[0]
        data_bipolar.append(an_data - ca_data)

    return np.array(data_bipolar, dtype=np.float32)


def detect(
    raw: mne.io.Raw,
    model: DenseNetClassifier | None = None,
    device: str | None = None,
    seizure_class_idx: int = 1,
    thresh: float = 0.5,
    window_sec: float = 10.0,
    stride_sec: float = 2.0,
    min_duration_sec: float = 4.0,
) -> pd.DataFrame:
    """Run SPaRCNet detector across sliding windows and return detected seizure intervals."""
    device = device or default_device()
    if model is None:
        model = load_model(device=device)

    bipolar_data = derive_bipolar_montage(raw)  # (16, n_samples)
    n_samples = bipolar_data.shape[1]
    win_len = int(window_sec * SFREQ)
    stride = int(stride_sec * SFREQ)

    if n_samples < win_len:
        return pd.DataFrame(columns=["onset_sec", "offset_sec", "duration_sec", "confidence", "label"])

    starts = np.arange(0, n_samples - win_len + 1, stride)
    windows = np.array([bipolar_data[:, s : s + win_len] for s in starts])

    tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    batch_size = 32
    probs = []

    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i : i + batch_size]
            logits, _ = model(batch)
            prob = F.softmax(logits, dim=-1)[:, seizure_class_idx]
            probs.append(prob.cpu().numpy())

    probs = np.concatenate(probs)
    seizure_mask = probs >= thresh

    # Convert positive windows to continuous intervals
    events = []
    in_seizure = False
    cur_start = 0.0
    cur_end = 0.0
    cur_confs = []

    for idx, (is_sz, s_samp) in enumerate(zip(seizure_mask, starts)):
        t_start = s_samp / SFREQ
        t_end = t_start + window_sec

        if is_sz:
            if not in_seizure:
                in_seizure = True
                cur_start = t_start
                cur_end = t_end
                cur_confs = [probs[idx]]
            else:
                cur_end = t_end
                cur_confs.append(probs[idx])
        else:
            if in_seizure:
                dur = cur_end - cur_start
                if dur >= min_duration_sec:
                    events.append({
                        "onset_sec": float(cur_start),
                        "offset_sec": float(cur_end),
                        "duration_sec": float(dur),
                        "confidence": float(np.mean(cur_confs)),
                        "label": "seizure",
                    })
                in_seizure = False

    if in_seizure:
        dur = cur_end - cur_start
        if dur >= min_duration_sec:
            events.append({
                "onset_sec": float(cur_start),
                "offset_sec": float(cur_end),
                "duration_sec": float(dur),
                "confidence": float(np.mean(cur_confs)),
                "label": "seizure",
            })

    return pd.DataFrame(events) if events else pd.DataFrame(columns=["onset_sec", "offset_sec", "duration_sec", "confidence", "label"])
