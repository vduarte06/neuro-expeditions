"""DINO-EEG seizure detector (Fan et al., 2026): DINO object detector (DETR family) on per-channel spectrograms.

Seizures are treated as "objects" in time: each TCP bipolar channel is detected separately, then detections are
merged across channels into recording-level events (designed for montage robustness). Trained on TUSZ v2.0.3 only;
Siena was used by the authors only as an external test set.

Weights: https://zenodo.org/records/19159105 (CC BY 4.0, 2.6 GB). Code: https://github.com/XploreAI-Lab/DINO-EEG,
cloned to data/external/DINO-EEG (commit 60da643). Its compiled CUDA attention op is replaced by a pure-PyTorch
equivalent (models/_msda_pytorch.py), so it runs on CPU / Apple MPS.

Input: the referential 19-channel mne.Raw (volts), as the official `run_edf_to_tcp_predictions.py` reads the EDF, so
that signals can be perturbed first. No SzCORE conversion: the official pipeline has none, and its common average
would add a re-referencing step before pyprep's bad-channel detection. Then the official preprocessing: 200 Hz, 60 Hz notch, 0.1-70 Hz band-pass, bad-channel detection + interpolation (pyprep),
TCP bipolar montage, STFT (1 s window), per-time-step normalisation. The 60 Hz notch is the released setting
(TUSZ is US data), kept as is. detect_bad_channels=False skips the pyprep step (detection + interpolation), so a
perturbed electrode reaches the model as it is.

Long recordings are run in overlapping 1 h chunks (CHUNK_SEC / CHUNK_OVERLAP_SEC), events merged across chunks. The
official script feeds each channel's whole recording to the model as one spectrogram image, which needs ~30 GB
for a 13.5 h Siena file (PN01-1). Recordings up to 1 h (typical TUSZ file length) are processed unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

from neuro_expeditions import DATA_DIR
from neuro_expeditions.models import _msda_pytorch

REPO = DATA_DIR / "external" / "DINO-EEG"
WEIGHTS = DATA_DIR / "cache" / "models" / "dinoeeg" / "DINO_EEG.pth"
THRESHOLD = 0.37  # official default
CHUNK_SEC = 3600.0
CHUNK_OVERLAP_SEC = 300.0
MODERN_TO_OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
COLUMNS = ["onset_sec", "offset_sec", "duration_sec", "confidence", "label"]


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _runner():
    """The official inference script, imported as a module (for its preprocessing + merging functions)."""
    sys.modules.setdefault("MultiScaleDeformableAttention", _msda_pytorch)
    if str(REPO / "2_dino_eeg") not in sys.path:
        sys.path.insert(0, str(REPO / "2_dino_eeg"))
    spec = importlib.util.spec_from_file_location("dinoeeg_runner", REPO / "run_edf_to_tcp_predictions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess(raw: mne.io.Raw, r=None, detect_bad_channels: bool = True) -> tuple[dict[str, np.ndarray], list[str]]:
    """Signal preprocessing up to the TCP bipolar montage (before the STFT), as the detector runs it.

    Returns ({'FP1-F7': (1, n) array, ...}, bad channels found and interpolated, in the official old names).
    """
    r = r or _runner()
    raw = raw.copy()
    raw.rename_channels({c: MODERN_TO_OLD.get(c, c) for c in raw.ch_names})
    r.pick_channels(raw)
    raw.set_montage(r.get_montage())
    if raw.info["sfreq"] != r.FREQUENCY:
        raw.resample(r.FREQUENCY, verbose="error")
    raw.notch_filter(60, verbose="error")
    raw.filter(0.1, 70, verbose="error")
    bads: list[str] = []
    if detect_bad_channels and r.NoisyChannels is not None:
        try:
            nd = r.NoisyChannels(raw, do_detrend=True)
            nd.find_bad_by_deviation()
            nd.find_bad_by_nan_flat()
            bads = list(nd.get_bads())
            raw.info["bads"] = bads
            if bads:
                raw.interpolate_bads(verbose="error")
        except Exception as e:  # official behaviour: carry on without interpolation; but say so
            warnings.warn(f"DINO-EEG pyprep bad-channel step failed, no interpolation: {e!r}")
    return r.rereference(raw), bads


class DinoEEG:
    def __init__(self, device: str | None = None, threshold: float = THRESHOLD, detect_bad_channels: bool = True):
        self.r = _runner()
        self.threshold = threshold
        self.detect_bad_channels = detect_bad_channels
        self.device = torch.device(device or default_device())

        from models.dino.dino import build_dino

        checkpoint = self.r._safe_torch_load(WEIGHTS)
        ckpt_args = checkpoint["args"]
        self.base_args = dict(ckpt_args) if isinstance(ckpt_args, dict) else dict(vars(ckpt_args))
        self.base_args.update(dataset="tusz", eval=True, distributed=False, amp=False, num_workers=0,
                              test_batch_size=1, device=str(self.device), downsample_seed=42,
                              tusz_downsample_times=0, tusz_label_dir=None)  # fmt: skip
        self.nq = int(self.base_args["num_queries"]) + int(self.base_args["dn_number"]) * 2

        # build_dino first loads ImageNet Swin weights from the authors' machine (backbone_dir). Those are fully
        # overwritten by the DINO-EEG checkpoint below, so point it at an empty stand-in file instead.
        stub_dir = WEIGHTS.parent / "backbone_stub"
        stub_dir.mkdir(exist_ok=True)
        for fn in ("swin_tiny_patch4_window7_224.pth", "swin_base_patch4_window12_384.pth",
                   "swin_large_patch4_window12_384_22k.pth"):  # fmt: skip
            if not (stub_dir / fn).exists():
                torch.save({"model": {}}, stub_dir / fn)
        self.base_args["backbone_dir"] = str(stub_dir)

        self.model, _, self.postprocessors = build_dino(types.SimpleNamespace(**self.base_args))
        state = checkpoint.get("model", checkpoint)
        missing, unexpected = self.model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()},
                                                         strict=False)  # fmt: skip
        if missing or unexpected:
            raise RuntimeError(f"Weight mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        del checkpoint
        self.model.to(self.device).eval()

    def _write_inputs(self, raw: mne.io.Raw, name: str, root: Path) -> list[dict]:
        r = self.r
        signals, _ = preprocess(raw, r, self.detect_bad_channels)
        duration = raw.n_times / raw.info["sfreq"]
        names = list(signals)
        spec = r.apply_batch_stft_and_transform(np.concatenate([signals[c] for c in names], axis=0))

        out_dir = root / "data" / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = []
        for i, ch in enumerate(names):
            if spec[i : i + 1].shape[-1] < r.NQ_CUT_VAL:
                continue
            fn = f"{name}_{ch}.h5"
            with r.h5py.File(out_dir / fn, "w") as h:
                h.create_dataset("signal", data=spec[i : i + 1])
                h.create_dataset("label", data=np.full(signals[ch].shape[1], r.N_CLASSES_TUSZ, dtype=np.uint8))
            meta.append({"filename": fn, "duration": duration, "time_dim": int(spec.shape[-1]), "is_background": True})
        return meta

    def detect(self, raw: mne.io.Raw, name: str = "rec") -> pd.DataFrame:
        duration = raw.n_times / raw.info["sfreq"]
        if duration <= CHUNK_SEC:
            return self._detect_single(raw, name)
        step = CHUNK_SEC - CHUNK_OVERLAP_SEC
        starts = list(np.arange(0.0, duration - CHUNK_SEC, step)) + [duration - CHUNK_SEC]
        parts = []
        for t0 in starts:
            chunk = raw.copy().crop(t0, min(t0 + CHUNK_SEC, raw.times[-1]))
            df = self._detect_single(chunk, name)
            del chunk
            if len(df):
                parts.append(df.assign(onset_sec=df.onset_sec + t0, offset_sec=df.offset_sec + t0))
        return _merge_overlapping(parts)

    @torch.no_grad()
    def _detect_single(self, raw: mne.io.Raw, name: str = "rec") -> pd.DataFrame:
        from datasets import build_dataloader
        from util.misc import collate_fn

        r = self.r
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = self._write_inputs(raw, name.replace("_", "-"), root)
            if not meta:
                return pd.DataFrame(columns=COLUMNS)
            r.write_eval_index_files(meta, root / "txt", self.nq)
            args = types.SimpleNamespace(**{**self.base_args, "data_dir": str(root / "data"),
                                            "tusz_txt_dir": str(root / "txt")})  # fmt: skip
            loaders, _ = build_dataloader(collate_fn, args, stage="full")
            raw_preds = r.run_inference(self.model, self.postprocessors, loaders["test"], self.device, self.threshold)

        preds = r.apply_channelwise_nms(raw_preds, iou_threshold=0.0)
        merged = r.merge_predictions_across_channels(preds, score_threshold=self.threshold)
        rows = r.predictions_to_csv_rows(merged)
        if not rows:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(rows).rename(columns={"score": "confidence"})
        df["label"] = "seizure"
        return df[COLUMNS].sort_values("onset_sec", ignore_index=True)


def _merge_overlapping(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Union of overlapping events from different chunks (lowest confidence kept, as in the official merge)."""
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    events = pd.concat(parts).sort_values("onset_sec").to_dict("records")
    merged = [events[0]]
    for ev in events[1:]:
        last = merged[-1]
        if ev["onset_sec"] <= last["offset_sec"]:
            last["offset_sec"] = max(last["offset_sec"], ev["offset_sec"])
            last["confidence"] = min(last["confidence"], ev["confidence"])
        else:
            merged.append(ev)
    df = pd.DataFrame(merged, columns=COLUMNS)
    df["duration_sec"] = df.offset_sec - df.onset_sec
    return df.reset_index(drop=True)


_default: DinoEEG | None = None


def detect(raw: mne.io.Raw) -> pd.DataFrame:
    global _default
    _default = _default or DinoEEG()
    return _default.detect(raw)
