"""Gotman 1982 Classical Seizure Detector.

Heuristic rule-based seizure detection based on half-wave decomposition, relative amplitude
surges (>3x background), rhythmicity, and period regularity (CV < 0.6).
Requires zero training data.

Port of the official SzCORE baseline (github.com/esl-epfl/gotman_1982, commit 371d4b0), same behaviour:
    - input (`main.py`): the SzCORE-standardised EEG (19 channels, uV, 256 Hz, common average;
      `neuro_expeditions.szcore.to_szcore`), re-referenced to the 18-channel double-banana bipolar montage
      (`Eeg.reReferenceToBipolar`);
    - per channel (`gotman.py`): 60 Hz 4th-order Butterworth low-pass (zero phase), half-wave decomposition (peak
      prominence 10 uV), 2 s windows every 1 s from 28 s on (16 s background ending 12 s before the window), up to the
      end of the recording (the last windows are shorter). A window is detected if its mean half-wave amplitude is
      > 3x the background, its mean half-wave period is 0.025-0.150 s and the period CV is < 0.6. Known discrepancy,
      kept on purpose (we reproduce the SzCORE baseline): Gotman's published criterion is an average half-wave
      duration corresponding to 3-20 Hz (as the official code's comment says), but the official code measures
      maximum-to-maximum intervals (a full cycle) against half-wave bounds, so it accepts ~6.7-40 Hz.
    - across channels (`gotman.py`): a detection at window start i, on a channel where it is not that channel's last
      detection, marks [i, i + 1 s) if the same channel also detects at i + 1 s, or any other channel detects at i
      or at i + 1 s;
    - events (`main.py`, `Annotations.loadMask`): every run of marked samples, no minimum duration.
Only the cross-channel loop is rewritten (set lookups instead of four nested loops, same result) so that long
recordings stay fast. Departures only where the official code would crash or warn: a channel with a single
half-wave peak returns the filtered signal (official: interp1d error), and a zero mean period gives CV 1 (official:
0/0 = NaN; the window fails the period test either way).

Reference:
    Gotman J. "Automatic recognition of epileptic seizures in the EEG."
    Electroencephalography and Clinical Neurophysiology, 1982; 54(5):530-540.
    Implementation adapted from the official SzCORE baseline (esl-epfl/gotman_1982).
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d

from neuro_expeditions.szcore import to_szcore

DEFAULT_WINDOW_SEC = 2.0
DEFAULT_OVERLAP = 0.5
DEFAULT_BACKGROUND_SEC = 16.0
DEFAULT_TRANSITION_SEC = 12.0
PROMINENCE_UV = 10.0  # official half_wave(prominence=10), data in uV
MIN_PERIOD_SEC = 0.025  # official mean-period bounds
MAX_PERIOD_SEC = 0.150
COLUMNS = ["onset_sec", "offset_sec", "duration_sec", "confidence", "label"]


def _lowpass_filter(data: np.ndarray, sfreq: float, cutoff: float = 60.0) -> np.ndarray:
    """Butterworth 4th-order low-pass, zero phase (official `lowpass_filter`)."""
    b, a = signal.butter(4, cutoff, fs=sfreq, btype="low", analog=False)
    return signal.filtfilt(b, a, data, axis=-1)


def _half_wave(t: np.ndarray, filtered: np.ndarray, prominence: float = PROMINENCE_UV) -> np.ndarray:
    """Half-wave decomposition: linear interpolation through peaks and troughs (official `half_wave`)."""
    maxs, _ = signal.find_peaks(filtered, prominence=prominence)
    mins, _ = signal.find_peaks(-filtered, prominence=prominence)
    peaks = np.sort(np.concatenate((maxs, mins)))
    if len(peaks) <= 1:  # official: returns the signal for 0 peaks, crashes (interp1d) for 1
        return filtered
    return interp1d(t[peaks], filtered[peaks], fill_value="extrapolate", assume_sorted=True)(t)


def _calculate_period(t: np.ndarray, halfwave_segment: np.ndarray) -> np.ndarray:
    """Intervals between successive half-wave maxima, in seconds (official `calculate_period`)."""
    peaks, _ = signal.find_peaks(halfwave_segment)
    if len(peaks) <= 1:
        return np.array([t[-1] - t[0]])
    return np.diff(t[peaks])


def gotman_detect_raw(
    eeg_data: np.ndarray,
    sfreq: float,
    window_sec: float = DEFAULT_WINDOW_SEC,
    overlap: float = DEFAULT_OVERLAP,
    background_sec: float = DEFAULT_BACKGROUND_SEC,
    transition_sec: float = DEFAULT_TRANSITION_SEC,
) -> np.ndarray:
    """Official `gotman_algorithm` on (n_channels, n_samples) bipolar EEG in uV.

    Returns:
        Boolean mask of shape (n_samples,) indicating sample-level seizure detection.
    """
    n_channels, n_samples = eeg_data.shape
    t = np.arange(n_samples) / sfreq

    window_size = int(window_sec * sfreq)
    step = int(window_size * (1.0 - overlap))
    bg_size = int(background_sec * sfreq)
    tr_size = int(transition_sec * sfreq)

    # Per-channel window detections (window start sample indices, ascending)
    detections: list[list[int]] = []
    for ch_idx in range(n_channels):
        halfwaves = _half_wave(t, _lowpass_filter(eeg_data[ch_idx], sfreq))
        found = []
        for i in range(bg_size + tr_size, n_samples, step):
            bg_amp = np.mean(np.abs(halfwaves[i - (bg_size + tr_size) : i - tr_size]))
            win_amp = np.mean(np.abs(halfwaves[i : i + window_size]))
            periods = _calculate_period(t[i : i + window_size], halfwaves[i : i + window_size])
            mean_period = np.mean(periods)
            period_ok = MIN_PERIOD_SEC < mean_period < MAX_PERIOD_SEC
            variation_coeff = np.std(periods) / mean_period if mean_period > 0 else 1.0
            if win_amp > 3 * bg_amp and period_ok and variation_coeff < 0.6:
                found.append(i)
        detections.append(found)

    # Cross-channel criteria, as the official nested loops: each marking is [i, i + step) for a detection i that is
    # not the last one of its channel.
    sets = [set(d) for d in detections]
    seizure_mask = np.zeros(n_samples, dtype=bool)
    for ch, dets in enumerate(detections):
        others = [s for k, s in enumerate(sets) if k != ch]
        for i in dets[:-1]:
            if (
                (i + step) in sets[ch]  # next window on the same channel
                or any((i + step) in s for s in others)  # next window on another channel
                or any(i in s for s in others)  # same window on another channel
            ):
                seizure_mask[i : i + step] = True
    return seizure_mask


def mask_to_events(mask: np.ndarray, sfreq: float) -> pd.DataFrame:
    """Every run of detected samples as one event, no minimum duration (official `Annotations.loadMask`)."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    on = np.flatnonzero(edges == 1) / sfreq
    off = np.flatnonzero(edges == -1) / sfreq
    return pd.DataFrame({"onset_sec": on, "offset_sec": off, "duration_sec": off - on,
                         "confidence": 1.0, "label": "seizure"}, columns=COLUMNS)  # fmt: skip


def to_bipolar(raw: mne.io.Raw):
    """mne.Raw -> SzCORE-standardised Eeg re-referenced to the 18-channel double banana (uV, 256 Hz)."""
    eeg = to_szcore(raw)
    eeg.reReferenceToBipolar()
    return eeg


def detect(raw: mne.io.Raw) -> pd.DataFrame:
    """Run the Gotman 1982 seizure detector on an mne.Raw and return detected seizure events.

    Args:
        raw: mne.io.Raw recording in volts with the 19 10-20 channels (any reference).

    Returns:
        DataFrame with columns: [onset_sec, offset_sec, duration_sec, confidence, label]
    """
    eeg = to_bipolar(raw)
    return mask_to_events(gotman_detect_raw(eeg.data, eeg.fs), eeg.fs)
