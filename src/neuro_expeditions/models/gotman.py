"""Gotman 1982 Classical Seizure Detector.

Heuristic rule-based seizure detection based on half-wave decomposition, relative amplitude
surges (>3x background), rhythmicity (3-20 Hz), and period regularity (CV < 0.6).
Requires zero training data.

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

DEFAULT_WINDOW_SEC = 2.0
DEFAULT_OVERLAP = 0.5
DEFAULT_BACKGROUND_SEC = 16.0
DEFAULT_TRANSITION_SEC = 12.0


def _lowpass_filter(data: np.ndarray, sfreq: float, cutoff: float = 60.0) -> np.ndarray:
    """Butterworth 4th-order lowpass filter."""
    nyq = sfreq / 2.0
    if cutoff >= nyq:
        cutoff = nyq - 1.0
    b, a = signal.butter(4, cutoff, fs=sfreq, btype="low", analog=False)
    return signal.filtfilt(b, a, data, axis=-1)


def _half_wave(t: np.ndarray, filtered: np.ndarray, prominence: float = 1e-6) -> np.ndarray:
    """Decompose signal into half-waves via peak and trough interpolation."""
    maxs, _ = signal.find_peaks(filtered, prominence=prominence)
    mins, _ = signal.find_peaks(-filtered, prominence=prominence)
    peaks = np.sort(np.concatenate((maxs, mins)))

    if len(peaks) <= 1:
        return filtered

    f_halfwaves = interp1d(t[peaks], filtered[peaks], fill_value="extrapolate", assume_sorted=True)
    return f_halfwaves(t)


def _calculate_period(t: np.ndarray, halfwave_segment: np.ndarray) -> np.ndarray:
    """Calculate half-wave durations in seconds."""
    peaks, _ = signal.find_peaks(halfwave_segment)
    if len(peaks) <= 1:
        return np.array([t[-1] - t[0]]) if len(t) > 1 else np.array([0.1])
    return np.diff(t[peaks])


def gotman_detect_raw(
    eeg_data: np.ndarray,
    sfreq: float,
    window_sec: float = DEFAULT_WINDOW_SEC,
    overlap: float = DEFAULT_OVERLAP,
    background_sec: float = DEFAULT_BACKGROUND_SEC,
    transition_sec: float = DEFAULT_TRANSITION_SEC,
) -> np.ndarray:
    """Core Gotman detection logic operating on (n_channels, n_samples) EEG in Volts.

    Returns:
        Boolean mask of shape (n_samples,) indicating sample-level seizure detection.
    """
    n_channels, n_samples = eeg_data.shape
    t = np.arange(n_samples) / sfreq

    window_size = int(window_sec * sfreq)
    step = int(window_size * (1.0 - overlap))
    bg_size = int(background_sec * sfreq)
    tr_size = int(transition_sec * sfreq)
    min_start_idx = bg_size + tr_size

    if n_samples < min_start_idx + window_size:
        return np.zeros(n_samples, dtype=bool)

    channel_detections = np.zeros((n_channels, n_samples), dtype=bool)
    idx_detection: list[list[int]] = [[] for _ in range(n_channels)]

    for ch_idx in range(n_channels):
        raw_ch = eeg_data[ch_idx]
        filtered = _lowpass_filter(raw_ch, sfreq)
        halfwaves = _half_wave(t, filtered)

        for i in range(min_start_idx, n_samples - window_size, step):
            bg_amp = np.mean(np.abs(halfwaves[i - (bg_size + tr_size) : i - tr_size])) + 1e-12
            win_amp = np.mean(np.abs(halfwaves[i : i + window_size]))

            periods = _calculate_period(t[i : i + window_size], halfwaves[i : i + window_size])
            mean_period = np.mean(periods)

            # Rhythmic activity criteria: frequency between 3 and 20 Hz (0.05 to 0.33s period)
            period_ok = (mean_period > 0.050) and (mean_period < 0.333)
            variation_coeff = (np.std(periods) / mean_period) if mean_period > 0 else 1.0
            regularity_ok = variation_coeff < 0.6
            amplitude_ok = win_amp > (3.0 * bg_amp)

            if amplitude_ok and period_ok and regularity_ok:
                channel_detections[ch_idx, i : min(n_samples, i + step)] = True
                idx_detection[ch_idx].append(i)

    # Multi-channel clustering criteria
    seizure_mask = np.zeros(n_samples, dtype=bool)
    for ch1 in range(n_channels):
        ch1_events = idx_detection[ch1]
        for idx in range(len(ch1_events) - 1):
            # Consecutive epochs on the same channel
            if abs(ch1_events[idx + 1] - ch1_events[idx]) == step:
                seizure_mask[ch1_events[idx] : ch1_events[idx + 1] + step] = True

        for ch2 in range(ch1 + 1, n_channels):
            ch2_events = set(idx_detection[ch2])
            for i in ch1_events:
                # Same epoch or adjacent epoch across channels
                if i in ch2_events:
                    seizure_mask[i : min(n_samples, i + step)] = True
                if (i + step) in ch2_events or (i - step) in ch2_events:
                    seizure_mask[i : min(n_samples, i + step)] = True

    return seizure_mask


def mask_to_events(mask: np.ndarray, sfreq: float, min_duration_sec: float = 3.0) -> pd.DataFrame:
    """Convert a sample-level boolean detection mask into event intervals."""
    if not np.any(mask):
        return pd.DataFrame(columns=["onset_sec", "offset_sec", "duration_sec", "confidence", "label"])

    # Find contiguous True regions
    diff = np.diff(np.pad(mask.astype(np.int8), (1, 1), mode="constant"))
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]

    events = []
    for start, stop in zip(starts, stops):
        dur = (stop - start) / sfreq
        if dur >= min_duration_sec:
            events.append({
                "onset_sec": float(start / sfreq),
                "offset_sec": float(stop / sfreq),
                "duration_sec": float(dur),
                "confidence": 1.0,
                "label": "seizure",
            })

    return pd.DataFrame(events)


def detect(raw: mne.io.Raw, pick_type: str = "eeg", min_duration_sec: float = 3.0) -> pd.DataFrame:
    """Run Gotman 1982 seizure detector on an mne.Raw object and return detected seizure events.

    Args:
        raw: mne.io.Raw recording (in volts).
        pick_type: channel type to evaluate (default: "eeg").
        min_duration_sec: minimum event duration to retain.

    Returns:
        DataFrame with columns: [onset_sec, offset_sec, duration_sec, confidence, label]
    """
    raw_eeg = raw.copy().pick(pick_type)
    sfreq = float(raw_eeg.info["sfreq"])
    data = raw_eeg.get_data()  # shape (n_channels, n_samples)

    mask = gotman_detect_raw(data, sfreq)
    return mask_to_events(mask, sfreq, min_duration_sec=min_duration_sec)
