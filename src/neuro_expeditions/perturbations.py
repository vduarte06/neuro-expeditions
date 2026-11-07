"""Simulated technical EEG acquisition errors for robustness evaluation.

Implements realistic electrode-level technical errors applied to referential (unipolar)
raw EEG prior to downstream detector preprocessing and re-referencing.

Error types:
    - Mains interference (50 Hz sinusoid)
    - High electrode impedance (broadband thermal noise)
    - Disconnected / floating electrode (flatline, DC drift, open-lead mains saturation)
    - Muscle activity (EMG bursts at 30-100 Hz on temporal/frontal channels)
    - Electrode displacement / misplacement (channel mixing with 10-20 neighbor)
    - Perspiration / sweat artifact (ultra-slow <0.5 Hz high-amplitude drift)
    - Calibration / gain errors (channel amplitude scaling)
    - Electrode pop (sudden step jump with exponential decay)
"""

from __future__ import annotations

from typing import Any, Callable
import mne
import numpy as np
from scipy import signal

# Standard 10-20 nearest spatial neighbors for displacement simulation
NEIGHBORS_10_20 = {
    "Fp1": "F3", "Fp2": "F4", "F3": "C3", "F4": "C4",
    "C3": "P3", "C4": "P4", "P3": "O1", "P4": "O2",
    "F7": "T7", "F8": "T8", "T7": "P7", "T8": "P8",
    "T3": "C3", "T4": "C4", "T5": "P3", "T6": "P4",
    "Fz": "Cz", "Cz": "Pz", "Pz": "Cz", "O1": "P3", "O2": "P4",
}


def _resolve_target_channels(raw: mne.io.Raw, target: str | list[str] | None) -> list[str]:
    """Resolve channel name(s) against available channels in raw."""
    if target is None or target == "all":
        return list(raw.ch_names)
    if isinstance(target, str):
        target = [target]
    # Match case-insensitively or via synonyms
    matches = []
    ch_lower = {c.lower(): c for c in raw.ch_names}
    for t in target:
        if t.lower() in ch_lower:
            matches.append(ch_lower[t.lower()])
        else:
            # Check prefix/suffix
            for c in raw.ch_names:
                if t.lower() in c.lower():
                    matches.append(c)
                    break
    return list(dict.fromkeys(matches))


# -----------------------------------------------------------------------------
# Perturbation Generators
# -----------------------------------------------------------------------------

def perturb_mains_50hz(
    raw: mne.io.Raw,
    amplitude_uV: float = 50.0,
    channels: str | list[str] = "all",
    freq: float = 50.0,
    seed: int = 42,
    **kwargs,
) -> mne.io.Raw:
    """Inject sinusoidal 50 Hz power-line interference."""
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    sfreq = raw_mod.info["sfreq"]
    times = raw_mod.times
    rng = np.random.default_rng(seed)

    amp_V = amplitude_uV * 1e-6
    data = raw_mod.get_data(picks=target_chs)

    for i in range(len(target_chs)):
        phase = rng.uniform(0, 2 * np.pi)
        sine = amp_V * np.sin(2 * np.pi * freq * times + phase)
        data[i] += sine

    raw_mod._data[[raw_mod.ch_names.index(c) for c in target_chs]] = data
    return raw_mod


def perturb_high_impedance(
    raw: mne.io.Raw,
    noise_rms_uV: float = 30.0,
    channels: str | list[str] = "F3",
    seed: int = 42,
    **kwargs,
) -> mne.io.Raw:
    """Inject broadband Gaussian noise from poor electrode contact / high impedance."""
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    rng = np.random.default_rng(seed)
    rms_V = noise_rms_uV * 1e-6
    data = raw_mod.get_data(picks=target_chs)

    for i in range(len(target_chs)):
        noise = rng.normal(0.0, rms_V, size=raw_mod.n_times)
        data[i] += noise

    raw_mod._data[[raw_mod.ch_names.index(c) for c in target_chs]] = data
    return raw_mod


def perturb_disconnected_electrode(
    raw: mne.io.Raw,
    mode: str = "flat",  # 'flat', 'drift', 'saturation'
    channels: str | list[str] = "F3",
    seed: int = 42,
    **kwargs,
) -> mne.io.Raw:
    """Simulate a loose, fallen, or detached electrode."""
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    rng = np.random.default_rng(seed)
    sfreq = raw_mod.info["sfreq"]
    times = raw_mod.times
    n_times = raw_mod.n_times
    data = raw_mod.get_data(picks=target_chs)

    for i in range(len(target_chs)):
        if mode == "flat":
            # Channel is near-zero (tiny thermal floor of 0.5 µV)
            data[i] = rng.normal(0, 0.5e-6, size=n_times)
        elif mode == "drift":
            # Slow random walk / DC wander (up to 300 µV)
            walk = np.cumsum(rng.normal(0, 0.5e-6, size=n_times))
            # High-pass filter slightly so it stays within clinical amplifier bounds
            b, a = signal.butter(1, 0.05 / (sfreq / 2), btype="high")
            drift = signal.filtfilt(b, a, walk) * 10
            data[i] = drift
        elif mode == "saturation":
            # Floating lead picking up massive 50 Hz mains hum and clipping
            mains = 800e-6 * np.sin(2 * np.pi * 50.0 * times + rng.uniform(0, 2 * np.pi))
            data[i] = np.clip(mains + rng.normal(0, 10e-6, size=n_times), -500e-6, 500e-6)

    raw_mod._data[[raw_mod.ch_names.index(c) for c in target_chs]] = data
    return raw_mod


def perturb_muscle_emg(
    raw: mne.io.Raw,
    burst_amplitude_uV: float = 60.0,
    channels: str | list[str] = ["F7", "T7", "T8", "F8"],
    burst_rate_per_min: float = 6.0,
    burst_duration_sec: float = 2.5,
    seed: int = 42,
    **kwargs,
) -> mne.io.Raw:
    """Simulate high-frequency muscle (EMG) bursts on temporalis/frontalis electrodes."""
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    sfreq = raw_mod.info["sfreq"]
    n_times = raw_mod.n_times
    total_dur_min = n_times / (sfreq * 60.0)
    rng = np.random.default_rng(seed)

    # Design 30-100 Hz bandpass filter for EMG noise
    b, a = signal.butter(3, [30.0 / (sfreq / 2), min(100.0, sfreq / 2 - 1) / (sfreq / 2)], btype="band")

    n_bursts = int(total_dur_min * burst_rate_per_min)
    burst_len = int(burst_duration_sec * sfreq)

    data = raw_mod.get_data(picks=target_chs)
    for i in range(len(target_chs)):
        # Generate envelope of bursts
        emg_raw = rng.normal(0, 1.0, size=n_times)
        emg_filtered = signal.filtfilt(b, a, emg_raw)
        emg_filtered = emg_filtered / (np.std(emg_filtered) + 1e-12) * (burst_amplitude_uV * 1e-6)

        envelope = np.zeros(n_times)
        if n_bursts > 0:
            burst_starts = rng.integers(0, max(1, n_times - burst_len), size=n_bursts)
            window = np.hanning(burst_len)
            for bs in burst_starts:
                envelope[bs : bs + burst_len] = np.maximum(envelope[bs : bs + burst_len], window)

        data[i] += emg_filtered * envelope

    raw_mod._data[[raw_mod.ch_names.index(c) for c in target_chs]] = data
    return raw_mod


def perturb_electrode_displacement(
    raw: mne.io.Raw,
    mix_ratio: float = 0.4,
    channel: str = "F3",
    neighbor: str | None = None,
    **kwargs,
) -> mne.io.Raw:
    """Simulate electrode misplacement by blending channel with its spatial 10-20 neighbor."""
    raw_mod = raw.copy()
    ch_list = _resolve_target_channels(raw_mod, channel)
    if not ch_list:
        return raw_mod
    target = ch_list[0]

    if neighbor is None:
        neighbor_cand = NEIGHBORS_10_20.get(target, "Cz")
        neigh_list = _resolve_target_channels(raw_mod, neighbor_cand)
        if not neigh_list:
            return raw_mod
        neigh = neigh_list[0]
    else:
        neigh = neighbor

    idx_target = raw_mod.ch_names.index(target)
    idx_neigh = raw_mod.ch_names.index(neigh)

    target_data = raw_mod._data[idx_target]
    neigh_data = raw_mod._data[idx_neigh]

    # Blended signal
    raw_mod._data[idx_target] = (1.0 - mix_ratio) * target_data + mix_ratio * neigh_data
    return raw_mod


def perturb_perspiration_drift(
    raw: mne.io.Raw,
    drift_amplitude_uV: float = 120.0,
    channels: str | list[str] = "all",
    seed: int = 42,
    **kwargs,
) -> mne.io.Raw:
    """Simulate ultra-slow sweat / perspiration artifact (<0.3 Hz undulating baseline)."""
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    sfreq = raw_mod.info["sfreq"]
    times = raw_mod.times
    rng = np.random.default_rng(seed)
    amp_V = drift_amplitude_uV * 1e-6
    data = raw_mod.get_data(picks=target_chs)

    for i in range(len(target_chs)):
        # Superposition of 2 slow sinusoids + random drift
        f1 = rng.uniform(0.08, 0.25)
        f2 = rng.uniform(0.03, 0.10)
        p1 = rng.uniform(0, 2 * np.pi)
        p2 = rng.uniform(0, 2 * np.pi)
        slow_wave = 0.7 * np.sin(2 * np.pi * f1 * times + p1) + 0.3 * np.sin(2 * np.pi * f2 * times + p2)
        data[i] += amp_V * slow_wave

    raw_mod._data[[raw_mod.ch_names.index(c) for c in target_chs]] = data
    return raw_mod


def perturb_gain_calibration(
    raw: mne.io.Raw,
    scale_factor: float = 0.5,
    channels: str | list[str] = "F3",
    **kwargs,
) -> mne.io.Raw:
    """Simulate amplifier gain or impedance calibration error (scaling signal amplitude)."""
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    for ch in target_chs:
        idx = raw_mod.ch_names.index(ch)
        raw_mod._data[idx] *= scale_factor
    return raw_mod


# -----------------------------------------------------------------------------
# Perturbation Registry & Variation Catalog
# -----------------------------------------------------------------------------

PERTURBATION_FUNCTIONS: dict[str, Callable] = {
    "mains_50hz": perturb_mains_50hz,
    "high_impedance": perturb_high_impedance,
    "disconnected": perturb_disconnected_electrode,
    "muscle_emg": perturb_muscle_emg,
    "displacement": perturb_electrode_displacement,
    "perspiration": perturb_perspiration_drift,
    "gain": perturb_gain_calibration,
}

# Pre-configured standardized variations ready for benchmark evaluation
VARIATIONS_CATALOG: dict[str, dict[str, Any]] = {
    # 1. Mains 50 Hz interference
    "mains_single_25uV": {
        "func": "mains_50hz",
        "params": {"amplitude_uV": 25.0, "channels": "F3"},
        "desc": "50 Hz mains hum (25 µV) on single electrode (F3)",
    },
    "mains_single_100uV": {
        "func": "mains_50hz",
        "params": {"amplitude_uV": 100.0, "channels": "F3"},
        "desc": "50 Hz mains hum (100 µV) on single electrode (F3)",
    },
    "mains_global_50uV": {
        "func": "mains_50hz",
        "params": {"amplitude_uV": 50.0, "channels": "all"},
        "desc": "50 Hz mains hum (50 µV) across all electrodes",
    },
    "mains_global_150uV": {
        "func": "mains_50hz",
        "params": {"amplitude_uV": 150.0, "channels": "all"},
        "desc": "50 Hz mains hum (150 µV) across all electrodes",
    },

    # 2. High electrode contact impedance (broadband noise)
    "impedance_single_20uV": {
        "func": "high_impedance",
        "params": {"noise_rms_uV": 20.0, "channels": "F3"},
        "desc": "High impedance noise (20 µV RMS) on F3",
    },
    "impedance_single_60uV": {
        "func": "high_impedance",
        "params": {"noise_rms_uV": 60.0, "channels": "F3"},
        "desc": "High impedance noise (60 µV RMS) on F3",
    },
    "impedance_multi_30uV": {
        "func": "high_impedance",
        "params": {"noise_rms_uV": 30.0, "channels": ["F3", "C3", "P3"]},
        "desc": "High impedance noise (30 µV RMS) on 3 electrodes (F3, C3, P3)",
    },

    # 3. Disconnected / loose electrode
    "disconnected_flat_F3": {
        "func": "disconnected",
        "params": {"mode": "flat", "channels": "F3"},
        "desc": "Disconnected electrode (flatline on F3)",
    },
    "disconnected_flat_F3_T7": {
        "func": "disconnected",
        "params": {"mode": "flat", "channels": ["F3", "T7"]},
        "desc": "Disconnected electrodes (flatline on F3 and T7)",
    },
    "disconnected_drift_F3": {
        "func": "disconnected",
        "params": {"mode": "drift", "channels": "F3"},
        "desc": "Disconnected electrode with large baseline drift on F3",
    },
    "disconnected_saturation_F3": {
        "func": "disconnected",
        "params": {"mode": "saturation", "channels": "F3"},
        "desc": "Disconnected floating electrode with mains saturation on F3",
    },

    # 4. Muscle tension / EMG bursts
    "muscle_temporal_40uV": {
        "func": "muscle_emg",
        "params": {"burst_amplitude_uV": 40.0, "channels": ["F7", "T7", "F8", "T8"]},
        "desc": "Intermittent muscle EMG bursts (40 µV) on temporal channels",
    },
    "muscle_temporal_100uV": {
        "func": "muscle_emg",
        "params": {"burst_amplitude_uV": 100.0, "channels": ["F7", "T7", "F8", "T8"]},
        "desc": "Strong muscle EMG bursts (100 µV) on temporal channels",
    },

    # 5. Electrode displacement
    "displacement_mild_25pct": {
        "func": "displacement",
        "params": {"mix_ratio": 0.25, "channel": "F3"},
        "desc": "Electrode misplacement (25% cross-talk from C3 into F3)",
    },
    "displacement_moderate_50pct": {
        "func": "displacement",
        "params": {"mix_ratio": 0.50, "channel": "F3"},
        "desc": "Electrode misplacement (50% cross-talk from C3 into F3)",
    },

    # 6. Sweat / perspiration drift
    "perspiration_mild_75uV": {
        "func": "perspiration",
        "params": {"drift_amplitude_uV": 75.0, "channels": "all"},
        "desc": "Perspiration sweat artifact (75 µV ultra-slow wave, all channels)",
    },
    "perspiration_severe_200uV": {
        "func": "perspiration",
        "params": {"drift_amplitude_uV": 200.0, "channels": "all"},
        "desc": "Severe perspiration artifact (200 µV ultra-slow wave, all channels)",
    },

    # 7. Channel calibration gain error
    "gain_attenuation_0.5x": {
        "func": "gain",
        "params": {"scale_factor": 0.5, "channels": "F3"},
        "desc": "Amplifier gain error: 0.5x signal attenuation on F3",
    },
    "gain_amplification_2.0x": {
        "func": "gain",
        "params": {"scale_factor": 2.0, "channels": "F3"},
        "desc": "Amplifier gain error: 2.0x signal amplification on F3",
    },
}


def apply_variation(raw: mne.io.Raw, variation_name: str) -> mne.io.Raw:
    """Apply a named variation from VARIATIONS_CATALOG to an mne.Raw recording."""
    if variation_name not in VARIATIONS_CATALOG:
        raise ValueError(
            f"Unknown variation '{variation_name}'. Available: {list(VARIATIONS_CATALOG.keys())}"
        )

    var_info = VARIATIONS_CATALOG[variation_name]
    func_key = var_info["func"]
    func = PERTURBATION_FUNCTIONS[func_key]
    params = var_info["params"]
    return func(raw, **params)


def list_variations() -> list[dict[str, str]]:
    """Return a list of available variation identifiers and descriptions."""
    return [{"name": k, "description": v["desc"]} for k, v in VARIATIONS_CATALOG.items()]
