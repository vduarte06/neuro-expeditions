"""Simulated technical EEG acquisition errors for robustness evaluation.

Implements realistic electrode-level technical errors applied to referential (unipolar)
raw EEG prior to downstream detector preprocessing and re-referencing.

VARIATIONS_CATALOG is the benchmark grid: 5 error types x 3 severities (mild / moderate / severe). CORE_VARIATIONS
(the default of the core runs) keeps mild and severe only.

    Error          Simulation                                Electrodes                  Mild     Moderate  Severe
    mains          50 Hz sinusoid, peak amplitude, same      3 random                    5 µV     20 µV     50 µV
                   phase on all affected electrodes
    impedance      Gaussian noise band-limited 0.5-70 Hz,    2 random                    10 dB    5 dB      0 dB
                   SNR vs the electrode's own 0.5-70 Hz RMS
                   (median of 10 s windows)
    disconnected   flat line (0.5 µV noise floor)            1 / 3 / 5 random (nested)
    muscle         20-100 Hz noise in 2 s bursts covering    F7, F8, T7, T8, P7, P8      20 µV    50 µV     100 µV RMS
                   10% of the recording, same burst timing   (old names F7, F8, T3-T6)
                   on all 6 electrodes, flat top + 0.2 s
                   ramps (RMS holds on the flat top)
    displacement   x <- (1-a)x + a*neighbour (NEIGHBORS_10_20) 2 random                  a 0.25   a 0.50    a 0.75

Random electrodes are drawn per recording, seeded from the recording name and the error type, so a recording always
gets the same electrodes, the three severities of one error type share them (and the 1 / 3 / 5 disconnected
electrodes are nested), and the noise realization is shared too: only the intensity changes between severities.
variation_electrodes() returns the electrodes a variation uses for a recording.

FOCUS_VARIATIONS disconnects (flat line) temporal electrodes chosen by the patient's seizure side (Siena
subject_info.csv lateralization), with a control on the same electrodes of the other hemisphere:

    disconnected_focus_mild      mid-temporal electrode over the focus    T7 (left focus) / T8 (right focus)
    disconnected_focus_severe    temporal chain over the focus            F7, T7, P7 / F8, T8, P8
    disconnected_contra_mild     same electrode, other hemisphere         T8 / T7
    disconnected_contra_severe   same chain, other hemisphere             F8, T8, P8 / F7, T7, P7

Patients without a left or right focus (PN10, bilateral) get no electrodes: these conditions do not apply to them.
Random disconnected_* = typical cost of a failed electrode; focus = worst case; focus vs contra = the part of the
worst case that is due to location.

EXTRA_VARIATIONS keeps the other error types (drift, saturation, perspiration, gain) and the older muscle definition
used by the timing suite of run_extended_tests.py; they use fixed electrodes and seed 42, unchanged.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any, Callable
import mne
import numpy as np
from scipy import signal

# Standard 10-20 nearest spatial neighbors for displacement simulation
NEIGHBORS_10_20 = {
    "Fp1": "F3", "Fp2": "F4", "F3": "C3", "F4": "C4",
    "C3": "P3", "C4": "P4", "P3": "O1", "P4": "O2",
    "F7": "T7", "F8": "T8", "T7": "P7", "T8": "P8",
    "T3": "C3", "T4": "C4", "T5": "P3", "T6": "P4", "P7": "P3", "P8": "P4",
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
    same_phase: bool = False,
    **kwargs,
) -> mne.io.Raw:
    """Inject sinusoidal 50 Hz power-line interference (amplitude_uV = peak amplitude).

    same_phase=True gives every electrode the same phase (one mains source); otherwise each gets a random phase.
    """
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    sfreq = raw_mod.info["sfreq"]
    times = raw_mod.times
    rng = np.random.default_rng(seed)

    amp_V = amplitude_uV * 1e-6
    data = raw_mod.get_data(picks=target_chs)
    shared_phase = rng.uniform(0, 2 * np.pi)

    for i in range(len(target_chs)):
        phase = shared_phase if same_phase else rng.uniform(0, 2 * np.pi)
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


def _bandpass_sos(sfreq: float, low: float, high: float) -> np.ndarray:
    """4th-order Butterworth band-pass (second-order sections), upper edge kept below Nyquist."""
    return signal.butter(4, [low, min(high, 0.45 * sfreq)], btype="band", fs=sfreq, output="sos")


def perturb_impedance_snr(
    raw: mne.io.Raw,
    snr_db: float = 5.0,
    channels: str | list[str] = "F3",
    band: tuple[float, float] = (0.5, 70.0),
    window_sec: float = 10.0,
    seed: int = 42,
    **kwargs,
) -> mne.io.Raw:
    """High electrode impedance: add band-limited Gaussian noise at a set SNR.

    The signal level is the electrode's own RMS in `band`, taken as the median RMS over `window_sec` windows so that
    seizures and artefacts do not inflate it. Noise RMS = signal RMS / 10^(snr_db / 20); 0 dB = equal RMS.
    """
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, channels)
    if not target_chs:
        return raw_mod

    sfreq = raw_mod.info["sfreq"]
    sos = _bandpass_sos(sfreq, *band)
    win = int(window_sec * sfreq)
    rng = np.random.default_rng(seed)

    for ch in target_chs:
        idx = raw_mod.ch_names.index(ch)
        x = signal.sosfiltfilt(sos, raw_mod._data[idx])
        n_win = len(x) // win
        if n_win >= 1:
            signal_rms = float(np.median(np.sqrt(np.mean(x[: n_win * win].reshape(n_win, win) ** 2, axis=1))))
        else:
            signal_rms = float(np.sqrt(np.mean(x**2)))
        noise = signal.sosfiltfilt(sos, rng.normal(0.0, 1.0, size=raw_mod.n_times))
        noise *= (signal_rms / 10 ** (snr_db / 20)) / (np.std(noise) + 1e-20)
        raw_mod._data[idx] = raw_mod._data[idx] + noise
    return raw_mod


def perturb_muscle_bursts(
    raw: mne.io.Raw,
    rms_uV: float = 50.0,
    channels: str | list[str] = ("F7", "F8", "T7", "T8", "P7", "P8"),
    band: tuple[float, float] = (20.0, 100.0),
    time_fraction: float = 0.10,
    burst_sec: float = 2.0,
    ramp_sec: float = 0.2,
    seed: int = 42,
    **kwargs,
) -> mne.io.Raw:
    """Muscle (EMG) artefact: band-limited noise in bursts with the same timing on all target electrodes.

    Bursts of `burst_sec` cover `time_fraction` of the recording: the recording is split into equal slots and one
    burst is placed at a random position in each, so bursts never overlap. Each burst has a flat top with
    raised-cosine ramps of `ramp_sec`; the noise has `rms_uV` RMS on the flat top. Each electrode gets its own noise.
    """
    raw_mod = raw.copy()
    target_chs = _resolve_target_channels(raw_mod, list(channels))
    if not target_chs:
        return raw_mod

    sfreq = raw_mod.info["sfreq"]
    n_times = raw_mod.n_times
    rng = np.random.default_rng(seed)

    burst_len = min(int(burst_sec * sfreq), n_times)
    ramp_len = min(int(ramp_sec * sfreq), burst_len // 2)
    window = np.ones(burst_len)
    if ramp_len > 0:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ramp_len))
        window[:ramp_len] = ramp
        window[-ramp_len:] = ramp[::-1]

    n_bursts = max(1, int(round(time_fraction * n_times / burst_len)))
    slot = n_times // n_bursts
    envelope = np.zeros(n_times)
    for k in range(n_bursts):
        start = k * slot + int(rng.integers(0, max(1, slot - burst_len + 1)))
        stop = min(start + burst_len, n_times)
        envelope[start:stop] = window[: stop - start]

    sos = _bandpass_sos(sfreq, *band)
    for ch in target_chs:
        idx = raw_mod.ch_names.index(ch)
        emg = signal.sosfiltfilt(sos, rng.normal(0.0, 1.0, size=n_times))
        emg *= (rms_uV * 1e-6) / (np.std(emg) + 1e-20)
        raw_mod._data[idx] = raw_mod._data[idx] + emg * envelope
    return raw_mod


def perturb_displacement_multi(
    raw: mne.io.Raw,
    mix_ratio: float = 0.5,
    channels: str | list[str] = "F3",
    **kwargs,
) -> mne.io.Raw:
    """Electrode displacement on several electrodes: each one is mixed with its NEIGHBORS_10_20 neighbour.

    Mixing uses the original (unperturbed) neighbour signal, so the result does not depend on the electrode order.
    """
    raw_mod = raw.copy()
    original = raw.get_data()
    for ch in _resolve_target_channels(raw_mod, channels):
        neighbour = NEIGHBORS_10_20.get(ch)
        if neighbour not in raw_mod.ch_names:
            continue
        i, j = raw_mod.ch_names.index(ch), raw_mod.ch_names.index(neighbour)
        raw_mod._data[i] = (1.0 - mix_ratio) * original[i] + mix_ratio * original[j]
    return raw_mod


# -----------------------------------------------------------------------------
# Perturbation Registry & Variation Catalog
# -----------------------------------------------------------------------------

PERTURBATION_FUNCTIONS: dict[str, Callable] = {
    "mains_50hz": perturb_mains_50hz,
    "high_impedance": perturb_high_impedance,
    "impedance_snr": perturb_impedance_snr,
    "disconnected": perturb_disconnected_electrode,
    "muscle_emg": perturb_muscle_emg,
    "muscle_bursts": perturb_muscle_bursts,
    "displacement": perturb_electrode_displacement,
    "displacement_multi": perturb_displacement_multi,
    "perspiration": perturb_perspiration_drift,
    "gain": perturb_gain_calibration,
}

MUSCLE_ELECTRODES = ["F7", "F8", "T7", "T8", "P7", "P8"]  # F7, F8, T3-T6 in the old 10-20 names


def _grid(group: str, func: str, n_electrodes, levels: dict[str, dict], desc: str) -> dict[str, dict[str, Any]]:
    """Catalog entries <group>_<severity>. n_electrodes: int, per-severity dict, or None (fixed channels in params)."""
    out = {}
    for severity, params in levels.items():
        n = n_electrodes[severity] if isinstance(n_electrodes, dict) else n_electrodes
        out[f"{group}_{severity}"] = {
            "group": group,
            "func": func,
            "params": params,
            "n_electrodes": n,
            "desc": desc.format(n=n, **params),
        }
    return out


# Benchmark grid: 5 error types x 3 severities (see module docstring).
VARIATIONS_CATALOG: dict[str, dict[str, Any]] = {
    **_grid("mains", "mains_50hz", 3,
            {s: {"amplitude_uV": a, "same_phase": True} for s, a in [("mild", 5.0), ("moderate", 20.0), ("severe", 50.0)]},
            "50 Hz mains, {amplitude_uV:g} µV peak, same phase, on {n} random electrodes"),
    **_grid("impedance", "impedance_snr", 2,
            {s: {"snr_db": d} for s, d in [("mild", 10.0), ("moderate", 5.0), ("severe", 0.0)]},
            "High impedance: 0.5-70 Hz noise at SNR {snr_db:g} dB on {n} random electrodes"),
    **_grid("disconnected", "disconnected", {"mild": 1, "moderate": 3, "severe": 5},
            {s: {"mode": "flat"} for s in ("mild", "moderate", "severe")},
            "Disconnected electrode: flat line on {n} random electrode(s)"),
    **_grid("muscle", "muscle_bursts", None,
            {s: {"rms_uV": r, "channels": MUSCLE_ELECTRODES} for s, r in [("mild", 20.0), ("moderate", 50.0), ("severe", 100.0)]},
            "Muscle: 20-100 Hz bursts ({rms_uV:g} µV RMS, 2 s, 10% of the time, synchronous) on F7, F8, T7, T8, P7, P8"),
    **_grid("displacement", "displacement_multi", 2,
            {s: {"mix_ratio": m} for s, m in [("mild", 0.25), ("moderate", 0.50), ("severe", 0.75)]},
            "Displacement: {n} random electrodes mixed with their neighbour, alpha {mix_ratio:g}"),
}  # fmt: skip

# Other error types, outside the benchmark grid. Fixed electrodes, seed 42 (unchanged definitions);
# muscle_temporal_* and perspiration_severe_200uV are used by the timing suite of run_extended_tests.py.
EXTRA_VARIATIONS: dict[str, dict[str, Any]] = {
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

    # Sweat / perspiration drift
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

    # Channel calibration gain error
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


TEMPORAL_CHAINS = {"L": ["F7", "T7", "P7"], "R": ["F8", "T8", "P8"]}  # [1] = mid-temporal electrode

# Disconnected electrodes over the seizure focus, and the mirrored control (see module docstring).
FOCUS_VARIATIONS: dict[str, dict[str, Any]] = {
    f"disconnected_{hemisphere}_{severity}": {
        "group": "disconnected_focus",
        "func": "disconnected",
        "params": {"mode": "flat"},
        "n_electrodes": None,
        "hemisphere": hemisphere,  # 'focus' = seizure side, 'contra' = the other side
        "extent": extent,          # 'mid' = mid-temporal electrode, 'chain' = F7/T7/P7 or F8/T8/P8
        "desc": f"Disconnected electrode: flat line on {what}, "
                f"{'over the seizure focus' if hemisphere == 'focus' else 'on the side opposite the seizure focus'}",
    }
    for hemisphere in ("focus", "contra")
    for severity, extent, what in [("mild", "mid", "the mid-temporal electrode (T7/T8)"),
                                   ("severe", "chain", "the temporal chain (F7, T7, P7 / F8, T8, P8)")]
}  # fmt: skip

ALL_VARIATIONS: dict[str, dict[str, Any]] = {**VARIATIONS_CATALOG, **FOCUS_VARIATIONS, **EXTRA_VARIATIONS}

# Default set of the core runs (decided 2026-11-08 to cut compute): the 5 error types at the two ends of the
# intensity range, mild and severe. The moderate levels and the focus/contra conditions stay available by name
# (--variations); seizure side is covered by the `side` suite of run_extended_tests.py.
CORE_VARIATIONS: list[str] = [name for name in VARIATIONS_CATALOG if not name.endswith("_moderate")]


def get_variation(variation_name: str) -> dict[str, Any]:
    """Catalog entry of a benchmark or extra variation."""
    if variation_name not in ALL_VARIATIONS:
        raise ValueError(f"Unknown variation '{variation_name}'. Available: {list(ALL_VARIATIONS)}")
    return ALL_VARIATIONS[variation_name]


def _stable_seed(*parts: str) -> int:
    """Seed that is the same in every Python process (unlike hash())."""
    return zlib.crc32("|".join(parts).encode())


def _recording_name(raw: mne.io.Raw) -> str:
    filenames = [f for f in (raw.filenames or []) if f]
    if not filenames:
        raise ValueError("Cannot tell which recording this is; pass recording=<name> to apply_variation.")
    return Path(str(filenames[0])).stem


def variation_electrodes(
    variation_name: str, recording: str, ch_names: list[str], side: str | None = None
) -> list[str]:
    """Electrodes a variation perturbs in a recording (random ones are drawn reproducibly per recording).

    Focus variations need the seizure side ('L'/'R'); by default it is looked up in Siena's subject_info.csv.
    An empty list means the variation does not apply to this recording (no left or right focus).
    """
    spec = get_variation(variation_name)
    n = spec.get("n_electrodes")
    if "hemisphere" in spec:
        if side is None:
            from neuro_expeditions.datasets import siena

            side = siena.seizure_side(recording)
        if side not in TEMPORAL_CHAINS:
            return []
        if spec["hemisphere"] == "contra":
            side = {"L": "R", "R": "L"}[side]
        chain = TEMPORAL_CHAINS[side]
        return [c for c in (chain if spec["extent"] == "chain" else chain[1:2]) if c in ch_names]
    if n is None:  # fixed electrodes in params
        target = spec["params"].get("channels", spec["params"].get("channel", "all"))
        if target == "all":
            return list(ch_names)
        target = [target] if isinstance(target, str) else list(target)
        by_lower = {c.lower(): c for c in ch_names}
        return [by_lower[t.lower()] for t in target if t.lower() in by_lower]

    pool = sorted(ch_names)  # sorted, so the draw does not depend on the channel order
    if spec["func"] == "displacement_multi":
        pool = [c for c in pool if NEIGHBORS_10_20.get(c) in ch_names]
    rng = np.random.default_rng(_stable_seed(recording, spec["group"], "electrodes"))
    order = rng.permutation(len(pool))
    return [pool[i] for i in order[:n]]  # same permutation for every severity -> nested selections


def apply_variation(
    raw: mne.io.Raw, variation_name: str, recording: str | None = None, side: str | None = None
) -> mne.io.Raw:
    """Apply a named variation (VARIATIONS_CATALOG, FOCUS_VARIATIONS or EXTRA_VARIATIONS) to an mne.Raw recording.

    `recording` (e.g. 'PN00-1') seeds the electrode choice and noise of benchmark variations, and gives the seizure
    side of focus variations (or pass `side`); by default it is the stem of the file the Raw was read from, which
    survives copy() and crop(). Raises ValueError if a focus variation does not apply (no left or right focus).
    """
    spec = get_variation(variation_name)
    func = PERTURBATION_FUNCTIONS[spec["func"]]
    if variation_name in EXTRA_VARIATIONS:
        return func(raw, **spec["params"])

    recording = recording or _recording_name(raw)
    electrodes = variation_electrodes(variation_name, recording, raw.ch_names, side=side)
    if not electrodes:
        raise ValueError(f"{variation_name} does not apply to {recording}: no left or right seizure focus.")
    params = {k: v for k, v in spec["params"].items() if k != "channels"}
    return func(
        raw,
        channels=electrodes,
        seed=_stable_seed(recording, spec["group"], "noise"),
        **params,
    )


def list_variations(include_extra: bool = True) -> list[dict[str, str]]:
    """Variation names and descriptions; 'set' is 'core' (benchmark, run by default), 'benchmark', 'focus' or 'extra'."""
    rows = [{"name": k, "description": v["desc"], "set": "core" if k in CORE_VARIATIONS else "benchmark"}
            for k, v in VARIATIONS_CATALOG.items()]
    rows += [{"name": k, "description": v["desc"], "set": "focus"} for k, v in FOCUS_VARIATIONS.items()]
    if include_extra:
        rows += [{"name": k, "description": v["desc"], "set": "extra"} for k, v in EXTRA_VARIATIONS.items()]
    return rows
