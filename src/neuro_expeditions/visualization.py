"""EEG visualization utilities for Neuro-Expeditions.

Provides interactive MNE browsing and static overview plotting for EEG exams,
with automatic seizure centering, bipolar/referential montages, and model prediction overlays.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mne
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd

from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena

# Standard longitudinal double-banana bipolar montage
BIPOLAR_PAIRS = [
    ("Fp1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),
    ("Fp2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
   
]

# Average-reference montage: left chain, midline, right chain (Siena EDF names, old temporal labels).
AVERAGE_CHANNELS = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1",
    "Fz", "Cz", "Pz",
    "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2",
]  # fmt: skip


HEMISPHERE_COLORS = {"left": "#1f4fbf", "right": "#c0392b", "midline": "#1c2833"}


def channel_side(ch_name: str) -> str:
    """'left' (odd 10-20 number), 'right' (even) or 'midline' ('z'), from the first electrode in the name."""
    electrode = ch_name.split("-")[0].strip()
    match = re.search(r"(\d+)$", electrode)
    if match:
        return "left" if int(match.group(1)) % 2 else "right"
    return "midline"


def channel_colors(ch_names: list[str]) -> dict[str, str]:
    """Per-channel trace colors: left hemisphere blue, right red, midline dark."""
    return {ch: HEMISPHERE_COLORS[channel_side(ch)] for ch in ch_names}


def add_second_separators(ax: plt.Axes) -> None:
    """Vertical grid line every second, darker on labelled whole-second ticks. Follows scrolling."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 5, 10], integer=True))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which="minor", axis="x", color="#9aa0a6", lw=0.6, alpha=0.6)
    ax.grid(True, which="major", axis="x", color="#5f6368", lw=0.9, alpha=0.7)


def flip_polarity_for_display(raw: mne.io.Raw) -> mne.io.Raw:
    """Negate the signal IN PLACE so it plots negative-up (clinical EEG convention).

    Display only: call it on a Raw that is used just for plotting, never before analysis or detectors.
    """
    raw.apply_function(np.negative, channel_wise=False, verbose="error")
    return raw


def load_average_montage(exam_id_or_path: str | Path) -> mne.io.Raw:
    """Load AVERAGE_CHANNELS re-referenced to their common average, named '<ch>-avg'."""
    edf_path = siena.resolve_edf_path(exam_id_or_path)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    by_short = {name.replace("EEG", "").strip().lower(): name for name in raw.ch_names}
    missing = [c for c in AVERAGE_CHANNELS if c.lower() not in by_short]
    if missing:
        raise ValueError(f"{edf_path.name}: missing channels {missing}")

    originals = [by_short[c.lower()] for c in AVERAGE_CHANNELS]
    raw.pick(originals).load_data(verbose="error")
    raw.reorder_channels(originals)
    raw.rename_channels(dict(zip(originals, AVERAGE_CHANNELS)))
    raw.set_channel_types(dict.fromkeys(raw.ch_names, "eeg"))
    raw.set_eeg_reference("average", projection=False, verbose="error")
    raw.rename_channels({c: f"{c}-avg" for c in AVERAGE_CHANNELS})
    return raw


def format_seconds(seconds: float) -> str:
    """Format seconds into human-readable string (e.g. '1h 24m 30s' or '19m 03s')."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def get_exam_catalog(dataset: str = "siena") -> list[dict[str, Any]]:
    """Scan available recordings in dataset and return structured metadata.

    Returns a list of dicts with:
        id, patient, file_name, path, duration_sec, duration_str,
        sfreq, n_channels, seizures (list of onsets/offsets), predictions_available.
    """
    if dataset.lower() != "siena":
        raise ValueError(f"Unsupported dataset: {dataset}. Currently 'siena' is supported.")

    recordings = siena.list_recordings()
    catalog = []
    clean_cache = DATA_DIR / "cache" / "002-seizure-robustness" / "clean"

    for edf in recordings:
        try:
            raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
            dur = raw.n_times / raw.info["sfreq"]
            sfreq = float(raw.info["sfreq"])
        except Exception:
            dur = 0.0
            sfreq = 512.0

        try:
            raw_sz = siena.load_seizures(edf)
            seizures = [{"onset": on, "offset": off, "duration": off - on} for on, off in raw_sz]
        except Exception:
            seizures = []

        preds = {}
        for det in ("eventnet", "sparcnet", "gotman"):
            pred_file = clean_cache / det / f"{edf.stem}.tsv"
            if pred_file.is_file():
                try:
                    preds[det] = len(pd.read_csv(pred_file, sep="\t"))
                except Exception:
                    preds[det] = 0

        catalog.append({
            "id": edf.stem,
            "patient": siena.patient_id(edf),
            "file_name": edf.name,
            "path": edf,
            "duration_sec": dur,
            "duration_str": format_seconds(dur),
            "sfreq": sfreq,
            "n_channels": 19,
            "seizures": seizures,
            "predictions": preds,
        })

    return catalog


def format_exam_table(catalog: list[dict[str, Any]], title: str = "Available EEG Exams") -> str:
    """Format the catalog into a clean tabular ASCII display."""
    lines = []
    width = 96
    lines.append("=" * width)
    lines.append(f"{title}".center(width))
    lines.append("=" * width)
    lines.append(f"{'#':<4} {'Exam ID':<13} {'Patient':<9} {'Duration':<12} {'Sampling':<10} {'Seizures':<36}")
    lines.append("-" * width)

    for i, item in enumerate(catalog, start=1):
        sz_list = item["seizures"]
        n_sz = len(sz_list)
        if n_sz == 0:
            sz_str = "None"
        elif n_sz == 1:
            sz = sz_list[0]
            sz_str = f"1 sz @ {format_seconds(sz['onset'])} (dur {int(round(sz['duration']))}s)"
        else:
            onsets = ", ".join(format_seconds(s["onset"]) for s in sz_list[:2])
            if n_sz > 2:
                onsets += f" (+{n_sz - 2} more)"
            sz_str = f"{n_sz} sz @ {onsets}"

        # Truncate if too long
        if len(sz_str) > 42:
            sz_str = sz_str[:39] + "..."

        lines.append(
            f"[{i:2d}] {item['id']:<13} {item['patient']:<9} {item['duration_str']:<12} {int(item['sfreq']):<4} Hz   {sz_str}"
        )

    lines.append("-" * width)
    return "\n".join(lines)


def interactive_select_exam(
    catalog: list[dict[str, Any]],
    preselected: str | int | None = None,
) -> dict[str, Any]:
    """Resolve an exam choice from user input or interactive prompt."""
    if not catalog:
        raise RuntimeError("No EEG recordings found in dataset directory.")

    # 1. Check if preselected matches directly
    if preselected is not None:
        cand = str(preselected).strip()

        # If integer index
        if cand.isdigit():
            idx = int(cand)
            if 1 <= idx <= len(catalog):
                return catalog[idx - 1]

        # Exact ID match
        for item in catalog:
            if item["id"].lower() == cand.lower() or item["file_name"].lower() == cand.lower():
                return item

        # Patient prefix match
        patient_matches = [item for item in catalog if item["patient"].lower() == cand.lower()]
        if len(patient_matches) == 1:
            return patient_matches[0]
        if len(patient_matches) > 1:
            print(f"\nFound {len(patient_matches)} recordings for patient '{cand}':")
            return interactive_select_exam(patient_matches)

        # Substring search
        substr_matches = [item for item in catalog if cand.lower() in item["id"].lower()]
        if len(substr_matches) == 1:
            return substr_matches[0]

        print(f"Warning: Could not uniquely resolve '{preselected}'. Please select from the list:")

    # 2. Interactive prompt loop
    print(format_exam_table(catalog))
    while True:
        try:
            prompt_str = f"Select an exam [1-{len(catalog)}], ID (e.g. {catalog[0]['id']}), or 'q' to quit: "
            choice = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)

        if not choice:
            continue
        if choice.lower() in ("q", "quit", "exit"):
            print("Exiting.")
            sys.exit(0)

        # Check numeric index
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(catalog):
                return catalog[idx - 1]
            print(f"Index {idx} out of range [1-{len(catalog)}]. Please try again.")
            continue

        # Check ID match
        matched = [item for item in catalog if item["id"].lower() == choice.lower()]
        if matched:
            return matched[0]

        # Check patient filter
        matched_patient = [item for item in catalog if item["patient"].lower() == choice.lower()]
        if matched_patient:
            if len(matched_patient) == 1:
                return matched_patient[0]
            print(f"\nRecordings for patient '{choice}':")
            return interactive_select_exam(matched_patient)

        # Check substring match
        matched_sub = [item for item in catalog if choice.lower() in item["id"].lower()]
        if len(matched_sub) == 1:
            return matched_sub[0]
        elif len(matched_sub) > 1:
            print(f"\nMultiple exams match '{choice}':")
            return interactive_select_exam(matched_sub)

        print(f"Unrecognized exam '{choice}'. Please enter a valid number or ID.")


def _average_reference(raw: mne.io.Raw) -> mne.io.Raw:
    """Re-reference a siena.load_raw() Raw (modern names) to the common average, as load_average_montage does."""
    modern = [siena.CHANNELS_10_20[c] for c in AVERAGE_CHANNELS]
    raw = raw.copy().reorder_channels(modern)
    raw.set_eeg_reference("average", projection=False, verbose="error")
    raw.rename_channels({m: f"{c}-avg" for m, c in zip(modern, AVERAGE_CHANNELS)})
    return raw


def _add_prediction_annotations(raw: mne.io.Raw, pred_file: Path, label: str) -> None:
    """Append cached detector events (onset_sec/offset_sec TSV) to raw.annotations, if present."""
    if not pred_file.is_file():
        return
    try:
        df = pd.read_csv(pred_file, sep="\t")
    except pd.errors.EmptyDataError:
        return
    if len(df) == 0 or "onset_sec" not in df.columns:
        return
    raw.set_annotations(
        raw.annotations
        + mne.Annotations(
            onset=df["onset_sec"].tolist(),
            duration=(df["offset_sec"] - df["onset_sec"]).tolist(),
            description=[label] * len(df),
            orig_time=raw.annotations.orig_time,
        )
    )


def load_exam_raw(
    exam_id_or_path: str | Path,
    montage: str = "referential",
    show_models: bool = True,
    models: list[str] | None = None,
    perturbation: str | None = None,
) -> tuple[mne.io.Raw, list[tuple[float, float]]]:
    """Load raw EEG with annotations and optional montage, perturbation and model predictions.

    `perturbation` is a VARIATIONS_CATALOG name. It is applied to the referential signal before
    re-referencing, the same way run_perturbations.py feeds the detector.
    """
    raw = siena.load_raw(exam_id_or_path, attach_annotations=True)
    seizures = siena.load_seizures(exam_id_or_path)

    if perturbation:
        from neuro_expeditions.perturbations import apply_variation

        raw = apply_variation(raw, perturbation)

    # Optional model predictions overlay
    if show_models:
        edf_stem = siena.resolve_edf_path(exam_id_or_path).stem
        target_models = models or ["eventnet", "sparcnet", "gotman"]
        cache_root = DATA_DIR / "cache" / "002-seizure-robustness"

        for mod in target_models:
            _add_prediction_annotations(raw, cache_root / "clean" / mod / f"{edf_stem}.tsv", f"{mod}_pred")
        if perturbation:
            # run_perturbations.py caches each detector's events on the perturbed signal here
            for mod in target_models:
                _add_prediction_annotations(
                    raw,
                    cache_root / "perturbations" / mod / perturbation / f"{edf_stem}.tsv",
                    f"{mod}_perturbed_pred",
                )

    # Optional re-referencing
    if montage.lower() == "average":
        raw = _average_reference(raw)
    elif montage.lower() == "bipolar":
        anodes = [a for a, _ in BIPOLAR_PAIRS]
        cathodes = [c for _, c in BIPOLAR_PAIRS]
        ch_names = [f"{a}-{c}" for a, c in BIPOLAR_PAIRS]
        raw = mne.set_bipolar_reference(
            raw,
            anode=anodes,
            cathode=cathodes,
            ch_name=ch_names,
            drop_refs=True,
            verbose="error",
        )

    return raw, seizures


def plot_overview(
    raw: mne.io.Raw,
    seizures: list[tuple[float, float]],
    exam_id: str,
    window_start: float | None = None,
    window_duration: float = 30.0,
    save_path: Path | str | None = None,
    show: bool = True,
    compare_raw: mne.io.Raw | None = None,
) -> plt.Figure:
    """Create a high-quality static two-panel overview plot.

    compare_raw: the same exam without the perturbation (same montage and polarity). It is drawn in grey under
    `raw`, and the channels where the two differ get bold labels.

    Panel 1: Global Field Power (GFP) envelope across the entire exam,
             marking seizure intervals and the current window.
    Panel 2: Stacked multi-channel EEG trace for the chosen window.
    """
    sfreq = raw.info["sfreq"]
    total_dur = raw.times[-1]

    # Determine inspection window
    if window_start is None:
        if seizures:
            win_start = max(0.0, seizures[0][0] - 5.0)
        else:
            win_start = 0.0
    else:
        win_start = max(0.0, float(window_start))
    win_end = min(total_dur, win_start + window_duration)

    fig, (ax_env, ax_trace) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1, 2.5]}
    )

    # 1. Full recording envelope
    data = raw.get_data()  # (n_ch, n_times)
    gfp_uV = np.sqrt(np.mean(data**2, axis=0)) * 1e6
    ds_step = max(1, int(sfreq // 4))
    t_ds_min = raw.times[::ds_step] / 60.0

    if compare_raw is not None:
        gfp_clean = np.sqrt(np.mean(compare_raw.get_data() ** 2, axis=0)) * 1e6
        ax_env.plot(t_ds_min, gfp_clean[::ds_step], color="#9aa0a6", lw=0.9, label="Global Field Power (clean)")
    ax_env.plot(t_ds_min, gfp_uV[::ds_step], color="#2b5c8f", lw=0.9, label="Global Field Power")
    ax_env.axvspan(win_start / 60.0, win_end / 60.0, color="#f39c12", alpha=0.35, label="Zoomed Window")

    for i, (on, off) in enumerate(seizures):
        ax_env.axvspan(
            on / 60.0,
            off / 60.0,
            color="#e74c3c",
            alpha=0.45,
            label="Seizure (ground truth)" if i == 0 else None,
        )

    ax_env.set_title(
        f"Exam {exam_id} (Duration: {format_seconds(total_dur)}, {len(raw.ch_names)} Channels, {len(seizures)} Seizure(s)) — Full Envelope",
        fontsize=12,
        fontweight="bold",
    )
    ax_env.set_xlabel("Recording Time (minutes)", fontsize=10)
    ax_env.set_ylabel("RMS Amplitude (µV)", fontsize=10)
    ax_env.grid(True, linestyle="--", alpha=0.4)
    ax_env.legend(loc="upper right", framealpha=0.9)

    # 2. Zoomed multi-channel EEG trace
    start_idx = int(win_start * sfreq)
    end_idx = int(win_end * sfreq)
    win_times = raw.times[start_idx:end_idx]
    win_data_uV = data[:, start_idx:end_idx] * 1e6

    ch_names = raw.ch_names
    n_ch = len(ch_names)

    # Robust spacing based on signal standard deviation
    med_std = np.median(np.std(win_data_uV, axis=1))
    spacing = max(80.0, float(med_std * 5.0))

    colors = channel_colors(ch_names)
    y_ticks = [(n_ch - 1 - i) * spacing for i in range(n_ch)]
    changed: set[str] = set()
    if compare_raw is not None:
        clean_uV = compare_raw.get_data()[:, start_idx:end_idx] * 1e6
        changed = {ch_names[i] for i in np.flatnonzero(np.abs(win_data_uV - clean_uV).max(axis=1) > 1e-3)}
        for i in range(n_ch):
            ax_trace.plot(win_times, clean_uV[i] + y_ticks[i], color="#9aa0a6", lw=1.4, alpha=0.9,
                          label="Clean (no perturbation)" if i == 0 else None)
    for i in range(n_ch):
        offset = y_ticks[i]
        ax_trace.plot(win_times, win_data_uV[i] + offset, color=colors[ch_names[i]], lw=0.75)

    # Mark seizure span in zoomed trace
    for on, off in seizures:
        if off >= win_start and on <= win_end:
            ax_trace.axvspan(max(on, win_start), min(off, win_end), color="#e74c3c", alpha=0.2)
            if win_start <= on <= win_end:
                ax_trace.axvline(on, color="#c0392b", linestyle="--", lw=1.2, label=f"Seizure Onset ({on:.1f}s)")

    ax_trace.set_yticks(y_ticks)
    ax_trace.set_yticklabels(ch_names, fontsize=9)
    for label in ax_trace.get_yticklabels():
        label.set_color(colors[label.get_text()])
        if label.get_text() in changed:
            label.set_fontweight("bold")
    ax_trace.set_xlim(win_start, win_end)
    ax_trace.set_ylim(-spacing * 0.8, (n_ch) * spacing)
    ax_trace.set_xlabel("Time (seconds)", fontsize=10)
    ax_trace.set_title(
        f"Multi-Channel EEG Trace ({win_start:.1f}s – {win_end:.1f}s | {format_seconds(win_start)} – {format_seconds(win_end)})",
        fontsize=11,
    )
    add_second_separators(ax_trace)

    handles, labels = ax_trace.get_legend_handles_labels()
    if handles:
        ax_trace.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Saved overview plot to: {out}")

    if show:
        plt.show()

    return fig
