"""Siena Scalp EEG Database (PhysioNet, siena-scalp-eeg 1.0.0).

14 adult epilepsy patients, referential 10-20 EEG at 512 Hz, recordings of 1-12+ h, seizure times in
per-patient Seizures-list-PNxx.txt files. Those files contain typos, so seizure times are loaded with the
official SzCORE parser (`epilepsy2bids`), which carries the known corrections.

Data on PhysioNet: https://physionet.org/content/siena-scalp-eeg/1.0.0/
Open S3 mirror: https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0/
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import shutil
import time
import urllib.request
from pathlib import Path

import mne
import pandas as pd
from epilepsy2bids.load_annotations.siena import loadAnnotationsFromEdf

from neuro_expeditions import DATA_DIR

ROOT = DATA_DIR / "siena"
BASE_URL = "https://physionet-open.s3.amazonaws.com/siena-scalp-eeg/1.0.0"
FALLBACK_URL = "https://physionet.org/files/siena-scalp-eeg/1.0.0"

SUBJECTS = [
    "PN00", "PN01", "PN03", "PN05", "PN06", "PN07", "PN09",
    "PN10", "PN11", "PN12", "PN13", "PN14", "PN16", "PN17",
]

# The 19 channels of the standard 10-20 system, as named in the Siena EDFs (old temporal names)
# mapped to modern names (T3/T4/T5/T6 -> T7/T8/P7/P8).
CHANNELS_10_20 = {
    "Fp1": "Fp1", "Fp2": "Fp2", "F3": "F3", "F4": "F4", "C3": "C3", "C4": "C4", "P3": "P3", "P4": "P4",
    "O1": "O1", "O2": "O2", "F7": "F7", "F8": "F8", "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "Fz": "Fz", "Cz": "Cz", "Pz": "Pz",
}  # fmt: skip


def _download_file(rel_path: str, root: Path, expected_sha256: str | None = None, retries: int = 5) -> Path:
    dest = root / rel_path
    if dest.exists() and dest.stat().st_size > 0:
        if expected_sha256 is None:
            return dest
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        if h.hexdigest() == expected_sha256:
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    urls = [f"{BASE_URL}/{rel_path}", f"{FALLBACK_URL}/{rel_path}"]
    for attempt in range(retries):
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "NeuroExpeditions/0.1"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
                    shutil.copyfileobj(resp, f)
                break
            except Exception:
                continue
        if tmp.exists() and tmp.stat().st_size > 0:
            break
        time.sleep(2**attempt)

    if not tmp.exists() or tmp.stat().st_size == 0:
        raise RuntimeError(f"Failed to download {rel_path} after {retries} attempts.")

    tmp.rename(dest)
    return dest


def download(subjects: list[str] | None = None, root: Path = ROOT, max_workers: int = 6, verbose: bool = True) -> None:
    """Fetch Siena Scalp EEG recordings and metadata from PhysioNet/S3."""
    root.mkdir(parents=True, exist_ok=True)
    selected = set(subjects or SUBJECTS)

    for meta in ["subject_info.csv", "SHA256SUMS.txt", "LICENSE.txt"]:
        _download_file(meta, root)

    sums_file = root / "SHA256SUMS.txt"
    entries = []
    with open(sums_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                sha, path = parts[0], parts[1]
                subj = path.split("/")[0] if "/" in path else None
                if subj in selected or (subj is None and path.endswith(".txt")):
                    entries.append((path, sha))

    if verbose:
        print(f"Downloading {len(entries)} files for subjects: {sorted(selected)} with {max_workers} workers...", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_file, rel_path, root, sha): rel_path
            for rel_path, sha in entries
        }
        for future in concurrent.futures.as_completed(futures):
            rel_path = futures[future]
            try:
                future.result()
                if verbose:
                    print(f"  ✓ {rel_path}", flush=True)
            except Exception as e:
                print(f"  ✗ {rel_path}: {e}", flush=True)
                raise


def list_recordings(root: Path = ROOT) -> list[Path]:
    """Fully downloaded EDFs (partial `.edf.part` downloads are skipped)."""
    return sorted(root.glob("PN*/PN*.edf"))


def patient_id(edf: Path) -> str:
    return edf.parent.name


def seizure_side(recording: str | Path, root: Path = ROOT) -> str:
    """Seizure lateralization of a recording's patient from subject_info.csv: 'L', 'R' or 'Bilateral'.

    `recording` is a patient ID ('PN00'), a recording stem ('PN00-1') or an EDF path.
    """
    patient = Path(str(recording)).stem.split("-")[0]
    info = pd.read_csv(root / "subject_info.csv", skipinitialspace=True)
    sides = dict(zip(info.patient_id, info.lateralization))
    if patient not in sides:
        raise KeyError(f"{patient} not in {root / 'subject_info.csv'}")
    return str(sides[patient]).strip()


def resolve_edf_path(subject_or_edf: str | Path, run: int | str | None = None, root: Path = ROOT) -> Path:
    """Resolve an EDF path from a Path, filename, recording stem (e.g. 'PN00-1'), or (subject, run) pair."""
    if isinstance(subject_or_edf, Path) and subject_or_edf.is_file():
        return subject_or_edf

    s = str(subject_or_edf).strip()
    path_cand = Path(s)
    if path_cand.is_file():
        return path_cand

    if run is None:
        # Check direct relative path or with .edf extension
        if (root / path_cand).is_file():
            return root / path_cand
        if (root / f"{path_cand.stem}.edf").is_file():
            return root / f"{path_cand.stem}.edf"

        stem = path_cand.stem
        # Check if stem contains patient prefix, e.g. 'PN00-1' -> root / 'PN00' / 'PN00-1.edf'
        if "-" in stem:
            patient = stem.split("-")[0]
            cand = root / patient / f"{stem}.edf"
            if cand.is_file():
                return cand

        # Glob search for stem in patient folders
        matches = list(root.glob(f"*/{stem}.edf")) or list(root.glob(f"*/{stem}*.edf"))
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Recording '{subject_or_edf}' not found in {root}")

    # If run is specified
    subj_dir = root / s
    if not subj_dir.is_dir():
        cand_dirs = list(root.glob(f"*{s}*"))
        if cand_dirs:
            subj_dir = cand_dirs[0]

    if isinstance(run, int):
        candidates = list(subj_dir.glob(f"{subj_dir.name}-{run}.edf")) or list(subj_dir.glob(f"*{run}*.edf"))
    else:
        run_str = str(run).removesuffix(".edf")
        candidates = list(subj_dir.glob(f"{run_str}.edf")) or list(subj_dir.glob(f"*{run_str}*.edf"))

    if not candidates:
        raise FileNotFoundError(f"Recording run '{run}' for subject '{subject_or_edf}' not found in {subj_dir}")
    return candidates[0]


def load_raw(edf: Path | str, run: int | str | None = None, root: Path = ROOT, attach_annotations: bool = True) -> mne.io.Raw:
    """The 19 standard 10-20 channels, referential montage, in volts, with modern channel names."""
    edf_path = resolve_edf_path(edf, run, root)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="error")
    by_short = {name.replace("EEG", "").strip().lower(): name for name in raw.ch_names}
    missing = [c for c in CHANNELS_10_20 if c.lower() not in by_short]
    if missing:
        raise ValueError(f"{edf_path.name}: missing channels {missing}")
    raw.pick([by_short[c.lower()] for c in CHANNELS_10_20]).load_data(verbose="error")
    raw.rename_channels({by_short[c.lower()]: new for c, new in CHANNELS_10_20.items()})
    raw.set_channel_types(dict.fromkeys(raw.ch_names, "eeg"))
    raw.set_montage("standard_1020", on_missing="ignore", verbose="error")

    if attach_annotations:
        seizures = load_seizures(edf_path)
        if seizures:
            onsets, durations = [], []
            for onset, offset in seizures:
                onsets.append(onset)
                durations.append(offset - onset)
            ann = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=["seizure"] * len(onsets),
                orig_time=raw.annotations.orig_time if raw.annotations else None,
            )
            raw.set_annotations(ann)

    return raw


def load_seizures(edf: Path | str, run: int | str | None = None, root: Path = ROOT) -> list[tuple[float, float]]:
    """Seizure (onset, offset) in seconds from the start of the recording."""
    edf_path = resolve_edf_path(edf, run, root)
    events = loadAnnotationsFromEdf(str(edf_path)).events
    return [(e["onset"], e["onset"] + e["duration"]) for e in events if e["eventType"].name.startswith("sz")]


def load_annotations(edf: Path | str, run: int | str | None = None, root: Path = ROOT) -> pd.DataFrame:
    """Load annotations as a DataFrame with [onset_sec, offset_sec, duration_sec, description]."""
    edf_path = resolve_edf_path(edf, run, root)
    seizures = load_seizures(edf_path)
    records = []
    for onset, offset in seizures:
        records.append({
            "subject": patient_id(edf_path),
            "file_name": edf_path.name,
            "onset_sec": float(onset),
            "offset_sec": float(offset),
            "duration_sec": float(offset - onset),
            "description": "seizure",
        })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["subject", "file_name", "onset_sec", "offset_sec", "duration_sec", "description"])
