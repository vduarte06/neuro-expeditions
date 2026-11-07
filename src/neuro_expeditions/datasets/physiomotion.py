"""PhysioMotion Artifact (OpenNeuro ds006386).

30 healthy subjects x 6 runs, 59-channel 10-20 EEG at 1000 Hz (~140 MB per run), recorded
while performing artefact-inducing tasks (blinks, eye/head movements, chewing, ...).
Manual artefact annotations live in derivatives/Manual_Annotations/sub{N}_run{RR}.csv with
columns channel,start_time,stop_time,label (one row per bipolar channel, or "ALL").

License CC0. Paper: https://www.nature.com/articles/s41597-026-07120-7
"""

from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path

import mne
import pandas as pd

from neuro_expeditions import DATA_DIR

DATASET_ID = "ds006386"
BASE_URL = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"
ROOT = DATA_DIR / "physiomotion"

SUBJECTS = list(range(1, 31))
RUNS = list(range(1, 7))

# Raw annotation label -> coarse group used for analysis. Combined labels (e.g. blink_eyebrow)
# map to "mixed" so they can be excluded from clean single-class comparisons.
LABEL_GROUPS = {
    "open_base": "clean_eyes_open",
    "close_base": "clean_eyes_closed",
    "blink": "blink",
    "hor_eyem": "eye_movement",
    "ver_eyem": "eye_movement",
    "chew": "muscle",
    "eyebrow": "muscle",
    "tongue": "muscle",
    "swallow": "muscle",
    "hor_headm": "head_movement",
    "ver_headm": "head_movement",
}
MIXED = "mixed"


def _run_files(subject: int, run: int) -> dict[str, str]:
    stem = f"sub-{subject}/eeg/sub-{subject}_task-artifact_run-{run:02d}"
    return {
        "eeg": f"{stem}_eeg.edf",
        "channels": f"{stem}_channels.tsv",
        "events": f"{stem}_events.tsv",
        "annotations": f"derivatives/Manual_Annotations/sub{subject}_run{run:02d}.csv",
    }


def _download(rel_path: str, root: Path, retries: int = 5) -> Path:
    dest = root / rel_path
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{rel_path}", timeout=60) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    tmp.rename(dest)
    return dest


def download(subjects=SUBJECTS, runs=RUNS, root: Path = ROOT, verbose: bool = True) -> None:
    """Fetch EDF + sidecars + manual annotations for the given subjects/runs (skips existing files)."""
    for s in subjects:
        for r in runs:
            if verbose:
                print(f"sub-{s} run-{r:02d}", flush=True)
            for rel in _run_files(s, r).values():
                _download(rel, root)


def load_raw(subject: int, run: int, root: Path = ROOT) -> mne.io.Raw:
    """Raw EEG in volts, with the manual annotations attached as mne.Annotations."""
    raw = mne.io.read_raw_edf(root / _run_files(subject, run)["eeg"], preload=True, verbose="error")
    raw.set_montage("colin27_1005", on_missing="warn")
    ann = load_annotations(subject, run, root)
    raw.set_annotations(
        mne.Annotations(ann["start"], ann["stop"] - ann["start"], ann["label"], orig_time=raw.annotations.orig_time)
    )
    return raw


def load_annotations(subject: int, run: int, root: Path = ROOT) -> pd.DataFrame:
    """One row per artefact event: start, stop, label, group, and the channels it was marked on."""
    df = pd.read_csv(root / _run_files(subject, run)["annotations"])
    df["label"] = df["label"].str.strip()
    events = (
        df.groupby(["start_time", "stop_time", "label"], sort=False)["channel"]
        .agg(lambda c: sorted(set(c)))
        .reset_index()
        .rename(columns={"start_time": "start", "stop_time": "stop", "channel": "channels"})
        .sort_values("start", ignore_index=True)
    )
    events["group"] = events["label"].map(LABEL_GROUPS).fillna(MIXED)
    return events
