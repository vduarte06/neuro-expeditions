"""002: run the pretrained seizure detectors on Siena and score them with SzCORE (currently: clean, unperturbed data).

Detectors (used as released, no training on Siena): Gotman 1982, eventNet, SPaRCNet, DINO-EEG.

    python evaluate_detectors.py
    python evaluate_detectors.py --detectors gotman,eventnet --patients PN00

Predictions are cached per detector x recording (data/cache/002-seizure-robustness/clean/<detector>/<rec>.tsv),
so re-running only processes new recordings (e.g. after more Siena files finish downloading). A detector that
fails on a recording is logged to results/clean/errors.log and skipped; the rest keeps running.

Outputs (expeditions/002-seizure-robustness/results/clean/):
    metrics_by_recording.csv, metrics_by_patient.csv, metrics_overall.csv, run_info.json, errors.log
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from timescoring import scoring
from timescoring.annotations import Annotation

from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena

HERE = Path(__file__).parent
CACHE = DATA_DIR / "cache" / "002-seizure-robustness" / "clean"
RESULTS = HERE / "results" / "clean"
DETECTORS = ("gotman", "eventnet", "sparcnet", "dinoeeg")
# Official SzCORE event-scoring parameters
SZCORE = scoring.EventScoring.Parameters(
    toleranceStart=30, toleranceEnd=60, minOverlap=0, maxEventDuration=5 * 60, minDurationBetweenEvents=90
)


def get_detector(name: str):
    """Returns detect(raw) -> DataFrame[onset_sec, offset_sec, ...]. Models are loaded once."""
    if name == "gotman":
        from neuro_expeditions.models import gotman

        return gotman.detect
    if name == "eventnet":
        from neuro_expeditions.models import eventnet

        model = eventnet.load_model()
        return lambda raw: eventnet.detect(raw, model=model)
    if name == "sparcnet":
        from neuro_expeditions.models import sparcnet

        model = sparcnet.load_model()
        return lambda raw: sparcnet.detect(raw, model=model)
    if name == "dinoeeg":
        from neuro_expeditions.models import dinoeeg

        return dinoeeg.DinoEEG().detect
    raise ValueError(name)


def run_detection(detectors: list[str], recordings: list[Path]) -> None:
    loaded = {}
    for edf in recordings:
        todo = [d for d in detectors if not (CACHE / d / f"{edf.stem}.tsv").exists()]
        if not todo:
            continue
        raw = siena.load_raw(edf, attach_annotations=False)
        for d in todo:
            t0 = time.time()
            try:
                loaded[d] = loaded.get(d) or get_detector(d)
                events = loaded[d](raw)
            except Exception:
                with open(RESULTS / "errors.log", "a") as f:
                    f.write(f"--- {d} {edf.name} {time.strftime('%Y-%m-%d %H:%M')}\n{traceback.format_exc()}\n")
                print(f"{edf.name} [{d}] FAILED (see errors.log)", flush=True)
                continue
            dest = CACHE / d / f"{edf.stem}.tsv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            events.to_csv(dest, sep="\t", index=False)
            print(f"{edf.name} [{d}] {len(events)} event(s) in {time.time() - t0:.0f}s", flush=True)
        del raw
        gc.collect()


def to_mask(intervals, duration: float) -> np.ndarray:
    """1 Hz boolean mask from (onset, offset) intervals in seconds."""
    mask = np.zeros(int(np.ceil(duration)), dtype=bool)
    for on, off in intervals:
        mask[max(int(on), 0) : int(np.ceil(off))] = True
    return mask


def score(detectors: list[str], recordings: list[Path]) -> pd.DataFrame:
    rows = []
    for edf in recordings:
        duration = _duration(edf)
        ref = to_mask(siena.load_seizures(edf), duration)
        for d in detectors:
            f = CACHE / d / f"{edf.stem}.tsv"
            if not f.exists():
                continue
            try:
                events = pd.read_csv(f, sep="\t")
            except pd.errors.EmptyDataError:  # detector found nothing and wrote an empty frame
                events = pd.DataFrame(columns=["onset_sec", "offset_sec"])
            hyp = to_mask(zip(events.get("onset_sec", []), events.get("offset_sec", [])), duration)
            ev = scoring.EventScoring(Annotation(ref, 1), Annotation(hyp, 1), SZCORE)
            rows.append({"detector": d, "patient": siena.patient_id(edf), "recording": edf.stem,
                         "hours": duration / 3600, "n_seizures": int(ev.refTrue), "tp": int(ev.tp), "fp": int(ev.fp)})
    return pd.DataFrame(rows)


def _duration(edf: Path) -> float:
    import mne

    raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    return raw.n_times / raw.info["sfreq"]


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by).agg(recordings=("recording", "count"), hours=("hours", "sum"),
                           n_seizures=("n_seizures", "sum"), tp=("tp", "sum"), fp=("fp", "sum"))
    g["sensitivity"] = g.tp / g.n_seizures
    g["precision"] = g.tp / (g.tp + g.fp).replace(0, np.nan)
    g["f1"] = 2 * g.sensitivity * g.precision / (g.sensitivity + g.precision)
    g["fp_per_24h"] = g.fp / g.hours * 24
    return g.round(4).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--patients", default="", help="comma-separated, e.g. PN00,PN05 (default: all downloaded)")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    detectors = [d.strip() for d in args.detectors.split(",")]

    recordings = siena.list_recordings()
    if args.patients:
        recordings = [r for r in recordings if siena.patient_id(r) in args.patients.split(",")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"{len(recordings)} complete recordings, {len({siena.patient_id(r) for r in recordings})} patients", flush=True)

    if not args.score_only:
        run_detection(detectors, recordings)

    per_rec = score(detectors, recordings)
    per_rec.to_csv(RESULTS / "metrics_by_recording.csv", index=False)
    summarise(per_rec, ["detector", "patient"]).to_csv(RESULTS / "metrics_by_patient.csv", index=False)
    overall = summarise(per_rec, ["detector"])
    overall.to_csv(RESULTS / "metrics_overall.csv", index=False)
    (RESULTS / "run_info.json").write_text(json.dumps({
        "recordings": [r.name for r in recordings], "detectors": detectors,
        "szcore_params": vars(SZCORE), "finished": time.strftime("%Y-%m-%d %H:%M"),
    }, indent=2))
    print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
