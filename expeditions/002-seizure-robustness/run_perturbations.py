#!/usr/bin/env python3
"""Run the acquisition-error perturbations on whole recordings for one or more detectors, vs their clean baseline.

Usage (assumes poetry shell inside expeditions/002-seizure-robustness):
    python run_perturbations.py                    # eventNet, good-patient subset -> results/perturbations/good_patients/
    python run_perturbations.py --detectors gotman,sparcnet          # same 19 conditions, other detectors
    python run_perturbations.py --detectors all                      # gotman, eventnet, sparcnet, dinoeeg
    python run_perturbations.py --cohort all       # all 14 patients     -> results/perturbations/all_patients/
    python run_perturbations.py --variations mains_severe,impedance_mild
    python run_perturbations.py --patients PN00 --detectors dinoeeg  # one patient -> custom_patients/
    python run_perturbations.py --recordings PN00-1 --variations disconnected_mild   # one file -> custom_patients/
    python run_perturbations.py --list

Each recording is perturbed once and every selected detector runs on the same perturbed signal. A detector is
compared with its own clean predictions (the evaluate_detectors.py cache, clean/<detector>/), on the same recordings.
Perturbed predictions are cached in data/cache/002-seizure-robustness/perturbations/<detector>/<condition>/.
A detector failing on a recording is logged to <results>/errors.log and that recording is left out of its numbers.

Outputs (results/perturbations/<cohort>_patients/), one row per detector x condition (x recording), with a `detector`
column. Cohort runs merge into the existing files (rows of the same detector and condition are replaced), so models
run separately end up side by side; custom_patients/ is overwritten on every run.
    summary_by_variation.csv   recall, precision, f1, fa_per_24h, deltas vs clean
    metrics_by_recording.csv   per recording: hours, n_seizures, tp, fp, perturbed electrodes
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from timescoring import scoring
from timescoring.annotations import Annotation

# Setup path so neuro_expeditions is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_detectors import DETECTORS, get_detector
from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena
from neuro_expeditions.perturbations import (
    FOCUS_VARIATIONS,
    VARIATIONS_CATALOG,
    apply_variation,
    get_variation,
    list_variations,
    variation_electrodes,
)

ELECTRODES = list(siena.CHANNELS_10_20.values())  # channel names of siena.load_raw()

CACHE_DIR = DATA_DIR / "cache" / "002-seizure-robustness" / "perturbations"  # <detector>/<condition>/<rec>.tsv
CLEAN_CACHE_DIR = DATA_DIR / "cache" / "002-seizure-robustness" / "clean"  # <detector>/<rec>.tsv (evaluate_detectors)
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "perturbations"

# Baseline cohort ("good patients"). Inclusion criteria (README / TESTS.md): temporal seizure localization and at
# least 2 seizures. This list is the earlier working list, chosen from eventNet's first clean baseline (24/28 seizures
# detected) before the criteria were written down; it is used by run_extended_tests.py and audit_event_masking.py too.
GOOD_PATIENTS = ["PN00", "PN01", "PN05", "PN09", "PN12", "PN13", "PN14", "PN16", "PN17"]

SZCORE_PARAMS = scoring.EventScoring.Parameters(
    toleranceStart=30,
    toleranceEnd=60,
    minOverlap=0,
    maxEventDuration=300,
    minDurationBetweenEvents=90,
)


def to_mask(intervals, duration: float) -> np.ndarray:
    """1 Hz boolean mask from (onset, offset) intervals in seconds."""
    mask = np.zeros(int(np.ceil(duration)), dtype=bool)
    for on, off in intervals:
        mask[max(int(on), 0) : int(np.ceil(off))] = True
    return mask


def get_recording_duration(edf: Path) -> float:
    import mne
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    return raw.n_times / raw.info["sfreq"]


def score_events(ref_intervals, hyp_df: pd.DataFrame, duration: float) -> tuple[int, int, int]:
    """Score hypothesis detections against reference using SzCORE event scoring."""
    ref_mask = to_mask(ref_intervals, duration)
    hyp_intervals = list(zip(hyp_df.get("onset_sec", []), hyp_df.get("offset_sec", [])))
    hyp_mask = to_mask(hyp_intervals, duration)

    ev = scoring.EventScoring(Annotation(ref_mask, 1), Annotation(hyp_mask, 1), SZCORE_PARAMS)
    return int(ev.refTrue), int(ev.tp), int(ev.fp)



def read_events(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["onset_sec", "offset_sec"])


def aggregate(rows: list[dict]) -> dict:
    """Event metrics pooled over recordings."""
    tp, fp = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows)
    n_sz, hours = sum(r["n_seizures"] for r in rows), sum(r["hours"] for r in rows)
    recall = tp / max(1, n_sz)
    precision = tp / max(1, tp + fp)
    return {
        "recall": recall,
        "precision": precision,
        "f1": 2 * recall * precision / max(1e-6, recall + precision),
        "fa_per_24h": fp / max(hours, 1e-9) * 24.0,
        "tp": tp,
        "fp": fp,
        "total_seizures": n_sz,
        "n_recordings": len(rows),
    }


def save_merged(df: pd.DataFrame, path: Path, merge: bool) -> None:
    """Write df. With merge, rows of an existing file are kept unless df has the same (detector, condition)."""
    keys = ["detector", "condition"]
    if merge and path.is_file():
        old = pd.read_csv(path)
        if set(keys) <= set(old.columns):  # files from before the detector column are replaced
            new = set(map(tuple, df[keys].astype(str).values))
            old = old[[tuple(k) not in new for k in old[keys].astype(str).values]]
            df = pd.concat([old, df], ignore_index=True)
    df.round(4).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the perturbation conditions on whole recordings for one or more detectors, vs clean."
    )
    parser.add_argument(
        "--detectors",
        default="eventnet",
        help=f"Comma-separated detectors, or 'all' ({', '.join(DETECTORS)}). Default: eventnet.",
    )
    parser.add_argument(
        "--cohort",
        choices=["good", "all"],
        default="good",
        help="good: the 9 GOOD_PATIENTS (default); all: all 14 Siena patients. Results go to "
             "results/perturbations/<cohort>_patients/.",
    )
    parser.add_argument(
        "--variations",
        default="",
        help="Comma-separated list of variations to evaluate (default: the 15 benchmark variations, 5 error types x "
             "mild/moderate/severe, plus the 4 disconnected focus/contra variations; extra variations can be "
             "named here too).",
    )
    parser.add_argument(
        "--patients",
        default="",
        help="Comma-separated patient list; overrides --cohort (results go to results/perturbations/custom_patients/).",
    )
    parser.add_argument(
        "--recordings",
        default="",
        help="Comma-separated recording list (e.g. PN00-1,PN01-2); overrides --cohort and --patients "
             "(results go to results/perturbations/custom_patients/).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available variations and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute perturbed predictions even if cached (clean predictions are reused; delete "
             "data/cache/002-seizure-robustness/clean/<detector>/ or use evaluate_detectors.py to redo them).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available Technical Error Variations:")
        for v in list_variations():
            print(f"  • {v['name']:<28} [{v['set']}] : {v['description']}")
        return

    detectors = list(DETECTORS) if args.detectors.strip() == "all" else [
        d.strip() for d in args.detectors.split(",") if d.strip()
    ]
    unknown = [d for d in detectors if d not in DETECTORS]
    if unknown or not detectors:
        parser.error(f"unknown detector(s) {unknown}; choose from {', '.join(DETECTORS)} or 'all'")

    if args.patients:
        target_patients = [p.strip() for p in args.patients.split(",") if p.strip()]
        results_dir = RESULTS_DIR / "custom_patients"
    else:
        target_patients = GOOD_PATIENTS if args.cohort == "good" else siena.SUBJECTS
        results_dir = RESULTS_DIR / f"{args.cohort}_patients"
    if args.variations:
        target_variations = [v.strip() for v in args.variations.split(",") if v.strip()]
        for v in target_variations:
            get_variation(v)  # fail early on a typo
    else:
        target_variations = list(VARIATIONS_CATALOG) + list(FOCUS_VARIATIONS)

    all_recordings = siena.list_recordings()
    recordings = [r for r in all_recordings if siena.patient_id(r) in target_patients]
    if args.recordings:
        wanted = {r.strip() for r in args.recordings.split(",") if r.strip()}
        recordings = [r for r in all_recordings if r.stem in wanted]
        missing = wanted - {r.stem for r in recordings}
        if missing:
            parser.error(f"recording(s) not found: {', '.join(sorted(missing))}")
        results_dir = RESULTS_DIR / "custom_patients"
    merge = results_dir.name != "custom_patients"

    print("=" * 80)
    print(" Technical Error Robustness Benchmark")
    print("=" * 80)
    print(f" Detectors        : {', '.join(detectors)}")
    print(f" Recordings       : {len(recordings)} ({len({siena.patient_id(r) for r in recordings})} patients)")
    print(f" Variations       : {len(target_variations)} conditions")
    print(f" Cache Directory  : {CACHE_DIR}/<detector>/<condition>/")
    print(f" Results Directory: {results_dir}")
    print("=" * 80, flush=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    errors_log = results_dir / "errors.log"

    durations = {edf.stem: get_recording_duration(edf) for edf in recordings}
    ground_truths = {edf.stem: siena.load_seizures(edf) for edf in recordings}
    print(f"Total dataset: {sum(durations.values()) / 3600:.2f} hours, "
          f"{sum(len(s) for s in ground_truths.values())} true seizures across {len(recordings)} files.\n")

    loaded: dict = {}

    def predict(detector: str, raw, dest: Path, label: str) -> None:
        """Run a detector on raw and cache its events at dest; failures are logged and leave no cache file."""
        t0 = time.time()
        try:
            if detector not in loaded:
                print(f"     Loading {detector}...", flush=True)
                loaded[detector] = get_detector(detector)
            events = loaded[detector](raw)
        except Exception:
            dest.unlink(missing_ok=True)
            with open(errors_log, "a") as f:
                f.write(f"--- {detector} {label} {time.strftime('%Y-%m-%d %H:%M')}\n{traceback.format_exc()}\n")
            print(f"     {label} [{detector}] FAILED (see {errors_log.name})", flush=True)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(dest, sep="\t", index=False)
        print(f"     {label} [{detector}] {len(events)} event(s) in {time.time() - t0:.0f}s", flush=True)

    def score_row(detector: str, condition: str, edf: Path, events: pd.DataFrame, electrodes: str = "") -> dict:
        dur = durations[edf.stem]
        n_sz, tp, fp = score_events(ground_truths[edf.stem], events, dur)
        return {
            "detector": detector,
            "condition": condition,
            "recording": edf.stem,
            "patient": siena.patient_id(edf),
            "hours": dur / 3600.0,
            "n_seizures": n_sz,
            "tp": tp,
            "fp": fp,
            "electrodes": electrodes,
        }

    summary_records: list[dict] = []
    recording_rows: list[dict] = []

    def save() -> None:
        save_merged(pd.DataFrame(summary_records), results_dir / "summary_by_variation.csv", merge)
        save_merged(pd.DataFrame(recording_rows), results_dir / "metrics_by_recording.csv", merge)

    # 1. Clean baseline per detector (cache shared with evaluate_detectors.py)
    print("Clean baseline...", flush=True)
    clean_rows: dict[str, list[dict]] = {d: [] for d in detectors}
    for edf in recordings:
        dests = {d: CLEAN_CACHE_DIR / d / f"{edf.stem}.tsv" for d in detectors}
        pending = [d for d in detectors if not dests[d].is_file()]
        if pending:
            raw = siena.load_raw(edf, attach_annotations=False)
            for d in pending:
                predict(d, raw, dests[d], f"{edf.stem} clean")
            del raw
            gc.collect()
        for d in detectors:
            if dests[d].is_file():
                clean_rows[d].append(score_row(d, "clean", edf, read_events(dests[d])))

    for d in detectors:
        m = aggregate(clean_rows[d])
        summary_records.append({"detector": d, "condition": "clean", "description": "Clean unperturbed baseline",
                                **m, "delta_recall": 0.0, "delta_f1": 0.0, "delta_fa_24h": 0.0})
        recording_rows.extend(clean_rows[d])
        print(f"  [{d:<8}] Recall={m['recall']:.1%} ({m['tp']}/{m['total_seizures']}), "
              f"Precision={m['precision']:.1%}, F1={m['f1']:.3f}, FA/24h={m['fa_per_24h']:.1f}")
    print(flush=True)
    clean_stems = {d: {r["recording"] for r in clean_rows[d]} for d in detectors}

    # 2. Each condition: perturb each recording once, run every detector on it, score vs clean on the same recordings
    for v_idx, var_name in enumerate(target_variations, start=1):
        t0 = time.time()
        var_desc = get_variation(var_name)["desc"]
        print(f"[{v_idx}/{len(target_variations)}] Running: {var_name}")
        print(f"     Description: {var_desc}", flush=True)

        var_rows: dict[str, list[dict]] = {d: [] for d in detectors}
        for edf in recordings:
            electrodes = variation_electrodes(var_name, edf.stem, ELECTRODES)
            if not electrodes:  # focus variation, patient without a left/right focus (PN10)
                continue
            dests = {d: CACHE_DIR / d / var_name / f"{edf.stem}.tsv" for d in detectors}
            pending = [d for d in detectors if args.force or not dests[d].is_file()]
            if pending:
                raw = siena.load_raw(edf, attach_annotations=False)
                perturbed = apply_variation(raw, var_name)  # upstream, in the referential montage
                for d in pending:
                    predict(d, perturbed, dests[d], f"{edf.stem} {var_name}")
                del raw, perturbed
                gc.collect()
            for d in detectors:
                if dests[d].is_file() and edf.stem in clean_stems[d]:
                    var_rows[d].append(score_row(d, var_name, edf, read_events(dests[d]), " ".join(electrodes)))

        for d in detectors:
            rows = var_rows[d]
            if not rows:
                print(f"     [{d:<8}] no results")
                continue
            applied = {r["recording"] for r in rows}
            m = aggregate(rows)
            base = aggregate([r for r in clean_rows[d] if r["recording"] in applied])
            rec = {
                "detector": d,
                "condition": var_name,
                "description": var_desc,
                **m,
                "delta_recall": m["recall"] - base["recall"],
                "delta_f1": m["f1"] - base["f1"],
                "delta_fa_24h": m["fa_per_24h"] - base["fa_per_24h"],
            }
            summary_records.append(rec)
            recording_rows.extend(rows)
            note = f"  [{len(applied)}/{len(recordings)} recordings]" if len(applied) < len(recordings) else ""
            print(f"     [{d:<8}] Recall: {m['recall']:.1%} (Δ {rec['delta_recall']:+.1%}) | "
                  f"F1: {m['f1']:.3f} (Δ {rec['delta_f1']:+.3f}) | "
                  f"FA/24h: {m['fa_per_24h']:.1f} (Δ {rec['delta_fa_24h']:+.1f}){note}", flush=True)
        print(f"     [{time.time() - t0:.1f}s]\n", flush=True)
        save()

    # 3. Final summary
    save()
    summary_df = pd.DataFrame(summary_records).round(4)
    print("=" * 80)
    print(" ALL PERTURBATIONS COMPLETE — Summary Benchmark Table")
    print("=" * 80)
    display_cols = ["detector", "condition", "recall", "precision", "f1", "fa_per_24h",
                    "delta_recall", "delta_f1", "delta_fa_24h"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)
    print(f"Results saved to: {results_dir / 'summary_by_variation.csv'}\n")


if __name__ == "__main__":
    main()
