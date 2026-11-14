#!/usr/bin/env python3
"""Run the acquisition-error perturbations on whole recordings for one or more detectors, vs their clean baseline.

Usage (assumes poetry shell inside expeditions/002-seizure-robustness):
    python run_perturbations.py                    # eventNet, included-patient subset -> results/perturbations/included_patients/
    python run_perturbations.py --detectors gotman,sparcnet          # same 10 core conditions, other detectors
    python run_perturbations.py --detectors all                      # gotman, eventnet, sparcnet, dinoeeg
    python run_perturbations.py --cohort all       # all 14 patients     -> results/perturbations/all_patients/
    python run_perturbations.py --variations mains_severe,impedance_mild
    python run_perturbations.py --patients PN00 --detectors dinoeeg  # one patient -> custom_patients/
    python run_perturbations.py --recordings PN00-1 --variations disconnected_mild   # one file -> custom_patients/
    python run_perturbations.py --list
    python run_perturbations.py --detectors all --estimate   # plan + expected duration, runs nothing

Progress: after every detector run a line `[progress] jobs done/total | hours | elapsed | ETA` is printed. The ETA
uses each detector's measured seconds per recording-hour (a prior from 2026-11-14 timings until the first run of
that detector finishes), times the recording-hours still to do. Cached results count as done and cost nothing.

Each recording is perturbed once and every selected detector runs on the same perturbed signal. A detector is
compared with its own clean predictions (the evaluate_detectors.py cache, clean/<detector>/), on the same recordings.
Perturbed predictions are cached in data/cache/002-seizure-robustness/perturbations/<detector>/<condition>/.
A detector failing on a recording is logged to <results>/errors.log and that recording is left out of its numbers.

Outputs (results/perturbations/<cohort>_patients/, or results/perturbations/<--out>/), one row per detector x condition
(x recording), with a `detector` column. --out gives a run its own folder (run_batch.sh does this per batch), so its
tables hold only what it ran and never mix with older results; runs sharing an --out merge like cohort runs. With
--out the folder also gets predictions/<detector>/<condition>/<recording>.tsv: a copy of every prediction the tables
were scored from (computed now or reused from the cache), so the folder is complete on its own and can be copied
elsewhere; runs/<run>.json lists under computed_files the ones this run computed (new in the cache). Cohort runs merge into the existing files (rows of the same detector and condition are replaced), so models
run separately end up side by side; custom_patients/ is overwritten on every run.
    summary_by_variation.csv   recall, precision, f1, fa_per_24h, deltas vs clean
    metrics_by_recording.csv   per recording: hours, n_seizures, tp, fp, perturbed electrodes
    runs/<run>.json            what this run did: command, git commit, detectors, conditions, how many detector runs
                               were computed vs reused from the cache, failures, start/end, status
Every row has a `run` column (--run-label, default the start time, e.g. 2026-11-14_0736), so the rows a run wrote can
be found; rows written before 2026-11-14 are labelled "before_2026-11-14".
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
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
    CORE_VARIATIONS,
    apply_variation,
    get_variation,
    list_variations,
    variation_electrodes,
)

ELECTRODES = list(siena.CHANNELS_10_20.values())  # channel names of siena.load_raw()

CACHE_DIR = DATA_DIR / "cache" / "002-seizure-robustness" / "perturbations"  # <detector>/<condition>/<rec>.tsv
CLEAN_CACHE_DIR = DATA_DIR / "cache" / "002-seizure-robustness" / "clean"  # <detector>/<rec>.tsv (evaluate_detectors)
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "perturbations"

# Baseline cohort ("included patients"): the patients meeting the inclusion criteria (README / TESTS.md), temporal seizure
# localization (subject_info.csv) and at least 2 seizures. 11 patients, 111 h, 35 seizures. Set on 2026-11-08; it
# replaces an earlier list of 9 chosen from eventNet's first baseline (without PN03, PN06), which favoured eventNet.
# Used by run_extended_tests.py, audit_event_masking.py and preprocessing_survival.py too.
INCLUDED_PATIENTS = ["PN00", "PN01", "PN03", "PN05", "PN06", "PN09", "PN12", "PN13", "PN14", "PN16", "PN17"]

# Seconds of compute per hour of recording, per detector (incl. loading + perturbing), for the ETA before the first
# measurement. Measured 2026-11-14 on PN00-1 (44 min, Apple MPS): eventNet 9.5 s (8 s of it the SzCORE resampling),
# DINO-EEG 10 s; Gotman / SPaRCNet from the 2026-11-08 README check (4 s / 1 s). Replaced live by measured rates.
PRIOR_SEC_PER_HOUR = {"eventnet": 13.0, "dinoeeg": 14.0, "dinoeeg_norej": 14.0, "gotman": 6.0, "sparcnet": 2.0}


class Progress:
    """Counts detector runs still to do (weighted by recording hours) and prints an ETA."""

    def __init__(self, jobs: list[tuple[str, float]]):
        self.total_jobs, self.done_jobs = len(jobs), 0
        self.left_hours = {}
        for det, hours in jobs:
            self.left_hours[det] = self.left_hours.get(det, 0.0) + hours
        self.total_hours = sum(self.left_hours.values())
        self.spent = {d: 0.0 for d in self.left_hours}  # measured seconds
        self.measured_hours = {d: 0.0 for d in self.left_hours}
        self.t0 = time.time()

    def rate(self, det: str) -> float:
        if self.measured_hours.get(det, 0) > 0.5:  # enough to trust the measurement
            return self.spent[det] / self.measured_hours[det]
        return PRIOR_SEC_PER_HOUR.get(det, 15.0)

    def eta_seconds(self) -> float:
        return sum(h * self.rate(d) for d, h in self.left_hours.items())

    def summary(self) -> str:
        lines = [f"  {d:<14} {h:7.1f} h of recordings to run, ~{fmt_duration(h * self.rate(d))}"
                 for d, h in self.left_hours.items()]
        return "\n".join(lines + [f"  {'total':<14} {self.total_jobs} detector runs, "
                                   f"~{fmt_duration(self.eta_seconds())}"])

    def update(self, det: str, hours: float, seconds: float) -> None:
        self.done_jobs += 1
        self.left_hours[det] = max(0.0, self.left_hours[det] - hours)
        self.spent[det] += seconds
        self.measured_hours[det] += hours
        eta = self.eta_seconds()
        done_h = self.total_hours - sum(self.left_hours.values())
        finish = time.strftime("%a %H:%M", time.localtime(time.time() + eta))
        print(f"     [progress] {self.done_jobs}/{self.total_jobs} runs | {done_h:.0f}/{self.total_hours:.0f} h | "
              f"elapsed {fmt_duration(time.time() - self.t0)} | ETA {fmt_duration(eta)} (~{finish})", flush=True)


def fmt_duration(seconds: float) -> str:
    m = int(round(seconds / 60))
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


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
            if "run" not in old.columns:
                old["run"] = "before_2026-11-14"
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
        choices=["included", "all"],
        default="included",
        help="included: the 11 INCLUDED_PATIENTS (default); all: all 14 Siena patients. Results go to "
             "results/perturbations/<cohort>_patients/.",
    )
    parser.add_argument(
        "--variations",
        default="",
        help="Comma-separated list of variations to evaluate (default: CORE_VARIATIONS, the core conditions, every "
             "error type x mild/severe; the moderate levels, the 4 disconnected focus/contra variations and the "
             "extra variations can be named here).",
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
        "--out",
        default="",
        help="Results folder under results/perturbations/ (e.g. batches/2026-11-14_0800). Default: <cohort>_patients/.",
    )
    parser.add_argument(
        "--run-label",
        default=time.strftime("%Y-%m-%d_%H%M"),
        help="Label written in the `run` column of every row and as runs/<label>.json. Default: start time.",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print what would run (detector runs not cached yet) and the expected duration, then exit.",
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
        target_patients = INCLUDED_PATIENTS if args.cohort == "included" else siena.SUBJECTS
        results_dir = RESULTS_DIR / f"{args.cohort}_patients"
    if args.variations:
        target_variations = [v.strip() for v in args.variations.split(",") if v.strip()]
        for v in target_variations:
            get_variation(v)  # fail early on a typo
    else:
        target_variations = list(CORE_VARIATIONS)

    all_recordings = siena.list_recordings()
    recordings = [r for r in all_recordings if siena.patient_id(r) in target_patients]
    if args.recordings:
        wanted = {r.strip() for r in args.recordings.split(",") if r.strip()}
        recordings = [r for r in all_recordings if r.stem in wanted]
        missing = wanted - {r.stem for r in recordings}
        if missing:
            parser.error(f"recording(s) not found: {', '.join(sorted(missing))}")
        results_dir = RESULTS_DIR / "custom_patients"
    if args.out:
        results_dir = RESULTS_DIR / args.out
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

    # Plan: every (detector, recording) run not cached yet, clean baselines first
    jobs: list[tuple[str, float]] = []
    for edf in recordings:
        jobs += [(d, durations[edf.stem] / 3600) for d in detectors
                 if not (CLEAN_CACHE_DIR / d / f"{edf.stem}.tsv").is_file()]
    for var_name in target_variations:
        for edf in recordings:
            if variation_electrodes(var_name, edf.stem, ELECTRODES):
                jobs += [(d, durations[edf.stem] / 3600) for d in detectors
                         if args.force or not (CACHE_DIR / d / var_name / f"{edf.stem}.tsv").is_file()]
    progress = Progress(jobs)
    print("To run (cached results are reused):")
    print(progress.summary() if jobs else "  nothing, everything is cached", flush=True)
    print()
    if args.estimate:
        return

    run = args.run_label
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(PROJECT_ROOT / "src"), "."],
                                capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip())
    manifest = {
        "run": run, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None,
        "command": "python " + " ".join(sys.argv), "git_commit": commit + ("+uncommitted changes" if dirty else ""),
        "detectors": detectors, "conditions": target_variations, "recordings": [r.stem for r in recordings],
        "expected_duration": fmt_duration(progress.eta_seconds()),
        "computed": {}, "reused_from_cache": {}, "failed": {},  # per detector: number of (recording, condition) runs
        "computed_files": [],  # cache paths (relative to the cache root) this run wrote
    }
    manifest_path = results_dir / "runs" / f"{run}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def write_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def count(kind: str, detector: str) -> None:
        manifest[kind][detector] = manifest[kind].get(detector, 0) + 1

    def keep(src: Path, detector: str, condition: str) -> None:
        """With --out, copy a prediction the tables use into <results>/predictions/ (folder complete on its own)."""
        if args.out:
            dst = results_dir / "predictions" / detector / condition / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    write_manifest()
    print(f"Run label: {run}  (rows get run={run}; record: {manifest_path.relative_to(Path(__file__).parent)})\n")

    loaded: dict = {}

    def predict(detector: str, raw, dest: Path, label: str) -> None:
        """Run a detector on raw and cache its events at dest; failures are logged and leave no cache file."""
        t0 = time.time()
        hours = raw.n_times / raw.info["sfreq"] / 3600
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
            count("failed", detector)
            progress.update(detector, hours, time.time() - t0)
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(dest, sep="\t", index=False)
        print(f"     {label} [{detector}] {len(events)} event(s) in {time.time() - t0:.0f}s", flush=True)
        count("computed", detector)
        manifest["computed_files"].append(str(dest.relative_to(CACHE_DIR.parent)))
        progress.update(detector, hours, time.time() - t0)

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
            "run": run,
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
        for d in set(detectors) - set(pending):
            count("reused_from_cache", d)
        if pending:
            raw = siena.load_raw(edf, attach_annotations=False)
            for d in pending:
                predict(d, raw, dests[d], f"{edf.stem} clean")
            del raw
            gc.collect()
        for d in detectors:
            if dests[d].is_file():
                clean_rows[d].append(score_row(d, "clean", edf, read_events(dests[d])))
                keep(dests[d], d, "clean")

    for d in detectors:
        m = aggregate(clean_rows[d])
        summary_records.append({"detector": d, "condition": "clean", "description": "Clean unperturbed baseline",
                                **m, "delta_recall": 0.0, "delta_f1": 0.0, "delta_fa_24h": 0.0, "run": run})
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
            for d in set(detectors) - set(pending):
                count("reused_from_cache", d)
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
                    keep(dests[d], d, var_name)

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
                "run": run,
            }
            summary_records.append(rec)
            recording_rows.extend(rows)
            note = f"  [{len(applied)}/{len(recordings)} recordings]" if len(applied) < len(recordings) else ""
            print(f"     [{d:<8}] Recall: {m['recall']:.1%} (Δ {rec['delta_recall']:+.1%}) | "
                  f"F1: {m['f1']:.3f} (Δ {rec['delta_f1']:+.3f}) | "
                  f"FA/24h: {m['fa_per_24h']:.1f} (Δ {rec['delta_fa_24h']:+.1f}){note}", flush=True)
        print(f"     [{time.time() - t0:.1f}s]\n", flush=True)
        save()
        write_manifest()

    # 3. Final summary
    save()
    manifest.update(status="finished", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_manifest()
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
