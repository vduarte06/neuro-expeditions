#!/usr/bin/env python3
"""Run technical error perturbation experiments on eventNet across the Good Patient Subset.

Usage (assumes poetry shell inside expeditions/002-seizure-robustness):
    python run_perturbations.py
    python run_perturbations.py --variations mains_global_50uV,impedance_single_20uV
    python run_perturbations.py --list
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from timescoring import scoring
from timescoring.annotations import Annotation

# Setup path so neuro_expeditions is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena
from neuro_expeditions.models import eventnet
from neuro_expeditions.perturbations import VARIATIONS_CATALOG, apply_variation, list_variations

CACHE_DIR = DATA_DIR / "cache" / "002-seizure-robustness" / "perturbations"
CLEAN_CACHE_DIR = DATA_DIR / "cache" / "002-seizure-robustness" / "clean" / "eventnet"
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "perturbations"

# The 9 patients in the clean good subset (85.7% clean recall, 0.76 F1)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run perturbation experiments on eventNet for the Good Patient Subset."
    )
    parser.add_argument(
        "--variations",
        default="",
        help="Comma-separated list of variations to evaluate (default: all 19 in catalog).",
    )
    parser.add_argument(
        "--patients",
        default=",".join(GOOD_PATIENTS),
        help=f"Comma-separated patient list (default: {','.join(GOOD_PATIENTS)}).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available variations and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute predictions even if cached.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available Technical Error Variations:")
        for v in list_variations():
            print(f"  • {v['name']:<28} : {v['description']}")
        return

    target_patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    if args.variations:
        target_variations = [v.strip() for v in args.variations.split(",") if v.strip()]
    else:
        target_variations = list(VARIATIONS_CATALOG.keys())

    # Get recordings for target patients
    all_recordings = siena.list_recordings()
    recordings = [r for r in all_recordings if siena.patient_id(r) in target_patients]

    print("=" * 80)
    print(" eventNet Technical Error Robustness Benchmark")
    print("=" * 80)
    print(f" Cohort           : {len(target_patients)} patients ({len(recordings)} recordings)")
    print(f" Variations       : {len(target_variations)} conditions")
    print(f" Cache Directory  : {CACHE_DIR}")
    print(f" Results Directory: {RESULTS_DIR}")
    print("=" * 80, flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load eventNet model
    print("Loading eventNet pre-trained weights...", flush=True)
    model = eventnet.load_model()

    # Pre-cache durations and ground truth seizures
    durations = {}
    ground_truths = {}
    total_hours = 0.0
    for edf in recordings:
        dur = get_recording_duration(edf)
        durations[edf.stem] = dur
        ground_truths[edf.stem] = siena.load_seizures(edf)
        total_hours += dur / 3600.0

    total_seizures = sum(len(sz) for sz in ground_truths.values())
    print(f"Total dataset: {total_hours:.2f} hours, {total_seizures} true seizures across {len(recordings)} files.\n")

    # 2. Score Clean Baseline for reference
    clean_rows = []
    print("Evaluating Clean Baseline...", flush=True)
    for edf in recordings:
        pred_file = CLEAN_CACHE_DIR / f"{edf.stem}.tsv"
        if not pred_file.is_file() or args.force:
            raw = siena.load_raw(edf, attach_annotations=False)
            df = eventnet.detect(raw, model=model)
            CLEAN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(pred_file, sep="\t", index=False)
            del raw
            gc.collect()
        else:
            try:
                df = pd.read_csv(pred_file, sep="\t")
            except pd.errors.EmptyDataError:
                df = pd.DataFrame(columns=["onset_sec", "offset_sec"])

        dur = durations[edf.stem]
        ref_sz = ground_truths[edf.stem]
        n_sz, tp, fp = score_events(ref_sz, df, dur)
        clean_rows.append({
            "condition": "clean",
            "recording": edf.stem,
            "patient": siena.patient_id(edf),
            "hours": dur / 3600.0,
            "n_seizures": n_sz,
            "tp": tp,
            "fp": fp,
        })

    clean_summary_tp = sum(r["tp"] for r in clean_rows)
    clean_summary_fp = sum(r["fp"] for r in clean_rows)
    clean_recall = clean_summary_tp / total_seizures
    clean_prec = clean_summary_tp / max(1, clean_summary_tp + clean_summary_fp)
    clean_f1 = 2 * clean_recall * clean_prec / max(1e-6, clean_recall + clean_prec)
    clean_fa_24h = (clean_summary_fp / total_hours) * 24.0

    print(f"Clean Baseline: Recall={clean_recall:.1%} ({clean_summary_tp}/{total_seizures}), "
          f"Precision={clean_prec:.1%}, F1={clean_f1:.3f}, FA/24h={clean_fa_24h:.1f}\n")

    summary_records = [{
        "condition": "clean",
        "description": "Clean unperturbed baseline",
        "recall": clean_recall,
        "precision": clean_prec,
        "f1": clean_f1,
        "fa_per_24h": clean_fa_24h,
        "tp": clean_summary_tp,
        "fp": clean_summary_fp,
        "total_seizures": total_seizures,
        "delta_recall": 0.0,
        "delta_f1": 0.0,
        "delta_fa_24h": 0.0,
    }]

    all_recording_rows = list(clean_rows)

    # 3. Iterate through each perturbation variation
    for v_idx, var_name in enumerate(target_variations, start=1):
        t0 = time.time()
        var_desc = VARIATIONS_CATALOG[var_name]["desc"]
        print(f"[{v_idx}/{len(target_variations)}] Running: {var_name}")
        print(f"     Description: {var_desc}", flush=True)

        var_cache_dir = CACHE_DIR / var_name
        var_cache_dir.mkdir(parents=True, exist_ok=True)

        var_rows = []
        for edf in recordings:
            pred_file = var_cache_dir / f"{edf.stem}.tsv"
            if not pred_file.is_file() or args.force:
                raw = siena.load_raw(edf, attach_annotations=False)
                # Apply variation upstream in referential montage
                perturbed_raw = apply_variation(raw, var_name)
                # Run detector
                df = eventnet.detect(perturbed_raw, model=model)
                df.to_csv(pred_file, sep="\t", index=False)
                del raw
                del perturbed_raw
                gc.collect()
            else:
                try:
                    df = pd.read_csv(pred_file, sep="\t")
                except pd.errors.EmptyDataError:
                    df = pd.DataFrame(columns=["onset_sec", "offset_sec"])

            dur = durations[edf.stem]
            ref_sz = ground_truths[edf.stem]
            n_sz, tp, fp = score_events(ref_sz, df, dur)
            var_rows.append({
                "condition": var_name,
                "recording": edf.stem,
                "patient": siena.patient_id(edf),
                "hours": dur / 3600.0,
                "n_seizures": n_sz,
                "tp": tp,
                "fp": fp,
            })

        all_recording_rows.extend(var_rows)

        # Aggregate metrics
        tp_sum = sum(r["tp"] for r in var_rows)
        fp_sum = sum(r["fp"] for r in var_rows)
        recall = tp_sum / total_seizures
        prec = tp_sum / max(1, tp_sum + fp_sum)
        f1 = 2 * recall * prec / max(1e-6, recall + prec)
        fa_24h = (fp_sum / total_hours) * 24.0

        d_recall = recall - clean_recall
        d_f1 = f1 - clean_f1
        d_fa = fa_24h - clean_fa_24h

        summary_records.append({
            "condition": var_name,
            "description": var_desc,
            "recall": recall,
            "precision": prec,
            "f1": f1,
            "fa_per_24h": fa_24h,
            "tp": tp_sum,
            "fp": fp_sum,
            "total_seizures": total_seizures,
            "delta_recall": d_recall,
            "delta_f1": d_f1,
            "delta_fa_24h": d_fa,
        })

        elapsed = time.time() - t0
        print(f"     => Recall: {recall:.1%} (Δ {d_recall:+.1%}) | F1: {f1:.3f} (Δ {d_f1:+.3f}) | "
              f"FA/24h: {fa_24h:.1f} (Δ {d_fa:+.1f}) [{elapsed:.1f}s]\n", flush=True)

        # Periodically save intermediate summary CSV
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(RESULTS_DIR / "summary_by_variation.csv", index=False)

    # 4. Final summary and reporting
    summary_df = pd.DataFrame(summary_records).round(4)
    summary_df.to_csv(RESULTS_DIR / "summary_by_variation.csv", index=False)

    rec_df = pd.DataFrame(all_recording_rows).round(4)
    rec_df.to_csv(RESULTS_DIR / "metrics_by_recording.csv", index=False)

    print("=" * 80)
    print(" ALL PERTURBATIONS COMPLETE — Summary Benchmark Table")
    print("=" * 80)
    display_cols = ["condition", "recall", "precision", "f1", "fa_per_24h", "delta_recall", "delta_f1", "delta_fa_24h"]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)
    print(f"Results saved to: {RESULTS_DIR / 'summary_by_variation.csv'}\n")


if __name__ == "__main__":
    main()
