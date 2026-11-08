#!/usr/bin/env python3
"""How much of each perturbation survives each detector's preprocessing (no models are run).

The pipeline is: referential EEG -> perturbation -> detector preprocessing -> model. This script
runs everything up to the model input, on the clean and the perturbed signal, and compares the two:

    psr_in_db    perturbation-to-signal ratio at the perturbed electrodes (referential, as injected), above 0.5 Hz:
                 10*log10( sum(perturbed - clean)^2 / sum(clean)^2 ), both 0.5 Hz high-passed, because the raw
                 referential power is dominated by slow drift that every detector removes
    psr_out_db   the same ratio on the model input, on the worst (most affected) model-input channel
    survival_db  psr_out_db - psr_in_db: ~0 dB = reaches the model at full strength; -20 dB = 1% of its power
                 survives (filtered out); > 0 dB = amplified relative to the signal (e.g. by the montage)
    n_affected   model-input channels where the perturbation is >= 1% of the signal power (psr > -20 dB)
    dino_new_bads  electrodes DINO-EEG's pyprep flags and interpolates in the perturbed signal but not in the clean one

Model inputs: eventNet = SzCORE (19 ch, common average, 256 Hz); Gotman = SzCORE -> 18 bipolar -> 60 Hz low-pass;
SPaRCNet = referential -> 16 bipolar, 200 Hz, 60 Hz notch, 0.5-40 Hz, clip +-500 uV; DINO-EEG = referential -> 200 Hz,
60 Hz notch, 0.1-70 Hz, pyprep bad-channel interpolation -> 20 TCP bipolar (before its STFT).

Each recording is perturbed as a whole (as in run_perturbations.py) and then analysed on a segment around its first
seizure (default 600 s, so it contains the seizure). DINO-EEG's pyprep runs on that segment, not on its usual 1 h chunk.

    python preprocessing_survival.py                          # first seizure recording of each good patient, all 19
    python preprocessing_survival.py --recordings PN00-1 --variations mains_severe,disconnected_focus_severe

Outputs (results/preprocessing_survival/<good_patients|custom>/):
    survival_by_recording.csv   one row per recording x condition x detector
    survival_summary.csv        medians over recordings per condition x detector
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_expeditions.datasets import siena
from neuro_expeditions.models import dinoeeg, gotman, sparcnet
from neuro_expeditions.perturbations import (
    FOCUS_VARIATIONS,
    VARIATIONS_CATALOG,
    apply_variation,
    get_variation,
    variation_electrodes,
)
from neuro_expeditions.szcore import to_szcore
from run_perturbations import GOOD_PATIENTS

RESULTS = Path(__file__).resolve().parent / "results" / "preprocessing_survival"
DETECTORS = ("eventnet", "gotman", "sparcnet", "dinoeeg")
FLOOR_DB = -120.0  # reported instead of -inf when nothing survives
AFFECTED_DB = -20.0


def model_inputs(raw, dino_runner) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Each detector's model input (channels x samples) from a referential mne.Raw, plus DINO's interpolated bads."""
    out = {}
    eeg = to_szcore(raw)
    out["eventnet"] = (eeg.data, [])
    bip = gotman.to_bipolar(raw)
    out["gotman"] = (gotman._lowpass_filter(bip.data, bip.fs), [])
    out["sparcnet"] = (sparcnet.preprocess(raw), [])
    signals, bads = dinoeeg.preprocess(raw, dino_runner)
    out["dinoeeg"] = (np.concatenate([signals[k] for k in signals], axis=0), bads)
    return out


def psr_db(diff: np.ndarray, ref: np.ndarray) -> float:
    """10*log10(power of diff / power of ref); FLOOR_DB when diff is zero."""
    p_diff, p_ref = float(np.sum(diff**2)), float(np.sum(ref**2))
    if p_diff <= 0 or p_ref <= 0:
        return FLOOR_DB
    return max(FLOOR_DB, 10 * np.log10(p_diff / p_ref))


def segment(edf: Path, length: float, duration: float) -> tuple[float, float]:
    seizures = siena.load_seizures(edf)
    centre = seizures[0][0] if seizures else length / 2
    start = min(max(0.0, centre - length / 2), max(0.0, duration - length))
    return start, min(duration, start + length)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recordings", default="", help="comma-separated, e.g. PN00-1,PN01-1 "
                    "(default: the first recording with a seizure of each good patient)")  # fmt: skip
    ap.add_argument("--variations", default="", help="comma-separated (default: the 15 grid + 4 focus conditions)")
    ap.add_argument("--segment-sec", type=float, default=600.0)
    args = ap.parse_args()

    all_recs = siena.list_recordings()
    if args.recordings:
        wanted = [r.strip() for r in args.recordings.split(",") if r.strip()]
        recordings = [r for r in all_recs if r.stem in wanted]
        out_dir = RESULTS / "custom"
    else:
        recordings = []
        for p in GOOD_PATIENTS:
            first = next((r for r in all_recs if siena.patient_id(r) == p and siena.load_seizures(r)), None)
            if first is not None:
                recordings.append(first)
        out_dir = RESULTS / "good_patients"
    variations = ([v.strip() for v in args.variations.split(",") if v.strip()] if args.variations
                  else list(VARIATIONS_CATALOG) + list(FOCUS_VARIATIONS))  # fmt: skip
    for v in variations:
        get_variation(v)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(recordings)} recordings x {len(variations)} conditions x {len(DETECTORS)} detectors "
          f"(segments of {args.segment_sec:.0f} s around the first seizure)", flush=True)

    runner = dinoeeg._runner()
    electrodes_all = list(siena.CHANNELS_10_20.values())
    rows = []
    for edf in recordings:
        t0 = time.time()
        raw = siena.load_raw(edf, attach_annotations=False)
        start, stop = segment(edf, args.segment_sec, raw.times[-1])
        clean_seg = raw.copy().crop(start, stop)
        clean_in = model_inputs(clean_seg, runner)
        hp = signal.butter(4, 0.5, btype="high", fs=raw.info["sfreq"], output="sos")
        for var in variations:
            electrodes = variation_electrodes(var, edf.stem, electrodes_all)
            if not electrodes:  # focus condition, patient without a left/right focus
                continue
            pert_seg = apply_variation(raw, var).crop(start, stop)
            idx = [clean_seg.ch_names.index(c) for c in electrodes]
            c_ref = signal.sosfiltfilt(hp, clean_seg.get_data()[idx], axis=-1)
            p_ref = signal.sosfiltfilt(hp, pert_seg.get_data()[idx], axis=-1)
            psr_in = psr_db(p_ref - c_ref, c_ref)
            pert_in = model_inputs(pert_seg, runner)
            for det in DETECTORS:
                (c, c_bads), (p, p_bads) = clean_in[det], pert_in[det]
                n = min(c.shape[1], p.shape[1])
                per_ch = [psr_db(p[i, :n] - c[i, :n], c[i, :n]) for i in range(c.shape[0])]
                psr_out = max(per_ch)
                rows.append({
                    "recording": edf.stem,
                    "patient": siena.patient_id(edf),
                    "condition": var,
                    "detector": det,
                    "electrodes": " ".join(electrodes),
                    "psr_in_db": round(psr_in, 2),
                    "psr_out_db": round(psr_out, 2),
                    "survival_db": round(psr_out - psr_in, 2),
                    "n_affected": int(sum(v > AFFECTED_DB for v in per_ch)),
                    "n_channels": len(per_ch),
                    "dino_new_bads": " ".join(sorted(set(p_bads) - set(c_bads))) if det == "dinoeeg" else "",
                })
        print(f"{edf.stem}: {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "survival_by_recording.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "survival_by_recording.csv", index=False)
    summary = (
        df.assign(dino_interpolated=df.dino_new_bads.astype(str).str.len() > 0)
        .groupby(["condition", "detector"], sort=False)
        .agg(n_recordings=("recording", "nunique"), psr_in_db=("psr_in_db", "median"),
             psr_out_db=("psr_out_db", "median"), survival_db=("survival_db", "median"),
             n_affected=("n_affected", "median"), n_channels=("n_channels", "first"),
             dino_interpolated=("dino_interpolated", "sum"))  # fmt: skip
        .reset_index()
    )
    summary.to_csv(out_dir / "survival_summary.csv", index=False)

    order = [v for v in variations if v in set(summary.condition)]
    for col, title in [("psr_out_db", "Median perturbation-to-signal ratio at the model input, worst channel (dB)"),
                       ("survival_db", "Median survival_db = psr_out - psr_in (0 = full strength, -20 = 1% of power)")]:
        table = summary.pivot(index="condition", columns="detector", values=col).reindex(order)[list(DETECTORS)]
        table.insert(0, "psr_in_db", summary.groupby("condition").psr_in_db.first().reindex(order))
        print(f"\n{title}:")
        print(table.round(1).to_string())
    dino = summary[summary.detector == "dinoeeg"].set_index("condition").reindex(order)
    print("\nDINO-EEG: recordings where pyprep interpolated a perturbed electrode:")
    print(dino[["dino_interpolated", "n_recordings"]].to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
