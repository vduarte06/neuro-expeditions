"""002 test T1 (robustness_tests_literature.md): does SzCORE event scoring hide the effect of acquisition errors?

Re-scores the cached eventNet predictions of run_perturbations.py (no detector is run). For each condition:
    - SzCORE event scoring (as in run_perturbations.py) and SzCORE sample scoring (1 s resolution);
    - event F1 under a tolerance sweep (onset/offset tolerance 0/0, 5/10, 15/30, 30/60 s = SzCORE);
    - onset latency of detected seizures (first detection onset - annotated onset);
    - change vs clean on the same recording: detections lost / gained, boundary IoU and onset shift of the detections
      that persist (each clean detection matched to the perturbed detection it overlaps most).
"Masked" = event F1 identical to clean while detections moved or appeared/disappeared.

    python audit_event_masking.py                # good-patient subset
    python audit_event_masking.py --cohort all

Outputs (results/masking_audit/<cohort>_patients/): masking_by_condition.csv, masking_by_recording.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from timescoring import scoring
from timescoring.annotations import Annotation

from evaluate_detectors import SZCORE, _duration, to_mask
from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena
from run_perturbations import GOOD_PATIENTS

HERE = Path(__file__).parent
CACHE = DATA_DIR / "cache" / "002-seizure-robustness"
RESULTS = HERE / "results" / "masking_audit"
TOLERANCES = [(0, 0), (5, 10), (15, 30), (30, 60)]


def read_events(f: Path) -> list[tuple[float, float]]:
    try:
        df = pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        return []
    return sorted(zip(df.get("onset_sec", []), df.get("offset_sec", [])))


def overlap(a, b) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def compare_to_clean(clean: list, pert: list) -> dict:
    lost = sum(not any(overlap(c, p) > 0 for p in pert) for c in clean)
    gained = sum(not any(overlap(p, c) > 0 for c in clean) for p in pert)
    ious, shifts = [], []
    for c in clean:
        best = max(pert, key=lambda p: overlap(c, p), default=None)
        if best is not None and overlap(c, best) > 0:
            ov = overlap(c, best)
            ious.append(ov / (max(c[1], best[1]) - min(c[0], best[0])))
            shifts.append(abs(best[0] - c[0]))
    return {"n_clean_events": len(clean), "n_pert_events": len(pert), "lost": lost, "gained": gained,
            "iou_sum": sum(ious), "n_matched": len(ious), "shift_sum": sum(shifts),
            "changed": int(lost > 0 or gained > 0 or any(i < 0.999 for i in ious))}  # fmt: skip


def score_recording(ref_intervals, hyp: list, duration: float) -> dict:
    ref, mask = to_mask(ref_intervals, duration), to_mask(hyp, duration)
    out = {}
    for ts, te in TOLERANCES:
        p = scoring.EventScoring.Parameters(toleranceStart=ts, toleranceEnd=te, minOverlap=0,
                                            maxEventDuration=SZCORE.maxEventDuration,
                                            minDurationBetweenEvents=SZCORE.minDurationBetweenEvents)  # fmt: skip
        ev = scoring.EventScoring(Annotation(ref, 1), Annotation(mask, 1), p)
        out[f"tp_tol{ts}_{te}"], out[f"fp_tol{ts}_{te}"] = int(ev.tp), int(ev.fp)
        out["n_seizures"] = int(ev.refTrue)
    ss = scoring.SampleScoring(Annotation(ref, 1), Annotation(mask, 1))
    out.update(sample_tp=int(ss.tp), sample_fp=int(ss.fp), sample_ref=int(ss.refTrue))
    lat = [min(h[0] for h in hyp if overlap((on - SZCORE.toleranceStart, off + SZCORE.toleranceEnd), h) > 0) - on
           for on, off in ref_intervals
           if any(overlap((on - SZCORE.toleranceStart, off + SZCORE.toleranceEnd), h) > 0 for h in hyp)]  # fmt: skip
    out.update(latency_sum=sum(lat), n_latency=len(lat))
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("condition").sum(numeric_only=True)
    out = pd.DataFrame(index=g.index)
    for ts, te in TOLERANCES:
        tp, fp = g[f"tp_tol{ts}_{te}"], g[f"fp_tol{ts}_{te}"]
        rec, prec = tp / g.n_seizures, tp / (tp + fp).replace(0, np.nan)
        out[f"event_f1_tol{ts}_{te}"] = 2 * rec * prec / (rec + prec)
    out["event_recall_szcore"] = g["tp_tol30_60"] / g.n_seizures
    out["fa_per_24h_szcore"] = g["fp_tol30_60"] / g.hours * 24
    s_rec, s_prec = g.sample_tp / g.sample_ref, g.sample_tp / (g.sample_tp + g.sample_fp).replace(0, np.nan)
    out["sample_f1"] = 2 * s_rec * s_prec / (s_rec + s_prec)
    out["mean_onset_latency_sec"] = g.latency_sum / g.n_latency.replace(0, np.nan)
    out["events_lost_vs_clean"] = g.lost
    out["events_gained_vs_clean"] = g.gained
    out["mean_boundary_iou_vs_clean"] = g.iou_sum / g.n_matched.replace(0, np.nan)
    out["mean_onset_shift_sec_vs_clean"] = g.shift_sum / g.n_matched.replace(0, np.nan)
    out["recordings_changed_vs_clean"] = g.changed
    out = out.round(4)
    clean = out.loc["clean"]
    out["delta_event_f1_szcore"] = (out.event_f1_tol30_60 - clean.event_f1_tol30_60).round(4)
    out["delta_sample_f1"] = (out.sample_f1 - clean.sample_f1).round(4)
    out["masked"] = (out.delta_event_f1_szcore.abs() < 1e-9) & (out.recordings_changed_vs_clean > 0)
    out.loc["clean", "masked"] = False
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["good", "all"], default="good")
    args = ap.parse_args()
    patients = GOOD_PATIENTS if args.cohort == "good" else siena.SUBJECTS
    recordings = [r for r in siena.list_recordings() if siena.patient_id(r) in patients]
    conditions = ["clean"] + sorted(p.name for p in (CACHE / "perturbations").iterdir() if p.is_dir())
    folder = lambda c: CACHE / "clean" / "eventnet" if c == "clean" else CACHE / "perturbations" / c  # noqa: E731

    rows = []
    for edf in recordings:
        duration, ref = _duration(edf), siena.load_seizures(edf)
        clean_f = folder("clean") / f"{edf.stem}.tsv"
        if not clean_f.exists():
            continue
        clean = read_events(clean_f)
        for c in conditions:
            f = folder(c) / f"{edf.stem}.tsv"
            if not f.exists():
                continue
            hyp = read_events(f)
            rows.append({"condition": c, "patient": siena.patient_id(edf), "recording": edf.stem,
                         "hours": duration / 3600, **score_recording(ref, hyp, duration),
                         **compare_to_clean(clean, hyp)})  # fmt: skip
    per_rec = pd.DataFrame(rows)
    complete = per_rec.groupby("condition").recording.nunique() == per_rec[per_rec.condition == "clean"].recording.nunique()
    per_rec = per_rec[per_rec.condition.isin(complete[complete].index)]  # only conditions run on every recording
    out = RESULTS / f"{args.cohort}_patients"
    out.mkdir(parents=True, exist_ok=True)
    per_rec.to_csv(out / "masking_by_recording.csv", index=False)
    summary = summarise(per_rec)
    summary.to_csv(out / "masking_by_condition.csv", index=False)
    cols = ["condition", "event_f1_tol30_60", "event_f1_tol0_0", "sample_f1", "events_lost_vs_clean",
            "events_gained_vs_clean", "mean_boundary_iou_vs_clean", "mean_onset_shift_sec_vs_clean", "masked"]
    print(summary[cols].to_string(index=False))
    print(f"\n{int(summary.masked.sum())}/{len(summary) - 1} conditions: event F1 unchanged but detections changed")


if __name__ == "__main__":
    main()
