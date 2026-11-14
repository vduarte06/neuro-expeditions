"""Primary analysis of the 002 perturbation results (included cohort), from cached predictions only. No detector runs.

Pre-specified primary outcomes:
  1. False alarms per hour, condition minus clean. Unit of resampling = patient (11). 95% percentile bootstrap CI
     of the pooled difference (sum FP / sum hours). A condition is "significant" when its CI excludes 0.
  2. Seizure-level recall: exact McNemar on whole-recording detection of each seizure, condition vs clean.
     Holm across conditions within detector.

Everything else in compile_results.py is exploratory.

A seizure counts as detected when a detection overlaps it within the SzCORE tolerances (-30 s / +60 s), the same rule
as run_extended_tests.py. False alarms are detections that overlap no seizure under the same rule.

    python primary_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from timescoring import scoring
from timescoring.annotations import Annotation

from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena
from run_perturbations import INCLUDED_PATIENTS

HERE = Path(__file__).parent
CACHE = DATA_DIR / "cache" / "002-seizure-robustness"
OUT = HERE / "results" / "primary"
DETECTORS = ["dinoeeg", "eventnet"]
TOL_START, TOL_END = 30.0, 60.0
N_BOOT = 10000
RNG = np.random.default_rng(42)
SZCORE = scoring.EventScoring.Parameters(
    toleranceStart=30, toleranceEnd=60, minOverlap=0, maxEventDuration=300, minDurationBetweenEvents=90
)


def read_events(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["onset_sec", "offset_sec"])
    return pd.read_csv(path, sep="\t")


def score_recording(events: pd.DataFrame, seizures: list[tuple[float, float]], duration: float) -> dict:
    """Per-seizure detection (tolerance rule) and false alarms counted by the official SzCORE scorer.

    SzCORE caps events at 300 s, so one long false detection counts as several false-alarm events. Counting raw
    detections instead would undercount, which is why the false-alarm count comes from timescoring.
    """
    on = events.onset_sec.to_numpy(dtype=float) if len(events) else np.array([])
    off = events.offset_sec.to_numpy(dtype=float) if len(events) else np.array([])
    detected = []
    matched = np.zeros(len(on), dtype=bool)
    for s_on, s_off in seizures:
        hit = (on <= s_off + TOL_END) & (off >= s_on - TOL_START)
        detected.append(bool(hit.any()))
        matched |= hit
    ref = np.zeros(int(np.ceil(duration)), dtype=bool)
    hyp = np.zeros_like(ref)
    for s_on, s_off in seizures:
        ref[max(int(s_on), 0) : int(np.ceil(s_off))] = True
    for a, b in zip(on, off):
        hyp[max(int(a), 0) : int(np.ceil(b))] = True
    ev = scoring.EventScoring(Annotation(ref, 1), Annotation(hyp, 1), SZCORE)
    return {"detected": detected, "fp": int(ev.fp)}


def load_all() -> pd.DataFrame:
    recs = [r for r in siena.list_recordings() if siena.patient_id(r) in INCLUDED_PATIENTS]
    rows = []
    for det in DETECTORS:
        for edf in recs:
            seizures = siena.load_seizures(edf)
            duration = siena.load_raw(edf, attach_annotations=False).times[-1]
            hours = duration / 3600
            clean = score_recording(read_events(CACHE / "clean" / det / f"{edf.stem}.tsv"), seizures, duration)
            rows.append({"detector": det, "condition": "clean", "recording": edf.stem,
                         "patient": siena.patient_id(edf), "hours": hours, "seizures": seizures,
                         "fp": clean["fp"], "detected": clean["detected"]})
            for cond_dir in sorted((CACHE / "perturbations" / det).glob("*")):
                f = cond_dir / f"{edf.stem}.tsv"
                if not f.is_file():
                    continue
                s = score_recording(read_events(f), seizures, duration)
                rows.append({"detector": det, "condition": cond_dir.name, "recording": edf.stem,
                             "patient": siena.patient_id(edf), "hours": hours, "seizures": seizures,
                             "fp": s["fp"], "detected": s["detected"]})
    return pd.DataFrame(rows)


def holm(p: list[float]) -> list[float]:
    m = len(p)
    order = np.argsort(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def fa_bootstrap(df: pd.DataFrame, cond: str, det: str) -> dict:
    """Pooled FA/h difference (cond - clean) with a patient-level bootstrap CI."""
    clean = df[(df.detector == det) & (df.condition == "clean")].set_index("recording")
    cur = df[(df.detector == det) & (df.condition == cond)].set_index("recording")
    common = cur.index.intersection(clean.index)
    c, k = cur.loc[common], clean.loc[common]
    patients = sorted(c.patient.unique())

    def rate_diff(pts: list[str]) -> float:
        sel = c.patient.isin(pts)
        hours = c.hours[sel].sum()
        return (c.fp[sel].sum() - k.fp[sel].sum()) / hours * 24 if hours else np.nan

    point = rate_diff(patients)
    boots = []
    for _ in range(N_BOOT):
        drawn = RNG.choice(patients, size=len(patients), replace=True)
        boots.append(rate_diff(list(drawn)))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"detector": det, "condition": cond, "recordings": len(common),
            "dFA_per_24h": round(point, 2), "ci95_low": round(lo, 2), "ci95_high": round(hi, 2),
            "significant": bool(lo > 0 or hi < 0)}


def recall_mcnemar(df: pd.DataFrame, det: str) -> pd.DataFrame:
    """Seizure-level exact McNemar, condition vs clean, Holm across conditions."""
    rows = []
    clean = df[(df.detector == det) & (df.condition == "clean")].set_index("recording")
    for cond in [c for c in df[df.detector == det].condition.unique() if c != "clean"]:
        cur = df[(df.detector == det) & (df.condition == cond)].set_index("recording")
        common = cur.index.intersection(clean.index)
        a_clean, a_cond = [], []
        for rec in common:
            a_clean += clean.loc[rec].detected
            a_cond += cur.loc[rec].detected
        a_clean, a_cond = np.array(a_clean, bool), np.array(a_cond, bool)
        only_clean = int((a_clean & ~a_cond).sum())
        only_cond = int((~a_clean & a_cond).sum())
        n_disc = only_clean + only_cond
        p = binomtest(only_clean, n_disc, 0.5).pvalue if n_disc else 1.0
        rows.append({"detector": det, "condition": cond, "seizures": len(a_clean),
                     "detected_clean": int(a_clean.sum()), "detected_cond": int(a_cond.sum()),
                     "lost": only_clean, "gained": only_cond, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p.tolist())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    conds = [c for c in df.condition.unique() if c != "clean"]

    fa = pd.DataFrame([fa_bootstrap(df, c, d) for d in DETECTORS for c in conds])
    fa.to_csv(OUT / "primary_fa_per_hour_bootstrap.csv", index=False)

    rec = pd.concat([recall_mcnemar(df, d) for d in DETECTORS], ignore_index=True)
    rec.to_csv(OUT / "primary_recall_mcnemar.csv", index=False)

    pd.set_option("display.width", 250)
    print("=== Primary 1: false alarms per 24 h, condition minus clean (patient bootstrap, 95% CI) ===")
    print(fa.to_string(index=False))
    print("\n=== Primary 2: seizure-level recall, McNemar vs clean (Holm within detector) ===")
    print(rec.round(4).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
