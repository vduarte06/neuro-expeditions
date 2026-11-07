"""002 extended tests (see robustness_tests_literature.md): the acquisition errors of test 2, seizure side (T3),
seizure-timed errors (T5), and the single-electrode map (T2), for all four detectors.

Runs on seizure-centred windows (each seizure ±15 min, overlapping windows merged) instead of whole recordings,
which is ~5x faster and enough for recall / latency questions. False alarms are counted in those windows too, so
they are peri-ictal FA rates, comparable between conditions but not with the whole-recording numbers.

Suites:
    acquisition   the 15 acquisition-error conditions of run_perturbations.py (VARIATIONS_CATALOG), so every error
                  type is measured on every detector (test 2 itself runs eventNet only, on whole recordings).
    side          errors on the 3 temporal electrodes over the seizure focus (ipsi) vs the same 3 on the other
                  side (contra); lateralization from subject_info.csv, PN10 (frontal, bilateral) is skipped.
                  Flat electrode, high-impedance noise (60 µV RMS), 50% displacement towards the medial neighbour.
    timing        the same artefact applied over the whole window vs only from seizure onset -30 s to offset +30 s.
    electrode_map high-impedance noise at 0 dB (noise RMS = that electrode's own RMS) on one electrode at a time.

    python run_extended_tests.py --suite acquisition
    python run_extended_tests.py --suite side
    python run_extended_tests.py --suite timing --detectors eventnet,gotman
    python run_extended_tests.py --suite electrode_map --cohort all
    python run_extended_tests.py --list

--cohort good = the baseline cohort, patients meeting the inclusion criteria (temporal seizure localization, at least
2 seizures; see README), currently the working list GOOD_PATIENTS in run_perturbations.py. --cohort all = all 14.

Predictions are cached per condition x detector x window (data/cache/002-seizure-robustness/extended/), shared
between cohorts. Outputs (results/extended/<cohort>_patients/):
    <suite>_by_condition.csv   recall, precision, F1, FA/24h (peri-ictal), median onset latency per condition x detector
    <suite>_by_seizure.csv     one row per seizure x condition x detector: detected, onset latency
    <suite>_stats.csv          paired McNemar tests on seizure-level detection (side: ipsi vs contra; timing:
                               ictal-only vs whole; all suites: each condition vs clean)
"""

from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from timescoring import scoring
from timescoring.annotations import Annotation

from evaluate_detectors import DETECTORS, SZCORE, get_detector, to_mask
from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena
from neuro_expeditions.perturbations import (
    VARIATIONS_CATALOG,
    apply_variation,
    perturb_disconnected_electrode,
    perturb_electrode_displacement,
    perturb_high_impedance,
)
from run_perturbations import GOOD_PATIENTS

HERE = Path(__file__).parent
CACHE = DATA_DIR / "cache" / "002-seizure-robustness" / "extended"
RESULTS = HERE / "results" / "extended"
MARGIN_SEC = 15 * 60
ICTAL_PAD_SEC = 30
ELECTRODES = list(siena.CHANNELS_10_20.values())
TEMPORAL = {"L": ["F7", "T7", "P7"], "R": ["F8", "T8", "P8"]}
MEDIAL = {"F7": "F3", "T7": "C3", "P7": "P3", "F8": "F4", "T8": "C4", "P8": "P4"}


def lateralization() -> dict[str, str]:
    info = pd.read_csv(siena.ROOT / "subject_info.csv", skipinitialspace=True)
    return dict(zip(info.patient_id, info.lateralization))


# ---------------------------------------------------------------- conditions
# Each condition: f(raw_window, ctx) -> perturbed raw, or None when it does not apply to this patient.
# ctx: {"side": "L"/"R"/other, "seizures": [(on, off), ...] relative to the window start}


def _side_error(error: str, which: str):
    def f(raw, ctx):
        if ctx["side"] not in TEMPORAL:
            return None
        side = ctx["side"] if which == "ipsi" else {"L": "R", "R": "L"}[ctx["side"]]
        chans = TEMPORAL[side]
        if error == "flat":
            return perturb_disconnected_electrode(raw, mode="flat", channels=chans)
        if error == "impedance60uV":
            return perturb_high_impedance(raw, noise_rms_uV=60.0, channels=chans)
        out = raw
        for ch in chans:  # displace each lateral electrode halfway towards its medial neighbour
            out = perturb_electrode_displacement(out, mix_ratio=0.5, channel=ch, neighbor=MEDIAL[ch])
        return out

    return f


def _timed(variation: str, ictal_only: bool):
    def f(raw, ctx):
        out = apply_variation(raw, variation)
        if ictal_only:
            keep = np.zeros(raw.n_times, dtype=bool)
            for on, off in ctx["seizures"]:
                keep[raw.time_as_index(max(on - ICTAL_PAD_SEC, 0))[0] : raw.time_as_index(off + ICTAL_PAD_SEC)[0]] = True
            out._data[:, ~keep] = raw._data[:, ~keep]
        return out

    return f


def _electrode_noise(ch: str):
    def f(raw, ctx):
        i = raw.ch_names.index(ch)
        rms_uV = raw._data[i].std() * 1e6
        return perturb_high_impedance(raw, noise_rms_uV=rms_uV, channels=ch, seed=42 + i)

    return f


def _variation(name: str):
    return lambda raw, ctx: apply_variation(raw, name)


SUITES = {
    "acquisition": {name: _variation(name) for name in VARIATIONS_CATALOG},
    "side": {f"side_{w}_{e}": _side_error(e, w)
             for e in ("flat", "impedance60uV", "displace50pct") for w in ("ipsi", "contra")},
    "timing": {f"timing_{s}_{v}": _timed(v, s == "ictal")
               for v in ("muscle_temporal_100uV", "muscle_temporal_40uV", "perspiration_severe_200uV")
               for s in ("whole", "ictal")},
    "electrode_map": {f"electrode_noise0dB_{ch}": _electrode_noise(ch) for ch in ELECTRODES},
}  # fmt: skip


# ---------------------------------------------------------------- windows


def seizure_windows(edf: Path) -> list[dict]:
    """Seizure ±15 min windows, overlapping ones merged. Times in seconds from the recording start."""
    import mne

    info = mne.io.read_raw_edf(edf, preload=False, verbose="error")
    duration = info.n_times / info.info["sfreq"]
    windows = []
    for on, off in sorted(siena.load_seizures(edf)):
        start, end = max(on - MARGIN_SEC, 0.0), min(off + MARGIN_SEC, duration)
        if windows and start <= windows[-1]["end"]:
            windows[-1]["end"] = max(windows[-1]["end"], end)
            windows[-1]["seizures"].append((on, off))
        else:
            windows.append({"start": start, "end": end, "seizures": [(on, off)]})
    for k, w in enumerate(windows):
        w["id"] = f"{edf.stem}_w{k}"
        w["seizures_rel"] = [(on - w["start"], off - w["start"]) for on, off in w["seizures"]]
    return windows


# ---------------------------------------------------------------- run


def run(suite: str, detectors: list[str], recordings: list[Path], sides: dict[str, str], errors_log: Path) -> None:
    conditions = {"clean": lambda raw, ctx: raw, **SUITES[suite]}
    loaded = {}
    for edf in recordings:
        windows = seizure_windows(edf)
        todo = [(w, c, d) for w in windows for c in conditions for d in detectors
                if not (CACHE / c / d / f"{w['id']}.tsv").exists()]  # fmt: skip
        if not todo:
            continue
        raw = siena.load_raw(edf, attach_annotations=False)
        patient = siena.patient_id(edf)
        for w in windows:
            seg = raw.copy().crop(w["start"], min(w["end"], raw.times[-1]))
            ctx = {"side": sides.get(patient), "seizures": w["seizures_rel"]}
            for cond, perturb in conditions.items():
                pending = [d for d in detectors if not (CACHE / cond / d / f"{w['id']}.tsv").exists()]
                if not pending:
                    continue
                pert = perturb(seg, ctx)
                if pert is None:  # condition not applicable (e.g. side suite for a bilateral patient)
                    continue
                for d in pending:
                    t0 = time.time()
                    try:
                        loaded[d] = loaded.get(d) or get_detector(d)
                        events = loaded[d](pert)
                    except Exception:
                        with open(errors_log, "a") as f:
                            f.write(f"--- {d} {cond} {w['id']} {time.strftime('%Y-%m-%d %H:%M')}\n"
                                    f"{traceback.format_exc()}\n")  # fmt: skip
                        print(f"{w['id']} {cond} [{d}] FAILED (see {errors_log.name})", flush=True)
                        continue
                    dest = CACHE / cond / d / f"{w['id']}.tsv"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    events.to_csv(dest, sep="\t", index=False)
                    print(f"{w['id']} {cond} [{d}] {len(events)} event(s) in {time.time() - t0:.0f}s", flush=True)
                del pert
            del seg
            gc.collect()
        del raw
        gc.collect()


# ---------------------------------------------------------------- score


def _read_events(f: Path) -> list[tuple[float, float]]:
    try:
        df = pd.read_csv(f, sep="\t")
    except pd.errors.EmptyDataError:
        return []
    return list(zip(df.get("onset_sec", []), df.get("offset_sec", [])))


def score(suite: str, detectors: list[str], recordings: list[Path], sides: dict[str, str]) -> pd.DataFrame:
    """One row per seizure x condition x detector, plus per-window tp/fp (on the window's first seizure row)."""
    rows = []
    for edf in recordings:
        patient = siena.patient_id(edf)
        for w in seizure_windows(edf):
            duration = w["end"] - w["start"]
            ref = to_mask(w["seizures_rel"], duration)
            for cond in ["clean", *SUITES[suite]]:
                for d in detectors:
                    f = CACHE / cond / d / f"{w['id']}.tsv"
                    if not f.exists():
                        continue
                    hyp_events = _read_events(f)
                    ev = scoring.EventScoring(Annotation(ref, 1), Annotation(to_mask(hyp_events, duration), 1), SZCORE)
                    for k, (on, off) in enumerate(w["seizures_rel"]):
                        hits = [h_on for h_on, h_off in hyp_events
                                if h_on <= off + SZCORE.toleranceEnd and h_off >= on - SZCORE.toleranceStart]
                        rows.append({"detector": d, "condition": cond, "patient": patient, "side": sides.get(patient),
                                     "window": w["id"], "seizure": f"{w['id']}_sz{k}",
                                     "detected": bool(hits), "latency_sec": min(hits) - on if hits else np.nan,
                                     "window_hours": duration / 3600 if k == 0 else 0.0,
                                     "tp": int(ev.tp) if k == 0 else 0, "fp": int(ev.fp) if k == 0 else 0,
                                     "n_seizures": int(ev.refTrue) if k == 0 else 0})  # fmt: skip
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["detector", "condition"]).agg(
        seizures=("seizure", "count"), detected=("detected", "sum"), n_seizures=("n_seizures", "sum"),
        tp=("tp", "sum"), fp=("fp", "sum"), hours=("window_hours", "sum"),
        median_latency_sec=("latency_sec", "median"))  # fmt: skip
    g["recall"] = g.tp / g.n_seizures
    g["precision"] = g.tp / (g.tp + g.fp).replace(0, np.nan)
    g["f1"] = 2 * g.recall * g.precision / (g.recall + g.precision)
    g["fa_per_24h_periictal"] = g.fp / g.hours * 24
    return g.round(4).reset_index()


def mcnemar(df: pd.DataFrame, a: str, b: str, detector: str) -> dict:
    """Exact McNemar on seizure-level detection, condition a vs b, paired by seizure."""
    sub = df[(df.detector == detector) & df.condition.isin([a, b])]
    wide = sub.pivot_table(index="seizure", columns="condition", values="detected", aggfunc="first").dropna()
    if wide.empty or a not in wide or b not in wide:
        return {}
    only_a = int((wide[a].astype(bool) & ~wide[b].astype(bool)).sum())
    only_b = int((~wide[a].astype(bool) & wide[b].astype(bool)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"detector": detector, "a": a, "b": b, "paired_seizures": len(wide), "detected_a": int(wide[a].sum()),
            "detected_b": int(wide[b].sum()), "only_a": only_a, "only_b": only_b, "p_exact": round(p, 4)}  # fmt: skip


def stats(suite: str, df: pd.DataFrame, detectors: list[str]) -> pd.DataFrame:
    pairs = [(c, "clean") for c in SUITES[suite]]
    if suite == "side":
        pairs += [(c, c.replace("_ipsi_", "_contra_")) for c in SUITES[suite] if "_ipsi_" in c]
    if suite == "timing":
        pairs += [(c, c.replace("_ictal_", "_whole_")) for c in SUITES[suite] if "_ictal_" in c]
    return pd.DataFrame([r for d in detectors for a, b in pairs if (r := mcnemar(df, a, b, d))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES))
    ap.add_argument("--cohort", choices=["good", "all"], default="good")
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the conditions of every suite and exit")
    args = ap.parse_args()
    if args.list or not args.suite:
        for s, conds in SUITES.items():
            print(f"{s} ({len(conds)}): {', '.join(conds)}")
        return
    detectors = [d.strip() for d in args.detectors.split(",")]
    patients = GOOD_PATIENTS if args.cohort == "good" else siena.SUBJECTS
    recordings = [r for r in siena.list_recordings()
                  if siena.patient_id(r) in patients and siena.load_seizures(r)]  # fmt: skip
    sides = lateralization()
    if args.suite == "side":  # only patients with a left or right focus, so clean and perturbed cover the same seizures
        recordings = [r for r in recordings if sides.get(siena.patient_id(r)) in TEMPORAL]
    out = RESULTS / f"{args.cohort}_patients"
    out.mkdir(parents=True, exist_ok=True)
    print(f"suite {args.suite}: {len(SUITES[args.suite])} conditions + clean, {len(recordings)} recordings with "
          f"seizures, {len(patients)} patients, detectors {detectors}", flush=True)  # fmt: skip

    if not args.score_only:
        run(args.suite, detectors, recordings, sides, out / f"{args.suite}_errors.log")

    per_seizure = score(args.suite, detectors, recordings, sides)
    per_seizure.to_csv(out / f"{args.suite}_by_seizure.csv", index=False)
    summary = summarise(per_seizure)
    summary.to_csv(out / f"{args.suite}_by_condition.csv", index=False)
    stats(args.suite, per_seizure, detectors).to_csv(out / f"{args.suite}_stats.csv", index=False)
    cols = ["detector", "condition", "detected", "seizures", "recall", "f1", "fa_per_24h_periictal", "median_latency_sec"]
    print(summary[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
