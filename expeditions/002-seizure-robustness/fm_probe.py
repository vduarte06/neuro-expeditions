"""Side experiment: can frozen EEG foundation models detect seizures on Siena?

Foundation models are not seizure detectors, so each one gets a linear probe trained leave-one-patient-out on
Siena itself (every recording is scored by a probe that never saw that patient). This differs from the main
002 protocol (pretrained detectors, no training on Siena) and is reported separately.

Models: REVE (brain-bzh/reve-base) and CBraMod (ICLR 2025), both frozen.
Pipeline: 19-ch 10-20, model's own preprocessing, 10 s windows / 5 s stride -> mean-pooled embedding ->
logistic regression (LOPO) -> per-second seizure probability -> threshold -> SzCORE event scoring (timescoring,
official SzCORE parameters).

    poetry run python expeditions/002-seizure-robustness/fm_probe.py                # embed (cached) + evaluate
    poetry run python expeditions/002-seizure-robustness/fm_probe.py --stage evaluate

Outputs (expeditions/002-seizure-robustness/results/fm_probe/):
    metrics_by_patient.csv, metrics_overall.csv   event + window metrics per model
    events/<model>/<recording>.tsv                 predicted seizure events (onset, duration)
    scores/<model>/<recording>.npz                 per-second probability, threshold, reference mask
Embeddings are cached in data/cache/002-seizure-robustness/fm_probe/<model>/<recording>.npz.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from timescoring import scoring
from timescoring.annotations import Annotation

from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import siena
from neuro_expeditions.models import cbramod, reve
from neuro_expeditions.windows import make_windows

HERE = Path(__file__).parent
CACHE = DATA_DIR / "cache" / "002-seizure-robustness" / "fm_probe"
RESULTS = HERE / "results" / "fm_probe"
WINDOW_S, STRIDE_S = 10.0, 5.0
POS_MIN_OVERLAP = 0.5  # training: window is "seizure" if >= 50% inside a seizure, "background" if 0%
SZCORE = scoring.EventScoring.Parameters(
    toleranceStart=30, toleranceEnd=60, minOverlap=0, maxEventDuration=5 * 60, minDurationBetweenEvents=90
)
MODELS = ("reve", "cbramod")


# ----------------------------------------------------------------------------------------------- embedding


def seizure_overlap(starts: np.ndarray, seizures: list[tuple[float, float]]) -> np.ndarray:
    ends = starts + WINDOW_S
    cover = np.zeros(len(starts))
    for on, off in seizures:
        cover += np.clip(np.minimum(ends, off) - np.maximum(starts, on), 0, None)
    return np.clip(cover / WINDOW_S, 0, 1)


class Encoders:
    """Lazily loaded so `--models` can run one at a time."""

    def __init__(self):
        self._reve = self._cbramod = None

    def embed(self, model: str, raw) -> tuple[np.ndarray, np.ndarray]:
        if model == "reve":
            self._reve = self._reve or reve.ReveEncoder()
            pre = reve.preprocess(raw)
            windows, starts = make_windows(pre.get_data(), pre.info["sfreq"], WINDOW_S, STRIDE_S)
            emb = self._reve.embed(reve.to_model_units(windows), pre.ch_names)["mean"]
        else:
            self._cbramod = self._cbramod or cbramod.CBraModEncoder(len(raw.ch_names), int(WINDOW_S * cbramod.SFREQ))
            pre = cbramod.preprocess(raw)
            windows, starts = make_windows(pre.get_data(), pre.info["sfreq"], WINDOW_S, STRIDE_S)
            emb = self._cbramod.embed(cbramod.to_model_units(windows))["mean"]
        return emb, starts


def embed_all(models: list[str], recordings: list[Path]) -> None:
    encoders = Encoders()
    for edf in recordings:
        todo = [m for m in models if not (CACHE / m / f"{edf.stem}.npz").exists()]
        if not todo:
            continue
        t0 = time.time()
        raw = siena.load_raw(edf, attach_annotations=False)
        seizures = siena.load_seizures(edf)
        duration = raw.times[-1] + 1 / raw.info["sfreq"]
        for m in todo:
            emb, starts = encoders.embed(m, raw)
            dest = CACHE / m / f"{edf.stem}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dest,
                emb=emb.astype(np.float16),
                starts=starts,
                overlap=seizure_overlap(starts, seizures),
                seizures=np.array(seizures, dtype=float).reshape(-1, 2),
                duration=duration,
                patient=siena.patient_id(edf),
            )
        del raw
        gc.collect()
        print(f"{edf.name}: {duration / 3600:.1f} h, {len(seizures)} seizure(s), {'+'.join(todo)} in {time.time() - t0:.0f}s",
              flush=True)


# ---------------------------------------------------------------------------------------------- evaluation


def load_cache(model: str) -> dict[str, dict]:
    recs = {}
    for f in sorted((CACHE / model).glob("*.npz")):
        z = np.load(f)
        recs[f.stem] = {k: z[k] for k in z.files}
        recs[f.stem]["patient"] = str(z["patient"])
        recs[f.stem]["duration"] = float(z["duration"])
    return recs


def per_second(probs: np.ndarray, starts: np.ndarray, duration: float) -> np.ndarray:
    """Average the probabilities of all windows covering each second."""
    n = int(np.ceil(duration))
    total, count = np.zeros(n), np.zeros(n)
    for p, s in zip(probs, starts):
        a, b = int(s), min(int(s + WINDOW_S), n)
        total[a:b] += p
        count[a:b] += 1
    return np.divide(total, count, out=np.zeros(n), where=count > 0)


def reference_mask(seizures: np.ndarray, duration: float) -> np.ndarray:
    mask = np.zeros(int(np.ceil(duration)), dtype=bool)
    for on, off in seizures:
        mask[int(on) : int(np.ceil(off))] = True
    return mask


def mask_to_events(mask: np.ndarray) -> pd.DataFrame:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    on, off = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return pd.DataFrame({"onset": on.astype(float), "duration": (off - on).astype(float), "eventType": "sz"})


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 200)))
    return float(grid[np.argmax([f1_score(y, p >= t, zero_division=0) for t in grid])])


def evaluate(model: str) -> pd.DataFrame:
    recs = load_cache(model)
    patients = sorted({r["patient"] for r in recs.values()})
    rows = []
    for test_patient in patients:
        train = [r for r in recs.values() if r["patient"] != test_patient]
        X = np.concatenate([r["emb"] for r in train]).astype(np.float32)
        ov = np.concatenate([r["overlap"] for r in train])
        keep = (ov == 0) | (ov >= POS_MIN_OVERLAP)
        X, y = X[keep], (ov[keep] >= POS_MIN_OVERLAP).astype(int)

        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
        clf.fit(X, y)
        threshold = best_threshold(y, clf.predict_proba(X)[:, 1])  # chosen on training patients only

        for name, r in recs.items():
            if r["patient"] != test_patient:
                continue
            probs = clf.predict_proba(r["emb"].astype(np.float32))[:, 1]
            score = per_second(probs, r["starts"], r["duration"])
            ref = reference_mask(r["seizures"], r["duration"])
            hyp = score >= threshold
            ev = scoring.EventScoring(Annotation(ref, 1), Annotation(hyp, 1), SZCORE)

            (RESULTS / "events" / model).mkdir(parents=True, exist_ok=True)
            (RESULTS / "scores" / model).mkdir(parents=True, exist_ok=True)
            mask_to_events(hyp).to_csv(RESULTS / "events" / model / f"{name}.tsv", sep="\t", index=False)
            np.savez_compressed(RESULTS / "scores" / model / f"{name}.npz", score=score, ref=ref, threshold=threshold)

            win_y = (r["overlap"] >= POS_MIN_OVERLAP).astype(int)
            rows.append({
                "model": model, "patient": test_patient, "recording": name, "hours": r["duration"] / 3600,
                "n_seizures": int(ev.refTrue), "tp": int(ev.tp), "fp": int(ev.fp), "threshold": threshold,
                "window_auroc": roc_auc_score(win_y, probs) if 0 < win_y.sum() < len(win_y) else np.nan,
                "window_auprc": average_precision_score(win_y, probs) if win_y.sum() else np.nan,
            })
        print(f"[{model}] {test_patient} done", flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by).agg(recordings=("recording", "count"), hours=("hours", "sum"), n_seizures=("n_seizures", "sum"),
                           tp=("tp", "sum"), fp=("fp", "sum"), window_auroc=("window_auroc", "mean"),
                           window_auprc=("window_auprc", "mean"))
    g["sensitivity"] = g.tp / g.n_seizures
    g["precision"] = g.tp / (g.tp + g.fp).replace(0, np.nan)
    g["f1"] = 2 * g.sensitivity * g.precision / (g.sensitivity + g.precision)
    g["fp_per_24h"] = g.fp / g.hours * 24
    return g.round(4).reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--stage", choices=["all", "embed", "evaluate"], default="all")
    ap.add_argument("--patients", default="", help="comma-separated, e.g. PN00,PN05 (default: all downloaded)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",")]

    recordings = siena.list_recordings()
    if args.patients:
        recordings = [r for r in recordings if siena.patient_id(r) in args.patients.split(",")]
    print(f"{len(recordings)} complete recordings, {len({siena.patient_id(r) for r in recordings})} patients", flush=True)
    if args.stage in ("all", "embed"):
        embed_all(models, recordings)
    if args.stage in ("all", "evaluate"):
        RESULTS.mkdir(parents=True, exist_ok=True)
        per_rec = pd.concat([evaluate(m) for m in models], ignore_index=True)
        per_rec.to_csv(RESULTS / "metrics_by_recording.csv", index=False)
        summarise(per_rec, ["model", "patient"]).to_csv(RESULTS / "metrics_by_patient.csv", index=False)
        overall = summarise(per_rec, ["model"])
        overall.to_csv(RESULTS / "metrics_overall.csv", index=False)
        (RESULTS / "run_info.json").write_text(json.dumps({
            "recordings": [r.name for r in recordings], "window_s": WINDOW_S, "stride_s": STRIDE_S,
            "szcore_params": vars(SZCORE), "finished": time.strftime("%Y-%m-%d %H:%M"),
        }, indent=2))
        print(overall.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
