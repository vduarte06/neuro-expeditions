"""Expedition 001: window PhysioMotion runs and cache frozen-REVE embeddings + band-power baseline.

    poetry run python expeditions/001-reve-artifacts/extract.py --subjects 1-6

Writes one .npz per run to data/cache/001-reve-artifacts/ with:
    starts (N,), labels (N,), reve_mean (N, 512), reve_attn (N, 512), bandpower (N, C*5)
Every window is kept (including "unlabeled"/"ambiguous") so predictions can be overlaid on full runs.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from neuro_expeditions import DATA_DIR
from neuro_expeditions.datasets import physiomotion as pm
from neuro_expeditions.features.bandpower import log_bandpower
from neuro_expeditions.models import reve
from neuro_expeditions.windows import label_windows, make_windows

CACHE = DATA_DIR / "cache" / "001-reve-artifacts"
WINDOW_S, STRIDE_S = 2.0, 1.0


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def extract_run(encoder: reve.ReveEncoder, subject: int, run: int, line_freq: float) -> None:
    dest = CACHE / f"sub{subject}_run{run:02d}.npz"
    if dest.exists():
        return
    t0 = time.time()
    raw = reve.preprocess(pm.load_raw(subject, run), line_freq=line_freq)
    ch_names = encoder.supported(raw.ch_names)
    raw.pick(ch_names)

    windows, starts = make_windows(raw.get_data(), raw.info["sfreq"], WINDOW_S, STRIDE_S)
    labels = label_windows(starts, WINDOW_S, pm.load_annotations(subject, run))
    emb = encoder.embed(reve.to_model_units(windows), ch_names)

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        starts=starts,
        labels=labels.astype(str),
        ch_names=np.array(ch_names),
        reve_mean=emb["mean"].astype(np.float32),
        reve_attn=emb["attn"].astype(np.float32),
        bandpower=log_bandpower(windows, raw.info["sfreq"]).astype(np.float32),
    )
    print(f"sub-{subject} run-{run:02d}: {len(windows)} windows in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-6", help="e.g. 1-6 or 1,3,5-8")
    ap.add_argument("--runs", default="1-6")
    ap.add_argument("--line-freq", type=float, default=50.0)
    ap.add_argument("--download", action="store_true", help="fetch missing files first")
    args = ap.parse_args()

    subjects, runs = parse_range(args.subjects), parse_range(args.runs)
    if args.download:
        pm.download(subjects, runs)
    encoder = reve.ReveEncoder()
    for s in subjects:
        for r in runs:
            extract_run(encoder, s, r, args.line_freq)


if __name__ == "__main__":
    main()
