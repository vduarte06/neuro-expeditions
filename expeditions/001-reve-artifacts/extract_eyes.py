"""Follow-up: eyes open vs eyes closed. Embed only the resting-baseline windows, twice:
with all 59 channels, and with posterior channels only (no frontal/eye signal, so the model
has to rely on what the occipital cortex does, e.g. alpha).

    poetry run python expeditions/001-reve-artifacts/extract_eyes.py --subjects 1-30

Writes one .npz per run to data/cache/001-reve-artifacts/eyes/.
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

CACHE = DATA_DIR / "cache" / "001-reve-artifacts" / "eyes"
WINDOW_S, STRIDE_S = 2.0, 1.0
CLASSES = ["clean_eyes_open", "clean_eyes_closed"]
POSTERIOR = ["P3", "P4", "P5", "P6", "P7", "P8", "Pz", "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "Oz", "O1", "O2"]


def parse_range(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def extract_run(encoder: reve.ReveEncoder, subject: int, run: int) -> None:
    dest = CACHE / f"sub{subject}_run{run:02d}.npz"
    if dest.exists():
        return
    t0 = time.time()
    raw = reve.preprocess(pm.load_raw(subject, run))
    ch_names = encoder.supported(raw.ch_names)
    raw.pick(ch_names)

    windows, starts = make_windows(raw.get_data(), raw.info["sfreq"], WINDOW_S, STRIDE_S)
    labels = label_windows(starts, WINDOW_S, pm.load_annotations(subject, run))
    keep = np.isin(labels, CLASSES)
    windows, starts, labels = windows[keep], starts[keep], labels[keep]

    dest.parent.mkdir(parents=True, exist_ok=True)
    if not keep.any():
        np.savez_compressed(dest, starts=starts, labels=labels.astype(str), ch_names=np.array(ch_names))
        print(f"sub-{subject} run-{run:02d}: no baseline windows", flush=True)
        return

    post = [ch_names.index(c) for c in POSTERIOR]
    x = reve.to_model_units(windows)
    emb_all = encoder.embed(x, ch_names)
    emb_post = encoder.embed(x[:, post], POSTERIOR)

    np.savez_compressed(
        dest,
        starts=starts,
        labels=labels.astype(str),
        ch_names=np.array(ch_names),
        reve_all_mean=emb_all["mean"].astype(np.float32),
        reve_all_attn=emb_all["attn"].astype(np.float32),
        reve_post_mean=emb_post["mean"].astype(np.float32),
        reve_post_attn=emb_post["attn"].astype(np.float32),
        bandpower=log_bandpower(windows, raw.info["sfreq"]).astype(np.float32),
    )
    print(f"sub-{subject} run-{run:02d}: {keep.sum()} baseline windows in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-30")
    ap.add_argument("--runs", default="1-6")
    args = ap.parse_args()
    encoder = reve.ReveEncoder()
    for s in parse_range(args.subjects):
        for r in parse_range(args.runs):
            if (pm.ROOT / pm._run_files(s, r)["eeg"]).exists():
                extract_run(encoder, s, r)


if __name__ == "__main__":
    main()
