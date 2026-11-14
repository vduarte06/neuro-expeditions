"""Compile clean baseline + perturbation results into one set of tables.

Reads the outputs already written by run_perturbations.py (whole recordings) and
run_extended_tests.py --suite side (peri-ictal windows), recomputes the paired tests from the
per-recording / per-seizure rows, and writes results/compiled/<cohort>/ (or results/compiled/<batch name>/ with
--perturbations batches/<name>, a run_batch.sh folder):

  performance.csv      detector x condition: recall, precision, F1, FA/24h, deltas vs clean,
                       Wilcoxon on per-recording dTP / dFP, Holm within (detector, family)
  detector_contrast.csv  H2: per-recording change of eventNet vs change of DINO-EEG (Wilcoxon on the
                       difference of differences), Holm within family
  side_periictal.csv   test 3 peri-ictal metrics + McNemar per seizure, Holm within detector
  summary.md           the same tables, readable

No model is run.

    python compile_results.py
    python compile_results.py --perturbations batches/2026-11-14_0800_<name> --contrast dinoeeg_norej,eventnet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).parent
CORE = [f"{e}_{lvl}" for e in ["mains", "impedance", "disconnected", "muscle", "displacement"] for lvl in ["mild", "severe"]]
SIDE = [f"side_{s}_{e}" for e in ["flat", "impedance60uV", "displace50pct"] for s in ["ipsi", "contra"]]
DETECTOR_NAMES = {"eventnet": "eventNet", "dinoeeg": "DINO-EEG", "sparcnet": "SPaRCNet", "gotman": "Gotman"}


def holm(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    order = np.argsort(p.values)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p.values[idx]))
        adj[idx] = running
    return pd.Series(adj, index=p.index)


def paired_wilcoxon(diff: np.ndarray) -> float:
    if np.all(diff == 0):
        return 1.0
    return float(wilcoxon(diff, zero_method="wilcox").pvalue)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    return 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)


def aggregate(g: pd.DataFrame) -> pd.Series:
    tp, fp, n, h = g.tp.sum(), g.fp.sum(), g.n_seizures.sum(), g.hours.sum()
    rec = tp / n if n else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return pd.Series({"n_recordings": len(g), "seizures": n, "tp": tp, "fp": fp, "recall": rec,
                      "precision": prec, "f1": f1, "fa_per_24h": fp / h * 24})


def whole_recording(rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for det, d in rec.groupby("detector"):
        clean = d[d.condition == "clean"].set_index("recording")
        c = aggregate(clean.reset_index())
        for cond, g in d.groupby("condition"):
            a = aggregate(g)
            row = {"detector": det, "condition": cond, "family": family(cond), **a.to_dict()}
            if cond != "clean":
                g = g.set_index("recording").loc[clean.index]
                dtp, dfp = (g.tp - clean.tp).values, (g.fp - clean.fp).values
                row.update(delta_recall=a.recall - c.recall, delta_precision=a.precision - c.precision,
                           delta_f1=a.f1 - c.f1, delta_fa_24h=a.fa_per_24h - c.fa_per_24h,
                           total_dTP=int(dtp.sum()), total_dFP=int(dfp.sum()),
                           recordings_more_fp=int((dfp > 0).sum()), recordings_fewer_tp=int((dtp < 0).sum()),
                           p_TP=paired_wilcoxon(dtp), p_FP=paired_wilcoxon(dfp))
            rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["p_TP", "p_FP"]:
        out[col + "_holm"] = np.nan
        for _, idx in out[out.condition != "clean"].groupby(["detector", "family"]).groups.items():
            out.loc[idx, col + "_holm"] = holm(out.loc[idx, col]).values
    return sort(out)


def detector_contrast(rec: pd.DataFrame, pair: list[str] | None = None) -> pd.DataFrame:
    """H2: is the per-recording change different between the two detectors? (pair: needed with > 2 detectors)"""
    dets = sorted(pair or rec.detector.unique())
    if len(dets) != 2:
        print(f"detector contrast skipped: {len(dets)} detectors ({', '.join(dets)}); choose two with --contrast a,b")
        return pd.DataFrame()
    rec = rec[rec.detector.isin(dets)]
    a, b = dets
    piv = rec.pivot_table(index=["condition", "recording"], columns="detector", values=["tp", "fp"])
    rows = []
    for cond in rec.condition.unique():
        if cond == "clean":
            continue
        cur, base = piv.loc[cond], piv.loc["clean"].loc[piv.loc[cond].index]
        d = cur - base
        row = {"condition": cond, "family": family(cond), "n_recordings": len(d)}
        for m in ["tp", "fp"]:
            diff = (d[(m, a)] - d[(m, b)]).values
            row[f"total_d{m.upper()}_{a}"] = int(d[(m, a)].sum())
            row[f"total_d{m.upper()}_{b}"] = int(d[(m, b)].sum())
            row[f"p_{m.upper()}"] = paired_wilcoxon(diff)
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ["p_TP", "p_FP"]:
        out[col + "_holm"] = np.nan
        for _, idx in out.groupby("family").groups.items():
            out.loc[idx, col + "_holm"] = holm(out.loc[idx, col]).values
    return sort(out)


def side_periictal(cond_df: pd.DataFrame, seiz: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = ["patient", "window", "seizure"]
    for det, s in seiz.groupby("detector"):
        det_s = s.set_index(["condition"] + key).detected.astype(bool)
        clean = det_s.loc["clean"]
        pairs = [(c, "clean") for c in SIDE] + [(f"side_ipsi_{e}", f"side_contra_{e}") for e in ["flat", "impedance60uV", "displace50pct"]]
        for x, y in pairs:
            if x not in det_s.index.get_level_values(0) or y not in det_s.index.get_level_values(0):
                continue
            ax, by = det_s.loc[x], (clean if y == "clean" else det_s.loc[y])
            ax, by = ax.align(by, join="inner")
            oa, ob = int((ax & ~by).sum()), int((~ax & by).sum())
            rows.append({"detector": det, "a": x, "b": y, "paired_seizures": len(ax), "detected_a": int(ax.sum()),
                         "detected_b": int(by.sum()), "only_a": oa, "only_b": ob, "p": mcnemar_exact(oa, ob)})
    stats = pd.DataFrame(rows)
    stats["p_holm"] = stats.groupby("detector").p.transform(lambda p: holm(p.reset_index(drop=True)).values)
    return cond_df, stats


def family(cond: str) -> str:
    return "clean" if cond == "clean" else ("core" if cond in CORE else "side")


def sort(df: pd.DataFrame) -> pd.DataFrame:
    order = {c: i for i, c in enumerate(["clean"] + CORE + SIDE)}
    keys = ["condition"] if "detector" not in df else ["detector", "condition"]
    return df.sort_values(keys, key=lambda s: s.map(order) if s.name == "condition" else s).reset_index(drop=True)


def fmt_p(p: float) -> str:
    return "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write_md(perf: pd.DataFrame, contrast: pd.DataFrame, side_cond: pd.DataFrame, side_stats: pd.DataFrame,
             path: Path, cohort: str) -> None:
    out = [f"# Compiled results ({cohort})", "",
           f"Generated by `compile_results.py`. Whole recordings: {int(perf.n_recordings.max())} recordings, "
           f"{int(perf.seizures.max())} seizures, detectors: {', '.join(DETECTOR_NAMES.get(d, d) for d in perf.detector.unique())}. "
           f"p values: Wilcoxon signed-rank on per-recording change vs clean, Holm within detector x family "
           "({len(CORE)} core, {len(SIDE)} side). Side peri-ictal: exact McNemar per seizure, Holm within detector.", ""]
    for det, d in perf.groupby("detector", sort=False):
        t = pd.DataFrame({
            "condition": d.condition,
            "recall": [f"{r:.2f} ({int(tp)}/{int(n)})" for r, tp, n in zip(d.recall, d.tp, d.seizures)],
            "precision": d.precision.map("{:.2f}".format),
            "F1": d.f1.map("{:.3f}".format),
            "ΔF1": d.delta_f1.map(lambda v: "" if pd.isna(v) else f"{v:+.3f}"),
            "FA/24h": d.fa_per_24h.map("{:.1f}".format),
            "ΔTP": d.total_dTP.map(lambda v: "" if pd.isna(v) else f"{int(v):+d}"),
            "ΔFP": d.total_dFP.map(lambda v: "" if pd.isna(v) else f"{int(v):+d}"),
            "p TP (Holm)": d.p_TP_holm.map(fmt_p),
            "p FP (Holm)": d.p_FP_holm.map(fmt_p),
        })
        out += [f"## {DETECTOR_NAMES.get(det, det)} — whole recordings", "", md_table(t), ""]
    if not contrast.empty:
        t = contrast.copy()
        for c in t.columns:
            if c.startswith("p_"):
                t[c] = t[c].map(fmt_p)
        t = t[[c for c in t.columns if c not in ("family", "n_recordings", "p_TP", "p_FP")]]
        out += ["## Detector contrast (H2) — per-recording change, eventNet vs DINO-EEG", "",
                "Wilcoxon on (change in detector A − change in detector B) per recording, Holm within family.", "",
                md_table(t), ""]
    t = side_cond.copy()
    t = t[["detector", "condition", "detected", "seizures", "recall", "fp", "fa_per_24h_periictal", "median_latency_sec"]]
    t["recall"] = t.recall.map("{:.2f}".format)
    t["fa_per_24h_periictal"] = t.fa_per_24h_periictal.map("{:.1f}".format)
    t["median_latency_sec"] = t.median_latency_sec.map("{:.1f}".format)
    s = side_stats.copy()
    s["p"], s["p_holm"] = s.p.map(fmt_p), s.p_holm.map(fmt_p)
    out += ["## Seizure side (test 3) — peri-ictal windows", "", md_table(t), "", md_table(s), ""]
    path.write_text("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default="included_patients")
    ap.add_argument("--perturbations", default="",
                    help="whole-recording results folder under results/perturbations/ (default: --cohort), e.g. a "
                         "batch folder batches/<name>; the side tables still come from results/extended/<cohort>/")
    ap.add_argument("--contrast", default="", help="two detectors for the H2 contrast, e.g. dinoeeg_norej,eventnet "
                                                     "(default: the two detectors in the results, if exactly two)")
    args = ap.parse_args()
    res = HERE / "results"
    pert = args.perturbations or args.cohort
    rec = pd.read_csv(res / "perturbations" / pert / "metrics_by_recording.csv")
    side_cond = pd.read_csv(res / "extended" / args.cohort / "side_by_condition.csv")
    side_seiz = pd.read_csv(res / "extended" / args.cohort / "side_by_seizure.csv")

    perf = whole_recording(rec)
    contrast = detector_contrast(rec, [d.strip() for d in args.contrast.split(",")] if args.contrast else None)
    side_cond, side_stats = side_periictal(side_cond, side_seiz)

    name = Path(pert).name if args.perturbations else args.cohort
    out = res / "compiled" / name
    out.mkdir(parents=True, exist_ok=True)
    perf.round(4).to_csv(out / "performance.csv", index=False)
    if not contrast.empty:
        contrast.round(4).to_csv(out / "detector_contrast.csv", index=False)
    side_stats.round(4).to_csv(out / "side_periictal_stats.csv", index=False)
    write_md(perf, contrast, side_cond, side_stats, out / "summary.md", name)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
