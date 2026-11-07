# 002 · Robustness of seizure detectors to EEG acquisition errors

*Robustez de algoritmos de deteção automática de crises epiléticas a erros técnicos de aquisição do EEG*

**Status:** clean baseline done (3 detectors × 41 Siena recordings). Perturbations not implemented yet.

**Background.** Automatic seizure detectors lose a lot of performance outside the data they were developed on.
In the largest independent challenge so far, the best algorithm reached an event F1 of 32%, and the authors name
artefacts in real clinical data as one of the causes. How much of that comes specifically from technical
acquisition errors (the part that depends on the EEG technician's work) is not known.

**Question.** What is the impact of technical acquisition errors on the performance of pre-trained seizure
detectors, and does that impact differ between a classical method and a deep learning method?

**Hypotheses.**
- **H1:** technical errors reduce performance, and the drop depends on error type and intensity.
- **H2:** the deep learning detector and the classical detector differ in robustness.

**Study type.** Experimental, quantitative, secondary public data, paired design: each recording is its own
control (clean vs perturbed).

## Setup
| | |
|---|---|
| Data | Siena Scalp EEG Database (PhysioNet): 14 adult patients, 41 recordings, 141 h, 47 seizures. Referential 10-20, 19 channels used. Seizure times via the official SzCORE parser (`epilepsy2bids`) |
| Detectors | **eventNet** (deep learning, U-Net, trained on TUSZ only) · **SPaRCNet** (deep learning, DenseNet, MGH/BWH ICU data) · **Gotman 1982** (classical, rule-based, no training) |
| Training | None. All used as released; Siena is in none of their training sets |
| Metrics | SzCORE event scoring (`timescoring`): tolerance −30 s / +60 s, max event 5 min, events < 90 s apart merged. Recall (sensitivity), precision, F1, false alarms per 24 h |

Literature behind the choices (datasets and model alternatives): [literature_investigation.md](literature_investigation.md).

## Intervention: simulated acquisition errors
Errors are applied **per electrode, before re-referencing**. The detectors re-reference internally (common
average or bipolar), so one bad electrode leaks into several derived channels. This is a central point of the
study and should be shown explicitly (clean vs perturbed signal on one electrode and on the derived channels).

| Error | Simulation | Levels |
|---|---|---|
| Mains interference | 50 Hz sinusoid | 3 amplitudes |
| High impedance | broadband noise | 3 SNRs |
| Disconnected electrode | flat signal or drift | 1, 2 or 3 electrodes |
| Muscle artefact | high-frequency noise in bursts | 3 intensities |
| Electrode displacement | mix with neighbouring electrode | 3 proportions |

5 error types × 3 levels = 15 conditions + clean baseline, per recording, per detector.

## Open items
- **Primary outcome:** the design text was cut off at "Variáveis de resultado". Pick the primary metric (event F1?).
- **Statistical analysis:** paired per-recording differences vs clean (Wilcoxon, or a mixed model with patient as
  random effect) for H1, detector × condition interaction for H2, correction across 15 conditions.
- **Detector preprocessing:** the eventNet and SPaRCNet adapters are not yet checked line by line against the
  official code (re-referencing, filtering).
- **Level calibration:** concrete amplitudes, SNRs, burst parameters and mixing proportions, anchored to clinical
  ranges and each recording's own signal power.
- **Disconnected electrode:** which electrodes (random, fixed, all combinations); flat vs drift.
- **Displacement:** define "neighbour" on the 10-20 layout.
- **Random seeds:** fixed seed per recording × condition, possibly several repeats.

## Run
*Assumes an active `poetry shell` inside `expeditions/002-seizure-robustness`.*

```bash
# Run all detectors on all recordings and score them (cached; re-runs only process new files)
python evaluate_detectors.py
python evaluate_detectors.py --score-only   # re-score cached predictions

# Inspect an exam: ground truth in red, detector outputs overlaid
python plot_exam.py                 # pick from a numbered list
python plot_exam.py PN00-1 --bipolar
python plot_exam.py PN00-1 --save overview.png
```
Results: `results/clean/` (`metrics_overall.csv`, `metrics_by_patient.csv`, `metrics_by_recording.csv`).

On hold: `fm_probe.py` (REVE / CBraMod embeddings + leave-one-patient-out probe). It trains on Siena, so it is
outside this study's protocol.

## Log

### 2026-11-07 · clean baseline
| Detector | Recall | Precision | F1 | False alarms / 24 h |
|---|---|---|---|---|
| eventNet | 0.57 (27/47) | 0.41 | **0.48** | 6.6 |
| SPaRCNet | 0.72 (34/47) | 0.04 | 0.07 | 157 |
| Gotman | 0.26 (12/47) | 0.03 | 0.05 | 70 |

- eventNet is the only detector usable as released. Its F1 is in line with the best SzCORE challenge results.
- SPaRCNet and Gotman fail on precision: 96–97% of their alarms are false.
- eventNet is very patient-dependent. On 9 patients (PN00, PN01, PN05, PN09, PN12, PN13, PN14, PN16, PN17) it finds
  24/28 seizures (recall 0.86, precision 0.69, F1 0.76, 3.5 FA/24 h). On the other 5 (PN03, PN06, PN07, PN10, PN11)
  it finds 3/19 (F1 0.12). PN10 (1/10) and PN06 (1/5) are the hardest. The subset was picked after seeing results,
  but it is where perturbation effects on recall will be most visible.
- Bug fixed before the run: the eventNet and SPaRCNet adapters passed volts, but both models expect µV. In volts,
  eventNet never crossed its threshold and SPaRCNet output a constant probability.
