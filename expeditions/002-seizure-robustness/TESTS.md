# 002 · Tests: what is run, how, and how to read the results

This file documents every test of expedition 002: the question it answers, the exact command, which patients and
detectors it covers, where its results go and how to read them. The study design is in [README.md](README.md). The
literature behind tests 3–6 is in [robustness_tests_literature.md](robustness_tests_literature.md) (its test ids
T1–T11 are given in brackets below).

## Overview

| # | Test | Question | Script | Patients | Detectors | Results folder | Status (2026-11-07, 14:30) |
|---|---|---|---|---|---|---|---|
| 1 | Clean baseline | What can each detector detect on unmodified EEG? | `evaluate_detectors.py` | all 14 | all 4 | `results/clean/` | **done** (all 41 recordings) |
| 2a | Acquisition errors, good patients | How much do 19 simulated errors change eventNet where it works well? | `run_perturbations.py --cohort good` | 9 good | eventNet | `results/perturbations/good_patients/` | stopped: perturbation design under review (3/19 conditions cached) |
| 2b | Acquisition errors, all patients | Same, on everyone | `run_perturbations.py --cohort all` | all 14 | eventNet | `results/perturbations/all_patients/` | not started |
| 2c | Acquisition errors, all detectors | Do the detectors differ in how much the same 19 errors change them (H2)? | `run_extended_tests.py --suite acquisition` | 9 good | all 4 | `results/extended/good_patients/acquisition_*` | not started |
| 3 | Seizure side [T3] | Does an error over the seizure focus hurt more than the same error on the other side? | `run_extended_tests.py --suite side` | 9 good | all 4 | `results/extended/good_patients/side_*` | stopped: perturbation design under review (~2/3 of windows cached) |
| 4 | Seizure timing [T5] | Does an artefact during the seizure only cause misses that the same artefact spread over time does not? | `run_extended_tests.py --suite timing` | 9 good | all 4 | `results/extended/good_patients/timing_*` | not started |
| 5 | Single-electrode map [T2] | Which electrode, when noisy, does most damage, and does it differ between common-average and bipolar detectors? | `run_extended_tests.py --suite electrode_map` | 9 good | all 4 | `results/extended/good_patients/electrode_map_*` | not started |
| 6 | Event-masking audit [T1] | Does the standard SzCORE event score hide changes caused by the errors? | `audit_event_masking.py` | 9 good / all 14 | eventNet | `results/masking_audit/` | not started (needs test 2) |

**Good patients** are the patients who meet the study's inclusion criteria:

| Criterion | Why it is defensible for a baseline |
|---|---|
| **Temporal seizure localization** | 13 of the 14 patients are temporal, so this makes the cohort clinically homogeneous. Only PN10 (frontal, bilateral FBTC) is excluded by it |
| **At least 2 seizures** | A baseline needs more than one event per patient to see variation between patients |

The aim of the baseline: a perturbation test needs a clean reference in which the detector works, so that a seizure
lost after adding an error can be attributed to the error and not to the detector already failing on that patient.
The runs started on 2026-11-07 use the working list `GOOD_PATIENTS` in `run_perturbations.py`:
PN00, PN01, PN05, PN09, PN12, PN13, PN14, PN16, PN17: about 76 h of EEG with 28 seizures, on which eventNet detected
24/28 seizures in the first baseline (README log). The other 5 patients (PN03, PN06, PN07, PN10, PN11) are included
in "all". Because this list was chosen after seeing results, before the criteria above were written down, every
perturbation test is also run, or can be run, on all patients, and detector outputs are cached per recording: after a revision,
re-running a script only computes the recordings of newly added patients and re-scores the rest.

## Common setup

| | |
|---|---|
| Data | Siena Scalp EEG: 14 patients, 41 recordings, 141 h, 47 seizures. 19 electrodes of the 10-20 system, referential, 512 Hz |
| Detectors | **eventNet** (U-Net, common-average reference), **DINO-EEG** (DETR detector, bipolar montage, built-in bad-channel interpolation), **SPaRCNet** (DenseNet, bipolar, 10 s windows), **Gotman 1982** (classical, rule-based). All used as released, never trained on Siena |
| Where errors are added | On individual electrodes of the referential signal, **before** each detector re-references it. One bad electrode therefore leaks into every common-average channel (eventNet) but only into the 2–4 bipolar channels that use it (DINO-EEG, SPaRCNet) |
| Scoring | SzCORE event scoring (`timescoring`): a detection counts if it overlaps the seizure from 30 s before onset to 60 s after offset; detections longer than 5 min are split; detections < 90 s apart are merged |
| Caching | Every detector output is cached per detector × condition × recording under `data/cache/002-seizure-robustness/`, so re-running a script only computes what is missing |

## How to read the results

| Metric | Meaning | Good value |
|---|---|---|
| `recall` / `sensitivity` | Seizures detected ÷ seizures annotated | high |
| `precision` | Detections that are real seizures ÷ all detections | high |
| `f1` | Harmonic mean of recall and precision; the SzCORE headline number | high |
| `fp_per_24h` / `fa_per_24h` | False alarms per 24 h of EEG | low; clinically, a few per day at most |
| `delta_*` | Perturbed minus clean, same recordings. Negative `delta_recall` = seizures lost; positive `delta_fa_24h` = alarms added | 0 = robust |
| `latency_sec` | First detection onset minus annotated seizure onset (negative = detected before the annotated onset, within the 30 s tolerance) | small |

Read the numbers with these caveats:
- **Small counts per patient.** Most patients have 2–5 seizures, so one detection moves a patient's recall by 0.2–0.5.
  Pooled numbers are more reliable than per-patient ones.
- **Precision is undefined (`NaN`, shown as "–") when a detector raises no alarm at all** for a patient: 0 of 0.
- **Recall and false alarms are the two separate failure modes.** An error can make a detector miss seizures
  (recall drops) or flood it with alarms (false alarms rise). F1 mixes both, so look at recall and false alarms first.
- **SPaRCNet and Gotman already raise ~80–140 false alarms per day on clean data**, so added false alarms barely show
  for them. Their robustness is best judged on recall.
- **Event scoring is tolerant.** Errors can move a detection by tens of seconds without changing any event metric;
  test 6 measures how often that happens.

## Test 1 · Clean baseline

**Question.** What does each detector detect on unmodified Siena EEG, overall and per patient? This is the reference
every other test compares against (H2 needs it for all detectors).

**What is done.** Each of the 4 detectors runs on all 41 recordings; detections are scored against the annotations.

```bash
python evaluate_detectors.py                          # all detectors, all patients
python evaluate_detectors.py --detectors gotman,eventnet --patients PN00
python evaluate_detectors.py --score-only             # re-score cached predictions
```

**Results** (`results/clean/`)

| File | Content |
|---|---|
| `metrics_overall.csv` | one row per detector: recordings, hours, n_seizures, tp, fp, sensitivity, precision, f1, fp_per_24h |
| `metrics_by_patient.csv` | same, per detector × patient |
| `metrics_by_recording.csv` | per detector × recording: hours, n_seizures, tp, fp |
| `run_info.json` | recordings, detectors, scoring parameters, finish time |
| `errors.log` | only present if a detector failed on a recording |
| `run.log`, `run_dinoeeg.log` | console output of the run (DINO-EEG was run separately because it is the slowest) |

## Test 2 · Acquisition errors (eventNet)

**Question.** H1: do technical errors reduce performance, and does the drop depend on error type and intensity?

**What is done.** Each of the 19 errors below is added to every recording, eventNet runs on the perturbed signal, and
the result is compared with the clean result on the same recordings (paired design). 2a uses the 9 good patients,
where eventNet works, so lost seizures are visible. 2b repeats it on all 14 patients, which shows whether the effect
depends on how well eventNet handles a patient to begin with. 2b reuses 2a's cached outputs.

```bash
python run_perturbations.py --cohort good    # 2a
python run_perturbations.py --cohort all     # 2b
python run_perturbations.py --list           # the 19 conditions
python run_perturbations.py --variations mains_global_150uV,disconnected_flat_F3
```

| Error | Conditions (name: what is added) |
|---|---|
| Mains interference | `mains_single_25uV`, `mains_single_100uV`: 50 Hz sinusoid on F3 · `mains_global_50uV`, `mains_global_150uV`: on all electrodes |
| High impedance | `impedance_single_20uV`, `impedance_single_60uV`: broadband noise (RMS) on F3 · `impedance_multi_30uV`: on F3, C3, P3 |
| Disconnected electrode | `disconnected_flat_F3`, `disconnected_flat_F3_T7`: flat line · `disconnected_drift_F3`: large slow drift · `disconnected_saturation_F3`: floating lead picking up clipped mains |
| Muscle artefact | `muscle_temporal_40uV`, `muscle_temporal_100uV`: 30–100 Hz bursts (6/min, 2.5 s) on F7, T7, F8, T8 |
| Electrode displacement | `displacement_mild_25pct`, `displacement_moderate_50pct`: F3 mixed with C3 |
| Perspiration | `perspiration_mild_75uV`, `perspiration_severe_200uV`: < 0.3 Hz slow waves on all electrodes |
| Gain error | `gain_attenuation_0.5x`, `gain_amplification_2.0x`: F3 scaled |

**Results** (`results/perturbations/<good|all>_patients/`)

| File | Content |
|---|---|
| `summary_by_variation.csv` | one row per condition (first row = clean): recall, precision, f1, fa_per_24h, tp, fp, and `delta_recall`, `delta_f1`, `delta_fa_24h` vs clean |
| `metrics_by_recording.csv` | per condition × recording: patient, hours, n_seizures, tp, fp (for paired statistics) |
| `run.log` | console output |

## Test 2c · Acquisition errors on all detectors

**Question.** H2: do the detector families differ in robustness? Test 2 only runs eventNet, so on its own it cannot
compare detectors. Test 2c applies the same 19 conditions (same catalog, same seed) to all 4 detectors, so every error
type is measured on every detector. Mains interference, gain errors and the drift / saturation disconnections are
covered by no other test for DINO-EEG, SPaRCNet and Gotman.

**What is done.** Like tests 3–5 it runs on the seizure windows, not whole recordings (see below), so its numbers are
comparable between detectors and conditions, but not directly with test 2. eventNet appears in both, which shows how
much the window design itself changes the result.

```bash
python run_extended_tests.py --suite acquisition
python run_extended_tests.py --suite acquisition --cohort all
```

Results: `results/extended/<good|all>_patients/acquisition_*`, same files and columns as tests 3–5.

## Test 3 · Seizure side [T3]

**Question.** Is an error worse when it sits over the seizure focus? This uses each patient's seizure lateralization
(`data/siena/subject_info.csv`: all good patients have temporal seizures, 7 left and 2 right) and tests whether the
detectors rely on focal or diffuse evidence. Not done in the literature found so far.

**What is done.** The same error is put on the 3 temporal electrodes over the focus (**ipsi**: F7, T7, P7 for left;
F8, T8, P8 for right) or on the same 3 electrodes of the other hemisphere (**contra**). Every seizure is thus seen
clean, with an ipsilateral error and with a contralateral error.

| Condition | Error on the 3 electrodes |
|---|---|
| `side_ipsi_flat`, `side_contra_flat` | flat line (disconnected) |
| `side_ipsi_impedance60uV`, `side_contra_impedance60uV` | 60 µV RMS broadband noise |
| `side_ipsi_displace50pct`, `side_contra_displace50pct` | each electrode mixed 50% with its medial neighbour (F7→F3, T7→C3, P7→P3, F8→F4, T8→C4, P8→P4) |

## Test 4 · Seizure timing [T5]

**Question.** Patients often move or chew during a seizure. Does an artefact that occurs only during the seizure make
detectors miss it, compared with the same artefact spread over the whole recording? This separates "the error hides
the seizure" from "the error creates false alarms".

**What is done.** Each artefact is applied either to the whole window (**whole**) or only from 30 s before seizure
onset to 30 s after offset (**ictal**). The artefact samples inside the seizure are identical in both versions; only
the background differs.

| Condition | Artefact |
|---|---|
| `timing_whole_muscle_temporal_100uV`, `timing_ictal_muscle_temporal_100uV` | strong muscle bursts on F7, T7, F8, T8 |
| `timing_whole_muscle_temporal_40uV`, `timing_ictal_muscle_temporal_40uV` | mild muscle bursts |
| `timing_whole_perspiration_severe_200uV`, `timing_ictal_perspiration_severe_200uV` | 200 µV slow waves on all electrodes |

## Test 5 · Single-electrode map [T2]

**Question.** The central claim of the study: one bad electrode leaks into the derived channels differently depending
on the montage. Which electrodes matter most, and is the pattern different for eventNet (common average: the error
reaches all 19 channels at 1/19 strength) than for DINO-EEG and SPaRCNet (bipolar: 2–4 channels at full strength)?

**What is done.** 19 conditions, `electrode_noise0dB_<electrode>`: broadband noise on that electrode only, with the same
RMS as the electrode's own signal (signal-to-noise ratio 0 dB). The per-detector results can be drawn as scalp maps.

## Tests 2c–5 · How they are run and read

```bash
python run_extended_tests.py --suite acquisition     # test 2c
python run_extended_tests.py --suite side            # test 3
python run_extended_tests.py --suite timing          # test 4
python run_extended_tests.py --suite electrode_map   # test 5
python run_extended_tests.py --suite side --cohort all --detectors eventnet,gotman
python run_extended_tests.py --list
```

- **Seizure windows, not whole recordings.** To keep the cost low, each test runs on windows of ±15 min around every
  seizure (overlapping windows merged), about 15 h of EEG for the good patients, plus a clean run on the same windows
  as the paired reference. Recall and latency are fully comparable with test 2. False alarms are counted only inside
  these windows (`fa_per_24h_periictal`): compare them between conditions, not with test 1 or 2.
- **All 4 detectors** run on every condition, so each test also answers H2 (do the detector families differ).
- **Paired statistics.** For each seizure we know whether it was detected under each condition. `*_stats.csv` gives an
  exact McNemar test on these pairs: `only_a` = seizures detected under condition a but not b, `only_b` the reverse,
  `p_exact` = two-sided p-value. Pairs tested: every condition vs `clean`; for test 3 also ipsi vs contra; for test 4
  also ictal vs whole. With 28 seizures, only fairly large effects reach significance; no multiple-comparison
  correction is applied in the file.

**Results** (`results/extended/<good|all>_patients/`, `<suite>` = `acquisition`, `side`, `timing` or `electrode_map`)

| File | Content |
|---|---|
| `<suite>_by_condition.csv` | per detector × condition (including `clean`): `detected` / `seizures` (seizure-level), recall, precision, f1 (SzCORE event scoring on the windows), `fa_per_24h_periictal`, `median_latency_sec` |
| `<suite>_by_seizure.csv` | one row per detector × condition × seizure: patient, side, window, `detected`, `latency_sec` |
| `<suite>_stats.csv` | McNemar tests described above |
| `<suite>_run.log`, `<suite>_errors.log` | console output; errors only if a detector failed |

`detected` counts a seizure as found if any detection overlaps it within the SzCORE tolerance. `recall` uses the full
SzCORE event scoring (with merging and splitting), so the two can differ by a seizure in rare cases.

## Test 6 · Event-masking audit [T1]

**Question.** In a first check on PN00, errors moved eventNet's detections (e.g. 1207–1282 s became 1196–1335 s) but
every event metric stayed the same, because of the −30/+60 s tolerance. How often does the standard score hide a
change, and what do stricter or finer metrics show? No detector is run: the test re-scores the cached outputs of
test 2.

```bash
python audit_event_masking.py --cohort good
python audit_event_masking.py --cohort all
```

**Results** (`results/masking_audit/<good|all>_patients/`)

| Column in `masking_by_condition.csv` | Meaning |
|---|---|
| `event_f1_tol30_60` | standard SzCORE event F1 (same as test 2) |
| `event_f1_tol15_30`, `_tol5_10`, `_tol0_0` | event F1 with stricter tolerances (seconds before onset / after offset) |
| `sample_f1` | SzCORE sample scoring: each second of EEG scored as seizure / not seizure |
| `mean_onset_latency_sec` | mean latency of detected seizures |
| `events_lost_vs_clean`, `events_gained_vs_clean` | detections that disappear / appear compared with clean, same recording |
| `mean_boundary_iou_vs_clean` | overlap (intersection over union) between each clean detection and its perturbed counterpart; 1 = unchanged |
| `mean_onset_shift_sec_vs_clean` | how far detection onsets moved |
| `recordings_changed_vs_clean` | recordings where any detection changed |
| `delta_event_f1_szcore`, `delta_sample_f1` | change vs clean |
| `masked` | **True** when the standard event F1 is unchanged although detections changed |

The last line of `run.log` gives the headline: how many of the 19 conditions are masked. `masking_by_recording.csv`
has the per-recording counts behind every column.

## Results folder layout

```
results/
├── clean/                         test 1
├── perturbations/
│   ├── good_patients/             test 2a
│   └── all_patients/              test 2b
├── extended/
│   └── good_patients/             tests 2c, 3, 4, 5 (acquisition_*, side_*, timing_*, electrode_map_*)
└── masking_audit/
    ├── good_patients/             test 6
    └── all_patients/              test 6
```
`results/` is not committed (`.gitignore`). Detector outputs are cached in `data/cache/002-seizure-robustness/`
(`clean/`, `perturbations/<condition>/`, `extended/<condition>/<detector>/`); papers are in
`data/papers/002-seizure-robustness/`. Neither is committed.

## Environment notes

- Python **3.13** Poetry environment (several dependencies, including `eventnet` and `pyprep`, are pinned to
  Python < 3.14). Activate with `eval $(poetry env activate)` (Poetry 2 has no `poetry shell`).
- **DINO-EEG on long recordings:** the official code feeds a channel's whole recording to the model at once, which
  needs ~30 GB for the 13.5 h PN01-1. The adapter (`src/neuro_expeditions/models/dinoeeg.py`) runs recordings longer
  than 1 h in 1 h chunks with 5 min overlap and merges the detections (4.4 GB, ~1 min per 2.7 h). Recordings up to
  1 h are processed exactly as in the official code.
