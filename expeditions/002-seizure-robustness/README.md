# 002 · Robustness of seizure detectors to EEG acquisition errors

*Robustez de algoritmos de deteção automática de crises epiléticas a erros técnicos de aquisição do EEG*

**Status:** clean baseline and 19 perturbation conditions being recomputed with the official eventNet code
(see log 2026-11-07). DINO-EEG set up, not yet run on all recordings.

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

**Inclusion criteria (baseline cohort, "good patients").**
- **Temporal seizure localization:** 13 of the 14 patients are temporal, so this makes the cohort clinically
  homogeneous. Only PN10 (frontal, bilateral FBTC) is excluded by it.
- **At least 2 seizures:** a baseline needs more than one event per patient to see variation between patients.

The runs started on 2026-11-07 use an earlier working list (`GOOD_PATIENTS` in `run_perturbations.py`, chosen from
eventNet's first baseline before these criteria were written down). Every perturbation test is also run on all 14
patients. Details per test: [TESTS.md](TESTS.md).

## Setup
| | |
|---|---|
| Data | Siena Scalp EEG Database (PhysioNet): 14 adult patients, 41 recordings, 141 h, 47 seizures. Referential 10-20, 19 channels used. Seizure times via the official SzCORE parser (`epilepsy2bids`) |
| Detectors | **eventNet** (deep learning, U-Net, trained on TUSZ only) · **SPaRCNet** (deep learning, DenseNet, MGH/BWH ICU data) · **DINO-EEG** (deep learning, DETR-style detector per channel, trained on TUSZ only) · **Gotman 1982** (classical, rule-based, no training) |
| Implementation | **eventNet:** official SzCORE package (`eventnet`, pinned commit) incl. its SzCORE preprocessing; only the batching loop is ours (the published one cannot run). **DINO-EEG:** official repo cloned, its own preprocessing/merging functions called; CUDA attention op swapped for an identical PyTorch version. **SPaRCNet:** model + weights official; preprocessing copied line by line from `runSPaRCNet.py`; window→event rule is ours (P(seizure) ≥ 0.5). **Gotman:** port of `esl-epfl/gotman_1982` (commit 371d4b0) with the same output: official input (SzCORE format → 18-ch double banana, µV, 256 Hz), thresholds, window loop, cross-channel rule and events (no minimum duration); only the cross-channel loop is rewritten for speed. Sample-identical to a transcription of the official code on 7 recordings (9.5 h) |
| Training | None. All used as released; Siena is in none of their training sets |
| Metrics | SzCORE event scoring (`timescoring`): tolerance −30 s / +60 s, max event 5 min, events < 90 s apart merged. Recall (sensitivity), precision, F1, false alarms per 24 h |

Literature behind the choices (datasets and model alternatives): [literature_investigation.md](literature_investigation.md).

## Pipeline
```
Siena EDF → 19 referential channels (512 Hz) → error injection → each detector's input format → its own
preprocessing → model → detections (.tsv) → SzCORE evaluation
```
Each detector gets the input format its official code expects. eventNet and Gotman: the SzCORE format (official
epilepsy2bids conversion: 256 Hz, common average, µV); eventNet was trained on it and the official Gotman baseline
reads it. SPaRCNet and DINO-EEG: the referential signal, as their official scripts read it. Their bipolar montages
cancel any common reference, so SzCORE conversion would only add a resampling step for SPaRCNet, and for DINO-EEG it
would hide flat electrodes from pyprep (see 2026-11-08). Errors are injected **before** any conversion, on the
electrodes, because that is where acquisition errors happen: a disconnected electrode is flat in the referential
signal, not in a derived channel, and a faulty electrode leaks into the common average.

Each detector's own preprocessing, as released: eventNet none; Gotman 18 bipolar + 60 Hz low-pass; SPaRCNet 16
bipolar, 200 Hz, 60 Hz notch, 0.5–40 Hz, clip ±500 µV; DINO-EEG 200 Hz, 60 Hz notch, 0.1–70 Hz, pyprep bad-channel
interpolation, TCP bipolar. **Note on the 60 Hz notch:** SPaRCNet and DINO-EEG were built for US data (60 Hz mains)
and are run with their released 60 Hz notch, although Siena (Italy) has 50 Hz mains. So the 50 Hz mains errors are
not notched out; SPaRCNet removes them anyway with its 40 Hz low-pass, DINO-EEG keeps them. The detectors are
treated as fixed tools, run as released: the study is about the technician's role in delivering a clean signal, so
what a released detector does not remove is the technician's to prevent. When reporting mains results, note that
DINO-EEG's vulnerability comes partly from a US default running on European data, not from its architecture alone.
How much of each error reaches each model is measured by `preprocessing_survival.py` (no models run).

## Intervention: simulated acquisition errors
Errors are applied **per electrode, before re-referencing**. The detectors re-reference internally (common
average or bipolar), so one bad electrode leaks into several derived channels. This is a central point of the
study and should be shown explicitly (clean vs perturbed signal on one electrode and on the derived channels).

| Error | Simulation | Electrodes | Mild | Moderate | Severe |
|---|---|---|---|---|---|
| Mains interference | 50 Hz sinusoid, peak amplitude, same phase | 3 random | 5 µV | 20 µV | 50 µV |
| High impedance | 0.5–70 Hz noise, SNR vs the electrode's own RMS | 2 random | 10 dB | 5 dB | 0 dB |
| Disconnected electrode | flat signal | random, nested | 1 | 3 | 5 |
| Muscle artefact | 20–100 Hz bursts, 10% of the time, synchronous | F7, F8, T3–T6 | 20 µV RMS | 50 µV RMS | 100 µV RMS |
| Electrode displacement | mix with neighbouring electrode | 2 random | α 0.25 | α 0.50 | α 0.75 |

5 error types × 3 levels = 15 conditions + clean baseline, per recording, per detector. Plus 4 targeted
disconnection conditions: flat mid-temporal electrode (mild) or temporal chain (severe) over the seizure focus, and
the same electrodes on the other side as control. Full definitions: [TESTS.md](TESTS.md) test 2 and
`src/neuro_expeditions/perturbations.py`.

## Open items
- **Primary outcome:** the design text was cut off at "Variáveis de resultado". Pick the primary metric (event F1?).
- **Statistical analysis:** paired per-recording differences vs clean (Wilcoxon, or a mixed model with patient as
  random effect) for H1, detector × condition interaction for H2, correction across 15 conditions.
- **SPaRCNet event rule:** the official code only outputs a probability per 10 s window. Our rule (any window
  ≥ 0.5) is likely why it raises ~140 false alarms/24 h. Needs a justified rule (smoothing, consecutive windows).
- **DINO-EEG defaults:** fixed 60 Hz notch (Siena mains is 50 Hz) and automatic bad-channel interpolation (pyprep),
  which may undo "disconnected electrode" perturbations. Decide: part of the detector, or switched off.
- **Gotman:** now exactly the official `esl-epfl/gotman_1982` (2026-11-08); the results below predate this and must
  be rerun.
- **Random seeds:** one fixed draw per recording × error type (decided 2026-11-07); several repeats still open.
- **Seizure focus:** Siena gives only the side (L/R), not the electrode; "focus" = the temporal chain on that side.

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

# Perturbations
python run_perturbations.py --list                                    # list all perturbations (grid, focus, extras)
python run_perturbations.py                                           # run the 19 default conditions with eventNet
python run_perturbations.py --detectors gotman,sparcnet               # same conditions, other detectors (or: all)
python run_perturbations.py --patients PN00 --variations disconnected_mild,disconnected_focus_severe
python plot_exam.py PN00-1 --perturb disconnected_focus_severe --compare   # perturbed signal over clean (grey)
python preprocessing_survival.py                                      # how much of each error reaches each model
```
Results: `results/clean/` (`metrics_overall.csv`, `metrics_by_patient.csv`, `metrics_by_recording.csv`).

On hold: `fm_probe.py` (REVE / CBraMod embeddings + leave-one-patient-out probe). It trains on Siena, so it is
outside this study's protocol.

## Log

### 2026-11-07 · clean baseline (first run)
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
- **Superseded:** the eventNet numbers above come from a rewritten adapter without the common-average reference;
  SPaRCNet's from an adapter without filtering/clipping. See 2026-11-07.

### 2026-11-07 · detector implementations corrected
- **SPaRCNet:** added the official 60 Hz notch, 0.5–40 Hz band-pass and ±500 µV clipping. Recall 0.72 → 0.79
  (37/47), but still 140 FA/24 h (F1 0.08), so the remaining problem is the window→event rule.
- **eventNet:** replaced the rewritten adapter with the official package. The rewrite skipped the SzCORE common-average
  reference and used its own windowing/peak rules. Clean baseline and all perturbations are being recomputed;
  old predictions kept in `data/cache/002-seizure-robustness/backup_rewritten_eventnet/`.
- **DINO-EEG:** added (official code). Detects the PN00-2 seizure (1217–1307 s vs 1220–1274 s) in 11 s on MPS.

### 2026-11-07 · perturbation catalog redefined
- **Grid:** the 19 ad-hoc conditions were replaced by the 5 × 3 grid of the study design (table above, names
  `<error>_mild|moderate|severe`). Decisions: mains amplitude = peak, same phase on the 3 electrodes (one source);
  impedance SNR measured against each electrode's own 0.5–70 Hz RMS (median over 10 s windows, so seizures and
  artefacts do not inflate it); muscle bursts of 2 s with the same timing on all 6 electrodes and the stated RMS on
  the burst's flat top.
- **Electrodes:** random per recording, seeded from the recording name and error type. The three levels share
  electrodes and noise (only intensity changes), the 1 / 3 / 5 disconnected electrodes are nested, and the electrodes
  used are written to `metrics_by_recording.csv`.
- **Disconnected over the focus:** random disconnection rarely hits the seizure side (temporal electrodes over the
  focus in 5/26, 9/26, 13/26 good-patient recordings at mild / moderate / severe), so it cannot show a location
  effect. Added `disconnected_focus_mild|severe` (T7/T8, or F7-T7-P7 / F8-T8-P8, on the seizure side from
  `subject_info.csv`) and `disconnected_contra_mild|severe` (same electrodes, other side). They report three
  numbers: typical cost of a failed electrode (random), worst case (focus), and the part due to location (focus vs
  contra). PN10 (bilateral) is excluded from them; their deltas are vs clean on the same recordings. Test 3
  (`run_extended_tests.py --suite side`) remains the multi-detector version of the same question.
- **Kept outside the grid** (`EXTRA_VARIATIONS`, not run by default): drift, saturation, perspiration, gain, and the
  old muscle definition used by test 4.
- **Perturbed signals are not stored:** regenerated in < 0.5 s per recording; only detector outputs are cached.
  `plot_exam.py --perturb <condition>` shows any condition on any exam, with the perturbed electrodes listed.
- Cached outputs of the old condition names are no longer used; the new conditions need a full run.
- **All detectors on whole recordings:** `run_perturbations.py --detectors <list|all>` runs the same conditions on
  Gotman, SPaRCNet and DINO-EEG too, each against its own clean baseline; results carry a `detector` column. Cache
  moved to `perturbations/<detector>/<condition>/` (eventNet's existing outputs moved, not recomputed). Checked on
  PN00-1 × `disconnected_focus_severe`: all 4 run (per 44 min recording: Gotman 4 s, SPaRCNet 1 s, DINO-EEG 16 s).

### 2026-11-08 · same SzCORE input for all detectors; error survival through preprocessing
- **Pipeline:** SPaRCNet and DINO-EEG built their montages straight from the referential signal; now, like eventNet and
  Gotman, they receive the SzCORE-format EEG (official conversion, after error injection; see Pipeline). Their bipolar
  montages cancel the common average, so their model input changes by only 0.2–0.3% RMS (extra 256 Hz resampling
  step; PN00-1). Clean baselines being recomputed; old outputs in `data/cache/002-seizure-robustness/backup_pre_szcore_input/`.
- **60 Hz notch kept** as released (option A, see Pipeline note).
- **Test 2s** (`preprocessing_survival.py`, 9 recordings, median `survival_db`; 0 = reaches the model at full strength):

  | Error | eventNet | Gotman | SPaRCNet | DINO-EEG |
  |---|---|---|---|---|
  | mains | −2.1 | +3.2 | **−24.1** (40 Hz low-pass removes it) | +4.0 to +4.7 |
  | impedance | −5.4 | +0.8 | +3.9 | +2.1 |
  | disconnected (random 1 / 3 / 5) | +0.4 / +3.0 / +2.5 | +2.5 / +5.7 / +5.7 | +3.0 / +5.3 / +5.3 | +2.9 / +6.4 / +6.4 |
  | muscle | +0.4 | +2.6 | +5.3 | +4.5 |
  | displacement | +1.7 | +2.0 | +1.4 to +1.8 | +2.4 |

  - Only SPaRCNet's preprocessing removes an error (mains). Everything else reaches every model at full strength or
    stronger: bipolar derivations cancel EEG shared by neighbouring electrodes but not the error, so relative to the
    signal the error grows by 2–6 dB (Gotman, SPaRCNet, DINO-EEG). The common average (eventNet) dilutes it slightly.
  - Survival does not depend on the level (filters and montages are linear), except where pyprep steps in.
  - DINO-EEG's pyprep interpolates the mains electrodes in 7/9 recordings at `mains_severe` (1/9 moderate) and an
    impedance electrode in 1/9. Interpolation replaces the electrode's EEG with its neighbours', so the input moves
    further from clean (+6.5 dB), not closer. It never catches a disconnected electrode (0/9 at every level): after
    the common average a flat electrode is no longer flat. So DINO-EEG does not "repair" disconnections.
  - The expected table (before measuring) was wrong for muscle and impedance on SPaRCNet: its 0.5–40 Hz band-pass
    removes part of the error, but just as much of the EEG, so relative to what the model sees they are not reduced.
- **Reverted the same day:** SzCORE input only for eventNet and Gotman (the detectors whose official code takes it);
  SPaRCNet and DINO-EEG back on the referential signal, as their official scripts. Reason: with SzCORE input, DINO-EEG's
  pyprep never caught a disconnected electrode (above), which is a side effect of our conversion, not of DINO-EEG.
  Gotman is now exactly the official algorithm: input (SzCORE → 18-ch double banana), thresholds (prominence 10 µV,
  not 1 µV; period 0.025–0.150 s), window loop, cross-channel rule, and no ≥ 3 s minimum event length. The old
  departures had no documented reason. Output is sample-identical to the official code (transcribed) on 7 recordings;
  clean SzCORE score there: 7/7 seizures, 274 false alarms/24 h.
  The DINO-EEG column of Test 2s above was measured with SzCORE input and must be rerun. All prediction caches deleted
  (`backup_pre_szcore_input/` kept).

### 2026-11-08 · notch: 50 Hz considered, 60 Hz kept
- Considered moving SPaRCNet and DINO-EEG to a 50 Hz notch (Siena mains). Rejected: the study is about the
  technician's role, so detectors stay fixed tools, run as released (see Pipeline note). With a 50 Hz notch, a
  synthetic 50 µV peak 50 Hz error would leave 0.24 µV RMS after DINO-EEG preprocessing (−43 dB), i.e. the mains
  test would be trivial for it.
