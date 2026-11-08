# Robustness tests in the literature, and candidate tests for Expedition 002

*How published work stress-tests pretrained classifiers/detectors with corrupted input signals, and which of those test
designs fit our setup (Siena, pretrained eventNet / DINO-EEG / SPaRCNet / Gotman, per-electrode errors before
re-referencing, SzCORE event scoring).*

Builds on [literature_investigation.md](literature_investigation.md) (datasets, models) and
[signal_processing_enhancements.md](signal_processing_enhancements.md) (front-end filtering). Those topics are not
repeated here. PDFs are in `data/papers/002-seizure-robustness/` (not committed; `data/` is git-ignored). Every entry below was checked against the PDF or the arXiv/Crossref
record. If something could not be checked, the text says so.

---

## 1. Papers

### 1a. EEG and seizure detection

| # | Citation | Local PDF | Perturbations / tests | Metrics | Key finding (one line) |
|---|---|---|---|---|---|
| E1 | Mohammadi & Zarei, *RobustSeiz*, arXiv:2609.04007 (2026) | `mohammadi2026_robustseiz.pdf` | Grid sweeps, applied in memory at inference: AWGN (SNR 0–100 dB), 60 Hz mains scaled to signal RMS, impulse noise, notch/mains mismatch, ADC bits, band-pass profiles, low-freq "impedance" noise, drift (4 modes), sampling-rate round trip, named (O1/O2, T7/T8) and random channel zeroing, FGSM/PGD. Data are **converted to CAR before perturbation**. Demo: **one model (SeizureTransformer) on TUSZ**, which is in-distribution | SzCORE sample + event Se/Prec/F1/FP24h; Lead/Lag onset timing; MC-dropout agreement φ stratified by TP/FP/FN/background | Most environment shifts left event metrics **exactly unchanged** (band-pass, CAR, drift, impedance: same numbers as baseline). Severe AWGN made the model *more conservative* (F1 0.62→0.52, FP/24h 16.7→1.5). Zeroing O1/O2 or T7/T8 gave F1 = 0, a degenerate result |
| E2 | Širca, Alimardani, Zafeiriou, Barmpas, *Beyond Accuracy: Robustness, Interpretability and Expressiveness of EEG FMs*, arXiv:2605.17562 (2026) | `sirca2026_eeg-fm-robustness.pdf` | 6 EEG-FMs + EEGNet **fine-tuned per task** (Siena included as a 4 s window task). White/pink noise at SNR {10…−5, −15} dB; random channel dropout p∈{0.1,0.3,0.5}, **zero-padded vs removed**; region dropout/noise (fixed "primary" vs "control" region per task). CAR applied **before** perturbation | Balanced accuracy, Δ percentage points | No model is best on every perturbation. The most noise-robust model (BrainOmni) is the most fragile under zero-padded dropout, and most of that fragility goes away when the channels are removed instead. The authors list muscle/eye artefacts and electrode shift as untested |
| E3 | Wagh et al., *Evaluating latent space robustness and uncertainty of EEG-ML models under realistic distribution shifts*, NeurIPS 2022, arXiv:2209.11233 | `wagh2022_eeg-latent-robustness.pdf` | 4 "domain-guided" transforms: band-pass (fL, fH), quantization precision, 0–1 Hz impedance noise, broadband noise (3 strengths each) | Task AUROC; latent-space integrity (Delaunay graph heterogeneity); MC-dropout uncertainty | Latent-space distortion and uncertainty under synthetic shifts can warn of degradation without external data |
| E4 | Banville et al., *Robust learning from corrupted EEG with dynamic spatial filtering*, NeuroImage 2022, doi:10.1016/j.neuroimage.2022.118994 (arXiv:2105.12916) | `banville2022_dynamic-spatial-filtering.pdf` | Channel corruption model X̃=(1−η)X+ηZ on a random channel mask. **2-D sweep:** noise strength η at fixed corruption prob. 0.5, and number of corrupted channels at fixed η. Compared against interpolation baselines (Autoreject / spherical spline). Natural corruption in at-home mobile EEG | Balanced accuracy per recording (TUAB pathology, PC18 sleep) | Without a channel-handling mechanism, accuracy collapses with corrupted channels. Attention-based channel reweighting gains up to 29.4% under heavy corruption |
| E5 | Bigdely-Shamlo et al., *The PREP pipeline*, Front. Neuroinform. 2015, doi:10.3389/fninf.2015.00016 | `bigdelyshamlo2015_prep-pipeline.pdf` | Interaction between noisy channels, average reference and bad-channel detection. Robust (iterative) reference | Channel-wise noise statistics, correlation, SNR | Average reference helps SNR, **but one noisy channel contaminates every channel**. Which channels look "bad" depends on the reference. This is the mechanism behind our pre-reference design, and DINO-EEG's pyprep comes from here |
| E6 | Kojima, Shi, Jaikumar … Conrad, *How much does the reduced EEG montage matter for seizure detection?*, medRxiv 2026, doi:10.64898/2026.05.05.26352477 | `kojima2026_reduced-montage.pdf` | Simulated sub-scalp reduced montages from 10-20 (436 patients, 1683 seizures), 3 detectors (one-class SVM, SPaRCNet, LSTM-NDD) | Event F1; **linear mixed-effects with variance partition**; laterality × montage interaction | Montage explains 0.4% of variance, patient 29.2%, detector 10.3%. For SPaRCNet, laterality and laterality × montage were significant. Missed seizures had lower amplitude and band power |
| E7 | Thodoroff, Pineau, Lim, *Learning robust features using deep learning for automatic seizure detection*, MLHC 2016, arXiv:1608.00220 | `thodoroff2016_seizure-missing-channels.pdf` | Missing channels at test time (CHB-MIT), variable montage; RCNN vs Shoeb-style SVM | Sensitivity, FP/h | Early evidence that a spatial (topographic-image) deep model holds up better than a hand-crafted SVM when channels are missing |
| E8 | Dan et al., *SzCORE*, Epilepsia 2024, doi:10.1111/epi.18113 (arXiv:2402.13005) | `dan2024_szcore.pdf` | No perturbations. Defines input content (19-ch 10-20, CAR) and event scoring (−30/+60 s tolerance, 5-min max, merge < 90 s) | Sample- and event-based Se/Prec/F1/FP24h | This is the scoring we use. Its tolerances are the source of the "masking" effect |
| E9 | Dan, Shahbazinia, Kechris, Atienza, *Quantifying the generalization gap in seizure detection (SzCORE challenge)*, ICML 2026 (PMLR 306), arXiv:2505.18191 | `dan2025_szcore-challenge.pdf` | No perturbations. 28 algorithms on 4,360 h held-out | Event F1; **Friedman → Wilcoxon signed-rank with Holm, Nemenyi critical-difference diagram, Cliff's δ** | Best F1 32%. Over 91% of false detections are shared by < 25% of algorithms, so FPs are idiosyncratic rather than driven by universal artefacts. Hard seizures are shorter (median 48 s) |
| E10 | Ziyabari, Shah, Golmohammadi, Obeid, Picone, *Objective evaluation metrics for automatic classification of EEG events*, arXiv:1712.10107 (2017); later a book chapter, doi:10.1007/978-3-030-67494-6_8 | `shah2017_eeg-event-scoring-metrics.pdf` (filename keeps the earlier agent's name; first author is Ziyabari) | Compares scoring methods on the same hypotheses: any-overlap (OVLP), epoch, DP alignment, ATWV, **TAES** | Per-method Se/FA24h | OVLP is very permissive: a hypothesis that covers a small part of a seizure counts as a full TP. TAES weights each event by its temporal overlap |
| E11 | Zhang et al., *EEGdenoiseNet*, J. Neural Eng. 2021, doi:10.1088/1741-2552/ac2bf8 (arXiv:2009.11662) | `zhang2021_eegdenoisenet.pdf` | Semi-synthetic: 4514 clean EEG, 3400 EOG, 5598 EMG single-channel 2 s segments, mixed at SNR −7…2 dB | RRMSE, correlation | **A library of real recorded EMG/EOG segments.** EMG is stored at 512 Hz (Siena's native rate) and EOG at 256 Hz. Usable for real-artefact injection |
| E12 | Klados & Bamidis, *A semi-simulated EEG/EOG dataset…*, Data in Brief 2016, doi:10.1016/j.dib.2016.06.032 (PMC4969208) | not downloaded (OA, but PMC/EuropePMC/CORE returned HTML to curl) | Clean EEG + real EOG via a head-model contamination equation | Ground-truth comparison | Design pattern: keep the clean original so the ground truth under the artefact is known |
| E13 | Hamid et al., *The Temple University Artifact Corpus (TUAR)*, IEEE SPMB 2020, doi:10.1109/SPMB50085.2020.9353647 | not OA | Annotated real multichannel artefacts (chewing, electrode, eye movement, muscle, shiver) | – | A source of *multichannel* real artefacts. Caveat: TUAR comes from TUH EEG, so it may overlap with eventNet's/DINO-EEG's training distribution |

### 1b. Other biosignals

| # | Citation | Local PDF | Perturbations / tests | Metrics | Key finding |
|---|---|---|---|---|---|
| B1 | Moody, Muldrow, Mark, *A noise stress test for arrhythmia detectors*, Computers in Cardiology 11:381–384 (1984). Database: MIT-BIH NSTDB, PhysioNet | not OA (no DOI; database is open at physionet.org/content/nstdb) | **Real recorded noise** (baseline wander `bw`, muscle `ma`, electrode motion `em`) added to clean records 118/119 at **SNR 24, 18, 12, 6, 0, −6 dB**, in alternating clean/noisy 2-min blocks. Reference annotations copied from the clean record | Beat-detection Se/+P per SNR | The canonical "real noise × calibrated SNR × unchanged labels" design. Electrode motion is the worst noise because it mimics ectopic beats |
| B2 | Venton et al., *Robustness of convolutional neural networks to physiological electrocardiogram noise*, Phil. Trans. R. Soc. A 2021, doi:10.1098/rsta.2020.0262 (arXiv:2108.01995) | `venton2021_ecg-noise-robustness.pdf` | NSTDB noise types (bw, em, ma, and all three combined) adapted to 12-lead ECG at SNR 5–10 dB. **Train-clean/test-noisy vs train-noisy/test-noisy cross design**. Two input transforms | F1 per noise type | Within-condition drop < 0.05 F1, but **clean-trained → noisy-tested** drops up to 0.18. The effect depends on noise type × input representation |
| B3 | Voisin et al., *Ambulatory atrial fibrillation monitoring using wearable PPG with deep learning*, arXiv:1811.07774 (2018; venue not verified) | `voisin2019_ppg-af-motion.pdf` | **Natural artefact stratification**, no injection: records ranked by accelerometer motion intensity, AUC computed on the cleanest c% for c=10…100 | **AUC–coverage curve** (flat curve = robust) | The deep model's AUC stays flat as noisier records are included. The feature baseline does not |
| B4 | Duggal et al., *REST: robust and efficient neural networks for sleep monitoring in the wild*, WWW 2020, doi:10.1145/3366423.3380241 (arXiv:2001.11363) | `duggal2020_rest-sleep-noise.pdf` | Sleep staging (Sleep-EDF, SHHS) under Gaussian, shot and adversarial noise at increasing strength | Macro-F1, Cohen's κ vs noise level | Standard sleep-staging nets collapse under noise (macro-F1 0.39 vs 0.67 for the robust model under Gaussian noise). Adversarial training + spectral regularisation helps |

### 1c. Fields with mature corruption-robustness methods

| # | Citation | Local PDF | Perturbations / tests | Metrics | Key finding |
|---|---|---|---|---|---|
| G1 | Hendrycks & Dietterich, *Benchmarking neural network robustness to common corruptions and perturbations* (ImageNet-C/-P), ICLR 2019, arXiv:1903.12261 | `hendrycks2019_imagenet-c.pdf` | 15 corruptions × **5 severities** in 4 families (noise, blur, weather, digital). -P: small perturbation *sequences* | **CE** = Σ_s error / Σ_s baseline error (AlexNet-normalised); **mCE**; **relative mCE** = degradation *beyond clean error*; **flip probability / flip rate** for prediction stability | Clean-accuracy gains largely carry over to mCE, but *relative* robustness barely improved from AlexNet to ResNet. Absolute and relative robustness have to be reported separately |
| G2 | Taori et al., *Measuring robustness to natural distribution shifts in image classification*, NeurIPS 2020, arXiv:2007.00644 | `taori2020_effective-robustness.pdf` | 204 models × synthetic corruptions vs natural shifts | **Effective robustness** = shifted accuracy above the linear (logit) trend predicted from clean accuracy | Robustness to synthetic corruptions **does not transfer** to natural shifts. Most "robust" models only sit on the clean-accuracy trend line |
| G3 | Ovadia et al., *Can you trust your model's uncertainty?*, NeurIPS 2019, arXiv:1906.02530 | `ovadia2019_uncertainty-dataset-shift.pdf` | Increasing shift intensity (rotations, ImageNet-C severities, OOD) for vanilla, temperature scaling, MC-dropout, ensembles, SVI, last-layer methods | Accuracy, **ECE, Brier, NLL vs shift intensity**; accuracy vs confidence | Calibration degrades steadily with shift. Temperature scaling fitted in-distribution does not survive shift. Ensembles are the most robust |
| G4 | Di Salvo, Doerrich, Ledig, *MedMNIST-C*, MICCAI 2024 ADSMI workshop, arXiv:2406.17536 | `disalvo2024_medmnist-c.pdf` | 12 datasets / 9 modalities. **Modality-specific** corruptions (e.g. dermatoscope black corners) × 5 severities, alongside generic noise/blur/colour | Balanced-error versions of CE / mCE and relative mCE (normalised by a ResNet-18 baseline) | Corruptions designed for the acquisition modality are needed, and as augmentation they beat generic MixUp/CutMix/RandAugment |
| G5 | Kong et al., *Robo3D*, ICCV 2023, doi:10.1109/ICCV51070.2023.01830 (arXiv:2303.17597) | `kong2023_robo3d.pdf` | LiDAR: weather, external disturbance and **internal sensor failure (beam missing, crosstalk, incomplete echo, cross-sensor)** × 3 severities, 34 models | mCE (baseline-normalised) and **mRR = mean resilience rate = Σ_l Acc_l / (L·Acc_clean)** | Models with similar clean accuracy differ widely in resilience. A sensor-failure taxonomy on a fixed geometric array, analogous to bad electrodes |
| G6 | Hirsch & Pearce, *The AURORA experimental framework for… speech recognition under noisy conditions*, ISCA ITRW ASR2000, pp. 181–188 | `hirsch2000_aurora.pdf` | **Real recorded noises** added at SNR 20, 15, 10, 5, 0, −5 dB + clean. SNR is defined from the *speech* level (ITU P.56) and a random cut of a long noise recording. Test set A = noises seen in multi-condition training, **B = unseen noises**, **C = channel (frequency-response) mismatch** | Word accuracy per SNR, averaged over 0–20 dB | Defines the matched / unmatched noise / channel-mismatch split. Results are averaged over an SNR range instead of read at one point |
| G7 | Radford et al., *Robust speech recognition via large-scale weak supervision* (Whisper), ICML 2023, arXiv:2212.04356 | `radford2022_whisper-noise-robustness.pdf` | §3.7: white noise vs natural "pub" noise, SNR set **per example from its own signal power**, Whisper vs 14 LibriSpeech-trained models | WER vs SNR | Specialists win at high SNR but **cross over** and lose below ~10 dB, especially under natural noise. Rankings at clean ≠ rankings under noise |

Not downloaded: Moody 1984 (no OA PDF), Hamid 2020 TUAR (IEEE, not OA), Klados 2016 (OA, but no PDF could be fetched by
curl).

---

## 2. Methodology patterns

| Pattern | What it is | Used by | Relevance to us |
|---|---|---|---|
| **Severity sweep + degradation curve** | Each corruption at 3–6 graded levels, results summarised as a curve or area (AUDC), or averaged over a range (Aurora 0–20 dB) | G1, G4, G5, G6, G7, B1, B4, E1, E2, E3, E4 | Our 2–3 hand-picked levels per type are too few to find the "breaking point" |
| **SNR relative to each record's own signal** | Noise scaled from the individual signal power, not as an absolute amplitude | G6 (P.56 speech level), G7 (per-example power), B1, E11, E1 (mains ∝ RMS) | Our levels are absolute µV. Per-electrode relative SNR makes patients comparable. Siena amplitudes vary a lot |
| **Real recorded noise at calibrated SNR, labels unchanged** | Add recorded artefact (not synthetic) to clean data, keep reference annotations | B1, B2, G6, G7 (pub noise), E11, E12 | Our EMG/drift are synthetic. G2 and G7 show synthetic and natural robustness can disagree |
| **Seen vs unseen noise / channel mismatch splits** | Separate in-training-distribution noise from novel noise and from transfer-function change | G6 (sets A/B/C), B2 (train-clean vs train-noisy cross) | Our detectors saw TUH artefacts in training. Siena-specific errors (50 Hz, displacement) are "unseen" |
| **Baseline-normalised aggregate (CE, mCE) vs relative (relative mCE, mRR)** | Normalising by a reference model removes difficulty differences between corruptions. Relative metrics separate robustness from clean skill | G1, G4, G5 | Gotman/SPaRCNet have near-zero clean precision, so raw Δ-metrics are floor-limited. We need relative metrics |
| **Effective robustness** | Is the shifted score above what clean score predicts across many models/units? | G2 | With 4 detectors we can use per-patient points instead of models |
| **Per-sensor / per-region ablation maps** | Corrupt one channel or region at a time, map the damage | E2 (region), E4 (#channels), G5 (beam missing, cross-sensor), E1 (named dropout), E6 (montage subsets), E7 | Nobody has mapped single electrodes before re-referencing |
| **Missing-channel handling as a confound** | Zero-fill vs removal vs interpolation changes conclusions | E2 (zero vs remove), E4 (interpolation baselines), E5, E1 (zero-filling degenerate) | DINO-EEG's pyprep interpolation is exactly this confound |
| **Prediction stability / flip rate** | Fraction of predictions that change under small perturbations, independent of correctness | G1 (ImageNet-P flip rate), E1 (MC-dropout φ) | Event flips and boundary shifts between clean and perturbed runs, which are hidden by tolerant scoring |
| **Calibration / confidence under shift** | ECE, Brier, NLL, confidence distributions as shift grows | G3, E1, E3 | eventNet and SPaRCNet output probabilities. Threshold drift explains FA changes |
| **Sample- vs event-level vs timing scoring** | Same hypotheses scored several ways. Tolerant scoring hides partial/shifted detections | E10 (OVLP vs TAES), E8, E1 (Lead/Lag) | Directly addresses our boundary-shift observation |
| **Natural artefact stratification (no injection)** | Rank real records by measured artefact level, compute performance on cleanest c% | B3 | Tests whether simulated sensitivity predicts behaviour on Siena's own artefacts |
| **Paired non-parametric statistics + effect sizes** | Friedman → Wilcoxon signed-rank (Holm), Nemenyi CD diagram, Cliff's δ | E9 | H2 (detector comparison) across patients |
| **Mixed-effects models with variance partition** | Patient random effect. Proportion of variance from patient / detector / condition. Interaction tests | E6 | H1/H2 with 14 patients, where patient variance will dominate |

---

## 3. Candidate tests for our setup (ranked)

**Compute unit (U):** one condition on the 9-patient ~76 h subset with eventNet ≈ 15 min. The full 141 h cohort is ≈ 28
min. DINO-EEG ≈ 3–4 U. SPaRCNet and Gotman ≤ 1 U (not timed). **Seizure-centred windows** (each seizure ±15 min) are
about 14 h for the 28 included-patient seizures, so ≈ 3 min/condition for eventNet. Use them when only recall/latency
matter.

**What the two closest recent papers already did.** These overlap checks recur in the tests below.
- **RobustSeiz (E1):** one detector on its own training corpus (TUSZ). Perturbations applied *after* CAR, global or zeroed
  channels. SNR sweeps, mains, notch mismatch, ADC, drift, channel zeroing, adversarial. Sample + event SzCORE metrics,
  Lead/Lag, MC-dropout.
- **Širca 2026 (E2):** models *fine-tuned on Siena*, scored as 4 s window classification. CAR *before* perturbation. Global
  white/pink SNR sweep, random/region dropout (zero vs remove), region noise with a fixed task region.
- **Neither paper** covers: per-electrode errors before re-referencing, CAR vs bipolar spreading, patient-specific
  lateralization, real artefacts, displacement or label swaps, classical vs DL on a held-out set, seizure-timed
  artefacts, or the event-tolerance masking effect.

### T1. Event-tolerance masking audit (re-scoring only)
- **Hypothesis:** SzCORE event scoring (−30/+60 s, merge < 90 s) hides much of the perturbation-induced change. Boundary,
  latency and sample metrics show graded degradation where event F1 is flat.
- **Procedure:** for all cached clean/perturbed predictions (all detectors), re-score with (a) SzCORE event,
  (b) SzCORE sample-based, (c) any-overlap without tolerance, (d) TAES, (e) a tolerance sweep (0/0, 5/10, 15/30,
  30/60 s), (f) onset latency (Lead/Lag) per TP, (g) clean-vs-perturbed **event flip rate** (events gained/lost) and
  boundary IoU between the clean and the perturbed detection of the same event. Report the fraction of conditions where
  event F1 is unchanged but (f)/(g) changed.
- **Metrics:** ΔF1 per scorer, onset latency shift (s), flip rate, boundary IoU.
- **Compute:** ~0 (re-scoring, minutes).
- **Why publishable:** a methods result: "SzCORE event metrics are insensitive to X% of acquisition-error effects".
  RobustSeiz Table 6 shows identical numbers for whole perturbation families and does not analyse why. Clinically, onset
  latency matters for alarms.
- **Motivated by:** E10, E8, E1 (Lead/Lag), G1 (flip rate).
- **Overlap:** low. RobustSeiz reports sample + event + Lead/Lag but does not study masking. Ziyabari compared scorers on
  clean output only.

### T2. Single-electrode leakage map: CAR vs bipolar vs classical
- **Hypothesis:** one bad electrode hurts a CAR detector (eventNet) through all 19 channels at ≈ 1/19 amplitude, plus its
  own channel at 18/19. In a bipolar detector (DINO-EEG, SPaRCNet) it hits only the 2 (up to 4) derivations that use it,
  at full amplitude. So the damage topography differs by montage. Also, a flat electrode in bipolar turns its
  derivations into referential signals, which exposes the recording reference.
- **Procedure:** for each of the 19 electrodes separately, inject one fixed error (high-impedance noise at 0 dB relative
  to that electrode's RMS; optionally a flat electrode) before re-referencing. Run all 4 detectors. Plot ΔFA/24h and
  Δrecall as scalp topomaps per detector. Add the analytic leakage matrix (perturbation energy per derived channel) as a
  detector-free figure.
- **Metrics:** ΔFA/24h, Δrecall, plus T1 metrics. Per-electrode effect with a patient-bootstrap CI.
- **Compute:** 19 conditions ≈ 4.75 h eventNet, 14–19 h DINO-EEG, ≤ 5 h each for SPaRCNet/Gotman. Doubles with the flat
  variant. Seizure windows + a 10 h background sample cut this about 3×.
- **Why publishable:** the central claim of this study, and no paper has done it. It turns the README's "one bad
  electrode leaks into several channels" into a measured, montage-specific vulnerability map, which is useful for
  technicians (which electrodes to check first).
- **Motivated by:** E5 (noisy channel contaminates the average), G5 (per-sensor failure), E4, E2, E6.
- **Overlap:** RobustSeiz zeroes named pairs *after* CAR, on one model. Širca dropout is post-CAR. **Novel.**

### T3. Seizure-side (ipsilateral vs contralateral) corruption using `subject_info.csv`
- **Hypothesis:** corrupting electrodes over the seizure focus (e.g. F7/T3/T5 for left temporal) reduces recall much more
  than corrupting the homologous contralateral electrodes, which mostly add false alarms. Bipolar detectors are more
  side-specific than CAR detectors, and Gotman (per-channel) is the most side-specific.
- **Procedure:** 13/14 patients are temporal (9 L, 4 R). PN10 is frontal-bilateral, so exclude it or analyse it
  separately. For each patient, apply flat, high-impedance and displacement errors to the 3 ipsilateral temporal
  electrodes vs the 3 contralateral ones, at one or two severities. The paired contrast is within-patient
  (ipsi − contra). Use the 11 included patients for recall (8 L, 3 R).
- **Metrics:** seizure-level detected/missed (paired McNemar, clean vs ipsi, ipsi vs contra), Δrecall, onset latency,
  ΔFA/24h. Mixed model `metric ~ side × detector + (1|patient)`.
- **Compute:** 3 errors × 2 sides × 1–2 levels = 6–12 conditions ≈ 1.5–3 h eventNet, 4.5–12 h DINO-EEG. Seizure windows
  are enough for recall (≈ 20–40 min eventNet).
- **Why publishable:** clinically sharp ("an error on the electrode that matters most") and uses patient metadata no
  benchmark uses. It tests whether the detectors rely on focal or diffuse evidence.
- **Motivated by:** E2 (primary vs control region, but fixed per task), E6 (laterality × montage interaction for
  SPaRCNet).
- **Overlap:** Širca's region test is task-level, not patient-specific, and uses fine-tuned models. **Novel.**

### T4. Real vs synthetic artefacts at matched SNR
- **Hypothesis:** real recorded EMG/EOG harms detectors differently (likely more FAs for DL) than our synthetic
  band-limited bursts at the same SNR. Detector rankings may flip (G7 crossover, G2 non-transfer).
- **Procedure:** inject EEGdenoiseNet EMG segments (512 Hz, the Siena rate) on temporal/frontal electrodes and EOG
  segments (256 Hz, resample) on Fp1/Fp2/F7/F8 in bursts with the same timing as `perturb_muscle_emg`, at SNR {6, 0, −6}
  dB relative to each electrode's RMS. Run the matched synthetic EMG at the same SNRs. Optional: TUAR multichannel
  artefacts (needs a TUH licence; note the possible overlap with the training distribution, i.e. Aurora set A vs B).
- **Metrics:** ΔFA/24h (primary), Δrecall, real-vs-synthetic difference per detector, rank change.
- **Compute:** 2 artefact types × 3 SNR × (real + synthetic) ≈ 12 conditions ≈ 3 h eventNet, 9–12 h DINO-EEG.
- **Why publishable:** shows whether the common practice of synthetic EEG perturbation (used by E1, E2, E3) is valid.
  Širca names muscle/eye artefacts as untested.
- **Motivated by:** B1, B2, G6, G7, E11, E12.
- **Overlap:** none in E1/E2. **Novel.**

### T5. Seizure-timed vs background-only errors
- **Hypothesis:** artefacts confined to the ictal period (patients move and chew during seizures) cause misses that
  whole-recording perturbations dilute. Artefacts confined to the background only change FAs. Separating the two
  explains which failure mode (missed seizure vs alarm flooding) each error causes.
- **Procedure:** apply EMG bursts, electrode pop/drift, or high impedance only from onset −30 s to offset +30 s ("ictal
  only"), and compare with the existing whole-record runs. Run on seizure-centred windows.
- **Metrics:** seizure-level miss rate, onset latency, boundary IoU (T1). FA rates come from the background runs.
- **Compute:** ~3 min/condition for eventNet, ~10–12 min for DINO-EEG. 6 conditions take under 1.5 h for all detectors.
- **Why publishable:** realistic and cheap. It separates the two clinically consequential failure modes that E1 names.
- **Motivated by:** E1 (missed seizures vs alarm flooding), E9 (hard seizures are short), B1 (alternating clean/noisy
  blocks).
- **Overlap:** none. **Novel.**

### T6. Built-in defences: bad-channel interpolation and notch (DINO-EEG pyprep on/off, eventNet + PREP)
- **Hypothesis:** robustness differences between detectors come partly from preprocessing, not architecture. pyprep
  catches flat/noisy electrodes (which restores performance) but not displacement/cross-talk (which is highly correlated
  with its neighbour). DINO-EEG's 60 Hz notch leaves 50 Hz untouched.
- **Procedure:** (a) run pyprep alone on each perturbed condition and record its bad-channel detection rate per error
  type and severity (no detector needed). (b) DINO-EEG with and without pyprep on flat, impedance, displacement and mains
  conditions. (c) eventNet with a PREP robust-reference + interpolation front end vs as released.
- **Metrics:** pyprep detection sensitivity per error; Δ metrics with vs without the defence.
- **Compute:** (b) 4–8 DINO runs ≈ 4–8 h. (c) 5 × U plus pyprep time. pyprep RANSAC on 141 h is not timed and may be the
  bottleneck.
- **Why publishable:** settles an open item in the README and gives a practical answer ("which errors do standard
  pipelines catch?").
- **Motivated by:** E5, E4 (interpolation baselines), E2 (zero vs remove changes conclusions), E1 (notch mismatch).
- **Overlap:** partial. RobustSeiz tested notch/mains mismatch globally, and Širca tested missing-channel handling. The
  pre-reference + pyprep interaction is new.

### T7. Recording-relative SNR sweep with degradation-curve summaries (H1 backbone)
- **Hypothesis:** each error type has a breaking-point SNR that differs between detectors (H1/H2). Rankings cross over
  between high and low SNR (G7).
- **Procedure:** re-parameterise the 5 main error types as per-electrode SNR {24, 18, 12, 6, 0, −6} dB, the NSTDB grid.
  Keep flat/disconnected as a separate categorical level.
- **Metrics:** curve per type. Area under the degradation curve (AUDC). Relative CE and mRR (normalise each detector to
  its own clean score, because Gotman/SPaRCNet clean precision is near zero). Effective robustness from per-patient
  clean vs perturbed scatter. Friedman + Wilcoxon-Holm + Cliff's δ across patients.
- **Compute:** 30 conditions ≈ 7.5 h eventNet, 22–30 h DINO-EEG (reduce DINO-EEG to 3 levels ≈ 11–15 h), ≤ 7.5 h each
  for SPaRCNet/Gotman.
- **Why publishable:** this is the expected core result and the backbone for the stats, but on its own it is the least
  novel test.
- **Motivated by:** G1, G4, G5, G6, G7, B1, E9.
- **Overlap:** **high in form.** RobustSeiz and Širca both ran SNR sweeps (global, post-CAR). Our novelty here is only
  pre-reference per-electrode injection, held-out pretrained detectors, and the classical vs DL comparison.

### T8. Natural-artefact stratification on clean Siena (AUC–coverage style)
- **Hypothesis:** FAs on *clean* Siena concentrate in windows with high natural artefact load. The detector most
  sensitive to simulated EMG (T4/T7) is also the most sensitive to natural EMG. This links simulation to reality.
- **Procedure:** per 10 s window, compute artefact indices: >30 Hz power ratio (EMG), 50 Hz power, line length, number of
  flat/high-variance channels (pyprep criteria). Rank windows and compute FA rate on the cleanest c% for c = 10…100.
  Correlate each detector's FA rate with the indices.
- **Metrics:** FA/24h–coverage curve, Spearman ρ between simulated sensitivity and natural sensitivity across detectors
  and patients.
- **Compute:** no detector runs. Features over 141 h take minutes to about an hour on CPU.
- **Why publishable:** gives the simulations external validity and connects to E9's finding that FPs are idiosyncratic.
- **Motivated by:** B3, G2, E9.
- **Overlap:** none. **Novel.**

### T9. "Silent" technician errors: displacement and electrode label swap
- **Hypothesis:** errors that bad-channel detectors cannot see (a neighbour-mixed electrode; swapped leads such as
  T3↔T4, or F3↔C3) degrade bipolar detectors (a derivation between near-identical electrodes goes to ~0 and focal phase
  reversal is lost) more than CAR detectors. A left/right swap moves the apparent focus but may leave CAR detection
  unchanged.
- **Procedure:** displacement on ipsilateral temporal electrodes at 3 mix ratios (combine with T3), adjacent swap, and
  homologous L/R swap.
- **Metrics:** Δrecall, ΔFA, pyprep detection rate (expected ≈ 0), onset latency.
- **Compute:** 5–6 conditions ≈ 1.5 h eventNet, 4.5–6 h DINO-EEG.
- **Why publishable:** realistic acquisition errors that QC pipelines cannot detect. The analogy is cross-talk /
  cross-sensor faults in G5.
- **Motivated by:** G5, E5.
- **Overlap:** Širca lists electrode shift as future work. **Novel.**

### T10. Threshold-free and calibration metrics under corruption
- **Hypothesis:** perturbations shift the output probability distribution (Ovadia), so the released fixed thresholds
  become mis-set. Part of the FA change is threshold drift that a threshold-free metric does not show.
- **Procedure:** save the continuous outputs (eventNet probability, SPaRCNet window P(seizure), DINO-EEG box scores).
  This needs a small adapter change; the current eventNet adapter returns events only. Compute sample-level AUROC/AUPRC,
  ECE/Brier on 1 s bins, and the per-condition optimal threshold vs the released threshold.
- **Metrics:** ΔAUROC, ΔAUPRC, ΔECE, threshold shift.
- **Compute:** 0 extra if outputs are saved during the T2/T7 runs, otherwise 1 U per condition.
- **Why publishable:** separates "the model's evidence degraded" from "the operating point moved". The latter can be fixed
  post hoc without retraining.
- **Motivated by:** G3, E1 (MC-dropout φ), E3.
- **Overlap:** RobustSeiz did MC-dropout agreement on TUSZ. Calibration and threshold shift on held-out data are new.

### T11. Dose-response in the number of bad electrodes: random vs clustered
- **Hypothesis:** CAR damage grows with the total bad-electrode energy wherever the electrodes are. Bipolar damage grows
  faster when bad electrodes are clustered on one chain (e.g. the patient lies on one side).
- **Procedure:** k ∈ {1, 2, 4, 8} flat or noisy electrodes, random vs neighbouring clusters, 3 seeds.
- **Metrics:** ΔF1, ΔFA/24h vs k, per layout.
- **Compute:** about 24 conditions ≈ 6 h eventNet, 18–24 h DINO-EEG.
- **Why publishable:** moderate novelty. Extends E4's "number of corrupted channels" sweep to pre-reference errors.
- **Motivated by:** E4, E2, E1.
- **Overlap:** **moderate–high.** Random channel dropout appears in both E1 and E2 (post-CAR, zeroing).

### Suggested order and budget
T1 → T8 cost almost nothing and should come first. Then T3 + T5 (cheap on seizure windows), then T2 (the main figure),
T4, T9 and T6. T7 is the confirmatory backbone, and T10 should run alongside T2/T7. eventNet for T1–T9 is roughly 20–25 h
of compute. DINO-EEG is the bottleneck: use seizure windows plus a background sample for it.

### Statistical notes (from E6, E9, and standard practice)
- The unit is the patient (n = 14, or 9 for recall analyses). Use paired Wilcoxon signed-rank per condition vs clean with
  Holm correction, and Friedman + Nemenyi for detector rankings (E9). Report Cliff's δ.
- Model `metric ~ condition × detector + (1|patient)` with a variance partition (E6). Expect patient variance to dominate,
  as it did in E6.
- Seizure-level detected/missed is paired binary: McNemar (clean vs perturbed). With 47 seizures, power is limited.
  Report exact counts.
- FA/24h is a rate. Use log-ratio vs clean, or a Poisson/negative-binomial model with log(duration) offset.
- Seeds: perturbations with random phase or bursts need ≥ 3 seeds on at least one condition to show that between-seed
  variance is small relative to effects.
