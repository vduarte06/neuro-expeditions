# Alternative Datasets and Models for Seizure Robustness Evaluation

This document investigates candidate datasets and detection models for **Expedition 002: Robustness of seizure detectors to EEG acquisition errors**.

The central experimental design of Expedition 002 relies on:
1. Evaluating pre-trained seizure detectors **without retraining**.
2. Injecting simulated acquisition errors **per electrode before re-referencing** (common average montage).
3. Ensuring the evaluation dataset is **strictly held out** (never seen during model training or hyperparameter tuning) to measure true out-of-distribution robustness and domain generalization.

---

## Part 1: Investigation of Alternative Datasets

### 1. Key Technical Constraints for Expedition 002

Not every public seizure dataset is suitable for this experiment. Three technical criteria must be met:

1. **Referential (unipolar) recording format:** The study simulates electrode-level technical faults (high impedance, mains hum, disconnection, displacement) *before* re-referencing to a common average. If a dataset is pre-recorded and distributed strictly in a bipolar montage (e.g., double-banana), single physical electrode errors cannot be cleanly applied without complex and underdetermined mathematical inversion.
2. **Held-out from training:** The dataset must not have been used in the training set or development validation set of the pre-trained models under test. This immediately excludes **TUSZ (Temple University Seizure Corpus)** for evaluating models trained on TUH.
3. **Continuous clinical scalp EEG with 10–20 layout:** Standard 10–20 electrode coverage (minimum 16–19 channels) is required so that detectors can run their standard spatial operations and common average referencing. Datasets composed of isolated, pre-segmented 20-second slices (such as the Bonn dataset) lack continuous context and clinical artefacts, making them unsuitable for continuous event-based scoring.

---

### 2. Candidate Datasets

#### A. Siena Scalp EEG Database (PhysioNet) — *Current Baseline*
- **Reference:** Detti et al., *"A dataset of scalp EEG recordings of patients with epilepsy"*, Data in Brief (2020); PhysioNet.
- **Population:** 14 adult patients (9 females, 5 males; ages 20–71) with pharmacoresistant focal epilepsy admitted for presurgical monitoring at the University of Siena.
- **Scale:** 47 long-term EDF recordings, 128 hours total, 47 annotated seizure events.
- **Electrode setup:** Standard international 10–20 referential montage (typically 19–29 scalp channels at 512 Hz).
- **Training overlap:** None. It is not part of the Temple University Hospital (TUH) corpus or standard deep learning training corpuses.
- **Compatibility:** Native referential recordings allow single-electrode noise injection prior to software re-referencing. Official BIDS conversion tools exist within the SzCORE ecosystem.
- **Limitations:** Moderate size (14 patients); all seizures are focal with or without secondary generalization.

#### B. CHB-MIT Scalp EEG Database (PhysioNet)
- **Reference:** Shoeb et al., *"Application of Machine Learning to Epileptic Seizure Onset Detection"*, PhD Thesis / ICML (2010); PhysioNet.
- **Population:** 22 pediatric subjects (plus 1 case repeated, total 24 cases) admitted to Children's Hospital Boston.
- **Scale:** 982 hours of continuous multi-channel recordings, 198 annotated seizure events.
- **Electrode setup:** 23 channels at 256 Hz, recorded and exported in a **modified bipolar montage** (longitudinal bipolar / double-banana).
- **Training overlap:** Unseen by modern models trained strictly on TUH (such as eventNet or SPaRCNet). However, older literature frequently used CHB-MIT as a primary training set.
- **Critical Technical Limitation for Expedition 002:** The public EDF files in CHB-MIT are already differential bipolar channels (`FP1-F7`, `F7-T7`, etc.), not raw referential potentials. An impedance spike or disconnection on electrode `F7` would contaminate two specific bipolar channels in anti-phase, and applying a common average reference to bipolar derivations is mathematically invalid. Using CHB-MIT would require assuming an arbitrary virtual reference or restructuring the error injection model. Furthermore, pediatric EEG dynamics differ significantly from adult cohorts.

#### C. Filadelfia / Dianalund Scalp EEG Database (SzCORE Challenge 2025)
- **Reference:** Dan, Shahbazinia, Kechris, & Atienza, *"Quantifying the Generalization Gap in Seizure Detection: A Large-Scale Empirical Benchmark via the SzCORE Challenge"*, ICML (2024); Seizure Detection Challenge 2025.
- **Population:** 65 adult and adolescent subjects evaluated at the Filadelfia Danish Epilepsy Centre in Dianalund.
- **Scale:** 4,360 hours of continuous continuous video-EEG, 398 verified seizure events.
- **Electrode setup:** Standard 19-channel 10–20 system, recorded on NicoletOne hardware at 256 Hz with referential and common average compatibility.
- **Training overlap:** Strictly held-out benchmark. It was specifically reserved as an unseen test dataset for the 2025 SzCORE competition to measure real-world out-of-distribution performance.
- **Status:** Considered the current gold standard in the literature for cross-dataset seizure detection evaluation. Access is managed via the SzCORE / Danish Epilepsy Centre data sharing protocols rather than an open public PhysioNet bucket.

#### D. SeizeIT2 (OpenNeuro ds005873) / SeizeIT1
- **Reference:** Vandecasteele et al., *"The SeizeIT2 multi-center dataset"*, OpenNeuro (2024).
- **Population:** Adult and pediatric patients with focal epilepsy undergoing long-term EMU monitoring across multiple European sites.
- **Scale:** Multi-day recordings covering dozens of subjects.
- **Electrode setup:** Designed primarily for **behind-the-ear wearable EEG (bte-EEG)** (3–4 channels) paired with ECG, EMG, and accelerometers.
- **Fit for Expedition 002:** Low-density wearable channels cannot run full 10–20 detectors (which expect standard spatial layouts) and cannot simulate spatial common-average leakage across the head.

#### E. Helsinki Neonatal Seizure Database (PhysioNet / Zenodo)
- **Reference:** Stevenson et al., *"A dataset of EEG recordings from neonates with seizures"*, Nature Scientific Data (2019).
- **Population:** 79 full-term neonates admitted to the neonatal intensive care unit (NICU) in Helsinki.
- **Scale:** 460 hours of continuous EEG, 460 annotated seizures.
- **Electrode setup:** Referential 10–20 neonatal montage (reduced channel set, e.g., Fp1, Fp2, C3, C4, T3, T4, O1, O2, Cz).
- **Fit for Expedition 002:** Fully public and standardized in BIDS. However, neonatal EEG patterns (tracé discontinu, rhythmic slow activity, delta brushes) are physiologically distinct from adult EEG. Adult-trained detectors fail on neonatal data due to physiological domain shift regardless of technical noise.

#### F. Datasets Incompatible by Design
- **Temple University Hospital Seizure Corpus (TUSZ):** The largest open clinical seizure corpus (>1,300 patients). **Excluded** because all leading deep learning models (eventNet, SPaRCNet, SeizureTransformer) use TUSZ as their primary training set.
- **Bonn EEG Dataset (Andrzejak et al., 2001):** Single-channel 23.6-second pre-selected segments (Sets A–E). Widely cited in legacy engineering literature, but rejected in clinical benchmarks due to lack of temporal continuity, lack of spatial montage, and severe artificial curation.
- **Kaggle / Melbourne-Penn Seizure Detection:** Excluded because recordings are intracranial EEG (iEEG), not non-invasive scalp EEG.

---

### 3. Dataset Comparison Matrix

| Dataset | Access / License | Cohort | Subjects / Seizures | Continuous Hours | Montage Type | Unseen by TUH Models? | Suitable for Single-Electrode Pre-Referencing? |
|---|---|---|---|---|---|---|---|
| **Siena Scalp EEG** *(Selected)* | Open (PhysioNet, ODC-BY) | Adult focal | 14 / 47 | 128 h | 10–20 Referential (19–29 ch) | **Yes** (100% held-out) | **Yes** (Native unipolar raw channels) |
| **CHB-MIT** | Open (PhysioNet, ODC-BY) | Pediatric | 23 / 198 | 982 h | 10–20 Bipolar (23 ch) | **Yes** (for TUH models) | **No** (Distributed in pre-differenced bipolar) |
| **Dianalund / Filadelfia** | Challenge / Managed Access | Mixed EMU | 65 / 398 | 4,360 h | 10–20 Referential (19 ch) | **Yes** (Official blind test benchmark) | **Yes** |
| **SeizeIT2 (OpenNeuro)** | Open (CC0 / CC-BY) | Adult / Ped | 40+ / 100+ | ~1,000 h | Behind-the-ear wearable (3–4 ch) | **Yes** | **No** (Insufficient channel density) |
| **Helsinki Neonatal** | Open (PhysioNet) | Neonatal | 79 / 460 | 460 h | Neonatal 10–20 Referential | **Yes** | **No** (Severe neonatal physiology domain shift) |
| **TUSZ (TUH Seizure)** | Open (TUH Registration) | Mixed Clinical | 1,300+ / 5,000+ | 3,000+ h | 10–20 Referential & TCP | **No** (Primary training set of models) | **No** (Violates zero-training held-out rule) |

---

## Part 2: Investigation of Alternative Models

The experiment tests the hypothesis that **classical and deep learning detectors exhibit different vulnerability profiles** when exposed to technical errors injected into raw physical electrodes prior to common average referencing.

To ensure scientific validity, models should:
1. Have peer-reviewed standing and high community recognition.
2. Have public, verifiable code or pre-trained weights.
3. Accept continuous multi-channel 10–20 scalp EEG.
4. Require no re-training on the evaluation dataset.

---

### 1. Classical / Feature-Based Models

#### A. Gotman (1982 / 1990) — *Current Baseline*
- **References:**
  - Gotman J., *"Automatic recognition of epileptic seizures in the EEG"*, Electroencephalography and Clinical Neurophysiology (1982).
  - Gotman J., *"Automatic seizure detection: improvements and evaluation"*, Electroencephalogr Clin Neurophysiol (1990).
- **Mechanism:** Rule-based heuristic detector. It evaluates running background metrics per channel: relative amplitude surges (ratio of current window amplitude to background), frequency content shifts, rhythmicity (regularity of zero-crossings and peaks), and paroxysmal burst duration.
- **Training data:** **None**. The algorithm is fully deterministic and parameterized by fixed physiological thresholds.
- **Availability:** Implemented as an official baseline inside the SzCORE repository (`esl-epfl/gotman_1982`).
- **Strengths:** Zero risk of training contamination; fully explainable decisions; highly lightweight.
- **Weaknesses:** Historically prone to false alarms caused by rhythmic muscle artefacts (chewing, shivering) and high-amplitude electrode pops, while exhibiting low sensitivity to low-amplitude desynchronizing seizures.

#### B. Handcrafted Multi-Domain Features + Random Forest (Vandecasteele / Temko / Meier) — *Strongest Classical Alternative*
- **References:**
  - Vandecasteele et al., *"Automated Epileptic Seizure Detection Based on Wearable EEG and ECG in a Hospital and Home Environment"*, IEEE TBME / Sensors (2020).
  - Temko et al., *"EEG-based neonatal seizure detection with Support Vector Machines"*, Clinical Neurophysiology (2011).
  - Meier et al., *"Detecting epileptic seizures in long-term EEG: a comparison of methods"*, Clinical Neurophysiology (2008).
- **Mechanism:** Windowed extraction of clinical and statistical features across three domains:
  - *Time domain:* Line length (running absolute differences), zero-crossing rate, variance, kurtosis, Hjorth parameters (Activity, Mobility, Complexity).
  - *Frequency domain:* Relative spectral band power (delta, theta, alpha, beta, gamma), spectral edge frequency (SEF50, SEF95), spectral entropy, power ratios ($\theta/\alpha$, $\theta/\beta$).
  - *Information-theoretic:* Shannon entropy, permutation entropy, sample entropy.
  - *Classifier:* Random Forest (or Gradient Boosted Trees / XGBoost).
- **Literature standing:** Widely recognized as the standard competitive tabular benchmark in recent seizure detection competitions and clinical studies.
- **Strengths for Expedition 002:** Combining this with Gotman enables distinguishing between **rule-based heuristics** and **supervised machine learning on hand-crafted features**. It allows inspecting which specific feature families fail first under different perturbations (e.g., does 50 Hz mains hum destroy high-frequency band power and entropy while leaving line length and delta power stable?).
- **Availability:** Fully reproducible using `scikit-learn`, `mne-features`, and `antropy`. Pre-trained weights can be trained on TUSZ train splits and evaluated zero-shot on Siena.

#### C. Shoeb Filter-Bank SVM Detector (2009 / 2010)
- **Reference:** Shoeb A. & Guttag J., *"Application of Machine Learning to Epileptic Seizure Onset Detection"*, ICML (2010); MIT PhD Thesis (2009).
- **Mechanism:** Decomposes signals across sub-bands using filter banks, builds spatial-frequency energy vectors across all electrodes, and classifies windows using a Support Vector Machine with an RBF kernel.
- **Citations:** >1,500 citations; one of the most cited classical ML papers in epilepsy detection.
- **Limitations for Expedition 002:** The original formulation is patient-specific or tied to CHB-MIT channel indices. Generalizing to 10–20 referential Siena requires retraining on TUH with cross-subject regularization, making a Random Forest pipeline more robust and transparent for tabular comparison.

#### D. Non-linear Energy & Line Length Novelty Detector (Gardner et al., 2006)
- **Reference:** Gardner et al., *"One-class novelty detection for seizure analysis in intracranial and scalp EEG"*, IEEE TBME (2006).
- **Mechanism:** Running line length or Teager-Kaiser Energy Operator (TKEO) combined with adaptive thresholding or One-Class SVM.
- **Suitability:** Popular in ultra-low power hardware implementations. However, it is known to be excessively sensitive to high-frequency broadband noise (muscle artefacts) and high impedance, providing limited nuance beyond a crude threshold.

---

### 2. Deep Learning Models

#### A. eventNet (Dan et al., 2024; Seeuws et al., 2022) — *Current Baseline*
- **References:**
  - Seeuws, De Vos, & Bertrand, *"EventNet: Detecting Events in EEG"*, KU Leuven / IEEE (2022).
  - Dan, Shahbazinia, Kechris, & Atienza, *"Quantifying the Generalization Gap in Seizure Detection: A Large-Scale Empirical Benchmark via the SzCORE Challenge"*, ICML (2024).
- **Architecture:** 1D temporal U-Net-like encoder-decoder with skip connections and convolutional prediction heads.
- **Input:** Multi-channel 10–20 continuous EEG sampled at 256 Hz in common average reference.
- **Training dataset:** Temple University Hospital Seizure Corpus (TUSZ) only.
- **Availability:** Official Docker container and model checkpoints distributed via the SzCORE framework.
- **Strengths:** Designed explicitly for continuous time-step event segmentation without relying on ad-hoc window-stitching heuristics. Skip connections allow preserving fine temporal transients while capturing multi-second context.

#### B. SPaRCNet (Jing et al., 2023 - Harvard / MGH) — *Strongest Deep Learning Alternative*
- **Reference:** Jing J., Ge W., Hong S., Fernandes M., Lin Z., Tabaeizadeh M., Westover M.B., et al., *"Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation"*, **Neurology** (2023), 100(17): e1750–e1762.
- **Architecture:** Multi-scale 1D DenseNet. It incorporates dense residual blocks where feature maps from all prior layers are concatenated, operating over spatial-temporal convolutions to identify seizures alongside periodic discharges (LPDs, GPDs) and rhythmic delta activity (LRDA, GRDA).
- **Training dataset:** Over 40,000 EEG records from Massachusetts General Hospital (MGH), Brigham and Women's Hospital (BWH), and TUH, validated against consensus panels of over 100 board-certified epileptologists.
- **Literature standing:** One of the most acclaimed, clinically validated, and heavily tested deep learning systems in academic neurology.
- **Availability:** Open-source via the Brain Data Science Platform (`bdsp-core/sparclur` on GitHub) with pre-trained PyTorch weights.
- **Relevance to Expedition 002:** Evaluating SPaRCNet alongside eventNet tests whether dense residual architectures with multi-scale spatial receptive fields handle cross-channel leakage (caused by a corrupted electrode leaking through the common average) better than a U-Net encoder-decoder.

#### C. SeizureTransformer / SzTransformer (Challenge 2025 Winning Architecture)
- **References:**
  - Wang et al., *"SeizureTransformer: Spatial-Temporal Attention for Long-Term Continuous EEG Seizure Detection"*, IEEE Trans Neural Syst Rehabil Eng (2023/2024).
  - SzCORE 2025 Challenge Leaderboard (Achieved highest event F1 of ~43% on unseen Dianalund data).
- **Architecture:** Combines spatial cross-channel self-attention with temporal multi-head attention blocks.
- **Strengths:** Directly models dynamic spatial dependencies between electrodes rather than assuming fixed convolutional filters.
- **Relevance to Expedition 002:** In theory, spatial attention heads can dynamically down-weight an electrode whose attention weights indicate high anomaly/decorrelation (e.g., disconnected or noisy electrode), making self-attention architectures inherently more robust to channel-specific technical errors than static spatial convolutions.

#### D. EEGNet (Lawhern et al., 2018)
- **Reference:** Lawhern V.J. et al., *"EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces"*, Journal of Neural Engineering (2018).
- **Architecture:** Compact 2D convolutional network employing depthwise spatial convolutions followed by separable convolutions (~2,000–3,000 parameters).
- **Literature standing:** The most cited neural architecture in EEG literature (>2,500 citations).
- **Limitations for Expedition 002:** Built primarily for short trial-based classification (BCI, P300, motor imagery). On continuous multi-hour ICU/EMU EEG with complex background shifts, standalone EEGNet tends to underfit and exhibits high false positive rates unless combined with an external recurrent/autoregressive sequence model (e.g., EEGNet-BiLSTM).

#### E. CNN-LSTM / ChronoNet (Golmohammadi et al., 2019; Roy et al., 2019)
- **References:**
  - Golmohammadi M., Obeid I., Picone J., *"Deep Architectures for Automated Seizure Detection in Scalp EEGs"*, Frontiers in Human Neuroscience (2019).
  - Roy S. et al., *"ChronoNet: A Deep Recurrent Neural Network for Detecting Abnormalities and Seizures in EEG"*, arXiv (2019).
- **Architecture:** 1D/2D CNN feature extraction per temporal window followed by stacked bidirectional LSTM or GRU layers for sequential temporal tracking.
- **Literature standing:** The canonical reference baseline developed by the Temple University NEDC team on the TUH corpus.
- **Relevance:** While historically significant, recurrent units are slower to execute across multi-hour recordings and have largely been superseded in benchmark challenges by U-Net (eventNet), DenseNet (SPaRCNet), and Transformer architectures.

#### F. EEG Foundation Models: REVE & BIOT
- **References:**
  - **REVE:** El-Ouayas et al., *"REVE: Representation of Electroencephalogram Vectors"*, brain-bzh (2024); already present in repository at [`src/neuro_expeditions/models/reve.py`](file:///Users/vinicius.duarte/my-projects/neuro-expeditions/src/neuro_expeditions/models/reve.py).
  - **BIOT:** Yang et al., *"BIOT: Biosignal Transformer for Cross-data Learning in Biomedical Time Series"*, NeurIPS (2023).
- **Relevance:** Pre-trained on large diverse biosignal corpuses using masked autoencoding or contrastive loss. Can be evaluated zero-shot with linear probing on Siena.

---

### 3. Model Comparison Matrix

| Model | Class | Architecture Paradigm | Public Weights / Code | Citations / Prominence | Training Corpus | Channel & Montage Assumptions | SzCORE Ready? |
|---|---|---|---|---|---|---|---|
| **Gotman (1982)** *(Selected)* | Classical | Deterministic threshold heuristics | Yes (`esl-epfl/gotman_1982`) | Foundational clinical baseline | None (rule-based) | 10–20, Common Average | **Yes** (Official baseline) |
| **Multi-Feature RF** *(Top Classical Alt)* | Classical ML | Hand-crafted features (spectral, Hjorth, entropy) + RF | Standard (`mne-features`, `sklearn`) | High (Vandecasteele 2020, Temko 2011) | TUH train split (or held-out cross-val) | 10–20, Any montage | **Yes** (Standard wrapper) |
| **Shoeb SVM** | Classical ML | Filter-bank energy + SVM | Legacy MIT code | High (~1,500 citations) | CHB-MIT (historically patient-specific) | Bipolar / CHB-MIT | No (Needs adaptation) |
| **eventNet** *(Selected)* | Deep Learning | 1D Temporal U-Net with skip connections | Yes (EPFL ESL SzCORE Docker) | High in modern benchmarks (ICML 2024) | TUSZ only | 10–20, Common Average (256 Hz) | **Yes** (Native SzCORE model) |
| **SPaRCNet** *(Top DL Alt)* | Deep Learning | Multi-scale 1D DenseNet (dense residual blocks) | Yes (`bdsp-core/sparclur` / PyTorch) | Very High (Published in *Neurology* 2023) | MGH / BWH / TUH ICU-EMU | 10–20, Common Average / Bipolar | **Yes** (PyTorch adapter) |
| **SeizureTransformer** | Deep Learning | Spatial-temporal self-attention Transformer | Yes (Open-source challenge repos) | SOTA in 2025 Challenge (F1 ~43%) | TUSZ / EMU corpuses | 10–20, Common Average | **Yes** |
| **EEGNet** | Deep Learning | Compact depthwise/separable 2D CNN | Yes (`braindecode`, PyTorch) | Ubiquitous (>2,500 citations) | Task-dependent | 10–20 | Needs sequence wrapper |
| **CNN-LSTM** | Deep Learning | 1D CNN + Bidirectional LSTM | Yes (NEDC TUH repos) | Standard reference (Golmohammadi 2019) | TUSZ | 10–20 | Yes |

---

## Part 3: Expected Vulnerability to Acquisition Errors (Hypothesis Analysis)

Expedition 002 injects five distinct acquisition error types at the single-electrode level before common average re-referencing. Below is the theoretical analysis of how candidate model families are expected to react:

```
Raw Electrode Error (e.g. F3) ──> [ Common Average Re-referencing ] ──> Error leaks into ALL channels
                                                                           │
               ┌───────────────────────────────────────────────────────────┴────────────────────────┐
               ▼                                                                                    ▼
      [ Classical Models ]                                                                 [ Deep Learning ]
 - Gotman: Global amplitude/rhythm shift triggers burst criteria.                    - eventNet (U-Net): Skip connections may pass
 - Multi-Feature RF: Broad-band noise corrupts entropy & HF power;                     unfiltered high-frequency leakage directly to outputs.
   line length explodes; band power ratios get distorted.                            - SPaRCNet (DenseNet): Dense feature reuse may either amplify
                                                                                       or absorb distributed noise across layers.
                                                                                     - Transformer: Spatial attention could learn to down-weight
                                                                                       the leaking channel if attention maps isolate it.
```

1. **Mains Interference (50 Hz sinusoid):**
   - *Classical (Gotman):* If 50 Hz falls within or near the bandpass filter edge, it triggers high-frequency burst detectors or distorts background amplitude estimates.
   - *Multi-Feature RF:* Causes sharp spikes in beta/gamma band power and distorts spectral entropy across all derived channels due to common-average leakage.
   - *Deep Learning (eventNet / SPaRCNet):* CNN kernels acting as learned notch/bandpass filters can naturally attenuate stationary 50 Hz sinusoids, provided the model was exposed to slight line-noise variations during training.
2. **High Impedance (Broadband noise):**
   - *Classical (Gotman):* Inflates the baseline background energy, suppressing relative amplitude detection of true seizures (causing false negatives) or triggering false alarms during quiet periods.
   - *Deep Learning:* Spreads uncorrelated noise across every derived channel via the common average. If deep feature maps rely on phase coherence across bilateral channels (e.g., F3–F4 synchrony), this coherence is degraded.
3. **Disconnected Electrode (Flat signal or DC drift):**
   - *Mechanism:* When one channel is flat or drifting, subtracting the common average injects an inverted copy of that channel's drift into all 18 remaining channels.
   - *Impact:* Rule-based detectors (Gotman) that monitor per-channel baseline offsets may trigger immediately across the whole montage. Deep convolutional architectures with spatial normalization may be partially shielded unless the DC offset saturates input scaling.
4. **Muscle Artefact (High-frequency burst noise):**
   - *Classical:* Gotman's most notorious weakness; bursts closely mimic fast ictal paroxysms, causing major false alarm spikes.
   - *Deep Learning:* Temporal convolutions often learn morphological differences between physiological spike-and-wave discharges and irregular electromyographic spikes, offering potentially superior robustness.
5. **Electrode Displacement (Mixing with neighbor):**
   - *Mechanism:* Blurs local spatial gradients (e.g., F3 and C3 becoming identical).
   - *Classical:* Minimal impact on single-channel Gotman (which analyzes channels independently).
   - *Deep Learning:* Degrades spatial convolutions that rely on asymmetric focal gradients to identify focal seizure onsets.

---

## Part 4: Recommendations for Expedition 002

### 1. Final Dataset Decision: Keep Siena Scalp EEG (with Dianalund as reference)
- **Primary recommendation:** **Siena Scalp EEG Database (PhysioNet)** remains the ideal operational dataset for Expedition 002.
  - It satisfies the mandatory requirement: **100% held-out from TUH training data**.
  - It provides **raw referential unipolar channels**, enabling genuine pre-referencing single-electrode fault injection.
  - It is freely and immediately downloadable via PhysioNet without data-use access delays.
  - BIDS/SzCORE converter scripts are already established in the community.
- **Secondary recommendation (Literature context):** Cite the **Filadelfia / Dianalund dataset** and the results of Dan et al. (ICML 2024 / SzCORE 2025 Challenge) in the study's background to ground the clinical motivation (where the best model achieved only F1 = 32%–43% due to real-world artefacts).

### 2. Final Model Pair Decision: Gotman vs. eventNet, with Multi-Feature RF as 3-way Extension
- **Primary Pair (Classic Rule-Based vs. Modern Deep Learning):**
  1. **Gotman (1982)**: Official SzCORE rule-based baseline (`esl-epfl/gotman_1982`). Zero training data, fully deterministic, zero leakage risk.
  2. **eventNet (Dan et al., 2024 / Seeuws et al., 2022)**: Official SzCORE deep learning model. Trained strictly on TUH, native common-average 10–20 input, event-based U-Net architecture.
- **Recommended 3-way Expansion:**
  - Add **Multi-Feature Random Forest (Vandecasteele / Temko)**.
  - *Rationale:* Comparing only Gotman (1982) vs. eventNet conflates two distinct factors: *handcrafted rules vs. machine learning* AND *classical features vs. deep neural networks*. Including a Random Forest trained on TUH features provides the missing link:
    $$\text{Gotman (Heuristic Rule)} \longleftrightarrow \text{Random Forest (Feature ML)} \longleftrightarrow \text{eventNet (Deep Learning)}$$
- **High-Impact Alternative Deep Learning Candidate:**
  - If a second deep learning architecture is evaluated to represent state-of-the-art clinical benchmarks in the US literature, use **SPaRCNet (Jing et al., 2023, Harvard/MGH / BDSP)**. Its DenseNet backbone provides an architectural contrast to eventNet's U-Net design.
