# Signal Processing Enhancements for Scalp EEG Seizure Detection

*Research report on classical, spatial, and adaptive signal processing techniques to improve seizure detection sensitivity, reduce false alarms, and harden deep learning architectures against acquisition noise.*

---

## 1. Executive Summary & Problem Formulation

Automated seizure detectors based on deep neural networks—such as **eventNet** (a 1D Temporal U-Net) and **SPaRCNet** (DenseNet)—demonstrate remarkable feature-learning capacity when evaluated on in-distribution data (e.g., TUSZ or MGH ICU recordings). However, our empirical evaluation on the independent Siena Scalp EEG Database revealed two fundamental vulnerabilities:

1. **False Alarm Vulnerability (Specificity Collapse):**  
   Non-epileptic physiological and environmental transients (muscle EMG bursts, electrode pops, 50 Hz line hum, perspiration baseline sway) frequently mimic seizure morphology, triggering excessive false alarms in continuous recordings (e.g., SPaRCNet generated >150 false alarms per 24 hours).
2. **Subtle & Focal Seizure Blindness (Sensitivity Collapse):**  
   In focal or low-voltage fast onset seizures (e.g., patients `PN06`, `PN10`, `PN11`), localized ictal discharges are spatially diluted when pooled across 19 standard channels, remaining below global detection thresholds.

While deep learning operates directly on raw voltage streams, **classical signal processing front-ends and structured post-processing provide crucial inductive biases and artifact suppression** that raw end-to-end networks cannot easily learn from limited training datasets.

This document reviews the state-of-the-art literature (2020–2025) on signal processing enhancements for scalp EEG seizure detection, analyzes their physiological and mathematical foundations, and outlines concrete, testable enhancements to implement directly on `eventNet`.

---

## 2. Taxonomy of Signal Processing Enhancements

```
┌────────────────────────────────────────────────────────────────────────┐
│             Signal Processing Pipeline for Scalp EEG                   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Spatial          │     │ Temporal &       │     │ Subspace &       │
│ Filtering        │     │ Spectral         │     │ Multi-Resolution │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ • CAR            │     │ • 0.5–40 Hz BP   │     │ • DWT Soft Thresh│
│ • Surface        │     │ • 50/60 Hz Notch │     │ • Blind Source   │
│   Laplacian      │     │ • Savitzky-Golay │     │   Separation/wICA│
│ • Bipolar Pairs  │     │ • Hilbert Energy │     │ • CEEMDAN        │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Robust Normalization     │
                     ├──────────────────────────┤
                     │ • Running Median/IQR     │
                     │ • Z-score Standardization│
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Detector (eventNet U-Net)│
                     └────────────┬─────────────┘
                                  ▼
                     ┌──────────────────────────┐
                     │ Temporal Post-Processing │
                     ├──────────────────────────┤
                     │ • Dual Hysteresis Thresh │
                     │ • Minimum Duration Gate  │
                     │ • Event Consolidation    │
                     └──────────────────────────┘
```

---

## 3. Detailed Review of Enhancement Strategies

### 3.1 Spatial Filtering & Montage Optimization

Scalp EEG potentials are heavily distorted by **volume conduction** through the brain, cerebrospinal fluid, meninges, skull, and scalp. A localized electrical dipole in the cortex spreads broadly across multiple centimeters of the scalp.

#### A. Common Average Referencing (CAR)
* **Equation:**  
  $$V_i^{\text{CAR}}(t) = V_i(t) - \frac{1}{N}\sum_{j=1}^{N} V_j(t)$$
* **Mechanism:** Removes common-mode non-cerebral noise (such as far-field powerline hum and distant reference lead noise).
* **Clinical Trade-off:** While CAR is the standard in many automated challenges (including SzCORE), it has a severe drawback in clinical practice: **a single noisy or detached electrode contaminates every other channel** because the bad signal enters into the average.

#### B. Surface Laplacian (Current Source Density, CSD / Hjorth Laplacian)
* **Equation (Hjorth Discrete Approximation):**  
  $$L_i(t) = V_i(t) - \sum_{j \in \mathcal{N}_i} w_{ij} V_j(t)$$  
  where $\mathcal{N}_i$ denotes the set of nearest spatial neighbors to electrode $i$, and $w_{ij} \propto 1/d_{ij}$.
* **Mechanism:** Acts as a spatial high-pass filter by computing the second spatial derivative ($\nabla^2 V$).
* **Benefits for Seizure Detection:**
  - Strongly attenuates deep, widespread volume-conduction artifacts (such as broad movement artifacts and generalized background slowing).
  - Emphasizes radial cortical dipoles directly beneath the electrode array, boosting the signal-to-noise ratio of **focal seizure onsets** (e.g. temporal lobe IAS).
* **Literature Evidence:** Murro et al. and recent reviews (2024) demonstrate that Surface Laplacian increases the focal contrast of electrographic seizure onset by up to 40% compared to referential recording.

#### C. Longitudinal Bipolar Montage ("Double-Banana")
* **Mechanism:** Computes differences between adjacent antero-posterior pairs:
  $$(Fp1 - F7), (F7 - T7), (T7 - P7), (P7 - O1), \dots$$
* **Benefits:** Standard clinical reading montage. Eliminates common reference noise completely, localizes focal phase reversals, and prevents single-electrode drift from polluting distant lobes.

---

### 3.2 Temporal & Spectral Filtering

Clinical electrographic seizures are dominated by organized, rhythmic activity in the **1.0 to 25.0 Hz range** (rhythmic delta, theta, alpha, and evolving beta frequencies).

#### A. Bandpass Filtering (0.5 Hz – 40.0 Hz)
* **Low Cutoff (0.5 Hz, 2nd–4th order Butterworth/zero-phase IIR):**
  - Essential to eliminate DC offset, perspiration (galvanic skin response, typically 0.05–0.3 Hz), and slow mechanical electrode sway.
  - Setting the low cutoff higher (e.g., 1.0 Hz) can attenuate slow spike-and-wave discharges; 0.5 Hz provides the optimal clinical compromise.
* **High Cutoff (40.0 Hz):**
  - Scalp myographic (EMG) muscle activity is broadband, spanning 20 Hz to over 200 Hz, with maximum energy between 50 Hz and 100 Hz.
  - Setting a sharp high cutoff at 40.0 or 45.0 Hz eliminates the majority of temporal muscle shivering and mastication artifacts without harming seizure detection (clinical scalp EEG rarely contains interpretable cortical rhythms above 40 Hz due to skull low-pass filtering).

#### B. 50 Hz / 60 Hz Notch & Harmonic Comb Filtering
* **Mechanism:** Second-order infinite impulse response (IIR) notch filter with high quality factor ($Q \approx 30$):
  $$H_{\text{notch}}(z) = \frac{1 - 2\cos(\omega_0)z^{-1} + z^{-2}}{1 - 2r\cos(\omega_0)z^{-1} + r^2 z^{-2}}$$
* **Impact on Neural Networks:** In convolutional architectures, unfiltered 50 Hz sinusoidal noise creates persistent high-frequency activations across early convolutional kernels, degrading downstream temporal feature representations. Removing it upstream stabilizes kernel responses.

---

### 3.3 Wavelet Multi-Resolution Denoising (EMG Suppression)

Muscle artifacts (EMG from frontalis, temporalis, and masseter muscles) are the #1 source of false alarms in automated seizure detection. Because EMG bursts can occur at 30–80 Hz with high amplitude, standard linear filtering can either leave residual EMG or blur sharp epileptic spikes.

#### Discrete Wavelet Transform (DWT) Soft-Thresholding
1. **Decomposition:** Decompose the 256 Hz signal using Symlet (`sym4` or `sym8`) or Daubechies (`db4`) wavelets into 5 levels:
   - Level 1 detail ($D_1$): 64 – 128 Hz (almost pure EMG/noise)
   - Level 2 detail ($D_2$): 32 – 64 Hz (EMG mixed with fast spikes)
   - Level 3 detail ($D_3$): 16 – 32 Hz (beta, paroxysmal fast activity)
   - Level 4 detail ($D_4$): 8 – 16 Hz (alpha, sleep spindles, ictal rhythmic activity)
   - Level 5 detail ($D_5$): 4 – 8 Hz (theta, typical ictal frequency)
   - Level 5 approximation ($A_5$): 0 – 4 Hz (delta band)
2. **Thresholding Detail Coefficients:**
   Apply adaptive soft thresholding (Donoho & Johnstone universal threshold $\lambda = \sigma \sqrt{2 \ln N}$) specifically on $D_1$ and $D_2$:
   $$\hat{d} = \text{sign}(d) \cdot \max(|d| - \lambda, 0)$$
3. **Reconstruction:** Inverse DWT yields an EEG signal where muscle artifact energy is suppressed by 12–18 dB while preserving phase-aligned epileptic spikes and slow waves.

---

### 3.4 Robust Inter-Patient Amplitude Normalization

Scalp EEG amplitude varies dramatically across humans (ranging from 10–20 µV in elderly patients with cortical atrophy to >150 µV in young adults). Deep learning models trained on fixed microvolt scales (such as `eventNet`, which was trained on raw microvolts from TUSZ) are sensitive to this mismatch:
- If a patient has naturally high baseline amplitude, background rhythms cross the detection threshold $\rightarrow$ **excess false alarms**.
- If a patient has low-voltage baseline, genuine seizures fail to cross the threshold $\rightarrow$ **missed seizures**.

#### Running Median / Interquartile Range (IQR) Robust Scaling
* Instead of standard mean/variance ($z$-scoring, which is heavily skewed by high-amplitude seizure bursts):
  $$\tilde{x}_i(t) = \frac{x_i(t) - \text{median}(x_i)}{\text{IQR}(x_i) + \epsilon}$$
* Computed over a causal or centered rolling window (e.g., 60 seconds) or per recording.
* **Invariant property:** Scale-invariance ensures that a $3\times$ relative amplitude surge during a seizure produces identical normalized activations regardless of whether the patient's baseline is 15 µV or 80 µV.

---

### 3.5 Temporal Smoothing & Post-Processing Logic

`eventNet` outputs continuous center probabilities $p(t) \in [0, 1]$ and event duration predictions. The standard pipeline applies a simple static threshold ($\text{thresh} = 0.44$). However, simple thresholding leads to:
1. **Fragmented Detections:** A single long seizure dipping slightly below 0.44 mid-event is scored as two separate events (or penalized).
2. **Transient Spikes:** Brief (0.5 s) artifact peaks crossing 0.44 cause false alarms.

#### A. Dual-Threshold Hysteresis Triggering (Schmitt Trigger)
* **High Onset Threshold ($T_{\text{start}} = 0.50$):** Demands strong model confidence to declare a seizure onset.
* **Low Offset Threshold ($T_{\text{stop}} = 0.30$):** Once a seizure is active, it remains classified as an event until confidence drops below 0.30 for more than 5 consecutive seconds.
* **Result:** Eliminates event fragmentation, captures full seizure durations, and prevents brief transient artifact spikes from triggering false alarms.

#### B. Minimum Event Duration Gating
* Clinical seizures rarely last under 5–10 seconds (subclinical rhythmic discharges < 3s are usually classified as interictal bursts, not clinical seizures).
* Imposing a strict minimum duration gate ($t_{\text{duration}} \ge 5.0\text{ s}$) filters out isolated electrode pops and movement transients.

---

## 4. Priority Ranking for Immediate Experimentation

Based on computational complexity, clinical relevance, and compatibility with `eventNet`:

| Rank | Enhancement Technique | Target Vulnerability | Expected Benefit | Implementation Complexity |
|---|---|---|---|---|
| **1** | **Bandpass (0.5–40 Hz) + 50 Hz IIR Notch** | High-frequency noise, 50 Hz mains hum, DC drift | Drastically reduces false alarms from mains & EMG; stabilizes baseline | Low (10 lines of MNE) |
| **2** | **Dual-Threshold Hysteresis Triggering** | Fragmented detections, borderline threshold misses | Recovers borderline seizures (improves recall on `PN06`/`PN11`), prevents fragmented FP | Low (pure post-processing) |
| **3** | **Surface Laplacian Spatial Filtering** | Volume-conduction blurring, focal seizure dilution | Sharpen focal seizure onsets, suppress distant muscle artifacts | Medium |
| **4** | **Wavelet Soft-Thresholding (DWT D1/D2)** | Dense muscle EMG bursts | Preserves spikes while suppressing high-frequency muscle noise | Medium |
| **5** | **Robust Running Window Normalization** | Patient amplitude disparities | Invariance to low/high voltage patient baselines | Medium |

---

## 5. Next Steps

As soon as the background perturbation benchmark completes:
1. We will inspect the vulnerability profile from the perturbation benchmark (identifying which technical errors degrade `eventNet` the most: mains, impedance, drift, or muscle).
2. We will select the most impactful signal processing enhancement (e.g. **0.5–40 Hz bandpass + 50 Hz notch** and/or **dual-threshold hysteresis**) and evaluate its ability to:
   - Recover performance under simulated noise.
   - Boost clean baseline detection metrics on both the Good subset and the Hard subset.
