# 001 · What does frozen REVE know about EEG artefacts?

**Question.** Without fine-tuning, do REVE embeddings separate clean EEG from different artefact
types? Are those representations useful for downstream EEG analysis?

## Setup
| | |
|---|---|
| Model | REVE-base (`brain-bzh/reve-base`, 69M params), frozen. Adapter: `neuro_expeditions.models.reve` |
| Data | PhysioMotion Artifact, OpenNeuro [ds006386](https://openneuro.org/datasets/ds006386) (CC0). 59-ch, 1000 Hz, healthy adults doing artefact tasks, manual annotations |
| Preprocessing | avg reference, 0.3 Hz high-pass, 50 Hz notch, resample 200 Hz, µV/100, clip ±15 (REVE's own conventions) |
| Windows | 2 s, 1 s stride. Labelled if one group covers ≥50% and no other group covers >10% |
| Classes | `clean_eyes_open`, `clean_eyes_closed`, `blink`, `eye_movement`, `muscle` (chew/eyebrow/tongue/swallow), `head_movement`. Combined labels → `mixed` (excluded) |
| Probe | Standardised logistic regression, leave-one-subject-out |
| Baseline | Log band power, 5 bands × 59 channels |

Originally planned on TUAR. Switched because TUAR needs a Temple data-use registration. Side benefit:
PhysioMotion is likely not in REVE's pretraining corpus, unlike TUH data.

Caveat: "clean" here means resting baseline (eyes open/closed), so clean-vs-artefact is partly
"rest vs task", not only signal contamination.

## Run
```bash
poetry run python expeditions/001-reve-artifacts/extract.py --subjects 1-6 --download   # ~5 GB, ~10 s/run to embed
poetry run jupyter lab expeditions/001-reve-artifacts/explore.ipynb
```
Scale up with `--subjects 1-30` (~25 GB).

Eyes open vs closed follow-up (all 30 subjects are downloaded):
```bash
poetry run python expeditions/001-reve-artifacts/extract_eyes.py --subjects 1-30   # ~10 min
poetry run jupyter lab expeditions/001-reve-artifacts/eyes_open_closed.ipynb
```

## Log

### 2026-10-31 · first pass, subjects 1–6 (11,705 labelled windows, leave-one-subject-out)
| Features | Balanced acc | Macro F1 |
|---|---|---|
| REVE, mean pool | 0.703 | 0.655 |
| REVE, attention pool | **0.710** | **0.658** |
| Band power baseline | 0.689 | 0.621 |

Chance = 0.167.

- **REVE only slightly beats the classical baseline overall**, and per-subject results swing both ways
  (sub-2: REVE 0.62–0.68 vs band power 0.48; sub-3: REVE 0.60–0.64 vs 0.76). Six subjects is too few to call a winner.
- **Where REVE is stronger:** blinks (0.84–0.86 vs 0.69). Muscle is ~0.91 for both.
- **Where it is weaker:** eye movement (0.74–0.78 vs 0.88). REVE confuses eye movements with head movements (~20%).
- **Eyes open vs closed is barely separated by anyone** (~0.44 recall for closed). Surprising given alpha. Suspects:
  sub-5 has no eyes-closed windows, and the closed baseline may contain drowsiness or eye drift. Worth a per-subject alpha check.
- **Annotations are incomplete:** in sub-1 run-01, rhythmic blinks start ~5 s before the annotated blink block, and the
  REVE probe already flags them as blinks. So some "errors" are label noise, not model failures.
- **Subject identity is strong in the embedding:** PCA separates subjects about as clearly as labels.
  Consistent with the 2026 "identity trap" audits. Per-subject normalisation of embeddings is an easy next test.

### 2026-10-31 · follow-up: eyes open vs closed (`eyes_open_closed.ipynb`, `extract_eyes.py`)
28 subjects with both conditions (27,867 windows), leave-one-subject-out, binary probe. REVE embedded twice:
all 59 channels, and posterior only (17 P/PO/O channels, so no direct eye/blink signal).

| Features (per-subject norm) | Balanced acc | AUROC |
|---|---|---|
| Occipital alpha, 1 number | 0.876 | 0.905 |
| Band power, posterior | 0.890 | 0.935 |
| Band power, all channels | 0.892 | 0.948 |
| REVE, posterior only | 0.864 | 0.925 |
| REVE, all channels | **0.918** | **0.960** |

- **The data has textbook alpha:** 27/28 subjects show ~4–25× more occipital alpha with eyes closed.
  Sub-6 shows none (eyes verifiably shut: blinks vanish), a normal low-alpha trait.
- **REVE clearly encodes alpha without being trained to:** a linear readout of the frozen embedding predicts
  within-subject occipital alpha with R² = 0.77 (posterior) / 0.81 (all channels).
- **But from the occipital signal alone it does not beat one hand-computed alpha number** (0.864 vs 0.876,
  median per-subject −0.02, Wilcoxon p < 0.001).
- **REVE's best score comes from combining channels:** all-channel REVE beats its posterior-only self on 26/28
  subjects (p < 1e-6). It is only marginally ahead of the classical features (vs alpha number p = 0.12; vs all-channel band power p = 0.017).
  Clearest case is sub-6 (no alpha): all-channel REVE 0.79, posterior-only 0.53, alpha number 0.30.
  It likely picks up frontal/eye state (no blinks, no saccades) when alpha is absent.
- **Per-subject normalisation matters** (+4–5 points for REVE). Raw embeddings carry strong subject offsets.
- **Label issues:** sub-1 run-01 and sub-26 run-06 "eyes closed" show neither alpha nor fewer blinks. Likely eyes open.
- Why the 6-class probe looked bad on eyes open/closed: only 6 subjects, one without eyes-closed data, one non-alpha
  subject, one mislabelled run, no per-subject normalisation.

**Next ideas:** all 30 subjects; per-subject z-scoring of embeddings; 4 s windows; channel-level embeddings
(does REVE localise blinks to Fp1/Fp2?); a CBraMod adapter for a second foundation model.
