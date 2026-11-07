Neuro-Expeditions

A personal research and experimentation repository at the intersection of neurophysiology, neuroscience, and technology, using computational methods, signal processing, machine learning, and artificial intelligence.

Areas of Exploration

* EEG & Neurophysiology — EEG analysis, sleep studies, seizure detection, brain activity monitoring, and quantitative EEG.
* Foundational Models — Pretrained models for EEG and other biosignals, representation learning, transfer learning, and multimodal approaches.
* Signal Processing — Digital signal processing, spectral analysis, filtering, feature extraction, and time-frequency representations.
* Computational Neuroscience — Computational approaches to understanding neural activity and biological systems.
* AI & Machine Learning — Deep learning, neural networks, LLMs, model evaluation, and intelligent systems for biomedical applications.
* Biomedical Engineering — Connecting clinical knowledge, biosignals, hardware, and software engineering.

Philosophy

This is not intended to be just another collection of tutorials or code examples.

The objective is to understand, experiment, reproduce research, challenge assumptions, and explore how emerging technologies can advance our understanding of the nervous system.

Each expedition may involve:

* Exploring a new research paper or scientific concept.
* Reproducing an existing model or experiment.
* Implementing algorithms and signal-processing techniques.
* Evaluating emerging AI architectures and foundational models.
* Investigating potential clinical and real-world applications.

Repository Structure

The repository will evolve as new experiments are added.

neuro-expeditions/
├── src/neuro_expeditions/      # reusable code
│   ├── datasets/               # one loader per public dataset (download, load, labels)
│   ├── models/                 # one adapter per foundation model (preprocess, embed)
│   ├── features/               # classical baselines
│   └── windows.py
├── expeditions/                # one folder per experiment: README (question, findings log), scripts, notebooks
│   └── 001-reve-artifacts/
└── data/                       # downloads + caches (not committed)

| # | Expedition | Model | Data |
|---|---|---|---|
| 001 | [What does frozen REVE know about EEG artefacts?](expeditions/001-reve-artifacts/) | REVE | PhysioMotion Artifact |
| 002 | [Robustness of seizure detectors to EEG acquisition errors](expeditions/002-seizure-robustness/) | eventNet, SPaRCNet, Gotman | Siena Scalp EEG |
