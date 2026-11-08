"""SzCORE input format (Dan et al. 2024): the standardised EEG every SzCORE algorithm receives.

19 electrodes of the 10-20 system (old temporal names T3/T4/T5/T6), in uV, 256 Hz (resampy), common average
reference. Built with the official `epilepsy2bids` code (`Eeg.standardize`), starting from an mne.Raw so that
perturbations can be applied upstream.

Used by the detectors that take this format as input: eventNet (trained on it) and Gotman (the official SzCORE
baseline reads it). SPaRCNet and DINO-EEG take the referential signal, as their official code does.
"""

from __future__ import annotations

import mne
from epilepsy2bids.eeg import Eeg

FS = 256
MODERN_TO_OLD = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def to_szcore(raw: mne.io.Raw) -> Eeg:
    """mne.Raw -> official SzCORE-standardised Eeg (19 channels, uV, 256 Hz, common average)."""
    names = {MODERN_TO_OLD.get(c, c).lower(): c for c in raw.ch_names}
    picks = [names[e.lower()] for e in Eeg.ELECTRODES_10_20]
    eeg = Eeg(raw.get_data(picks=picks, units="uV"), list(Eeg.ELECTRODES_10_20), int(raw.info["sfreq"]))
    eeg.standardize(FS, list(Eeg.ELECTRODES_10_20), "Avg")
    return eeg

