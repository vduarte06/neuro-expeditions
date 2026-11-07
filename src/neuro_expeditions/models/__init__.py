"""Model adapters.

- reve: EEG foundation model encoder (windows -> embeddings)
- gotman: Gotman 1982 classical rule-based seizure detector (raw -> events)
- eventnet: eventNet U-Net deep learning seizure detector (raw -> events)
- sparcnet: SPaRCNet DenseNet deep learning seizure & continuum detector (raw -> events)
"""

from . import eventnet, gotman, reve, sparcnet

__all__ = ["reve", "gotman", "eventnet", "sparcnet"]
