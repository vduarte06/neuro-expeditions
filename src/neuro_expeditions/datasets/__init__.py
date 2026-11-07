"""Dataset loaders. Each module downloads, loads and labels one public EEG dataset."""

from . import physiomotion, siena

__all__ = ["physiomotion", "siena"]
