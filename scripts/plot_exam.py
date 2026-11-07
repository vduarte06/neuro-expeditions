#!/usr/bin/env python3
"""Convenience launcher for EEG exam plotting."""
import sys
from pathlib import Path

# Add project root and invoke plot_exam main
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "expeditions" / "002-seizure-robustness"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from plot_exam import main

if __name__ == "__main__":
    main()
