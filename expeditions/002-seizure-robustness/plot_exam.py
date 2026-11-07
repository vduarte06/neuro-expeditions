#!/usr/bin/env python3
"""Plot and interactively inspect EEG exams from the dataset.

Usage (assumes poetry shell inside expeditions/002-seizure-robustness):
    # 1. Interactive list selection (prompts you to pick an exam):
    python plot_exam.py

    # 2. Directly plot an exam by ID or catalog number:
    python plot_exam.py PN00-1
    python plot_exam.py 1

    # 3. Filter by patient:
    python plot_exam.py --patient PN00

    # 4. Use clinical longitudinal bipolar montage (double-banana):
    python plot_exam.py PN00-1 --bipolar

    # 5. Jump to a specific seizure or start time:
    python plot_exam.py PN01-1 --seizure 2
    python plot_exam.py PN00-1 --start 0

    # 6. Save a high-resolution static overview (envelope + seizure trace):
    python plot_exam.py PN00-1 --save overview.png
    python plot_exam.py PN00-1 --overview

    # 7. Print list of all exams and exit:
    python plot_exam.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so neuro_expeditions can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mne
from neuro_expeditions.visualization import (
    format_exam_table,
    format_seconds,
    get_exam_catalog,
    interactive_select_exam,
    load_exam_raw,
    plot_overview,
)


def parse_time_arg(arg: str | float | None) -> float | None:
    """Parse time string like '120', '20m', '1h30m', or '12:30' into seconds."""
    if arg is None:
        return None
    if isinstance(arg, (int, float)):
        return float(arg)
    s = str(arg).strip().lower()
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Unrecognized time format: '{arg}'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and plot EEG exams from the dataset with interactive selection and seizure centering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "exam",
        nargs="?",
        default=None,
        help="Exam ID (e.g. PN00-1), catalog number (e.g. 1), or patient ID (e.g. PN00). If omitted, displays interactive list.",
    )
    parser.add_argument(
        "--id", "-i",
        dest="exam_id",
        default=None,
        help="Exam ID or index (equivalent to positional exam argument).",
    )
    parser.add_argument(
        "--patient", "-p",
        default=None,
        help="Filter exam list to a specific patient (e.g. PN00).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Print table of all available recordings and exit.",
    )
    parser.add_argument(
        "--montage", "-m",
        choices=["referential", "bipolar"],
        default="referential",
        help="EEG montage: 'referential' (19 channels 10-20) or 'bipolar' (18 channels double-banana).",
    )
    parser.add_argument(
        "--bipolar",
        action="store_true",
        help="Convenience shortcut for --montage bipolar.",
    )
    parser.add_argument(
        "--start", "-s",
        default=None,
        help="Start time in seconds or format like '10m' or '12:00'. Default: 10s before seizure onset (or 0 if no seizure).",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=20.0,
        help="Time window duration in seconds (default: 20s).",
    )
    parser.add_argument(
        "--seizure",
        type=int,
        default=1,
        help="Which seizure to jump to if multiple exist (1-based index, default: 1).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="EEG vertical display scaling in µV (default: 100 µV).",
    )
    parser.add_argument(
        "--no-models",
        action="store_true",
        help="Do not overlay precomputed model predictions (eventNet, SPaRCNet, Gotman).",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Display static two-panel overview (full envelope + zoomed trace) instead of MNE browser.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Path to save static overview figure (e.g. 'exam_PN00-1.png').",
    )

    args = parser.parse_args()

    # 1. Load catalog
    catalog = get_exam_catalog("siena")
    if not catalog:
        print("Error: No Siena EEG recordings found in data/siena. Please check data directory.", file=sys.stderr)
        sys.exit(1)

    # 2. If --patient is supplied, filter catalog
    if args.patient:
        pat = args.patient.strip().lower()
        catalog = [item for item in catalog if item["patient"].lower() == pat]
        if not catalog:
            print(f"Error: No recordings found for patient '{args.patient}'.", file=sys.stderr)
            sys.exit(1)

    # 3. If --list is requested, just print table and exit
    if args.list:
        print(format_exam_table(catalog))
        return

    # 4. Resolve exam selection
    preselected = args.exam or args.exam_id
    selected = interactive_select_exam(catalog, preselected=preselected)

    exam_id = selected["id"]
    seizures = selected["seizures"]
    montage = "bipolar" if args.bipolar else args.montage

    print("\n" + "=" * 70)
    print(f" Loading EEG Exam: {exam_id}")
    print("=" * 70)
    print(f" Patient ID      : {selected['patient']}")
    print(f" File            : {selected['file_name']}")
    print(f" Total Duration  : {selected['duration_str']} ({selected['duration_sec']:.1f} s)")
    print(f" Sampling Rate   : {int(selected['sfreq'])} Hz")
    print(f" Montage         : {montage.capitalize()} ({'18 bipolar channels' if montage == 'bipolar' else '19 10-20 channels'})")
    print(f" Ground Truth Sz : {len(seizures)} seizure(s)")
    for idx, sz in enumerate(seizures, start=1):
        print(f"   [{idx}] Onset: {format_seconds(sz['onset'])} ({sz['onset']:.1f}s), Duration: {int(round(sz['duration']))}s")

    preds = selected.get("predictions", {})
    if preds and not args.no_models:
        pred_desc = ", ".join(f"{k}: {v} event(s)" for k, v in preds.items() if v > 0)
        print(f" Model Overlays  : {pred_desc if pred_desc else '0 cached detections'}")

    # Determine start time
    custom_start = parse_time_arg(args.start)
    if custom_start is not None:
        start_time = max(0.0, custom_start)
        print(f" Start Time      : {start_time:.1f}s ({format_seconds(start_time)}) [user specified]")
    elif seizures:
        # Jump to chosen seizure
        sz_idx = max(0, min(len(seizures) - 1, args.seizure - 1))
        target_sz = seizures[sz_idx]
        start_time = max(0.0, target_sz["onset"] - 10.0)
        print(f" Start Time      : {start_time:.1f}s ({format_seconds(start_time)}) [10s before seizure {sz_idx + 1} onset]")
    else:
        start_time = 0.0
        print(" Start Time      : 0.0s (beginning of recording)")

    print("-" * 70)

    # 5. Load raw data
    raw, sz_tuples = load_exam_raw(
        selected["path"],
        montage=montage,
        show_models=not args.no_models,
    )

    # 6. Plotting
    scale_val = args.scale * 1e-6  # µV to V

    # If save or static overview requested
    if args.overview or args.save:
        plot_overview(
            raw=raw,
            seizures=sz_tuples,
            exam_id=exam_id,
            window_start=start_time,
            window_duration=args.duration,
            save_path=args.save,
            show=args.overview or (args.save is None),
        )
        return

    # Interactive MNE raw browser
    print("\nOpening interactive MNE viewer...")
    print("Keyboard shortcuts:")
    print("  • Left / Right Arrow : Scroll backward / forward in time")
    print("  • Page Up / Page Down: Scroll by full page")
    print("  • + / - keys         : Increase / decrease vertical amplitude scale")
    print("  • 'a' key            : Toggle annotation mode")
    print("  • Close window or press 'q' to exit viewer.\n")

    scalings_dict = dict(eeg=scale_val)
    fig = raw.plot(
        start=start_time,
        duration=args.duration,
        n_channels=len(raw.ch_names),
        scalings=scalings_dict,
        title=f"EEG Exam: {exam_id} | {montage.capitalize()} | Seizure @ {format_seconds(seizures[0]['onset']) if seizures else 'None'}",
        show=True,
        block=True,
    )


if __name__ == "__main__":
    main()
