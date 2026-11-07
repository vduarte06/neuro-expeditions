#!/usr/bin/env bash
# scripts/commit_staged.sh
# Creates logical, clear, atomic commits for the neuro-expeditions repository.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== Running in DRY-RUN mode (no commits will be made) ==="
    echo
fi

commit_step() {
    local step_num="$1"
    local message="$2"
    shift 2
    local files=("$@")

    echo "--------------------------------------------------------------------------------"
    echo "Commit #$step_num: $message"
    echo "Files:"
    for f in "${files[@]}"; do
        echo "  + $f"
    done

    if [ "$DRY_RUN" = true ]; then
        echo "(Dry-run: skipping git add and git commit)"
        echo
        return 0
    fi

    # Stage files
    git add "${files[@]}"
    
    # Check if there is anything to commit
    if git diff --cached --quiet; then
        echo "  (No changes staged for this step, skipping)"
    else
        git commit -m "$message"
        echo "  ✓ Committed successfully"
    fi
    echo
}

echo "Starting atomic commits for neuro-expeditions repository..."
echo

# 1. Project setup, config, poetry dependencies, and gitignore
commit_step 1 "chore: initialize project configuration, poetry dependencies, and gitignore" \
    ".gitignore" \
    "pyproject.toml" \
    "poetry.lock" \
    "readme.md"

# 2. Core library base and Expedition 001 (Artifacts & Foundation Models)
commit_step 2 "feat(001): add core library components and expedition 001 reve-artifacts" \
    "src/neuro_expeditions/__init__.py" \
    "src/neuro_expeditions/windows.py" \
    "src/neuro_expeditions/datasets/__init__.py" \
    "src/neuro_expeditions/datasets/physiomotion.py" \
    "src/neuro_expeditions/features/__init__.py" \
    "src/neuro_expeditions/features/bandpower.py" \
    "src/neuro_expeditions/models/__init__.py" \
    "src/neuro_expeditions/models/reve.py" \
    "src/neuro_expeditions/models/cbramod.py" \
    "expeditions/001-reve-artifacts"

# 3. Siena Scalp EEG dataset integration
commit_step 3 "feat(dataset): add Siena Scalp EEG loader and annotation parser" \
    "src/neuro_expeditions/datasets/siena.py"

# 4. Seizure detector model adapters (Gotman 1982, eventNet, SPaRCNet)
commit_step 4 "feat(models): add Gotman 1982, eventNet, and SPaRCNet detector adapters" \
    "src/neuro_expeditions/models/gotman.py" \
    "src/neuro_expeditions/models/eventnet.py" \
    "src/neuro_expeditions/models/sparcnet.py"

# 5. Expedition 002: study protocol, literature review, and clean baseline evaluation
commit_step 5 "feat(002): add seizure robustness protocol, literature survey, and baseline evaluation" \
    "expeditions/002-seizure-robustness/original_study_idea.md" \
    "expeditions/002-seizure-robustness/literature_investigation.md" \
    "expeditions/002-seizure-robustness/evaluate_detectors.py" \
    "expeditions/002-seizure-robustness/fm_probe.py" \
    "expeditions/002-seizure-robustness/README.md"

# 6. Interactive EEG exam visualization tools
commit_step 6 "feat(viz): add interactive EEG exam plotting and inspection tool" \
    "src/neuro_expeditions/visualization.py" \
    "expeditions/002-seizure-robustness/plot_exam.py" \
    "scripts/plot_exam.py"


echo "================================================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run complete. To execute the commits, run:"
    echo "    bash scripts/commit_staged.sh"
else
    echo "All commits created successfully! Git log:"
    echo
    git log --oneline --decorate --graph -n 10
    echo
    echo "Git working tree status:"
    git status
fi
echo "================================================================================"
