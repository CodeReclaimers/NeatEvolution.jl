#!/bin/bash
# Cleanup script for Inverted Pendulum (CartPole) example generated files
# This removes all output files and directories created by the evolution examples

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning up Inverted Pendulum example output files..."
echo

# Remove individual output files in the directory
FILES_TO_REMOVE=(
    "*.png"
    "*.gif"
    "*.csv"
    "*.json"
)

REMOVED_COUNT=0

for pattern in "${FILES_TO_REMOVE[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            rm -f "$file"
            echo "  ✓ Removed $file"
            REMOVED_COUNT=$((REMOVED_COUNT + 1))
        fi
    done
done

# Remove the output directory and all its contents
if [ -d "output" ]; then
    FILE_COUNT=$(find output -type f 2>/dev/null | wc -l)
    rm -rf output
    echo "  ✓ Removed output/ directory ($FILE_COUNT files)"
    REMOVED_COUNT=$((REMOVED_COUNT + FILE_COUNT))
fi

# Remove any checkpoint files
if [ -d "checkpoints" ]; then
    FILE_COUNT=$(find checkpoints -type f 2>/dev/null | wc -l)
    rm -rf checkpoints
    echo "  ✓ Removed checkpoints/ directory ($FILE_COUNT files)"
    REMOVED_COUNT=$((REMOVED_COUNT + FILE_COUNT))
fi

echo
if [ $REMOVED_COUNT -eq 0 ]; then
    echo "No generated files found. Directory is already clean."
else
    echo "Cleanup complete! Removed $REMOVED_COUNT file(s)."
fi
