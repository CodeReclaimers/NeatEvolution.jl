#!/bin/bash
# Cleanup script for XOR example generated files
# This removes all output files and directories created by the XOR evolution examples

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Cleaning up XOR example output files..."
echo

# Remove individual output files in the xor directory
FILES_TO_REMOVE=(
    "xor_activation_heatmap.png"
    "xor_evolution.gif"
    "xor_fitness.csv"
    "xor_fitness.png"
    "xor_speciation.csv"
    "xor_species_fitness.csv"
    "xor_species.png"
    "xor_winner.png"
)

REMOVED_COUNT=0

for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ Removed $file"
        REMOVED_COUNT=$((REMOVED_COUNT + 1))
    fi
done

# Remove the output directory and all its contents
if [ -d "output" ]; then
    FILE_COUNT=$(find output -type f | wc -l)
    rm -rf output
    echo "  ✓ Removed output/ directory ($FILE_COUNT files)"
    REMOVED_COUNT=$((REMOVED_COUNT + FILE_COUNT))
fi

echo
if [ $REMOVED_COUNT -eq 0 ]; then
    echo "No generated files found. Directory is already clean."
else
    echo "Cleanup complete! Removed $REMOVED_COUNT file(s)."
fi
