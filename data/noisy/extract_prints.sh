#!/bin/bash

# Conda environment
CONDA_ENV="mvs"

# Input/output folders
INPUT_DIR="noiseprint_clean"
OUTPUT_DIR="print_clean"
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Loop over all PNG files and run Python script
for infile in "$INPUT_DIR"/*.png; do
    filename=$(basename "$infile")
    outfile="$OUTPUT_DIR/$filename"
    python /home/dangeo314/Documents/class/ECE253Project/src/noiseprint/main_extraction.py "$infile" "$outfile"
done

echo "Processing complete. Outputs saved in $OUTPUT_DIR."

