#!/bin/bash

# Check if an argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

INPUT_FILE="$1"

# Check if the file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found!"
    exit 1
fi

# Process the file and save the output with the modified filename
OUTPUT_FILE="${INPUT_FILE%.*}_loss.txt"

grep "train_loss=" "$INPUT_FILE" | awk -F'train_loss=|val_loss=' '{print $2 ", " $3}' > "$OUTPUT_FILE"

echo "Processing complete. Output saved to $OUTPUT_FILE"