#!/bin/bash

# Gzip all CSV files in the specified directories
for csv_file in /dev/shm/outputs/story_experiments/*/*/data/*.csv; do
    if [ -f "$csv_file" ]; then
        echo "Compressing $csv_file..."
        gzip "$csv_file"
    else
        echo "No CSV files found in $csv_file"
    fi
done