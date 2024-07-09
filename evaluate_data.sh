#!/bin/bash

for file in /dev/shm/outputs/story_experiments/*/base_70.00_util_30.00_long_tasks_532508/data/latency_statistics.txt; do
    echo "Contents of $file:"
    tail "$file"
    echo
done

