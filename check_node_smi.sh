#!/bin/bash

output_file="nvidia_smi_output.txt"

# Clear the file if it already exists
> "$output_file"

for i in {1..10}; do
    echo -e "======================\nNode ${i}" >> "$output_file"
    srun -w slurmnode${i} nvidia-smi --format=csv --query-gpu=memory.free >> "$output_file"
done