#!/bin/bash

output_file="nvidia_smi_output.txt"

# Clear the file if it already exists
> "$output_file"

for i in {1..10}; do
    # Run the srun command and store the output in a variable
    srun_output=$(srun -w slurmnode${i} nvidia-smi 2>&1)

    # Check if the srun output contains the message "queued and waiting for resources"
    if ! echo "$srun_output" | grep -q "queued and waiting for resources"; then
        echo -e "======================\nNode ${i}\n" >> "$output_file"
        echo "$srun_output" >> "$output_file"
    fi
done