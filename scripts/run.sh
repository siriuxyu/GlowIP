#!/bin/sh

# Check arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <mode> <job_id>"
  exit 1
fi

mode="$1"
job_id="$2"

# Compose identifiers
job_name="${mode}_${job_id}"
job_script="scripts/job_${mode}.sh"
output_file="results/output_${job_name}.txt"
error_file="results/errput_${job_name}.txt"
log_file="${job_name}_log.txt"

# Check if the job script exists
if [ ! -f "$job_script" ]; then
  echo "Job script '$job_script' not found!"
  exit 1
fi

# Remove previous logs
rm -f "$output_file" "$error_file" "$log_file"

# Submit the job
bsub < "$job_script"

# Monitor job status
watch -n 2 bjobs

