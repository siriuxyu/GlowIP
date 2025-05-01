#!/bin/sh

if [ $# -lt 1 ]; then
  echo "Usage: $0 <mode>"
  exit 1
fi

mode="$1"

# Remove corresponding output files
rm -f output_"$mode".txt
rm -f errput_"$mode".txt
rm -f "${mode}_log.txt"

# Submit the job
bsub < scripts/job_"$mode".sh

# Check bjobs status every 2 seconds
watch -n 2 bjobs

