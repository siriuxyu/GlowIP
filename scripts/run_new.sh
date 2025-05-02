f [ $# -lt 2 ]; then
  echo "Usage: $0 <mode> <job_id>"
  exit 1
fi

mode="$1"
job_id="$2"
dataset="$3"
batchsize="$4"
size="$5"
job_name="${mode}_${job_id}"

output_file="results/output_${job_name}.txt"
error_file="results/errput_${job_name}.txt"
log_file="results/${job_name}_log.txt"
script_template="scripts/job_template.sh"
temp_script="scripts/job_${job_name}_temp.sh"

# Substitute 
sed "s/{SIZE}/${size}/g; s/{BATCHSIZE}/${batchsize}/g; s/{JOBNAME}/${job_name}/g; s/{DATASET}/${datasize}/g; s/{LOGFILE}/${log_file}/g" "$script_template" > "$temp_script"

# Remove old logs
rm -f "$output_file" "$error_file" "$log_file"

# Submit temp script
bsub < "$temp_script"

# Watch every 2 secs
watch -n 2 bjobs

