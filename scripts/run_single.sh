if [ $# -lt 5 ]; then
  echo "Usage: $0 <mode> <dataset> <batchsize> <size> <job_id>"
  exit 1
fi

mode="$1"
dataset="$2"
batchsize="$3"
size="$4"
job_id="$5"
job_name="${mode}_${dataset}_${size}_${job_id}"


output_file="results/output_${job_name}.txt"
error_file="results/errput_${job_name}.txt"
log_file="results/${job_name}_log.txt"
temp_script="scripts/job_${job_name}_temp.sh"

if [ "$mode" == "train" ]; then
  script_template="scripts/train_template.sh"
elif [ "$mode" == "solve" ]; then
  script_template="scripts/solve_template.sh"
else
  echo "Invalid mode. Use 'train' or 'solve'."
  exit 1
fi

# Substitute placeholders in the job template
sed "s#{SIZE}#${size}#g; \
     s#{BATCHSIZE}#${batchsize}#g; \
     s#{JOBNAME}#${job_name}#g; \
     s#{DATASET}#${dataset}#g; \
     s#{LOGFILE}#${log_file}#g; \
     s#{JOBID}#${job_id}#g" "$script_template" > "$temp_script"

sed -i 's/\r//' "$temp_script"

# Remove old logs
rm -f "$output_file" "$error_file" "$log_file"

# Submit temp script
bsub < "$temp_script"

# Watch every 2 secs
# watch -n 2 bjobs
