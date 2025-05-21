if [ $# -lt 5 ]; then
  echo "Usage: $0 <mode> <dataset> <batchsize> <size> <job_id>"
  exit 1
fi

mode="$1"
dataset="$2"
batchsize="$3"
size="$4"
job_id="$5"
coupling_bias="$6"
job_name="${mode}_${dataset}_${size}_${job_id}"


output_file="results/output_${job_name}.txt"
error_file="results/errput_${job_name}.txt"
log_file="results/${job_name}_log.txt"
script_template="scripts/${mode}_template_ddp.sh"
temp_script="scripts/job_${job_name}_temp.sh"


# Substitute placeholders in the job template
sed "s#{SIZE}#${size}#g; \
     s#{BATCHSIZE}#${batchsize}#g; \
     s#{JOBNAME}#${job_name}#g; \
     s#{DATASET}#${dataset}#g; \
     s#{LOGFILE}#${log_file}#g; \
     s#{JOBID}#${job_id}#g; \
     s#{COUPLING_BIAS}#${coupling_bias}#g" \
     "$script_template" > "$temp_script"


sed -i 's/\r//' "$temp_script"

# Remove old logs
rm -f "$output_file" "$error_file" "$log_file"

# Submit as a job array with 4 tasks (for 4 nodes)
bsub -J "${job_name}[1-4]" < "$temp_script"

echo "Submitted job array ${job_name}[1-4] for 4-node distributed training"

