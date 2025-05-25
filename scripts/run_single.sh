# if [ $# -lt 5 ]; then
#   echo "Usage: $0 <mode> <dataset> <batchsize> <size> <job_id>"
#   exit 1
# fi

mode="$1"
dataset="$2"
batchsize="$3"
size="$4"
job_id="$5"
coupling_bias="$6"
job_name="${mode}_${dataset}_${size}_${job_id}"


output_file="results/${mode}/output_${job_name}.txt"
error_file="results/${mode}/errput_${job_name}.txt"
log_file="results/logs/${job_name}_log.txt"
script_template="scripts/${mode}_template.sh"
temp_script="scripts/${mode}/job_${job_name}_temp.sh"

# Substitute placeholders in the job template
sed "s#{SIZE}#${size}#g; \
     s#{MODE}#${mode}#g; \
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

# Submit temp script
bsub < "$temp_script"

