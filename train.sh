rm output.txt
rm errput.txt
rm py_output.txt
rm ip_log.txt

bsub < job.sh

watch -n 2 bjobs

