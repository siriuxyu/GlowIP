#!/bin/bash
#BSUB -J {JOBNAME}
#BSUB -q gpuq
#BSUB -n 4
#BSUB -R "select[ngpus>0] rusage[ngpus_shared=2]"
#BSUB -o /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/{MODE}/output_{JOBNAME}.txt
#BSUB -e /gpfsdata/home/Zhaobo_hengjia21/GlowIP/results/{MODE}/errput_{JOBNAME}.txt

export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/bin:$PATH
export PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$PATH
export LD_LIBRARY_PATH=/gpfsdata/home/Zhaobo_hengjia21/anaconda3/envs/gpu122/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64/:$LD_LIBRARY_PATH

cd /gpfsdata/home/Zhaobo_hengjia21/GlowIP
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu122

python solve_cs.py -experiment {EXPERIMENT} \
        -lr 0.05 -steps 50 \
        -dataset {DATASET} \
        -model {DATASET} \
        -job_id {JOBID} \
        -size {SIZE} \
        -batchsize {BATCHSIZE}  >> {LOGFILE}
